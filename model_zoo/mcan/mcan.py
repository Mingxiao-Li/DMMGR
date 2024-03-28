from x.modules.attention import SelfAttention, CrossAttention
from x.modules.linear import MLP, LayerNorm
from x.core.registry import registry
from x.common.util import get_numpy_word_embed
import torch.nn.functional as F
import torch.nn as nn
import torch
import os 
import json 


class SA(nn.Module):
    def __init__(self, config):
        super(SA, self).__init__()
        self.self_attention = SelfAttention(
            hidden_size=config.hidden_size,
            multi_head=config.multi_head,
            hidden_size_head=config.hidden_size_head,
            dropout_r=config.dropout_r,
            use_ffn=True,
            mid_size=config.mid_size,
        )

    def forward(self, x, mask):
        return self.self_attention(x, mask)


class SGA(nn.Module):
    def __init__(self, config):
        super(SGA, self).__init__()

        self.experimental_setup = config.experimental_setup
        self.self_attention = SelfAttention(
            hidden_size=config.hidden_size,
            multi_head=config.multi_head,
            hidden_size_head=config.hidden_size_head,
            dropout_r=config.dropout_r,
            use_ffn=False,
        )

        self.ques_img_att = CrossAttention(
            hidden_size=config.hidden_size,
            multi_head=config.multi_head,
            hidden_size_head=config.hidden_size_head,
            dropout_r=config.dropout_r,
            mid_size=config.mid_size,
            use_ffn=self.experimental_setup != "kg_include",
            act_fun="relu",
        )
        if self.experimental_setup == "kg_include":
            self.kg_img_att = CrossAttention(
                hidden_size=config.hidden_size,
                multi_head=config.multi_head,
                hidden_size_head=config.hidden_size_head,
                dropout_r=config.dropout_r,
                use_ffn=True,
                act_fun="relu",
                mid_size=config.mid_size,
            )

    def forward(self, ques_feat, img_feat, kg_feat, ques_mask, kg_mask):

        x = self.self_attention(img_feat)

        x = self.ques_img_att(ques_feat, ques_feat, x, ques_mask)

        if self.experimental_setup == "kg_include":
            x = self.kg_img_att(kg_feat, kg_feat, x, kg_mask)

        return x


class AttFlat(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        flat_mlp_size: int,
        flat_glimpses: int,
        flat_outsize: int,
        dropout_r: float,
        act_fn: str = "relu",
    ):
        super(AttFlat, self).__init__()

        self.flat_glimpses = flat_glimpses
        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            dropout_r=dropout_r,
            act_fun=act_fn,
        )

        self.linear_merge = nn.Linear(hidden_size * flat_glimpses, flat_outsize)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2) == 1, 1e-32)
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(torch.sum(att[:, :, i : i + 1] * x, dim=1))
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)
        return x_atted


class MCA_ED(nn.Module):
    def __init__(self, config):
        super(MCA_ED, self).__init__()
        self.enc_list = nn.ModuleList([SA(config) for _ in range(config.num_layers)])
        self.dec_list = nn.ModuleList([SGA(config) for _ in range(config.num_layers)])

    def forward(self, ques, ques_mask, kg, kg_mask, img_feat):

        for enc in self.enc_list:
            ques = enc(ques, ques_mask)

        for dec in self.dec_list:
            img_feat = dec(ques, img_feat, kg, ques_mask, kg_mask)

        return ques, img_feat


@registry.register_model(name="MCANet")
class MACNet(nn.Module):
    def __init__(self, config, pretrained_embed=None):
        super(MACNet, self).__init__()

        if not config.use_glove_emb:
            self.embedding = nn.Embedding(config.vocab_size, config.word_embed_size)
        elif config.use_glove_emb:
            word2index_path = os.path.join(config.parent_path, config.word2index_path)
            assert os.path.exists(word2index_path)

            with open(word2index_path, "r") as f:
                word2index = json.load(f)

            pretrained_word_emb = get_numpy_word_embed(
                word2index,
                os.path.join(config.parent_path, config.pretrained_word_path),
            )

            num_words, word_dim = pretrained_word_emb.shape

            self.embedding = nn.Embedding(num_words, word_dim)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_word_emb))
            self.embedding.weight.requires_grad = config.fine_tune

        self.lstm = nn.LSTM(
            input_size=config.word_embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.rnn_num_layers,
            batch_first=True,
        )

        self.img_feat_linear = nn.Linear(config.img_feat_size, config.hidden_size)
        self.kg_feat_liner = nn.Linear(config.word_embed_size, config.hidden_size)

        self.backbone = MCA_ED(config)

        self.attflat_img = AttFlat(
            hidden_size=config.hidden_size,
            flat_mlp_size=config.flat_mlp_size,
            flat_outsize=config.flat_out_size,
            flat_glimpses=config.flat_glimpses,
            dropout_r=config.dropout_r,
        )

        self.attflat_lang = AttFlat(
            hidden_size=config.hidden_size,
            flat_mlp_size=config.flat_mlp_size,
            flat_outsize=config.flat_out_size,
            flat_glimpses=config.flat_glimpses,
            dropout_r=config.dropout_r,
        )

        self.proj_norm = LayerNorm(config.flat_out_size)
        self.proj = nn.Linear(config.flat_out_size, config.answer_size)

    def forward(self, ques, kg, ques_mask, kg_mask, img_feat):

        # Pre-process language feature
        lang_feat = self.embedding(ques)
        lang_feat, _ = self.lstm(lang_feat)

        if kg is not None:
            kg_feat = self.embedding(kg)
            kg_feat = self.kg_feat_liner(kg_feat)
        else:
            kg_feat = None
        img_feat = self.img_feat_linear(img_feat)

        lang_feat, img_feat = self.backbone(
            lang_feat, ques_mask, kg_feat, kg_mask, img_feat
        )

        lang_feat = self.attflat_lang(lang_feat, ques_mask)
        img_feat = self.attflat_img(img_feat, None)

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat
