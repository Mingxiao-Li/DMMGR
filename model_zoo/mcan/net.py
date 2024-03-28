import torch.nn as nn
import torch
import torch.nn.functional as F

from model_zoo.mcan.net_utils import MLP, LayerNorm
from backup_code.mca_ import MCA_ED
from x.core.registry import registry

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------
class AttFlat(nn.Module):
    def __init__(self, config):
        super(AttFlat, self).__init__()
        self.config = config

        self.mlp = MLP(
            in_size=config.hidden_size,
            mid_size=config.flat_mlp_size,
            out_size=config.flat_glimpses,
            dropout_r=config.dropout_r,
            use_relu=True,
        )

        self.linear_merge = nn.Linear(
            config.hidden_size * config.flat_glimpses, config.flat_out_size
        )

    def forward(self, x, x_mask):

        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2) == 1, -1e9)

        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.config.flat_glimpses):
            att_list.append(torch.sum(att[:, :, i : i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------
@registry.register_model(name="MCAN")
class Net(nn.Module):
    def __init__(self, config, pretrained_embed=None):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.token_size, embedding_dim=config.word_embedsize
        )

        # Loading the GloVe embedding weights:
        if config.use_glove:
            assert (
                pretrained_embed is not None
            ), "Pretraiend emebding should not be None"
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.lstm = nn.LSTM(
            input_size=config.word_embedsize,
            hidden_size=config.hidden_size,
            num_layers=config.rnn_num_layers,
            batch_first=True,
        )

        self.img_feat_linear = nn.Linear(config.img_feat_size, config.hidden_size)

        self.loc_feat_linear = nn.Linear(config.img_loc_size, config.hidden_size)

        self.kg_proj = nn.Linear(config.word_embedsize, config.hidden_size)
        self.backbone = MCA_ED(config)

        self.attflat_img = AttFlat(config)
        self.attflat_lang = AttFlat(config)

        self.proj_norm = LayerNorm(config.flat_out_size)
        self.proj = nn.Linear(config.flat_out_size, config.answer_size)

    def forward(self, ques, ques_mask, kg, kg_mask, img_feat, img_loc):

        # Pre-process language feature
        lang_feat = self.embedding(ques)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre_process kg triplet feature
        kg_feat = self.embedding(kg)
        kg_feat = self.kg_proj(kg_feat)
        # kg_feat, _ =  self.lstm(kg_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)
        img_loc_feat = self.loc_feat_linear(img_loc)
        img_feat = img_feat + img_loc_feat

        lang_feat, img_feat = self.backbone(
            lang_feat,
            ques_mask,
            kg_feat,
            kg_mask,
            img_feat,
        )

        lang_feat = self.attflat_lang(lang_feat, ques_mask)
        img_feat = self.attflat_img(img_feat, None)

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat
