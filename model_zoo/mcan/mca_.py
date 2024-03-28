from model_zoo.mcan.net_utils import FC, MLP, LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
class MHAtt(nn.Module):
    def __init__(self, config):
        super(MHAtt, self).__init__()
        self.config = config

        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_merge = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = (
            self.linear_v(v)
            .view(n_batches, -1, self.config.multi_head, self.config.hidden_size_head)
            .transpose(1, 2)
        )

        k = (
            self.linear_k(k)
            .view(n_batches, -1, self.config.multi_head, self.config.hidden_size_head)
            .transpose(1, 2)
        )

        q = (
            self.linear_q(q)
            .view(n_batches, -1, self.config.multi_head, self.config.hidden_size_head)
            .transpose(1, 2)
        )

        atted = self.att(v, k, q, mask)
        atted = (
            atted.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.config.hidden_size)
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask = [0,0,0,1,1,1]
            mask = mask.unsqueeze(1).unsqueeze(2).expand(scores.shape)
            scores = scores.masked_fill(mask == 1, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------
class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=config.hidden_size,
            mid_size=config.ff_size,
            out_size=config.hidden_size,
            dropout_r=config.dropout_r,
            use_relu=True,
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------
class SA(nn.Module):
    def __init__(self, config):
        super(SA, self).__init__()

        self.mhatt = MHAtt(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.dropout_r)
        self.norm1 = LayerNorm(config.hidden_size)

        self.dropout2 = nn.Dropout(config.dropout_r)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(self, x, x_mask):

        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------
class SGA(nn.Module):
    def __init__(self, config):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(config)
        self.mhatt2 = MHAtt(config)
        self.mhatt3 = MHAtt(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.dropout_r)
        self.norm1 = LayerNorm(config.hidden_size)

        self.dropout2 = nn.Dropout(config.dropout_r)
        self.norm2 = LayerNorm(config.hidden_size)

        self.dropout3 = nn.Dropout(config.dropout_r)
        self.norm3 = LayerNorm(config.hidden_size)

        self.dropout4 = nn.Dropout(config.dropout_r)
        self.norm4 = LayerNorm(config.hidden_size)

    def forward(self, img_feat, ques, ques_mask, kg, kg_mask):
        # image self attention
        x = img_feat
        y = ques
        z = kg
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, None)))

        # question image cross attention
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, ques_mask)))

        # knowledge image cross attention
        x = self.norm3(x + self.dropout3(self.mhatt3(z, z, x, kg_mask)))

        x = self.norm4(x + self.dropout4(self.ffn(x)))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------
class MCA_ED(nn.Module):
    def __init__(self, config):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(config) for _ in range(config.num_layers)])
        self.dec_list = nn.ModuleList([SGA(config) for _ in range(config.num_layers)])

    def forward(self, ques, ques_mask, kg, kg_mask, img_feat):
        x = ques
        y = img_feat
        z = kg

        for enc in self.enc_list:
            x = enc(x, ques_mask)

        for dec in self.dec_list:
            y = dec(y, x, ques_mask, z, kg_mask)

        return x, y
