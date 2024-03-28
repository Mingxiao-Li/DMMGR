import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from x.modules.linear import MLP, LayerNorm, FFN
import pdb

class SelfAttention(nn.Module):
    r"""
    Regular self attention module in transformer
    """

    def __init__(
        self,
        hidden_size: int,
        multi_head: int,
        hidden_size_head: int,
        dropout_r: float,
        use_ffn: bool = True,
        mid_size: int = None,
        act_fun: str = "relu",
        output_attn=False,
    ):
        super(SelfAttention, self).__init__()

        self.output_attn = output_attn
        self.mhatt = MultiHeadAttention(
            hidden_size=hidden_size,
            multi_head=multi_head,
            hidden_size_head=hidden_size_head,
            dropout_r=dropout_r,
        )
        self.use_ffn = use_ffn
        if self.use_ffn:
            assert mid_size is not None, "Mid size should not be None"
            assert act_fun is not None, "Activitaion function should not be None"
            self.ffn = FFN(
                in_size=hidden_size,
                mid_size=mid_size,
                out_size=hidden_size,
                dropout_r=dropout_r,
                act_fun=act_fun,
            )

        self.dropout = nn.Dropout(dropout_r)
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, mask=None):

        x1, attn = self.mhatt(x, x, x, mask)
        x = self.norm(x + self.dropout(x1))
        if self.use_ffn:
            x = self.ffn(x)

        if self.output_attn:
            return x, attn
        return x


class MultiHeadAttention(nn.Module):
    r"""
    Regular multihead attention in transformers
    """

    def __init__(self, hidden_size, multi_head, hidden_size_head, dropout_r):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.multi_head = multi_head
        self.hidden_size_head = hidden_size_head

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        # k: key, v: value q:query
        batch_size = q.size(0)
        v = (
            self.linear_v(v)
            .view(batch_size, -1, self.multi_head, self.hidden_size_head)
            .transpose(1, 2)
        )
        # origin shape (batch_size, sequence_len, hidden_size)
        # v shape (batch_size, sequence_len, multi_head, hidden_size_head).transpose(1,2)
        # v shape (batch_size, multi_head, sequence_len, hidden_size_head)

        k = (
            self.linear_k(k)
            .view(batch_size, -1, self.multi_head, self.hidden_size_head)
            .transpose(1, 2)
        )

        q = (
            self.linear_q(q)
            .view(batch_size, -1, self.multi_head, self.hidden_size_head)
            .transpose(1, 2)
        )

        output, attn = self.attention(v, k, q, mask)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        )
        output = self.linear_merge(output)
        return output, attn

    def attention(self, value, key, query, mask):
        d_k = query.size(-1)  # hideen_size_head

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # query shape =[batch_size, multi_head, sequence_len hidden_size_head]
        # key.transpose(-2,-1) shape =[batch_size, multi_head, hidden_size_head, sequence_len)
        # score shape = [batch_size, multi_head, sequence_len, sequence_len

        if mask is not None:
            # mask = [0,0,0,1,1,1]
            mask = mask.unsqueeze(1).unsqueeze(2).expand(scores.shape)
            scores = scores.masked_fill(mask == 1, -1e32)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        output = torch.matmul(att_map, value)
        return output, att_map


class CrossAttention(nn.Module):
    r"""
    A cross attention module
    --> To be able to stack any number of cross attention
    eg: cross_atten(x1,x1,y,m) + cross_attn(x2,x2,y,m)+ ....
    * if ffn is used, then it is a regular cross attention module in transoformer
    k,v are from one source and q is from another source
    """

    def __init__(
        self,
        hidden_size: int,
        multi_head: int,
        hidden_size_head: int,
        dropout_r: float,
        use_ffn: bool = False,
        mid_size: int = None,
        act_fn: str = None,
        output_attn=False,
    ):
        super(CrossAttention, self).__init__()

        self.output_attn = output_attn
        self.mhatt = MultiHeadAttention(
            hidden_size=hidden_size,
            multi_head=multi_head,
            hidden_size_head=hidden_size_head,
            dropout_r=dropout_r,
        )
        self.dropout = nn.Dropout(dropout_r)
        self.norm = LayerNorm(hidden_size)

        self.use_ffn = use_ffn
        if self.use_ffn:
            assert mid_size is not None, "Mid size should not be None"
            assert act_fn is not None, "Activigation function should not be None"
            self.ffn = FFN(
                in_size=hidden_size,
                mid_size=mid_size,
                out_size=hidden_size,
                dropout_r=dropout_r,
                act_fun=act_fn,
            )

    def forward(self, x_v, x_k, y_q, mask):

        y1, attn = self.mhatt(x_v, x_k, y_q, mask)
        y = self.norm(y_q + self.dropout(y1))

        if self.use_ffn:
            y = self.ffn(y)
        if self.output_attn:
            return y, attn
        return y
