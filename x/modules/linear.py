import torch.nn as nn
import torch.nn.functional as F
import torch
from x.modules.utils import gelu_new, gelu, swish

ACT2FN = {
    "gelu": gelu,
    "gelu_new": gelu_new,
    "relu": F.relu,
    "swish": swish,
    "tanh": F.tanh,
}


class FC(nn.Module):
    r"""
    A fully connected layer (can choose with or without using dropout and activation function)
    relu is default activation function
    """

    def __init__(
        self, in_size: int, out_size: int, dropout_r: float = 0, act_fun: str = "relu"
    ):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.act_fun = act_fun

        self.linear = nn.Linear(in_size, out_size)
        if self.act_fun is not None:
            self.activation = ACT2FN[self.act_fun]

        if self.dropout_r > 0:
            self.dropout = nn.Dropout(self.dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.act_fun is not None:
            x = self.activation(x)

        if self.dropout_r > 0:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    r"""
    A layer with two linear layer
    x => linear layer(one) + act_fun + (dropout) + linear layer (two)
    input_size -> mid_size -> output_size
    """

    def __init__(
        self,
        in_size: int,
        mid_size: int,
        out_size: int,
        dropout_r: float = 0,
        act_fun: str = "relu",
    ):

        super(MLP, self).__init__()
        self.fc = FC(
            in_size=in_size, out_size=mid_size, dropout_r=dropout_r, act_fun=act_fun
        )
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    r"""
    Layer normalization
    """

    def __init__(self, size: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FFN(nn.Module):
    r"""
    FeedFord net in transformer (including skip connection)
    """

    def __init__(
        self,
        in_size: int,
        mid_size: int,
        out_size: int,
        dropout_r: float = 0,
        act_fun: str = "relu",
    ):
        super(FFN, self).__init__()
        self.mlp = MLP(
            in_size=in_size,
            mid_size=mid_size,
            out_size=out_size,
            dropout_r=dropout_r,
            act_fun=act_fun,
        )

        self.norm = LayerNorm(out_size)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        return self.norm(x + self.dropout(self.mlp(x)))
