import torch
import torch.nn as nn
import math 

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        text_vocab_embedding,
        text_emb_dim,
        n_input,
        n_head,
        n_hid,
        n_layers,
        dropout=0.5,
    ):
        super(TransformerEncoder, self).__init__()
        self.n_input = n_input
        self.text_vocab_embedding = text_vocab_embedding
        self.emb_proj = nn.Linear(text_emb_dim, n_input)
        self.pos_encoder = PositionalEncoding(n_input, dropout)
        encoder_layers = nn.TransformerEncoderLayer(n_input, n_head, n_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, n_layers, norm=nn.LayerNorm(n_input)
        )
    
    def forward(self, src):
        
        src = self.text_vocab_embedding(src)
        src = self.emb_proj(src) * math.sqrt(self.n_input)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output