import torch.nn as nn
import torch
from x.modules.dynamic_rnn import DynamicRNN
from x.core.registry import registry
from x.modules.attention import SelfAttention, CrossAttention


@registry.register_model(name="Seq2Seq")
class Seq2Seq(nn.Module):

    def __init__(self, config, pretrain_embed = None):
        super(Seq2Seq, self).__init__()
        self.config = config
        encoder = registry.get("Encoder", config.encoder.name)
        decoder = registry.get("Decoder", config.decoder.name)

        self.encoder = encoder(config.encoder, pretrain_embed)
        self.decoder = decoder(config.decoder)

    def forward(self, en_input_seq, de_input_seq):
        if "Rnn" in self.config.encoder.name:
            _, hidden, cell = self.encoder(en_input_seq)
            decoder_output = self.decoder(de_input_seq, hidden,cell)
            return decoder_output
        elif "Transformer" in self.config.encoder.name:
            return 1
        else:
            raise ValueError("The encoder name should either contain 'Rnn' or 'Transformer' ")

@registry.register(type="Encoder", name="RnnEncoder")
class RNNEncoder(nn.Module):

    def __init__(self, config, pretrain_embed):
        super(RNNEncoder,self).__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.embeded_size)
        if config.use_glove_emb:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrain_embed))

        assert config.rnn_type in ["lstm", "gru"],"rnn_type {} should be 'lstm' or 'gru'".format(config.rnn_type)
        if config.rnn_type == "lstm":
            rnn = nn.LSTM(input_size = config.input_size,
                          hidden_size = config.hidden_size,
                          num_layers = config.num_layers,
                          bidirectional = config.bidirectional,
                          dropout = config.dropout,
                          batch_first = True)
        elif config.rnn_type == "gru":
            rnn = nn.GRU(input_size = config.input_size,
                         hidden_size = config.hidden_size,
                         num_layers = config.num_layers,
                         bidriectional = config.bidriectional,
                         dropout = config.dropout,
                         batch_first = True)

        self.encoder_rnn = DynamicRNN(rnn, output_last_layer=False)

    def forward(self, input_seq):
        # input_seq: batch, length
        # seq_len: batch,
        seq_len = torch.sum(input_seq != 1,dim=1)
        word_emb = self.word_embedding(input_seq)
        output, (h,c) = self.encoder_rnn(word_emb, seq_len)
        return output,h,c


@registry.register(type="Decoder", name="RnnDecoder")
class RNNDecoder(nn.Module):

    def __init__(self, config):

        super(RNNDecoder,self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embeded_size)
        # out_put size -> number token numbers
        if config.rnn_type == "lstm":
            rnn = nn.LSTM(input_size = config.input_size,
                          hidden_size = config.hidden_size,
                          num_layers = config.num_layers,
                          dropout = config.dropout,
                          bidirectional = False,
                          batch_first = True)

        elif config.rnn_type == "gru":
            rnn = nn.GRU(input_size = config.input_size,
                         hidden_size = config.hidden_size,
                         num_layers = config.num_layers,
                         dropout = config.dropout,
                         bidriectional = False,
                         batch_first = True)
        self.fc_out = nn.Linear(config.hidden_size, config.vocab_size)
        if config.use_emb_out:
            self.fc_out.weight.data.copy_(self.embedding.weight.data.permute(1,0))
        self.decoder = rnn

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input)
        output, (_, _) = self.decoder(embedded,(hidden,cell))
        prediction = self.fc_out(output)
        return prediction


@registry.register(type="Encoder", name="TransformerEncoder")
class TransformerEncoder(nn.Module):

    def __init__(self):
        super(TransformerEncoder, self).__init__()

    def forward(self):
        pass


@registry.register(type="Decoder", name="TransformerDecoder")
class TransformerDecoder(nn.Module):

    def __init__(self):
        super(TransformerDecoder, self).__init__()

    def forward(self):
        pass



class GPT(nn.Module):

    def __init__(self):
        super(GPT,self).__init__()

    def forward(self):
        pass


