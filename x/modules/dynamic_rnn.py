import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb


class DynamicRNN(nn.Module):
    """
    Example:
        lstm = nn.LSTM()
        rnn = DynamicRNN(lstm)
    """

    def __init__(self, rnn_model, output_last_layer):
        super().__init__()
        self.rnn_model = rnn_model
        self.output_last_layer = output_last_layer

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.
        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.
        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True
        )

        if initial_state is not None:
            hx = initial_state
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None

        self.rnn_model.flatten_parameters()
        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        h_n = h_n.index_select(dim=1, index=bwd_order)
        c_n = c_n.index_select(dim=1, index=bwd_order)

        outputs, lens_unpack = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )

        return outputs, lens_unpack, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().view(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
