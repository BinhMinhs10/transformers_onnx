from d2l import torch as d2l
import torch
from torch import nn


# decoder output fro previous t-1 used as query
class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 dropout=0,
                 **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = d2l.MLPAttention(
            num_hiddens,
            num_hiddens,
            num_hiddens,
            dropout
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        # transpose outputs to (batch_size, seq_len, num_hiddens)
        return (outputs.permute(1, 0, 2),
                hidden_state,
                enc_valid_len)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs = []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention_cell(
                query, enc_outputs, enc_outputs, enc_valid_len
            )
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_len]


if __name__ == '__main__':
    encoder = d2l.Seq2SeqEncoder(vocab_size=10,
                                 embed_size=8,
                                 num_hiddens=16,
                                 num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10,
                                      embed_size=8,
                                      num_hiddens=16,
                                      num_layers=2)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    out, state = decoder(X, state)
    print(
        out.shape, len(state),
        state[0].shape, len(state[1]),
        state[1][0].shape
    )
