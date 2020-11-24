"""component transformer architecture"""
# transformer differs seq2seq with attention
# transformer block: RNN replaced by transformer block (multi-head + position wise feed forward)
# position encoding: add seq infor each item\

from d2l import torch as d2l
import math
import torch
from torch import nn


# self-attention outputs a same-length sequential output for each input item
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def transpose_qkv(self, X, num_heads):
        # batch_size, seq_len, num_heads, num_hiddens/num_heads
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        # batch_size, num_heads, seq_len, num_hiddens/num_heads
        X = X.permute(0, 2, 1, 3)
        output = X.reshape(-1, X.shape[2], X.shape[3])
        return output

    def transpose_output(self, X, num_heads):
        # reversed version of transpose_qkv
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, query, key, value, valid_len):
        query = self.transpose_qkv(self.W_q(query), self.num_heads)
        key = self.transpose_qkv(self.W_q(key), self.num_heads)
        value = self.transpose_qkv(self.W_q(value), self.num_heads)

        if valid_len is not None:
            if valid_len.ndim == 1:
                valid_len = valid_len.repeat(self.num_heads)
            else:
                valid_len = valid_len.repeat(self.num_heads, 1)
        output = self.attention(query, key, value, valid_len)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input,
                 ffn_num_hiddens, pw_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, pw_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape,
                 dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens,
                 dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000,
                      torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

if __name__ == "__main__":
    cell = MultiHeadAttention(5, 5, 5, 100, 10, 0.5)
    cell.eval()
    X = torch.ones((2, 4, 5))
    valid_len = torch.tensor([2, 3])
    print(cell(X, X, X, valid_len).shape)

    # position-wise feed forward
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    print(ffn(torch.ones((2, 3, 4)))[0])

    # addnorm
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    print(
        add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape
    )

    # position encoding
    pe = PositionalEncoding(20, 0)
    pe.eval()
    Y = pe(torch.zeros((1, 100, 20)))
    d2l.plot(torch.arange(100), Y[0, :, 4:8].T, figsize=(6, 2.5),
             legend=["dim %d" % p for p in [4, 5, 6, 7]])