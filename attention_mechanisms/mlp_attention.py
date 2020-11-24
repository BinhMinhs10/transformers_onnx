import torch.nn as nn
import torch
from batch_dot_product import masked_softmax


class MLPAttention(nn.Module):
    def __init__(self, key_size, query_size, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # y = x * A^T + b
        self.w_k = nn.Linear(key_size, units, bias=False)
        self.w_q = nn.Linear(query_size, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len):
        # expand query to (batch_size, queries, 1, units)
        # key to (batch_size, 1, kv_pairs, units)
        query, key = self.w_q(query), self.w_k(key)
        features = query.unsqueeze(2) + key.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.v(features).squeeze(-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return torch.bmm(attention_weights, value)


if __name__ == "__main__":
    atten = MLPAttention(key_size=2, query_size=2, units=8, dropout=0.1)
    atten.eval()
    keys = torch.ones(2, 10, 2)
    # shape = (2, 10, 4)
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

    result = atten(torch.ones(2, 1, 2), keys, values, torch.tensor([2, 6]))
    print(result)
