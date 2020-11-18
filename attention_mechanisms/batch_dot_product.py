import torch
import torch.nn as nn
import math
from d2l import torch as d2l


def masked_softmax(X, valid_len):
    """Perform softmax by filtering out some elements."""
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            # repeat value [2, 2, 3, 3]
            valid_len = torch.repeat_interleave(
                valid_len,
                repeats=shape[1],
                dim=0
            )
        else:
            valid_len = valid_len.reshape(-1)
        # print(X.reshape(-1, shape[-1]))
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_len, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# multiple matrix
# print(torch.ones(2, 1, 3))
# result = torch.bmm(torch.ones(2, 1, 3), torch.ones(2, 3, 2))
# print(result.shape)
# print(result)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len=None):
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        print(attention_weights.shape)
        print(values.shape)
        return torch.bmm(attention_weights, value)


if __name__ == "__main__":
    atten = DotProductAttention(dropout=0.5)
    atten.eval()
    keys = torch.ones(2, 10, 2)
    # shape = (2, 10, 4)
    print(torch.arange(40, dtype=torch.float32))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    result = atten(torch.ones(2, 1, 2), keys, values, torch.tensor([2, 6]))
    print(result)
