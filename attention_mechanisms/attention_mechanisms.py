import torch
from batch_dot_product import masked_softmax


tensor = torch.rand(2, 3, 4)
# x = torch.tensor([[1, 2, 3, 4]])
# print(x.unsqueeze(1))
print(tensor)
print(tensor.unsqueeze(1))
print(masked_softmax(tensor, torch.tensor([2, 3])))

