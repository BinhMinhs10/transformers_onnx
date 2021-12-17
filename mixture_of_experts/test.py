import torch
from torch import nn
from torch.optim import Adam
from mixture_of_experts.mlp import MLP
from mixture_of_experts.moe import MoE


def train(x, y, model, loss_fn, optim):
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model


def eval(x, y, model, loss_fn):
    model.eval()
    y_hat, aux_loss = model(x.float())
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))


def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x, y


input_size = 1000
num_classes = 20
num_experts = 10
hidden_size = 64
batch_size = 5
k = 4

model = MLP(input_size, output_size=num_classes, hidden_size=hidden_size)
# model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True)


loss_fn = nn.NLLLoss()
optim = Adam(model.parameters())

x, y = dummy_data(batch_size, input_size, num_classes)

model = train(x, y, model, loss_fn, optim)

x, y = dummy_data(batch_size, input_size, num_classes)
eval(x, y, model, loss_fn)

