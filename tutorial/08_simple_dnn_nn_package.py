# -*- coding: utf-8 -*-
"""
This is a simple example of DNN using Pytorch's nn package
"""

import torch
from torch.autograd import Variable

# data type: CPU or GPU
# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

# N batch: size
# D_in: input size
# D_out: output size
# H: hidden size
N, D_in, H, D_out = 16, 1024, 256, 128

# create input and output data
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# using nn.Sequential() to create a model instance
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
# send the model to GPU
model.cuda()

# define a MSE loss function using nn
loss_fn = torch.nn.MSELoss(size_average=False)

# Training
learning_rate = 1e-5
for t in range(1000):
    # forward
    y_pred = model(x)

    # loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t+1, loss.data[0])
    
    # zeros the grads
    model.zero_grad()

    # backward
    loss.backward()

    # update the parameters in the model
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

