# -*- coding: utf-8 -*-
"""
This is a simple example of DNN using Pytorch's Variable and autograd
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
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# define two weight matrixes
wh = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
wo = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

# Trainig
learning_rate = 1e-6
for t in range(1000):    
    # forward
    h = x.mm(wh)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(wo)

    # loss
    loss = (y_pred - y).pow(2).sum()
    if t%100 == 99:
        print(t+1, loss.data[0])

    # using autograd to backward
    loss.backward()

    # update
    wh.data -= learning_rate * wh.grad.data
    wo.data -= learning_rate * wo.grad.data

    # clear the grad of weight matrx
    wh.grad.data.zero_()
    wo.grad.data.zero_()


