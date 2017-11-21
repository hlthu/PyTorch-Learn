# -*- coding: utf-8 -*-
"""
This is a simple example of DNN using Pytorch's Tensor
"""

import torch

# data type: CPU or GPU
# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

# N batch: size
# D_in: input size
# D_out: output size
# H: hidden size
N, D_in, H, D_out = 16, 1024, 256, 128

# create random input and output
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# define two weight matrx
wh = torch.randn(D_in, H).type(dtype)
wo = torch.randn(H, D_out).type(dtype)

# Training
learning_rate = 1e-6
for t in range(1000):
    # forward
    h = x.mm(wh)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(wo)

    # loss
    loss = (y_pred - y).pow(2).sum()
    if t%100 == 99:
        print (t+1, loss)

    # backup
    grad_y_pred = 2.0 * (y_pred - y)
    grad_wo = h_relu.t().mm(grad_y_pred) 
    grad_h_relu = grad_y_pred.mm(wo.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_wh = x.t().mm(grad_h)

    # update
    wh -= learning_rate * grad_wh
    wo -= learning_rate * grad_wo
