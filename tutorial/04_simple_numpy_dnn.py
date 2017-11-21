# -*- coding: utf-8 -*-
"""
This is a simple example of DNN using numpy
"""

import numpy as np 

# N batch: size
# D_in: input size
# D_out: output size
# H: hidden size
N, D_in, H, D_out = 64, 256, 128, 16

# create random input and output
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly init the weight of hidden layer and output layer
wh = np.random.randn(D_in, H)
wo = np.random.randn(H, D_out)

# forward and backword
learning_rate = 1e-6
for t in range(1000):
    ## forword
    h = x.dot(wh)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(wo)

    ## get loss
    loss = np.square(y_pred - y).sum()
    if t%100 == 99:
        print (t+1, loss)

    ## backword
    grad_y_pred = 2 * (y_pred - y)
    grad_wo = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(wo.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_wh = x.T.dot(grad_h)

    ## update
    wh -= learning_rate * grad_wh
    wo -= learning_rate * grad_wo


