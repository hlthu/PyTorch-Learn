# -*- coding: utf-8 -*-
"""
This is a simple example of DNN using Pytorch's Variable and autograd

In PyTorch we can easily define our own autograd operator by 
defining a subclass of torch.autograd.Function and implementing 
the forward and backward functions. We can then use our new 
autograd operator by constructing an instance and calling it 
like a function, passing Variables containing input data.

In this example we define our own custom autograd function for
performing the ReLU nonlinearity, and use it to implement our
two-layer network:
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

# define a class containning backward and foward
class MyReLU(torch.autograd.Function):
    # the forward function
    def forward(self, input):
        # save the input for using in backword
        self.save_for_backward(input)
        return input.clamp(min=0)

    # backward
    def backward(self, grad_output):
        # load the saved input above
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# create input and output
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# create two weight matrxes
wh = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
wo = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

## Training
learning_rate = 1e-6
for t in range(1000):
    # create an instance of MyReLU
    relu = MyReLU()

    # forward
    y_pred = relu(x.mm(wh)).mm(wo)

    # loss
    loss = (y_pred - y).pow(2).sum()
    if t%100 == 99:
        print(t+1, loss.data[0])
    
    # use autograd to backward
    loss.backward()

    # update weights
    wh.data -= learning_rate * wh.grad.data
    wo.data -= learning_rate * wo.grad.data

    # zeros the grad of weights
    wh.grad.data.zero_()
    wo.grad.data.zero_()

