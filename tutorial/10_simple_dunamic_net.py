# -*- coding: utf-8 -*-
"""
This is a simple example of dynamic DNN
"""

import random
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

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.hidden_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.hidden_linear(h_relu).clamp(min=0)
        output = self.output_linear(h_relu)
        return output

# create a model instance and send it to GPU
model = DynamicNet(D_in, H, D_out)
model.cuda()

# some training criterons
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for t in range(1000):
    # forward
    y_pred = model(x)
    
    # loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t+1, loss.data[0])

    # backward and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()