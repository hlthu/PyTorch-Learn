# -*- coding: utf-8 -*-
"""
This is a simple example of RNN
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#################################
# define some variables
batch_size = 10
STEPS = 5
data_size = 50
hidden_size = 20
output_size = 10

# generate a bacth of data
batch = Variable(torch.randn(batch_size, data_size).cuda())
# init the hidden states with zeros
hidden = Variable(torch.randn(batch_size, hidden_size).cuda())
# generate a random target
target = Variable(torch.randn(batch_size, output_size).cuda())

##################################
# define a RNN model
class RNN(nn.Module):

    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        input_size = data_size + hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = F.relu(self.i2h(input))
        output = F.softmax(self.h2o(hidden))
        return hidden, output

# init a model and send it to GPU
rnn = RNN(data_size, hidden_size, output_size)
rnn.cuda()

##################################
# trainning
# first define a MSE loss
loss_fn = nn.MSELoss()
loss = 0

# sum up the loss and then backword
for t in range(STEPS):
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)

loss.backward()


