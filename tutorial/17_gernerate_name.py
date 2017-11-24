# -*- coding: utf-8 -*-
"""
This is an simple example of classifying country of names using RNN
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from io import open
import glob
import string
import unicodedata
import random
import time
import math
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

## convert unicode to ascii
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1
def UnicodeToAscii(str):
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', str)
        if unicodedata.category(ch) != 'Mn'
        and ch in all_letters
)
print(UnicodeToAscii('Ślusàrski'))


## read a file
def ReadFile(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [UnicodeToAscii(line) for line in lines]

## read the files
category_lines = {}
all_category = []
for filename in glob.glob('data/names/*.txt'):
    # print(filename)
    category = filename.split('/')[-1].split('.')[0]
    all_category.append(category)
    lines = ReadFile(filename)
    category_lines[category] = lines

n_category = len(all_category)
print(category_lines['Japanese'][:5])

## define the gernerative model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_category + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_category + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = F.sigmoid(self.i2h(input_combined))
        output = F.sigmoid(self.i2o(input_combined))
        output_combined = torch.cat((hidden, output), 1)
        output = F.log_softmax(self.dropout(self.o2o(output_combined)))
        return output, hidden
    def InitHidden(self):
        return Variable(torch.zeros(1, self.hidden_size).cuda())

## get pairs of (category, line)
def RandomChoice(l):
    return l[random.randint(0, len(l)-1)]
def RandomTrainPair():
    category = RandomChoice(all_category)
    line = RandomChoice(category_lines[category])
    return category, line

## one-hot vector for category
def CategoryTensor(category):
    li = all_category.index(category)
    tensor = torch.zeros(1, n_category)
    tensor[0][li] = 1
    return tensor

## one-hot matrix for input, no EOS
def InputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

## one-hot matrx for output, include EOS
def OutputTensor(line):
    letter_index = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_index.append(n_letters - 1) ## EOS
    return torch.LongTensor(letter_index)

## random example for training
def RandomTrainExample():
    category, line = RandomTrainPair()
    category_tensor = Variable(CategoryTensor(category).cuda())
    input_tensor = Variable(InputTensor(line).cuda())
    output_tensor = Variable(OutputTensor(line).cuda())
    return category_tensor, input_tensor, output_tensor

## train one iteration
rnn = RNN(n_letters, 128, n_letters)
rnn.cuda()
loss_fn = nn.NLLLoss()
learning_rate = 0.00666
def Train(category_tensor, input_tensor, output_tensor):
    hidden = rnn.InitHidden()
    rnn.zero_grad()
    loss = 0
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_tensor[i], hidden)
        loss += loss_fn(output, output_tensor[i])
    loss.backward()
    for p in rnn.parameters():
        p.data -= learning_rate * p.grad.data
    return output, loss.data[0]/input_tensor.size()[0]

## get the running time
def GetTime(time_start):
    now = time.time()
    s = now - time_start
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' %(m, s)

## train
print_interval = 5000
n_iters = 100000
start = time.time()
loss_interval = 0
for iter in range(1, n_iters+1):
    category_tensor, input_tensor, output_tensor = RandomTrainExample()
    output, loss = Train(category_tensor, input_tensor, output_tensor)
    loss_interval += loss
    if iter % print_interval == 0:
        print(iter, '\t', loss_interval/print_interval, '\t', GetTime(start))
        loss_interval = 0

## sample from a category and starting letter
max_len = 15
def sample(category, start_letter='A'):
    category_tensor = Variable(CategoryTensor(category).cuda())
    input_tensor = Variable(InputTensor(start_letter).cuda())
    hidden = rnn.InitHidden()
    output_name = start_letter
    for i in range(max_len):
        output, hidden = rnn(category_tensor, input_tensor[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
            input_tensor = Variable(InputTensor(letter).cuda())
    return output_name

def samples(category, start_letters='ABC'):
    print('-'*5, category, '-'*5)
    for start_letter in start_letters:
        print(sample(category, start_letter))
    print()

samples('Chinese', 'HLZ')
samples('Russian', 'RUS')
samples('German', 'GER')


