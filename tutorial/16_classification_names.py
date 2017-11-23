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

## convert unicode to ascii
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
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

## get the index a letter
def LetterToIndex(letter):
    return all_letters.find(letter)

## turn a letter into Tensor (one-hot)
def LetterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][LetterToIndex(letter)] = 1
    return tensor

## turn a Name to Tensor
def NameToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        tensor[i][0][LetterToIndex(letter)] = 1
    return tensor

# print (LetterToTensor('H'))
# print (NameToTensor('Lu'))

## define a RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = F.log_softmax(output)
        return output, hidden

    def InitHidden(self):
        return Variable(torch.zeros(1, self.hidden_size).cuda())

## RNN instance
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_category)
rnn.cuda()

## get the most likely index from output
def CategoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    pr = torch.exp(top_n)
    category_i = top_i[0][0]
    return all_category[category_i], pr[0][0]

## randomly choose for a list
def RandomChoice(l):
    return l[random.randint(0, len(l)-1)]

## randomly choose a train example
def RandomTrainingExample():
    category = RandomChoice(all_category)
    line = RandomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_category.index(category)]).cuda())
    line_tensor = Variable(NameToTensor(line).cuda())
    return category, line, category_tensor, line_tensor

## test the random choices
for i in range(10):
    category, line, category_tensor, line_tensor = RandomTrainingExample()
    print('Class =', category, '/ Name =', line)

## train a example
loss_fn = nn.NLLLoss()
lr = 0.0033
# optimizer = optim.SGD(rnn.parameters(), lr=lr)
def Train(category_tensor, line_tensor):
    hidden = rnn.InitHidden()
    rnn.zero_grad()

    # forward
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = loss_fn(output, category_tensor)
    loss.backward()
    # optimizer.step()
    for param in rnn.parameters():
        param.data -= lr * param.grad.data
    return output, loss.data[0]

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
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = RandomTrainingExample()
    output, loss = Train(category_tensor, line_tensor)
    if iter % print_interval == 0:
        guess, pr = CategoryFromOutput(output)
        flag = '×'
        if guess == category:
            flag = '√'
            print(iter, loss, '  (', GetTime(start), ') ', line, '/', guess, pr, flag)
        else:
            print(iter, loss, '  (', GetTime(start), ') ', line, '/', guess, pr, flag, 'True is', category) 
        
## evaluation
def Evaluate(line_tensor):
    hidden = rnn.InitHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

# test
n_test = 10000
confusion = torch.zeros(n_category, n_category)
for i in range(n_test):
    category, line, category_tensor, line_tensor = RandomTrainingExample()
    output = Evaluate(line_tensor)
    guess, pr = CategoryFromOutput(output)
    category_i = all_category.index(category)
    guess_i = all_category.index(guess)
    confusion[category_i][guess_i] += 1
for i in range(n_category):
    confusion[i] = confusion[i]/confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

