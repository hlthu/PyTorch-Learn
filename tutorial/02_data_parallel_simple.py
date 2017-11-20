# -*- coding: utf-8 -*-
"""
This is a simple code to use multiple GPUs
"""


####################################
# First we nedd to import something and define some variable
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# imput and output size
input_size = 5
output_size = 2

# batch and data size
batch_size = 20
data_size = 100

#####################################
# define a class to generate random data
class RandDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# get a data loader
rand_dataset = RandDataset(input_size, data_size)
rand_loader = DataLoader(dataset=rand_dataset, batch_size=batch_size, shuffle=True)

#######################################
# Define a simple model, just one linear layer
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, input):
        output = self.fc(input)
        print("  In model, input size", input.size(), "output size", output.size())
        return output

# get a model and init it
model = Model(input_size, output_size)

# to check if there are more than 1 GPUs
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# send the model to GPU
if torch.cuda.is_available():
    model.cuda()


#######################################
# Let's run the model
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)
    output = model(input_var)
    print("Outside model, input size", input_var.size(), "output size", output.size())

print("=============Finished!=============")