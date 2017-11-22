# -*- coding: utf-8 -*-
"""
This is an simple example of classifying bag of word
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

################################
# create train data and test data
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

################################
# map each word to an integer
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

################################
VOCAB_SIZE = len(word_to_ix)
LABEL_SIZE = 2
# define the classifcation model
class BowClassifier(nn.Module):
    def __init__(self, VOCAB_SIZE, LABEL_SIZE):
        super(BowClassifier, self).__init__()
        self.fc = nn.Linear(VOCAB_SIZE, LABEL_SIZE)
    
    def forward(self, input):
        return F.softmax(self.fc(input))

model = BowClassifier(VOCAB_SIZE, LABEL_SIZE)
model.cuda()

################################
# make BoW to a vector
def make_bow_vector(sent, word_to_ix):
    vec = torch.zeros(1, len(word_to_ix))
    for word in sent:
        vec[0, word_to_ix[word]] += 1.0
    return vec

# make Target to an integer
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}
def make_label(label, label_to_ix):
    vec = torch.LongTensor(1)
    vec[0] = label_to_ix[label]
    return vec

###################################
# train and test
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    epoch_loss = 0
    for sent, label in data:
        # init the grad
        model.zero_grad()

        # input and output
        input = make_bow_vector(sent, word_to_ix)
        output = make_label(label, label_to_ix)
        input = Variable(input.cuda(), requires_grad=True)
        output = Variable(output.cuda())

        # forward
        pred = model(input)

        # loss
        loss = loss_fn(pred, output)
        epoch_loss += loss.data[0]
        
        # backward
        loss.backward()
        optimizer.step()

    if epoch % 10 == 9:
        print('-'*40)
        print('Epoch', epoch+1, 'Loss', epoch_loss)
        ## test
        for sent, label in test_data:
            input = make_bow_vector(sent, word_to_ix)
            input = Variable(input.cuda())
            out = model(input)
            print('True', label, ', Predict Pr', out.data[0, label_to_ix[label]])

        


