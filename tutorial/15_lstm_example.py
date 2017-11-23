# -*- coding: utf-8 -*-
"""
This is an simple example of classifying bag of word
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

########################################
# process the sequence one by one
# using random data to use a LSTM mdoel
lstm = nn.LSTM(3, 3)
lstm.cuda()
inputs = [Variable(torch.randn((1, 3)).cuda()) for _ in range(5)]
hidden = (Variable(torch.randn(1, 1, 3).cuda()), Variable(torch.randn(1, 1, 3).cuda()))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print(out.data)

# process the sequence one time
inputs = Variable(torch.randn(5, 1, 3).cuda())
hidden = (Variable(torch.randn(1, 1, 3).cuda()), Variable(torch.randn(1, 1, 3).cuda()))
out, hidden = lstm(inputs, hidden)
print(out.data)

#######################################
# An LSTM for part of speech tagging
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor.cuda())

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
VOCAB_SIZE = len(word_to_ix)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
TARGET_SIZE = len(tag_to_ix)

# define a lstm model
class LSTMTagger(nn.Module):
    def __init__(self, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TARGET_SIZE):
        super(LSTMTagger, self).__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
        self.hidden_tag = nn.Linear(HIDDEN_DIM, TARGET_SIZE)
        self.hidden = self.init_hidden(HIDDEN_DIM)
    
    def init_hidden(self, HIDDEN_DIM):
        return (Variable(torch.zeros(1, 1, HIDDEN_DIM).cuda()), Variable(torch.zeros(1, 1, HIDDEN_DIM).cuda()))

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence),1,-1), self.hidden)
        tag_scores = F.softmax(self.hidden_tag(lstm_out.view(len(sentence), -1)))
        return tag_scores

# Train
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TARGET_SIZE)
model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1)
for epoch in range(300):
    epoch_loss = 0
    for sent, tags in training_data:
        # forward
        model.zero_grad()
        model.hidden = model.init_hidden(HIDDEN_DIM)
        sent_in = prepare_sequence(sent, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sent_in)
        # backward
        loss = loss_fn(tag_scores, targets)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    if epoch % 10 == 9:
        print(epoch+1, "Loss", epoch_loss)

# test
inputs = prepare_sequence(training_data[0][0], word_to_ix)
targets = prepare_sequence(training_data[0][1], tag_to_ix)
tag_scores = model(inputs)
print(tag_scores.data)
print(targets.data)