# -*- coding: utf-8 -*-
"""
This is an simple example of classifying bag of word
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

word_to_ix = {"hello":0, "world":1}
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(Variable(lookup_tensor))
print(hello_embed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

#############################################
# train data
test_sentence = """But these new vectors are a big pain:
 you could think of thousands of different semantic attributes 
 that might be relevant to determining similarity, and 
 how on earth would you set the values of the different 
 attributes? Central to the idea of deep learning is that 
 the neural network learns representations of the features, 
 rather than requiring the programmer to design them herself. 
 So why not just let the word embeddings be parameters in our model, 
 and then be updated during training? This is exactly what we will do. 
 We will have some latent semantic attributes that the network can, 
 in principle, learn. Note that the word embeddings will probably 
 not be interpretable. That is, although with our hand-crafted 
 vectors above we can see that mathematicians and physicists are 
 similar in that they both like coffee, if we allow a neural network 
 to learn the embeddings and see that both mathematicians and physicisits 
 have a large value in the second dimension, it is not clear what that means. 
 They are similar in some latent semantic dimension, but this probably has 
 no interpretation to us. and then infer that physicist is actually a good 
 fit in the new unseen sentence? This is what we mean by a notion of similarity: 
 we mean semantic similarity, not simply having similar orthographic representations. 
 It is a technique to combat the sparsity of linguistic data, by connecting 
 the dots between what we have seen and what we haven’t. This example of course 
 relies on a fundamental linguistic assumption: that words appearing in similar 
 contexts are related to each other semantically. This is called the distributional 
 hypothesis.""".split()

# trigrams and vocabulary
trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2])
    for i in range(len(test_sentence) - 2)]
for x in trigrams:
    print(x)
vocab = set(test_sentence)
word_to_ix = { word: i
    for i, word in enumerate(vocab)}
ix_to_word = { i: word
    for i, word in enumerate(vocab)}
VOCAB_SIZE = len(vocab)

#############################################
# define a class model
class NGramLM(nn.Module):
    def __init__(self, CONTEXT_SIZE, EMBEDDING_DIM, VOCAB_SIZE):
        super(NGramLM, self).__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.fc1 = nn.Linear(CONTEXT_SIZE * EMBEDDING_DIM, 128)
        self.fc2 = nn.Linear(128, VOCAB_SIZE)

    def forward(self, input):
        embeds = self.embeddings(input).view(1, -1)
        out = self.fc1(embeds)
        out = F.relu(out)
        out = self.fc2(out)
        return F.softmax(out)

#############################################
# training
model = NGramLM(CONTEXT_SIZE, EMBEDDING_DIM, VOCAB_SIZE)
model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

for epoch in range(20):
    epoch_loss = 0
    for context, target in trigrams:
        # preprocessing
        context_ids = [word_to_ix[w] for w in context]
        context_var = Variable(torch.LongTensor(context_ids).cuda())
        target_id = [word_to_ix[target]]
        target_var = Variable(torch.LongTensor(target_id).cuda())

        # forward
        model.zero_grad()
        output = model(context_var)

        # loss
        loss = loss_fn(output, target_var)
        epoch_loss += loss.data[0]

        # backward
        loss.backward()
        optimizer.step()
    print("Epoch", epoch+1, "Loss", epoch_loss)

###########################################
# test
word = ["But", "these"]
print(word)
for t in range(15):
    context = word
    context_ids = [word_to_ix[w] for w in context]
    context_var = Variable(torch.LongTensor(context_ids).cuda())

    output = model(context_var)
    pred = torch.max(output, 1)[1]

    new_word = ix_to_word[pred.data[0]]
    print(new_word)
    word = [word[1], new_word]
