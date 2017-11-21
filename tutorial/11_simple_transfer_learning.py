# -*- coding: utf-8 -*-
"""
This is a simple example of transfer learning
"""

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import time
import os

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()


# training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # train one epoch
    for epoch in range(num_epochs):
        print('-'*20)
        print('Epoch', epoch+1, '/', num_epochs)

        # train or validation
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            
            total_loss = 0.0
            total_acc = 0.0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs, labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                total_loss += loss.data[0]
                total_acc += torch.sum(preds == labels.data)

            epoch_loss = total_loss/dataset_sizes[phase]
            epoch_acc = total_acc/dataset_sizes[phase]

            print(phase, 'Loss: ', epoch_loss, '   Acc: ', epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    time_elapsed = time.time() - since
    print('Train Time: ', time_elapsed)
    print('Best Acc: ', best_acc)


    model.load_state_dict(best_model_wts)
    return model

## first we use the pretrained model to retrain
print('retrain the ResNet')
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 2)

if use_gpu:
    model_ft.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=10)

## then we use the pretrained model as feature extractor
print()
print('='*20)
print('using ResNet as a feature extractor....')

model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
model_conv.fc = torch.nn.Linear(num_ftrs, 2)

if use_gpu:
    model_conv.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer, scheduler, num_epochs=10)