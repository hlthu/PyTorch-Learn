# -*- coding: utf-8 -*-
"""
This is an example about how  to load and preprocess the data
note some of the display codes are commented
"""

import os
import torch
import pandas as pd 
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# read the landmarks
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

# show the last photo and landmarks
n = 65
img_name = landmarks_frame.ix[n, 0]
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
landmarks = landmarks.reshape(-1, 2)

print('Image name ', img_name)
print('landmarks shape ', landmarks.shape)
print('first 4 landmarks ', landmarks[:4])

# a function for show image
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='g')
    # plt.pause(0.001)

show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
plt.show()

# define a class for dataset
class FaceLandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

face_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv', 'data/faces')

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i+1, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i+1)
    ax.axis('off')
    show_landmarks(**sample)

    if i==3:
        plt.show()
        break

# Firstly, we need to rescale the figures
class Rescale(object):
    """
    Rescale the image to a given size,
    with width=output_size or height=output_size
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size   
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w/w, new_h/h]
        return {'image': img, 'landmarks': landmarks}

# Then, we need to randomly crop the figures
class RandomCrop(object):
    """
    crop the image randomly
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        image = image[top:top+new_h, left:left+new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks} 

# Finaly, we need to convent the sample to Tensors
class ToTensor(object):
    """
    convert the image of ndarray type into tensors
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap the color axis
        # numpy image: H * W * C
        # tensor image: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(255)])

sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    trans_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i+1)
    show_landmarks(**trans_sample)
plt.show()

## iterating throughthe dataset
transformed_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv', 'data/faces',
        transform=transforms.Compose([Rescale(256), RandomCrop(255), ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

## batching and shuffling the data     
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
