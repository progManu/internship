import torch
import os
from torch.utils.data import Dataset
import gzip
import numpy as np
import struct
from torchvision.transforms import v2
import random
import cv2


class RevMNIST(Dataset):
    def __init__(self, root='./data/', train=True, offset=None, filt_labels=None):

        self.root = root

        self.dataset = None
        self.labels = None

        if offset is None:
            self.offset = [0, 14]
        else:
            self.offset = offset

        self.filt_labels = filt_labels

        fileends = ['train-images.idx3-ubyte',
                'train-labels.idx1-ubyte',
                't10k-images.idx3-ubyte',
                't10k-labels.idx1-ubyte']
            
        self.filenames = [os.path.join(self.root, fileend) for fileend in fileends]

        train_images, train_labels, test_images, test_labels = self.parse_dataset_in_tensor()

        if train:
            self.dataset, self.labels = train_images, train_labels
        else:
            self.dataset, self.labels = test_images, test_labels
        
        mean, std = self.dataset.float().mean()/255, self.dataset.float().std()/255

        normal_transform = v2.Compose([
            v2.ToDtype(dtype=torch.float, scale=True),
            v2.Normalize((mean,), (std,))
        ])

        self.dataset = normal_transform(self.dataset)
    
    def parse_dataset_in_tensor(self):
        train_images = test_images = train_labels = test_labels = None

        with open(self.filenames[0], 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            train_images = torch.from_numpy(data.reshape((size, 1, nrows, ncols)))

        with open(self.filenames[1], 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            train_labels = torch.from_numpy(data.reshape((size,))) # (Optional)
        
        with open(self.filenames[2], 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            test_images = torch.from_numpy(data.reshape((size, 1, nrows, ncols)))
        
        with open(self.filenames[3], 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            test_labels = torch.from_numpy(data.reshape((size,))) # (Optional)
        
        return train_images, train_labels, test_images, test_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image1 = self.dataset[idx].permute(1, 2, 0)
        offset = random.randint(1, self.__len__() - 1)
        plus_or_minus = (-1)**offset

        new_idx = min(offset, idx + (plus_or_minus*offset))

   
        image2 = self.dataset[new_idx].permute(1, 2, 0)


        label1 = self.labels[idx]
        label2 =  self.labels[new_idx]

        rows, cols = image1.size()[0], image1.size()[1]


        # Calculate the transformation matrix using cv2.getAffineTransform()
        T1 = np.float32([[1, 0, random.randint(-10, 5)], [0, 1, random.randint(-5, 10)]])
        T2 = np.float32([[1, 0, random.randint(-10, 5)], [0, 1, random.randint(-5, 10)]])

        # Apply the affine transformation using cv2.warpAffine()
        image1 = cv2.warpAffine(image1.numpy(), T1, (cols,rows))
        image2 = cv2.warpAffine(image2.numpy(), T2, (cols,rows))

        image1 = torch.tensor(np.expand_dims(image1, axis=0))
        image2 = torch.tensor(np.expand_dims(image2, axis=0))

        return (image1, image2, label1, label2, torch.tensor(T1[:, 2]), torch.tensor(T2[:, 2]))