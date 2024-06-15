import torch
import os
from torch.utils.data import Dataset
import gzip
import numpy as np
import struct
from torchvision.transforms import v2


class ModMNIST(Dataset):
    def __init__(self, root='./data/', transform=None, target_transform=None, train=True):

        self.root = root

        self.transform = transform

        self.target_transform = target_transform

        self.dataset = None
        self.labels = None

        fileends = ['train-images.idx3-ubyte',
                'train-labels.idx1-ubyte',
                't10k-images.idx3-ubyte',
                't10k-labels.idx1-ubyte']
            
        self.filenames = [os.path.join(self.root, fileend) for fileend in fileends]

        train_images, train_labels, test_images, test_labels = self.parse_dataset_in_tensor()

        if train:
            self.dataset, self.labels = self.process_data(train_images, train_labels)
        else:
            self.dataset, self.labels = self.process_data(test_images, test_labels)
        
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
    
    def process_data(self, trainingdata, labeldata):

        transform = v2.Resize(size = (14,14))
        
        size, _, nrows, ncols = trainingdata.size()

        new_train_data = torch.zeros(4*size, 1, nrows, ncols, dtype=torch.uint8)
        new_label_data = torch.zeros(4*size, dtype=torch.uint8)

        for i in range(size):
            original_image = trainingdata[i, :, :, :]
            resized_image = transform(original_image)
            
            new_train_data[4*i + 0, :, 0:14, 0:14] = resized_image
            new_train_data[4*i + 1, :, 14:28, 0:14] = resized_image
            new_train_data[4*i + 2, :, 0:14, 14:28] = resized_image
            new_train_data[4*i + 3, :, 14:28, 14:28] = resized_image

            new_label_data[4*i + 0] = 4*labeldata[i] + 0 # 0 is the top-left corner
            new_label_data[4*i + 1] = 4*labeldata[i] + 1 # 1 is the top-right corner
            new_label_data[4*i + 2] = 4*labeldata[i] + 2 # 2 is the bottom-left corner
            new_label_data[4*i + 3] = 4*labeldata[i] + 3 # 3 is the bottom-right corner
        
        return new_train_data, new_label_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.dataset[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label