import torch
import os
from torch.utils.data import Dataset
import gzip
import numpy as np
import struct
from torchvision.transforms import v2


class ModMNIST(Dataset):
    def __init__(self, root='./data/', transform=None, target_transform=None, train=True, offset=None, filt_labels=None):

        self.root = root

        self.transform = transform

        self.target_transform = target_transform

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

        if self.filt_labels is not None:
            train_images, train_labels, test_images, test_labels = self.filter_tensors(train_images, train_labels, test_images, test_labels, self.filt_labels)

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
    
    def filter_tensors(self, train_images, train_labels, test_images, test_labels, filt_labels):
        filt_labels = sorted(filt_labels)

        filt_train_images = train_images[train_labels == filt_labels[0]]
        filt_train_labels = train_labels[train_labels == filt_labels[0]]
        filt_test_images = test_images[test_labels == filt_labels[0]]
        filt_test_labels = test_labels[test_labels == filt_labels[0]]

        for i in range(1, len(filt_labels)):
            lab = filt_labels[i]

            train_mask =  train_labels == lab
            test_mask = test_labels == lab

            filt_train_images = torch.vstack((filt_train_images, train_images[train_mask]))
            filt_train_labels = torch.hstack((filt_train_labels, train_labels[train_mask]))
            filt_test_images = torch.vstack((filt_test_images, test_images[test_mask]))
            filt_test_labels = torch.hstack((filt_test_labels, test_labels[test_mask]))
        
        return filt_train_images, filt_train_labels, filt_test_images, filt_test_labels


    
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

        positions = int(len(self.offset)**2)

        new_train_data = torch.zeros(positions*size, 1, nrows, ncols, dtype=torch.uint8)
        new_label_data = torch.zeros(positions*size, dtype=torch.uint8)

        for i in range(size):
            original_image = trainingdata[i, :, :, :]
            resized_image = transform(original_image)

            cont = 0

            for j in self.offset:
                for k in self.offset:
                    new_train_data[positions*i + cont, :, j:j+14, k:k+14] = resized_image
                    new_label_data[positions*i + cont] = positions*labeldata[i] + cont

                    cont = cont + 1
        
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