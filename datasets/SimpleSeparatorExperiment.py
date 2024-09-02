import torch
from sklearn.model_selection import KFold
import os
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# First, we import the `Subset` class
from torch.utils.data import Subset
import torch.nn.functional as F

# Get the parent directory path
parent_dir = os.path.abspath('..\\experiments')

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from NNUtilities import *

device = torch.device('cuda')

'''

class ComplexCNN(torch.nn.Module):
    def __init__(self, num_classes=11):
        super(ComplexCNN, self).__init__()

        self.alpha = torch.nn.Parameter(torch.FloatTensor(2))

        self.embedding = None

        self.features = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 3 * 3, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        self.embedding = x
        x = self.classifier(x)
        return x

'''

'''
    

class ComplexCNN(torch.nn.Module):
    def __init__(self, num_classes=11):
        super(ComplexCNN, self).__init__()

        self.alpha = torch.nn.Parameter(torch.FloatTensor(2))

        self.embedding = None

        self.features = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 1024),  # Reduced from 4096 to 1024
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),          # Reduced from 4096 to 512
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        self.embedding = x
        x = self.classifier(x)
        return x
'''

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.alpha = torch.nn.Parameter(torch.FloatTensor(2))

        self.embedding = None

        # Define the layers of the CNN
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=10, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=11, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=12, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=13, stride=1)
        
        # Define fully connected layers
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 512)  # Assuming input images are 32x32, after pooling it becomes 8x8
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 11)
        
    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor for the fully connected layers

        self.embedding = x
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        batch_size = dataloader.batch_size
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, _, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                predictions[i*batch_size:(i+1)*batch_size] = out.cpu()
                labels[i*batch_size:(i+1)*batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()