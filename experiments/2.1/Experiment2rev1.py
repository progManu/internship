import torch
import os
import sys

# Get the parent directory path
parent_dir = os.path.abspath('..')

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from NNUtilities import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BATCH_SIZE = 64

class Experiment2rev1:
    def __init__(self, trainloader, testloader, model=None, lr=1e-2, epochs=100):

        self.trainloader = trainloader
        self.testloader = testloader

        if model is None:
            raise ValueError("Model must be defined")
        
        self.model = model

        self.epochs = epochs

        self.optimizer = None
        self.lossfunc = None

        self.lr = lr

        self.loss = None
        self.test_accuracy = None
        
        self.label_map = [
            {
                0: 'A-left',
                1: 'A-right',
                2: 'B-left',
                3: 'B-right'
            },
            {
                0: 'A',
                1: 'B'
            },
            {
                0: 'left',
                1: 'right'
            }
        ]
    
    def run(self, re_train=True):
        if re_train:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.lossfunc = torch.nn.CrossEntropyLoss()
            self.loss, self.test_accuracy = train(model=self.model, optimizer=self.optimizer, trainloader=self.trainloader, testloader=self.testloader, lossfunc=self.lossfunc, epochs=self.epochs)
            self.trained = True

            display_losses_and_accuracies(loss=self.loss, accuracy=self.test_accuracy, epochs=self.epochs)
        
        test_labels_model, test_predictions_model = self.model.get_labels_and_predictions(dataloader=self.testloader)

        test_labels_model1 = [(lambda x: self.label_map[0][x])(x) for x in test_labels_model] # 4 CLASSES LABELS [A-L A-R B-L B-R]
        test_predictions_model1 = [(lambda x: self.label_map[0][x])(x) for x in test_predictions_model]

        test_labels_model2 = [(lambda x: self.label_map[1][x])(x//2) for x in test_labels_model] # 2 CLASS LABELS [A B]
        test_predictions_model2 = [(lambda x: self.label_map[1][x])(x//2) for x in test_predictions_model]

        test_labels_model3 = [(lambda x: self.label_map[2][x])(x % 2) for x in test_labels_model] # 2 CLASS LABELS [left right]
        test_predictions_model3 = [(lambda x: self.label_map[2][x])(x % 2) for x in test_predictions_model]

        cm1 = confusion_matrix(test_labels_model1, test_predictions_model1, labels=[self.label_map[0][i] for i in range(len(self.label_map[0].keys()))])

        cm2 = confusion_matrix(test_labels_model2, test_predictions_model2, labels=[self.label_map[1][i] for i in range(len(self.label_map[1].keys()))])

        cm3 = confusion_matrix(test_labels_model3, test_predictions_model3, labels=[self.label_map[2][i] for i in range(len(self.label_map[2].keys()))])

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Display confusion matrices
        ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=[self.label_map[0][i] for i in range(len(self.label_map[0].keys()))]).plot(ax=axes[0])
        axes[0].set_title('Class-Position confusion matrix')

        ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=[self.label_map[1][i] for i in range(len(self.label_map[1].keys()))]).plot(ax=axes[1])
        axes[1].set_title('Class confusion matrix')

        ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=[self.label_map[2][i] for i in range(len(self.label_map[2].keys()))]).plot(ax=axes[2])
        axes[2].set_title('Position confusion matrix')

        plt.tight_layout()
        plt.show()

class SimpleArchitecture(torch.nn.Module):
    def __init__(self, filter_size, input_size, output_size):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=filter_size),
            torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(input_size - filter_size + 1, output_size)
    
    def get_conv_layer(self):
        return self.conv
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                out = self(x)
                out = torch.argmax(out, axis=1)
                predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = out
                labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y
        return labels.tolist(), predictions.tolist()

class SimpleArchitectureWithPooling(torch.nn.Module):
    def __init__(self, filter_size, pooling, max_pool_size, input_size, fc_output_size=4):
        super().__init__()

        self.fc_output_size = fc_output_size

        if pooling is None:
            pooling = torch.nn.MaxPool1d
    
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=filter_size),
            torch.nn.ReLU(),
            pooling(max_pool_size),
            torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(int((input_size - filter_size + 1)/max_pool_size), self.fc_output_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()

        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)

        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                out = self(x)
                out = torch.argmax(out, axis=1)
                predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = out
                labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y
            
    
        return labels.tolist(), predictions.tolist()
    
        
    def get_output_embeddings(self, dataloader):
        self.eval()

        predictions = torch.empty((len(dataloader.dataset), self.fc_output_size))

        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                out = self(x)
                predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] = out
            
        return predictions.tolist()