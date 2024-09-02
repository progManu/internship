import torch
import sys
import os

# Get the parent directory path
parent_dir = os.path.abspath('..')

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from NNUtilities import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BATCH_SIZE = 64

class Experiment2:
    def __init__(self, trainloader, testloader, label_map, model=None, lr=1e-2, epochs=100):

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
        
        self.label_map = label_map
    
    def run(self, re_train=True):
        if re_train:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.lossfunc = torch.nn.CrossEntropyLoss()
            self.loss, self.test_accuracy = train(model=self.model, optimizer=self.optimizer, trainloader=self.trainloader, testloader=self.testloader, lossfunc=self.lossfunc, epochs=self.epochs)
            self.trained = True

            display_losses_and_accuracies(loss=self.loss, accuracy=self.test_accuracy, epochs=self.epochs)
        
        test_labels_model, test_predictions_model = self.model.get_labels_and_predictions(dataloader=self.testloader)

        test_labels_model = [(lambda x: self.label_map[x][0])(x) for x in test_labels_model]
        test_predictions_model = [(lambda x: self.label_map[x][0])(x) for x in test_predictions_model]

        cf_matrix = confusion_matrix(test_labels_model, test_predictions_model, labels=[self.label_map[i][0] for i in range(len(self.label_map.keys()))])
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=[self.label_map[i][0] for i in range(len(self.label_map.keys()))])
        disp.plot()
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

class SimpleConvWithPooling(torch.nn.Module):
    def __init__(self, filter_size, input_size, dataloader_for_ranges):
        super().__init__()
    
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=filter_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(input_size - filter_size + 1)
        )

        self.ranges = self.get_ranges(dataloader_for_ranges)
        
        
    def forward(self, x):
        with torch.no_grad():
            x = self.conv(x)
            x.apply_(lambda input: self.from_ranges_to_output(input=input))
            x = x.type(torch.uint8)
            return x
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()

        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        y_test = torch.empty(len(dataloader.dataset), dtype=torch.uint8)

        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                out = self(x).flatten()
                predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = out
                y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y
        
        return y_test.tolist(), predictions.tolist()
    
    def get_ranges(self, dataloader):
        self.eval()

        regression_output = {}

        for i in range(4):
            regression_output[i] = []

        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                preds = self.conv(x).flatten()
                labels = y
                for pred, label in zip(preds.tolist(), labels):
                    label = int(label)
                    regression_output[label].append(pred)
        
        ranges_output = {}

        for i in range(4):
            ranges_output[i] = (min(regression_output[i]), max(regression_output[i]))
        
        return ranges_output
    
    def from_ranges_to_output(self, input):
        for i, range in self.ranges.items():
            if input >= range[0] and input <= range[1]:
                return int(i)
        
        distances = np.empty(((2*4, 2)))

        for i, range in self.ranges.items():

            distances[2*i][0] = int(i)
            distances[2*i + 1][0] = int(i)

            distances[2*i][1] = (input**2 + range[0])**0.5
            distances[2*i + 1][1] = (input**2 + range[1])**0.5
        
        label_with_min_dist = distances[np.argmin(distances[:, 1]), 0]

        return label_with_min_dist

class SimpleArchitectureWithPooling(torch.nn.Module):
    def __init__(self, filter_size, max_pool_size, input_size, fc_output_size=4):
        super().__init__()

        self.fc_output_size = fc_output_size
    
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=filter_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(max_pool_size),
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