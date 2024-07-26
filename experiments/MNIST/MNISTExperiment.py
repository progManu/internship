import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import fmean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from torch.nn.utils import clip_grad_norm_

import sys
import os

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from NNUtilities import display_losses_and_accuracies, set_requires_grad

device = torch.device('cuda')

def get_accuracy(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct=0
        for x, y in iter(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            out = model(x)
            correct+=(torch.argmax(out, axis=1)==y).sum()
        return correct/len(dataloader.dataset)

def strange_get_accuracy(model, dataloader, devisor):
    model.eval()
    with torch.no_grad():
        correct=0
        for x, y in iter(dataloader):
            x = x.to(device)
            y = y.to(device)

            y = y // devisor
            
            out = model(x)
            correct+=(torch.argmax(out, axis=1)==y).sum()
        return correct/len(dataloader.dataset)

def save_grad_flow_v2(named_parameters, ave_grads, max_grads):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    
    if len(ave_grads.keys()) == 0 and len(max_grads.keys()) == 0:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                ave_grads[n] = []
                max_grads[n] = []
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            ave_grads[n].append(p.grad.cpu().abs().mean())
            max_grads[n].append(p.grad.cpu().abs().max())

def enable_retain_grad(model):
    for name, param in model.named_parameters():
        if ('weight' in name or 'bias' in name) and param.requires_grad:
            param.retain_grad()


def train(model, optimizer, trainloader, testloader, lossfunc, ave_grads, max_grads, epochs=5, print_grad=False, clip_grad=False):
    loss = []
    test_accuracy = []

    if print_grad:
        enable_retain_grad(model=model)

    for epoch in tqdm(range(epochs)):
        losses_per_epoch = []
        test_accuracy.append(get_accuracy(model, testloader).item())
        model.train()
        for x, y in iter(trainloader):
            x = x.to(device)
            y = y.to(device)
            
            out = model(x)
            l = lossfunc(out, y)
            losses_per_epoch.append(l)
            optimizer.zero_grad()
            l.backward()
            if clip_grad:
                clip_grad_norm_(model.parameters(), 0.75)
            if print_grad:
                save_grad_flow_v2(model.named_parameters(), ave_grads=ave_grads, max_grads=max_grads)
            optimizer.step()
        loss.append(fmean(losses_per_epoch))
    
    return (loss, test_accuracy)

def strange_train(model, optimizer, trainloader, testloader, lossfunc, ave_grads, max_grads, epochs=5, print_grad=False, clip_grad=False, digits=True): # used in order to train the digits and the position separately
    loss = []
    test_accuracy = []

    devisor = len(testloader.dataset.offset)**2 if digits else int((testloader.dataset.labels.max() + 1)/(len(testloader.dataset.offset)**2)) # devisor if we have 40 output we have 10 digits so 4 positions


    if print_grad:
        enable_retain_grad(model=model)

    for epoch in tqdm(range(epochs)):
        losses_per_epoch = []
        test_accuracy.append(strange_get_accuracy(model, testloader, devisor).item())
        model.train()
        for x, y in iter(trainloader):
            x = x.to(device)
            y = y.to(device)

            y = y // devisor
            
            out = model(x)
            l = lossfunc(out, y)
            losses_per_epoch.append(l)
            optimizer.zero_grad()
            l.backward()
            if clip_grad:
                clip_grad_norm_(model.parameters(), 0.75)
            if print_grad:
                save_grad_flow_v2(model.named_parameters(), ave_grads=ave_grads, max_grads=max_grads)
            optimizer.step()
        loss.append(fmean(losses_per_epoch))
    
    return (loss, test_accuracy)

class MNISTExperiment:
    def __init__(self, filtered_labels, trainloader, testloader, model, fc_out_size, print_grad=False, clip_grad=False):
        self.filtered_labels = filtered_labels

        self.print_grad = print_grad
        self.clip_grad = clip_grad

        if model is not None and trainloader is not None and testloader is not None:
            self.model = model
            self.trainloader = trainloader
            self.testloader = testloader
        else:
            raise RuntimeError("PARAMETERS not defined")

        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.lossfunc=torch.nn.CrossEntropyLoss()

        self.fc_out_size = fc_out_size

        self.max_grads = {}
        self.ave_grads = {}
    
    def dipaly_gradient_flow(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        for key, value in self.ave_grads.items():
            value = torch.stack(value)
            value = value.numpy()
            plt.plot([*range(len(value))], value, label=key)
        plt.title('Average Gradients')
        plt.xlabel('Training iterations')
        plt.ylabel('Gradient Magnitude')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        for key, value in self.max_grads.items():
            value = torch.stack(value)
            value = value.numpy()
            plt.plot([*range(len(value))], value, label=key)
        plt.title('Max Gradients')
        plt.xlabel('Training iterations')
        plt.ylabel('Gradient Magnitude')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def run(self, epochs=5):
        loss, test_accuracy = train(model=self.model, optimizer=self.optimizer, trainloader=self.trainloader, testloader=self.testloader, lossfunc=self.lossfunc, epochs=epochs, ave_grads=self.ave_grads, max_grads=self.max_grads, print_grad=self.print_grad, clip_grad=self.clip_grad)
        self.dipaly_gradient_flow()
        display_losses_and_accuracies(loss=loss, accuracy=test_accuracy, epochs=epochs)
    
    def weird_run(self, epochs=5, digits=True): # used in order to train the digits and the position separately
        loss, test_accuracy = strange_train(model=self.model, optimizer=self.optimizer, trainloader=self.trainloader, testloader=self.testloader, lossfunc=self.lossfunc, epochs=epochs, ave_grads=self.ave_grads, max_grads=self.max_grads, print_grad=self.print_grad, clip_grad=self.clip_grad, digits=digits)
        self.dipaly_gradient_flow()
        display_losses_and_accuracies(loss=loss, accuracy=test_accuracy, epochs=epochs)

    
    def display_confusion_matrices(self):
        test_labels_model, test_predictions_model = self.model.get_labels_and_predictions(dataloader=self.testloader)

        devisor = int(self.fc_out_size / len(self.filtered_labels))

        test_labels_model2 = [x//devisor for x in test_labels_model]
        test_predictions_model2 = [x//devisor for x in test_predictions_model]

        test_labels_model3 = [(x % devisor) for x in test_labels_model]
        test_predictions_model3 = [(x % devisor) for x in test_predictions_model]

        cf_matrix2 = confusion_matrix(test_labels_model2, test_predictions_model2, labels=[i for i in range(len(self.filtered_labels))])
        cf_matrix3 = confusion_matrix(test_labels_model3, test_predictions_model3, labels=[(i % devisor) for i in range(devisor)])

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Display confusion matrices
        ConfusionMatrixDisplay(confusion_matrix=cf_matrix2, display_labels=[i for i in range(len(self.filtered_labels))]).plot(ax=axes[0])
        axes[0].set_title('Class confusion matrix')
        ConfusionMatrixDisplay(confusion_matrix=cf_matrix3, display_labels=[(i % devisor) for i in range(devisor)]).plot(ax=axes[1])
        axes[1].set_title('Position confusion matrix')

        print(f'accuracy on classes: {np.trace(cf_matrix2)/cf_matrix2.sum()}, accuracy on positions: {np.trace(cf_matrix3)/cf_matrix3.sum()}')

        fig.show()
        

class CNN(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_kernel_size, fc_out_size, batch_size):
        super().__init__()

        self.batch_size = batch_size
    
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=conv_kernel_size), # torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=14),
                torch.nn.ReLU(),
                torch.nn.Flatten()
        )
        image_after_conv = (28 - self.conv[0].kernel_size[0] + 1)*(28 - self.conv[0].kernel_size[0] + 1)

        self.head = torch.nn.Linear(conv_out_channels*image_after_conv, fc_out_size)
        
    def forward(self, x):
        return self.head(self.conv(x))
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                
                predictions[i*self.batch_size:(i+1)*self.batch_size] = out.cpu()
                labels[i*self.batch_size:(i+1)*self.batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()

class CNNWithPooling(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_kernel_size, pool_size, pooling, fc_out_size, batch_size):
        super().__init__()

        self.batch_size = batch_size

        rows_image_after_conv = (28 - conv_kernel_size + 1)

        if pool_size is None:
            pool_size = rows_image_after_conv

        image_after_max_pool = (rows_image_after_conv - pool_size + 1)**2
    
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=conv_kernel_size), # torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=14),
                torch.nn.ReLU(),
                pooling(kernel_size=pool_size, stride=1),
                torch.nn.Flatten()
        )

        self.head = torch.nn.Linear(conv_out_channels*image_after_max_pool, fc_out_size)
        
    def forward(self, x):
        return self.head(self.conv(x))
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                
                predictions[i*self.batch_size:(i+1)*self.batch_size] = out.cpu()
                labels[i*self.batch_size:(i+1)*self.batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()

class CNNWithPoolingAug(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_kernel_size, pool_size, pooling, fc_out_size, batch_size):
        super().__init__()

        self.batch_size = batch_size

        rows_image_after_conv = (28 - conv_kernel_size + 1)

        if pool_size is None:
            pool_size = rows_image_after_conv

        image_after_max_pool = (rows_image_after_conv - pool_size + 1)**2
    
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=conv_kernel_size), # torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=14),
                torch.nn.ReLU(),
                pooling(kernel_size=pool_size, stride=1),
                torch.nn.Flatten()
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(conv_out_channels*image_after_max_pool, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, fc_out_size)
        )
        
    def forward(self, x):
        return self.head(self.conv(x))
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                
                predictions[i*self.batch_size:(i+1)*self.batch_size] = out.cpu()
                labels[i*self.batch_size:(i+1)*self.batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()
    
'''

class HistogramPooling(torch.nn.Module):
    def __init__(self, num_of_bins, learnable=True):
        super().__init__()

        self.thresholds = None
        if learnable:
            self.thresholds = torch.nn.Parameter(torch.randn(1, 1, num_of_bins, device=device)) # adding the dimensions in order to vectorize the operation of difference
        else:
            t = torch.tensor([i*0.05 for i in range(num_of_bins)])
            t = t.reshape(1, 1, len(t))
            self.thresholds =  torch.nn.Parameter(t.to(device), requires_grad=False)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
    
        # Compute the difference between each element in 'x' and 'thresholds_tensor'
        # 'x' is of shape [rows, cols] and 'thresholds_tensor' is of shape [num_of_bins]
        # Resulting 'diffs' tensor will be of shape [rows, cols, num_of_bins]
        diffs = x.unsqueeze(2) - self.thresholds
    
        # Apply ReLU to the differences
        relu_diffs = self.relu(diffs)
    
        # Sum along the 'cols' dimension
        y = relu_diffs.sum(dim=1)
    
        # Move the result to the desired device
        y = y.to(device)
    
        return y

'''

'''
class HistogramPooling(torch.nn.Module):
    def __init__(self, num_of_bins):
        super().__init__()
        self.num_of_bins = num_of_bins
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.vlinspace = torch.func.vmap(torch.linspace)

    
    def find_bin_edges(self, input, bins):
        min, max = input.min(dim=1).values, input.max(dim=1).values
        ranges = self.vlinspace(min, max, steps=bins + 1)
        return ranges

    def forward(self, x):
        ranges = self.find_bin_edges(x.cpu(), self.num_of_bins) # numeber of bins + 1 because the max isn't useful in the cdf

        ranges = ranges[:, :-1] # delete max
        reorder_ranges = torch.sort(ranges, descending=True).values
        reorder_ranges = reorder_ranges.to(device)
        diffs = x.unsqueeze(2) - reorder_ranges.unsqueeze(1) # the unsqueeze is in order to get a difference with dim (batch, rows, 1) - (batch, 1, thresholds) = (batch, rows, thresholds)

        relu_diffs = self.relu(diffs)
        # y = relu_diffs.sum(dim=1)

        y = relu_diffs.mean(dim=1)

        y = y.to(device) # return self.softmax(y) not use because of gradient vanishing
        return y
'''

class HistogramPooling(torch.nn.Module):
    def __init__(self, num_of_bins):
        super().__init__()
        self.num_of_bins = num_of_bins
        self.vhist = torch.func.vmap(lambda x: torch.histc(input=x, bins=self.num_of_bins))

    def forward(self, x):
        batch_size, channels, x_size, y_size = x.size()
        x = x.reshape(batch_size*channels, x_size, y_size)
        x = self.vhist(x)
        x = x.reshape(batch_size, channels, -1)
        return x

class CNNWithHistogramPooling(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_kernel_size, num_of_bins, fc_out_size, batch_size):
        super().__init__()

        self.batch_size = batch_size
    
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=conv_kernel_size), # torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=14),
                torch.nn.ReLU(),
                HistogramPooling(num_of_bins=num_of_bins),
                torch.nn.Flatten()
        )

        self.head = torch.nn.Linear(conv_out_channels*num_of_bins, fc_out_size)
        
    def forward(self, x):
        x = self.conv(x)
        return self.head(x)
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                
                predictions[i*self.batch_size:(i+1)*self.batch_size] = out.cpu()
                labels[i*self.batch_size:(i+1)*self.batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()

class BetterCNN(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_kernel_size, fc_out_size, batch_size):
        super().__init__()

        self.batch_size = batch_size
    
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=conv_kernel_size), # torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=14),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size),
                torch.nn.ReLU(),
                torch.nn.Flatten()
        )
        image_after_conv = (28 - 2*self.conv[0].kernel_size[0] + 2)*(28 - 2*self.conv[0].kernel_size[0] + 2)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(conv_out_channels*image_after_conv, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, fc_out_size)
        )
        
    def forward(self, x):
        return self.head(self.conv(x))
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                
                predictions[i*self.batch_size:(i+1)*self.batch_size] = out.cpu()
                labels[i*self.batch_size:(i+1)*self.batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()

class BetterCNNWithPooling(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_kernel_size, pool_size, pooling, fc_out_size, batch_size):
        super().__init__()

        self.batch_size = batch_size

        rows_image_after_conv = (28 - 2*conv_kernel_size + 2)

        if pool_size is None:
            pool_size = rows_image_after_conv

        image_after_max_pool = (rows_image_after_conv - pool_size + 1)**2
    
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=conv_kernel_size), # torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=14),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size),
                torch.nn.ReLU(),
                pooling(kernel_size=pool_size, stride=1),
                torch.nn.Flatten()
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(conv_out_channels*image_after_max_pool, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, fc_out_size)
        )
        
    def forward(self, x):
        return self.head(self.conv(x))
    
    def get_labels_and_predictions(self, dataloader):
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                out = self(x)
                out = torch.argmax(out, axis=1)
                
                predictions[i*self.batch_size:(i+1)*self.batch_size] = out.cpu()
                labels[i*self.batch_size:(i+1)*self.batch_size] = y.cpu()
        return labels.tolist(), predictions.tolist()


    