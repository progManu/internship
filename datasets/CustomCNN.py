import torch
from sklearn.model_selection import KFold
import os
import sys
from tqdm import tqdm

# Get the parent directory path
parent_dir = os.path.abspath('..\\experiments')

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from NNUtilities import *

def conv_output(input_size, kernel_size, stride_size, padding_size):
    out_size = int(((input_size + (2*padding_size) - kernel_size) / stride_size) + 1)
    return out_size

def reset_all_weights(model: torch.nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: torch.nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def get_accuracy(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for x, _, y in iter(dataloader):
            out = model(x)
            correct += (torch.argmax(out, axis=1) == y).sum()
        return float(correct/len(dataloader.dataset))

def train(model, optimizer, trainloader, testloader, lossfunc, epochs=10):
    loss = []
    test_accuracy = []

    for epoch in tqdm(range(epochs)):
        losses_per_epoch = []
        test_accuracy.append(get_accuracy(model, testloader))
        model.train()
        for x, _, y in iter(trainloader):
            out = model(x)
            l = lossfunc(out, y)
            losses_per_epoch.append(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        loss.append(fmean(losses_per_epoch))
    
    return (loss, test_accuracy)


class CustomCNNExperiment():
    def __init__(self, dataset, model=None, lr=1e-2, k_folds=5, batch_size=512):

        self.dataset = dataset

        if model is None:
            raise ValueError("Model must be defined")
        
        self.model = model

        self.optimizer = None
        self.lossfunc = None

        self.lr = lr

        self.loss = None
        self.test_accuracy = None

        self.batch_size = batch_size

        self.kfold = KFold(n_splits=k_folds, shuffle=True)
    
    def run(self, epochs=5):
        for fold, (train_ids, test_ids) in enumerate(self.kfold.split([*range(len(self.dataset))])):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.lossfunc = torch.nn.CrossEntropyLoss()

                # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                      self.dataset, 
                      batch_size=self.batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                      self.dataset,
                      batch_size=self.batch_size, sampler=test_subsampler)
            
            reset_all_weights(self.model)

            self.loss, self.test_accuracy = train(model=self.model, optimizer=self.optimizer, trainloader=trainloader, testloader=testloader, lossfunc=self.lossfunc, epochs=epochs)

            display_losses_and_accuracies(self.loss, self.test_accuracy, epochs)


class CustomCNN(torch.nn.Module):
    def __init__(self, out_channs, classes):
        super().__init__()

        self.nn_feat = {}
        self.nn_feat["l1"] = {}
        self.nn_feat["l2"] = {}
        self.nn_feat["l3"] = {}

        self.nn_feat["l1"]["conv_in_channels"] = 3
        self.nn_feat["l1"]["conv_out_channels"] = out_channs
        self.nn_feat["l1"]["conv_kernel_size"] = 3
        self.nn_feat["l1"]["pooling_kernel_size"] = 3

        self.nn_feat["l2"]["conv_in_channels"] = out_channs
        self.nn_feat["l2"]["conv_out_channels"] = 3*out_channs
        self.nn_feat["l2"]["conv_kernel_size"] = 4
        self.nn_feat["l2"]["pooling_kernel_size"] = 4

        self.nn_feat["l3"]["conv_in_channels"] = 3*out_channs
        self.nn_feat["l3"]["conv_out_channels"] = 3*out_channs
        self.nn_feat["l3"]["conv_kernel_size"] = 5
        self.nn_feat["l3"]["pooling_kernel_size"] = 5


        self.feature_ext = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.nn_feat["l1"]["conv_in_channels"], out_channels=self.nn_feat["l1"]["conv_out_channels"], kernel_size=self.nn_feat["l1"]["conv_kernel_size"], stride=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=self.nn_feat["l1"]["pooling_kernel_size"], stride=self.nn_feat["l1"]["pooling_kernel_size"]),
            torch.nn.Conv2d(in_channels=self.nn_feat["l2"]["conv_in_channels"], out_channels=self.nn_feat["l2"]["conv_out_channels"], kernel_size=self.nn_feat["l2"]["conv_kernel_size"], stride=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=self.nn_feat["l2"]["pooling_kernel_size"], stride=self.nn_feat["l1"]["pooling_kernel_size"]),
            torch.nn.Conv2d(in_channels=self.nn_feat["l3"]["conv_in_channels"], out_channels=self.nn_feat["l3"]["conv_out_channels"], kernel_size=self.nn_feat["l3"]["conv_kernel_size"], stride=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=self.nn_feat["l3"]["pooling_kernel_size"], stride=self.nn_feat["l1"]["pooling_kernel_size"])
        )

        conv1 = conv_output(224, self.nn_feat["l1"]["conv_kernel_size"], 1, 0)
        pool1 = conv_output(conv1, self.nn_feat["l1"]["pooling_kernel_size"], self.nn_feat["l1"]["pooling_kernel_size"], 0)

        conv2 = conv_output(pool1, self.nn_feat["l2"]["conv_kernel_size"], 1, 0)
        pool2 = conv_output(conv2, self.nn_feat["l2"]["pooling_kernel_size"], self.nn_feat["l1"]["pooling_kernel_size"], 0)

        conv3 = conv_output(pool2, self.nn_feat["l3"]["conv_kernel_size"], 1, 0)
        pool3 = conv_output(conv3, self.nn_feat["l3"]["pooling_kernel_size"], self.nn_feat["l1"]["pooling_kernel_size"], 0)

        output_size = pool3*pool3*self.nn_feat["l3"]["conv_in_channels"]

        self.output_size = output_size

        self.class_head = torch.nn.Sequential(
            torch.nn.Linear(output_size, output_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size // 2, output_size // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size // 4, classes)
        )
    
    

    def forward(self, x):
        x = self.feature_ext(x)
        x = x.flatten(start_dim=1)
        x = self.class_head(x)
        return x
    
    def get_labels_and_predictions(self, dataloader):
        batch_size = dataloader.batch_size
        self.eval()
        predictions = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        labels = torch.empty(len(dataloader.dataset), dtype=torch.uint8)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                out = self(x)
                out = torch.argmax(out, axis=1)
                predictions[i*batch_size:(i+1)*batch_size] = out
                labels[i*batch_size:(i+1)*batch_size] = y
        return labels.tolist(), predictions.tolist()