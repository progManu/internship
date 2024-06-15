import torch
from tqdm import tqdm
from statistics import fmean
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.colors as mcolors

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_accuracy(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for x, y in iter(dataloader):
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
        for x, y in iter(trainloader):
            out = model(x)
            l = lossfunc(out, y)
            losses_per_epoch.append(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        loss.append(fmean(losses_per_epoch))
    
    return (loss, test_accuracy)

def display_losses_and_accuracies(loss, accuracy, epochs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([*range(epochs)], loss, 'b-', label='Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot([*range(epochs)], accuracy, 'b-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def set_requires_grad(m, requires_grad):
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.requires_grad_(requires_grad)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad_(requires_grad)

def plot_clusters(data, n_clusters, figsize=None):
    pca = PCA(2)
    
    #Transform the data
    df = pca.fit_transform(data)
    
    #Initialize the class object
    kmeans = KMeans(n_clusters=n_clusters)
    
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    
    #Getting unique labels
    u_labels = np.unique(label)

    if figsize is None:
        fig, axes = plt.subplots(1, 1)
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    colors = get_cmap(n=n_clusters)
    
    #plotting the results:
    for i in u_labels:
        color = colors(i)
        axes.scatter(df[label == i , 0] , df[label == i , 1] , label = i, color=color)
    axes.legend()
    fig.show()

def plot3D_clusters(data, n_clusters, figsize=None):
    pca = PCA(3)  # Reduce to 3 components for 3D plotting
    
    # Transform the data
    df = pca.fit_transform(data)
    
    # Initialize the class object
    kmeans = KMeans(n_clusters=n_clusters)
    
    # Predict the labels of clusters
    label = kmeans.fit_predict(df)
    
    # Getting unique labels
    u_labels = np.unique(label)

    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    # Add a 3D subplot
    axes = fig.add_subplot(111, projection='3d')
    
    colors = get_cmap(n=n_clusters)
    
    # Plotting the results
    for i in u_labels:
        color = colors(i)
        axes.scatter(df[label == i, 0], df[label == i, 1], df[label == i, 2], label=i, color=color)
    
    axes.legend()
    plt.show()
