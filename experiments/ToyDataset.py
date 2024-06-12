import torch
from torch.utils.data import TensorDataset
import numpy as np
from scipy.linalg import circulant
import matplotlib.pyplot as plt

class ToyDataset(TensorDataset):
    def __init__(self, noise_level: np.uint8, class_values: tuple, input_size: int, inter_class_size: int, binary=False):

        if len(class_values) != 2:
            raise ValueError("Maximum 2 classes")
        else:
            self.class_values = class_values
        
        self.input_size = input_size
        
        template_a_class, template_b_class = self.create_class_input_templates(input_size=self.input_size) # creation of the template without noise
        
        self.template_a = template_a_class
        self.template_b = template_b_class

        dataset, labels = self.create_noised_samples(template_a_class=template_a_class, template_b_class=template_b_class, noise_level=noise_level, inter_class_size=inter_class_size, binary=binary) # creation of 2 class dataset with interclass noise

        reshape_dataset = dataset.reshape(dataset.shape[0], -1, dataset.shape[1]) # reshape in order to use it in TensorDataset
        reshape_labels = labels

        reshape_dataset.shape, reshape_labels.shape

        inps = torch.tensor(reshape_dataset, dtype=torch.float32)
        tgts = torch.tensor(reshape_labels, dtype=torch.uint8)

        self.label_map = None

        if not binary:
            self.label_map = {
                0: ['A-left', 'sx high-low'],
                1: ['A-right', 'dx high-low'],
                2: ['B-left', 'sx low-high'],
                3: ['B-right', 'dx low-high']
            }
        else:
            self.label_map = {
                0: ['A', 'high-low'],
                1: ['B', 'low-high']
            }

        super().__init__(inps, tgts)

    def add_noise(self, input: np.array, max_delta: int) -> np.array: # this function adds a random value to the original array
        delta = np.random.randint(-max_delta, max_delta)
        delta_matrix = delta*np.ones_like(input)
        return input + delta_matrix
    
    def create_noised_samples(self, template_a_class: np.array, template_b_class: np.array, noise_level: np.uint8, inter_class_size: int, binary: bool) -> tuple:

        a_samples = template_a_class
        b_samples = template_b_class

        half_samples = int(a_samples.shape[0]/2) # left samples counter

        input_labels = None

        if not binary:
            input_labels = np.hstack((np.zeros(half_samples), np.ones(half_samples))) # 4 classes
        else:
             input_labels = np.zeros(a_samples.shape[0]) # 2 classes

        a_labels = input_labels

        for idx in range(inter_class_size):
            a_samples = np.vstack((a_samples, self.add_noise(input=template_a_class, max_delta=noise_level)))
            a_labels = np.hstack((a_labels, input_labels))
            b_samples = np.vstack((b_samples, self.add_noise(input=template_b_class, max_delta=noise_level)))

        b_labels = None

        if not binary:
            b_labels = a_labels + 2 # 4 classes
        else:
            b_labels = a_labels + 1 # 2 classes

        dataset = np.vstack((a_samples, b_samples))
        labels = np.hstack((a_labels, b_labels))

        return (dataset, labels)
    
    def create_class_input_templates(self, input_size: int) -> tuple:
        if input_size % 2 == 0:
            raise ValueError("input_size must be odd in order to have left and right separation") # odd size in order to better distinguish right and left
        
        class_a_template = [0 for i in range(input_size)]
        class_a_template[0] = self.class_values[0]
        class_a_template[1] = self.class_values[1]

        class_b_template = [0 for i in range(input_size)]
        class_b_template[0] = self.class_values[1]
        class_b_template[1] = self.class_values[0]

        circulant_a = circulant(class_a_template)
        circulant_b = circulant(class_b_template)

        cleaned_circulant_a = circulant_a[1:, :]
        cleaned_circulant_b = circulant_b[1:, :]

        return (cleaned_circulant_a, cleaned_circulant_b)
    
    def show_templates(self):
    
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        pos1 = axes[0].imshow(self.template_a, cmap='hot')
        axes[0].set_title("A Template")
        axes[0].axis('off')

        pos2 = axes[1].imshow(self.template_b, cmap='hot')
        axes[1].set_title("B Template")
        axes[1].axis('off')

        # Create an invisible axis to which the colorbar will be attached
        cax = fig.add_axes([0.25, 0.01, 0.5, 0.03])  # [left, bottom, width, height]
    
        # Add colorbar for the whole figure
        plt.colorbar(pos1, cax=cax, orientation='horizontal')

        plt.show()
    
    def show_label_map(self, binary=False):
        for key, value in self.label_map.items():
            print(f'tag: {key}, class: {value}')

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)