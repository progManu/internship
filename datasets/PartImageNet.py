
from PINOtherPre import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class PartImageNet(PINOtherPre, Dataset):
    
    def __init__(self, root_folder='.', transform=None, target_transform=None, filter_labels=None):
        super().__init__(root_folder)
        self.root_folder = root_folder
        
        self.transform = transform
        self.target_transform = target_transform

        self.threshold = None

        # self.remove_unworth_labels() # remove unworth classes (so classes with very few images)

        filtered_labels = [] # remove classes that I don't want (I can just filter those)
        if filter_labels is not None:
            for i in range(len(self.annot)):
                if self.annot[i]['label'] in filter_labels:
                    filtered_labels.append(self.annot[i])
        
        if len(filtered_labels):
            self.annot = filtered_labels

        self.length = len(self.annot)
        self.labels = self.get_labels()

        self.labels.sort()

        self.map = {}
        for idx in range(len(self.labels)):
            self.map[self.labels[idx]] = idx
    
    def get_labels(self):
        labels = []

        for annot in self.annot:
            labels.append(annot['label'])
        return list(set(labels))
    
    def get_labels_hist(self):
        labels = []

        for annot in self.annot:
            labels.append(annot['label'])
        return labels
    
    def remove_unworth_labels(self):
        labels = np.array(self.get_labels())
        bins = np.max(labels) + 1
        hist = self.get_labels_hist()

        dataset_length = len(self.annot)

        dataset_histogram, _ = np.histogram(hist, bins=bins)

        most_freq_label = np.max(dataset_histogram)
        less_freq_label = np.min(dataset_histogram)

        threshold = None
        idxs_to_remove = None

        i = most_freq_label

        while threshold is None:
            if i >= most_freq_label*0.3 and i <= most_freq_label*0.8:
                remaining_datapoins = dataset_histogram[dataset_histogram >= i]
                remaining_datapoins_count = remaining_datapoins.sum()

                if remaining_datapoins_count >= 0.7*dataset_length:
                    threshold = i
                    self.threshold = threshold
                    idxs_to_remove = labels[dataset_histogram < i]
                    print(f'threshold: {i}, datapoints: {remaining_datapoins_count} <= {0.7*dataset_length}')
            i = max(less_freq_label, i - 1)
        
        filt_list = [ann for ann in self.annot if ann['label'] not in idxs_to_remove]

        self.annot = filt_list
    
    def get_mean_and_std(self):
        vec_img_size = int(self.max_img_size**2)
        full_colors = torch.empty(3, len(self)*vec_img_size)
        for i, ann in enumerate(self.annot):
            file_name = ann['img_name']
            folder = ann['folder']
            img_path = os.path.join(self.images_path, folder, file_name + '.JPEG')

            image = read_image(img_path) if self.transform is None else self.transform(read_image(img_path)) # here is the error

            image = image.flatten(start_dim=1)

            full_colors[:, i*vec_img_size:(i+1)*vec_img_size] = image
        
        return full_colors.mean(axis=1), full_colors.std(axis=1)



    
    def plot_labels_hist(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.get_labels_hist(), bins=len(self.get_labels()))
        plt.xlabel('Labels')
        plt.ylabel('Counts')
        plt.title('Histogram of Labels')
        plt.show()
        
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        file_name = self.annot[idx]['img_name']
        folder = self.annot[idx]['folder']
        img_path = os.path.join(self.images_path, folder, file_name + '.JPEG')
        image = read_image(img_path) if self.transform is None else self.transform(read_image(img_path)) # here is the error
        img_label = self.map[self.annot[idx]['label']]

        annotation_path = os.path.join(self.annot_path, folder, file_name + '.png')
        annotation_img = read_image(annotation_path)

        
        return (image, annotation_img, img_label)
