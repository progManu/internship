
from PINOtherPre import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class PartImageNet(PINOtherPre, Dataset):
    
    def __init__(self, root_folder='.', transform=None, target_transform=None):
        super().__init__(root_folder)

        self.supercategories = {
            "quadruped": [0, 9, 10, 11, 16, 17, 20, 26, 28, 30, 33, 40, 44, 48, 49, 53, 59, 61, 63, 65, 67, 77, 78, 79, 80, 83, 86, 88, 90, 96, 97, 99, 100, 102, 109, 114, 116, 121, 125, 128, 135, 136, 144, 145, 148, 149],
            "biped": [15, 19, 29, 32, 34, 50, 51, 57, 64, 71, 106, 110, 119, 133, 140, 152, 157],
            "fish": [5, 18, 41, 43, 45, 81, 82, 89, 141, 142],
            "bird": [6, 14, 22, 37, 42, 56, 66, 68, 85, 120, 123, 127, 132, 151],
            "snake": [1, 3, 35, 47, 60, 72, 75, 94, 122, 130, 131, 138, 139, 143, 155],
            "reptile": [2, 27, 38, 39, 54, 69, 74, 76, 84, 92, 95, 104, 105, 113, 117, 118, 124, 126, 147, 154],
            "car": [7, 8, 12, 21, 25, 31, 36, 46, 55, 62, 73, 93, 107, 108, 112, 115, 129, 137, 156],
            "bicycle": [13, 23, 101, 103, 134, 146],
            "boat": [4, 24, 91, 111],
            "aeroplane": [98, 150],
            "bottle": [52, 58, 70, 87, 153]
        }

        self.root_folder = root_folder
        
        self.transform = transform
        self.target_transform = target_transform

        self.threshold = None

        # self.remove_unworth_labels() # remove unworth classes (so classes with very few images)

        self.length = len(self.annot)
        self.labels = self.get_labels()
        self.labels.sort()
    
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
        img_label = self.annot[idx]['label']

        annotation_path = os.path.join(self.annot_path, folder, file_name + '.png')
        annotation_img = read_image(annotation_path)

        
        return (image, annotation_img, img_label)
