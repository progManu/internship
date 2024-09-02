from AggregationPreProcess import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class PartImageNetAgg(AggregationPreProcess, Dataset):
    
    def __init__(self, root_folder='.', transform=None, target_transform=None):
        super().__init__(root_folder)
        
        self.root_folder = root_folder
        
        self.transform = transform
        self.target_transform = target_transform

        self.length = len(self.annot)

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