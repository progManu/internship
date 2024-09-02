from PINOtherPre import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class SubSetImageNet(PINOtherPre, Dataset):
    
    def __init__(self, root_folder='.', transform=None, target_transform=None):
        super().__init__(root_folder)

        self.supercategories = {
            "quadruped": [0, 9, 10, 11, 16, 17, 20, 26, 28, 30, 33, 40, 44, 48, 49, 53, 59, 61, 63, 65],
            "reptile": [2, 27, 38, 39, 54, 69, 74, 76, 84, 92, 95, 104, 105, 113, 117, 118, 124, 126, 147, 154]
        } # this are balaced supercategories (in reality quadrupeds have more than 40 classes)

        self.smap = {} # from supercategories groups a map from normal class to corresponding supercategory
        
        for i, (key, subclasses_list) in enumerate(self.supercategories.items()):
            for subclass in subclasses_list:
                self.smap[subclass] = i
        
        self.root_folder = root_folder
        
        self.transform = transform
        self.target_transform = target_transform

        self.length = self.count_dataset_elements()

        quadruped_list = list(self.supercategories.values())[0]
        reptile_list = list(self.supercategories.values())[1]

        subset_classes = quadruped_list + reptile_list

        self.annot = [ann for ann in self.annot if ann['label'] in subset_classes]

        self.labels_count = len(list(self.supercategories.keys()))
    
    def count_dataset_elements(self):
        cont = 0
        for i, (key, subclasses_list) in enumerate(self.supercategories.items()):
            for elem in self.annot:
                if elem["label"] in subclasses_list:
                    cont += 1
        return cont

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        file_name = self.annot[idx]['img_name']
        folder = self.annot[idx]['folder']
        img_path = os.path.join(self.images_path, folder, file_name + '.JPEG')
        image = read_image(img_path) if self.transform is None else self.transform(read_image(img_path)) # here is the error
        img_label = self.smap[self.annot[idx]['label']]

        annotation_path = os.path.join(self.annot_path, folder, file_name + '.png')
        annotation_img = read_image(annotation_path)

        
        return (image, annotation_img, img_label)