import os
import zipfile
import torch
import torchvision

from torchvision.io import read_image
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from tqdm import tqdm

from PIL import Image

import json

part_image_map = {
	0: 0,
	1: 0,
	2: 0,
	3: 1,
	4: 2,
	5: 2,
	6: 2,
	7: 2,
	8: 3,
	9: 4,
	10: 4,
	11: 4,
	12: 5,
	13: 6,
	14: 6,
	15: 6,
	16: 6,
	17: 7,
	18: 8,
	19: 9,
	20: 10,
	21: 10,
	22: 10,
	23: 11,
	24: 12,
	25: 12,
	26: 13,
	27: 14,
	28: 14,
	29: 14,
	30: 15,
	31: 16,
	32: 17,
	33: 18,
	34: 18,
	35: 18,
	36: 18,
	37: 19,
	38: 20,
	39: 21,
	40: 22
}

scls_map = { #  this is the super-class map
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

class AggregationPreProcess():
    def __init__(self, root_folder='.'):
        self.root_folder = root_folder
        self.zip_path = 'PartImageNet_Seg.zip'
        self.folder_to_extract = 'PartImageNet/'  # Include trailing slash
        self.images_path = os.path.join(self.folder_to_extract, 'images')
        self.annot_path = os.path.join(self.folder_to_extract, 'annotations')
        self.ttv_paths = ['train', 'test', 'val']
        
        self.max_img_size = 112

        self.part_whole_bg = 158
        self.part_img_bg = 40

        self.annot = []

        self.annot_file_name = 'annotations.json'

        self.cls_to_scls_map = {}

        for i, (key, subclasses_list) in enumerate(scls_map.items()):
            for subclass in subclasses_list:
                self.cls_to_scls_map[subclass] = i

        if not os.path.isdir(os.path.join(root_folder, self.folder_to_extract)):
            self.unzip()
        else:
            print('Zip Archive already extracted')
        
        if not os.path.exists(os.path.join(root_folder, self.annot_file_name)):
            self.dataset_preprocess()
            print('Second Pass ...')
            self.second_pass() # used to filter of the residual 2 parts images
            with open(self.annot_file_name, "w") as outfile:
                json.dump(self.annot, outfile)
        else:
            with open(os.path.join(self.root_folder, self.annot_file_name)) as json_file:
                self.annot = json.load(json_file)
                print(f'Loaded: {os.path.join(self.root_folder, self.annot_file_name)}')
                
        self.part_img_bg = list(part_image_map.values())[-1]

    def unzip(self):
        print('Zip Archive Extraction ...')
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            members = [file for file in all_files if file.startswith(self.folder_to_extract)]
            zip_ref.extractall(members=members)
        print('Zip Archive Extraction Done')
        
    def rescale_image(self, image_path):
        image = read_image(image_path)
        image = image[:3, :, :] if image.size()[0] > 3 else image
        image = F.resize(image, [self.max_img_size, self.max_img_size], torchvision.transforms.InterpolationMode.NEAREST_EXACT)
        image = F.to_pil_image(image)
        image.save(image_path)
    
    def change_class_parts_image(self, image_path):
        image = read_image(image_path)
        image.apply_(part_image_map.get)
        image = F.to_pil_image(image)
        image.save(image_path)
    
    def second_pass(self): # used to filter of the residual 2 parts images
        new_annot = []
        for item in tqdm(self.annot):
            parts_path = os.path.join(self.annot_path, item['folder'], item['img_name'] + '.png')
            parts_image = read_image(parts_path)
            parts = parts_image.unique().size()[0]
            
            if parts > 2:
                new_annot.append(item)
        self.annot = new_annot
    
    def delete_background(self, image_path, parts_whole_image_path):

        image = read_image(image_path)
        parts_whole_image = read_image(parts_whole_image_path)

        if (parts_whole_image.size()[1] != image.size()[1]) and (parts_whole_image.size()[2] != image.size()[2]):
            raise RuntimeError("Images must have the same size")
    
        mask = (parts_whole_image != 158).repeat(3, 1, 1)

        db_image = F.to_pil_image(image*mask)

        db_image.save(image_path)
    
    def dataset_preprocess(self):
        with tqdm(self.ttv_paths) as ttv_paths:
            ttv_paths.set_description('filtering')
            for path in ttv_paths:
                image_folder = os.path.join(self.images_path , path)
                annots_folder = os.path.join(self.annot_path , path)
                files_without_ext = [f.strip('.JPEG') for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

                last_unsuccesful_filt = None

                for f in files_without_ext:
                    ttv_paths.set_postfix({'last unsuccess': {last_unsuccesful_filt}})
                    ttv_paths.refresh()

                    ext_change_part_whole_file = f + '.png'
                    class_seg_img_path = os.path.join(annots_folder + '_whole', ext_change_part_whole_file)
                    class_seg_parts_img_path = os.path.join(annots_folder, ext_change_part_whole_file)

                    real_image_path = os.path.join(image_folder, f + '.JPEG')

                    class_seg_img = read_image(class_seg_img_path)[0]

                    channel, _, _ = read_image(os.path.join(image_folder, f + '.JPEG')).size()

                    classes_in_img = class_seg_img.unique()

                    filt_class_idx = classes_in_img != self.part_whole_bg

                    try:
                        self.delete_background(real_image_path, class_seg_img_path)
                        
                        filt_class = classes_in_img[filt_class_idx].item() # don't use images like that

                        self.change_class_parts_image(class_seg_parts_img_path)

                        parts = read_image(class_seg_parts_img_path).unique().size()[0]

                        if (channel >= 3) and (parts > 2): # this excludes blank images and images with only one part
                            self.rescale_image(class_seg_img_path)
                            self.rescale_image(os.path.join(image_folder, f + '.JPEG'))
                            self.rescale_image(class_seg_parts_img_path)

                            self.annot.append({
                                'folder': path,
                                'img_name': f,
                                'label': self.cls_to_scls_map[filt_class]
                            })
                    except RuntimeError:
                        last_unsuccesful_filt = ext_change_part_whole_file