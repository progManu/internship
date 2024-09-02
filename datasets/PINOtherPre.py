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

# background = class 158 in whole image
# background = class 40 in part-image

class PINOtherPre():
    def __init__(self, root_folder):

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

        if not os.path.isdir(os.path.join(root_folder, self.folder_to_extract)):
            self.unzip()
        else:
            print('Zip Archive already extracted')
        
        if not os.path.exists(os.path.join(root_folder, self.annot_file_name)):
            self.filter_images()
            with open(self.annot_file_name, "w") as outfile:
                json.dump(self.annot, outfile)
        else:
            with open(os.path.join(self.root_folder, self.annot_file_name)) as json_file:
                self.annot = json.load(json_file)
                print(f'Loaded: {os.path.join(self.root_folder, self.annot_file_name)}')




    def unzip(self):
        print('Zip Archive Extraction ...')
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            members = [file for file in all_files if file.startswith(self.folder_to_extract)]
            zip_ref.extractall(members=members)
        print('Zip Archive Extraction Done')
    
    def process_image(self, image_path):

        image = read_image(image_path)

        image = image[:3, :, :] if image.size()[0] > 3 else image

        image = F.resize(image, [self.max_img_size, self.max_img_size], torchvision.transforms.InterpolationMode.NEAREST_EXACT)

        image = F.to_pil_image(image)

        image.save(image_path)


    
    def filter_images(self):
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

                    class_seg_img = read_image(class_seg_img_path)[0]

                    channel, _, _ = read_image(os.path.join(image_folder, f + '.JPEG')).size()

                    classes_in_img = class_seg_img.unique()

                    filt_class_idx = classes_in_img != self.part_whole_bg

                    try:
                        filt_class = classes_in_img[filt_class_idx].item() # don't use images like that

                        if channel >= 3:
                            self.process_image(class_seg_img_path)
                            self.process_image(os.path.join(image_folder, f + '.JPEG'))
                            self.process_image(class_seg_parts_img_path)

                            self.annot.append({
                                'folder': path,
                                'img_name': f,
                                'label': filt_class
                            })
                    except RuntimeError:
                        last_unsuccesful_filt = ext_change_part_whole_file