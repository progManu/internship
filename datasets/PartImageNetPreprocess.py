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

class PartImageNetPreprocess():
    def __init__(self, root_folder):

        self.root_folder = root_folder
        self.zip_path = 'PartImageNet_Seg.zip'
        self.folder_to_extract = 'PartImageNet/'  # Include trailing slash
        self.images_path = os.path.join(self.folder_to_extract, 'images')
        self.annot_path = os.path.join(self.folder_to_extract, 'annotations')
        self.ttv_paths = ['train', 'test', 'val']
        
        self.max_img_size = 224

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
    
    def get_subject_bounds(self, img_tensor):
        not_bg_x, not_bg_y = torch.where(img_tensor != self.part_whole_bg)
        
        left, right = not_bg_x.min().item(), not_bg_x.max().item()
        upper, lower = not_bg_y.min().item(), not_bg_y.max().item()

        return (left, upper, right, lower)
    
    def pad_image(self, image, bg):
        # Get current dimensions of the image
        _, current_height, current_width  = image.size()

        if current_width > self.max_img_size or current_height > self.max_img_size:
            return None

        # Determine padding and cropping
        pad_height = max(0, self.max_img_size - current_height)
        pad_width = max(0, self.max_img_size - current_width)

        padding = (
            pad_width // 2, 
            pad_height // 2,
            (pad_width + 1) // 2,
            (pad_height + 1) // 2
        )

        image = F.pad(image, padding, fill=bg)

        return image
    
    def process_image(self, class_seg_img, image_path):
        left, upper, right, lower = self.get_subject_bounds(class_seg_img)

        image = read_image(image_path)
        
        image = image[: ,left:right + 1, upper:lower + 1]

        bg = None

        if "annotations" in image_path and "_whole" in image_path:
            bg = 158
        elif "annotations" in image_path:
            bg = 40
        else:
            bg = 0

        image = self.pad_image(image, bg)

        if image is None:
            return False

        image = F.to_pil_image(image)

        image.save(image_path)

        return True


    
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

                        if self.process_image(class_seg_img, class_seg_img_path) and channel >= 3:

                            self.process_image(class_seg_img, os.path.join(image_folder, f + '.JPEG'))
                            self.process_image(class_seg_img, class_seg_parts_img_path)

                            self.annot.append({
                                'folder': path,
                                'img_name': f,
                                'label': filt_class
                            })
                    except RuntimeError:
                        last_unsuccesful_filt = ext_change_part_whole_file