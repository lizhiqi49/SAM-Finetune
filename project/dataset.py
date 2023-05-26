import os
import glob
import numpy as np
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from transformers import SamProcessor
from PIL import Image

from utils import *

data_root = "./data"
categories = ['AGC', 'ASC-H', 'ASCUS', 'HSIL', 'LSIL', 'N']

class CellSegDataset(Dataset):
    """
    Using all cell images and the training set of patch images to 
    perform fine-tuning
    """

    def __init__(
        self, 
        sam_processor: SamProcessor,
        split="train", 
        gt_mask_resize_reso=256,
    ):
        cell_image_paths = glob.glob(
            os.path.join(data_root, "TCT&LCT_CELL", "*", "img", "*.png")
        )
        patch_image_paths = glob.glob(
            os.path.join(data_root, "TCT&LCT_PATCH", split, "*", "img", "*.jpg")
        )
        if split == 'train':
            self.image_paths = cell_image_paths + patch_image_paths
        else:
            self.image_paths = patch_image_paths
        self.split = split
        self.gt_mask_resize_reso = gt_mask_resize_reso

        self.processor = sam_processor


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Path manipulation
        image_path = self.image_paths[idx]
        dir_name, image_name = os.path.split(image_path)

        tmp = os.path.dirname(dir_name)
        cate = os.path.basename(tmp)
        mask_path = os.path.join(tmp, "mask", image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess using SamProcessor
        inputs = self.processor(image, return_tensors='pt')
        # inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Load mask
        if cate == 'N':
            mask = np.zeros(image.size)
        else:
            mask = imageio.imread(mask_path)
        mask = resize_image(mask[..., None], self.gt_mask_resize_reso)
        mask = torch.FloatTensor(mask) / 255.0

        inputs['ground_truth_mask'] = mask

        inputs['category'] = cate

        return inputs, image_path
    

class CellClsfyDataset(Dataset):

    def __init__(
        self, 
        sam_processor: SamProcessor,
        split="train", 
    ):
        cell_image_paths = glob.glob(
            os.path.join(data_root, "TCT&LCT_CELL", "*", "img", "*.png")
        )
        patch_image_paths = glob.glob(
            os.path.join(data_root, "TCT&LCT_PATCH", split, "*", "roi", "*.jpg")
        )
        if split == 'train':
            self.image_paths = cell_image_paths + patch_image_paths
        else:
            self.image_paths = patch_image_paths
        self.split = split

        self.processor = sam_processor

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Path manipulation
        image_path = self.image_paths[idx]
        dir_name, image_name = os.path.split(image_path)

        tmp = os.path.dirname(dir_name)
        cate = os.path.basename(tmp)
        mask_path = os.path.join(tmp, "mask", image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess using SamProcessor
        inputs = self.processor(image, return_tensors='pt')
        # inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs['category_label'] = categories.index(cate)

        return inputs


        
