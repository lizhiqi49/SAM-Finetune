"""
Generate masks for cell PATCH images

"""

import os
import cv2
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import glob
from typing import Optional
from PIL import Image

from utils import *

from transformers import pipeline, SamModel, SamProcessor, SamImageProcessor


class SamMaskGenerator:

    def __init__(self, pretrained_sam_path="facebook/sam-vit-base"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SamModel.from_pretrained(pretrained_sam_path).to(self.device)
        self.processor = SamProcessor.from_pretrained(pretrained_sam_path)

    def generate_masks(
        self, 
        image: Image, 
        input_boxes: Optional[list[int]] = None,
        input_points: Optional[list[int]] = None,
        input_labels: Optional[list[int]] = None,
    ):
        # Preprocess
        inputs = self.processor(
            image, 
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        # Predict masks
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        return masks
    
def customize_prompts(image):
    # Pick bounding box
    image_orig = image.copy()
    mouse = DrawImageMouse()
    box = mouse.draw_image_rectangle_on_mouse(image, winname="pick bounding box")
    roi: np.ndarray = image_orig[box[1]:box[3], box[0]:box[2]]
    image = draw_image_boxes(image, [box], color=(0,0,255), thickness=2)
    # show the image
    cv2.imshow("pick bounding box", image)
    cv2.waitKey(0)

    # Pick points
    points, labels = draw_points_on_mouse(image)

    return [[box]], [points], [labels], roi

    

if __name__ == '__main__':
    pretrained_sam_path = "sam-ckpt/sam-vit-base"
    data_root = "./data/TCT&LCT_PATCH"
    cell_classes = ['AGC', 'ASC-H', 'ASCUS', 'HSIL', 'LSIL', 'N']
    # cell_classes = ['N']
    
    # Init SamMaskGenerator
    mask_generator = SamMaskGenerator(pretrained_sam_path)

    for c in cell_classes:

        image_paths = glob.glob(os.path.join(data_root, '*', c, 'img', '*.jpg'))
        for img_path in image_paths:
            # img_path = os.path.join(image_root, img_file)
            tmp_root, img_file = os.path.split(img_path)
            tmp_root = os.path.dirname(tmp_root)
            mask_root = os.path.join(tmp_root, 'mask')
            os.makedirs(mask_root, exist_ok=True)
            mask_path = os.path.join(mask_root, img_file)

            if os.path.exists(mask_path):
                continue
            else:
                # Load image
                image = Image.open(img_path).convert("RGB")
                image_cv2 = cv2.imread(img_path)

                # Show image and get point coordinates by clicking
                input_boxes, input_points, input_labels, roi = customize_prompts(image_cv2)
                
                # Generate masks
                masks = mask_generator.generate_masks(
                    image,
                    input_boxes,
                    input_points,
                    input_labels
                )

                # Show masks on image and choose the best one by keyboard input
                show_masks_on_image(image, masks)
                mask_id = eval(input("Please choose the prefered mask id:")) - 1
                mask = masks[0, mask_id, ...]

                # Save the chosen mask and roi
                mask_image = np.zeros(mask.shape)
                mask_image[mask] = 1
                mask_image *= 255
                mask_image = mask_image.astype(np.uint8)
                # save mask
                imageio.imwrite(mask_path, mask_image)
                print(f"Save mask image to {mask_path}.")

                # save roi
                roi_root = os.path.join(tmp_root, 'roi')
                os.makedirs(roi_root, exist_ok=True)
                roi_path = os.path.join(roi_root, img_file)
                imageio.imwrite(roi_path, roi[..., [2,1,0]])
                print(f"Save ROI image to {roi_path}.")
            











