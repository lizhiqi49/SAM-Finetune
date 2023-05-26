"""
Pipeline that perform segmentation using pretrained SAM model
and classification using pretrained classifier
"""

import os
import argparse
import numpy as np
import torch
import imageio
from PIL import Image

from utils import *
from classifier import ImageEmbeddingClassifier
from transformers import SamModel, SamProcessor

categories = ['AGC', 'ASC-H', 'ASCUS', 'HSIL', 'LSIL', 'N']

def main(
    image_path: str,
    pretrained_classifier_path: str,
    pretrained_sam_path: str,
    output_dir: str = './output'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Load SAM processor and model
    processor = SamProcessor.from_pretrained(pretrained_sam_path)
    sam_model = SamModel.from_pretrained(pretrained_sam_path)
    print(f"Load SAM processor and model from {pretrained_sam_path}.")

    # Load Classifier
    clsfier = ImageEmbeddingClassifier(ch=256, ch_mults=[1, 1, 2, 2], num_blocks_per_layer=2)
    clsfier.load_state_dict(torch.load(pretrained_classifier_path))

    # To device
    sam_model.to(device)
    clsfier.to(device)

    # Load image
    image = Image.open(image_path)
    inputs = processor(image, return_tensors='pt')

    # Predict masks
    with torch.no_grad():
        outputs = sam_model(
            pixel_values=inputs["pixel_values"].to(device),
            multimask_output=False,
            return_dict=True
        )

    # Postprocess
    mask = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0].squeeze()

    # Save mask
    mask_image = np.zeros(mask.shape)
    mask_image[mask] = 255
    mask_image = mask_image.astype(np.uint8)
    mask_path = os.path.join(output_dir, 'mask.png')
    imageio.imwrite(mask_path, mask_image)
    print(f"Save mask image to {mask_path}.")

    # ROI cropping from mask
    roi = bbox_cropping(np.array(image), mask)
    roi = Image.fromarray(roi)

    # Classification
    inputs = processor(roi, return_tensors='pt')
    image_embeds = sam_model.get_image_embeddings(inputs["pixel_values"].to(device))
    with torch.no_grad():
        pred = clsfier(image_embeds)[0]
    label = pred.argmax()
    print("Pathological class: ", categories[label])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, 
                        help="The path of image that should be segmented and classified.")
    parser.add_argument('--pretrained_classifier_path', type=str,
                        help="The path of pretrained classifier's state dict")
    parser.add_argument('--pretrained_sam_path', type=str,
                        help="""The local/huggingface path of pretrained SAM model,
                                notice that it should be a directory that contains 
                                config files and model state dict""")
    parser.add_argument('--output_dir', type=str,
                        help="The directory for saving output files")
    args = parser.parse_args()

    main(**vars(args))





