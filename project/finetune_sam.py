"""
Fine-tune Segment-Anything model on our generated masks
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
import monai
from tqdm.auto import tqdm, trange
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader

from utils import *
from dataset import CellSegDataset

from transformers import SamModel, SamProcessor


def main(
    pretrained_sam_path: str = "./sam-ckpt/sam-vit-base",
    train_batch_size: int = 8,
    test_batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 4e-5,
    num_epochs: int = 100,
    checkpointing_epoch_interv: int = 1,
    output_dir: str = "./output",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Load processor and model
    processor = SamProcessor.from_pretrained(pretrained_sam_path)
    model = SamModel.from_pretrained(pretrained_sam_path)
    print(f"Load SAM processor and model from {pretrained_sam_path}.")

    # Make sure we only compute the gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Create dataset and dataloader
    train_dataset = CellSegDataset(sam_processor=processor, split="train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_dataset = CellSegDataset(sam_processor=processor, split='test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, 
    )

    # Config optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.mask_decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # To device
    model.to(device)
    seg_loss.to(device)
    
    # Train
    print("Start training ...")
    for epoch in range(num_epochs):

        model.train()
        epoch_losses = []
        epoch_progress_bar = trange(len(train_dataloader), desc="Steps")
        for batch in train_dataloader:
            inputs, _ = batch
            # forward
            outputs = model(
                pixel_values=inputs["pixel_values"].to(device),
                multimask_output=False,
                return_dict=True
            )
            # apply sigmoid to the predicted mask
            predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
            # compute loss
            # predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = inputs["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

            # update progress bar
            epoch_progress_bar.update(1)
            logs = {"step_loss": loss.detach().item()}
            epoch_progress_bar.set_postfix(**logs)

        print(f"Epoch: {epoch}")
        print(f'Mean loss: {np.mean(epoch_losses)}')

        # Checkpointing & validation
        if (epoch+1) % checkpointing_epoch_interv == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.eval()
            # validation
            for batch_idx, batch in enumerate(test_dataloader):
                inputs, original_image_paths = batch
                with torch.no_grad():
                    outputs = model(
                        pixel_values=inputs["pixel_values"].to(device),
                        multimask_output=False
                    )
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                # dispaly and save
                fig, axes = plt.subplots(1, len(masks), figsize=(32, 5))
                for i, mask in enumerate(masks):
                    mask = mask.cpu().numpy().squeeze()
                    original_image = imageio.imread(original_image_paths[i])
                    axes[i].imshow(original_image)
                    show_mask(mask, axes[i])
                    axes[i].axis("off")
                fig.savefig(os.path.join(ckpt_dir, f"validation_batch{batch_idx}.png"))
                plt.close(fig)

            # save state dict
            ckpt_save_path = os.path.join(ckpt_dir, "ckpt.pkl")
            torch.save(model.state_dict(), ckpt_save_path)
            print(f"Save checkpoint to {ckpt_save_path}.")

    print("Training finished.")

if __name__ == '__main__':
    main()






