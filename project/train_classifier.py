"""
Fine-tune Segment-Anything model on our generated masks
"""

import os
import numpy as np
import torch
import torch.nn as nn
import imageio.v3 as imageio
from tqdm.auto import trange
from PIL import Image
from torch.utils.data import DataLoader

from utils import *
from dataset import CellClsfyDataset
from classifier import ImageEmbeddingClassifier

from transformers import SamModel, SamProcessor


def main(
    ch: int = 256,
    ch_mults: list[int] = [1, 1, 2, 2],
    pretrained_sam_path: str = "./sam-ckpt/sam-vit-base",
    train_batch_size: int = 8,
    test_batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 4e-5,
    num_epochs: int = 100,
    checkpointing_epoch_interv: int = 1,
    output_dir: str = "./output/train_classifier",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Load processor and model
    processor = SamProcessor.from_pretrained(pretrained_sam_path)
    sam_model = SamModel.from_pretrained(pretrained_sam_path)
    sam_model.requires_grad_(False)
    print(f"Load SAM processor and model from {pretrained_sam_path}.")

    # Init classifier
    model = ImageEmbeddingClassifier(
        ch, ch_mults, 
        in_reso=64, in_channels=256, out_channels=6, num_blocks_per_layer=2
    )


    # Create dataset and dataloader
    train_dataset = CellClsfyDataset(sam_processor=processor, split="train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_dataset = CellClsfyDataset(sam_processor=processor, split='test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, 
    )

    # Config optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    loss_f = nn.CrossEntropyLoss()

    # To device
    sam_model.vision_encoder.to(device)
    model.to(device)
    loss_f.to(device)
    
    # Train
    print("Start training ...")
    for epoch in range(num_epochs):

        model.train()
        epoch_losses = []
        epoch_accs = []
        epoch_progress_bar = trange(len(train_dataloader), desc="Steps")
        for batch in train_dataloader:
            inputs = batch
            # Image embedding
            image_embeds = sam_model.get_image_embeddings(inputs["pixel_values"].to(device))

            # forward
            pred = model(image_embeds)
            # compute loss
            y = inputs['category_label'].to(device)
            loss = loss_f(pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

            acc = (pred.detach().argmax(dim=-1) == y).sum().item() / len(y)
            epoch_accs.append(acc)

            # update progress bar
            epoch_progress_bar.update(1)
            logs = {"step_loss": loss.detach().item()}
            epoch_progress_bar.set_postfix(**logs)

        print(f"Epoch: {epoch}")
        print(f'Train/loss: {np.mean(epoch_losses)}')
        print(f'Train/accuracy: {np.mean(epoch_accs)}')

        # Checkpointing & validation
        if (epoch+1) % checkpointing_epoch_interv == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.eval()
            # validation
            losses = []
            accs = []
            for batch_idx, batch in enumerate(test_dataloader):
                inputs = batch
                # Image embedding
                image_embeds = sam_model.get_image_embeddings(inputs["pixel_values"].to(device))

                # forward
                # forward
                pred = model(image_embeds)
                # compute loss
                y = inputs['category_label'].to(device)
                loss = loss_f(pred, y)
                # compute accuracy
                y_pred = pred.argmax(dim=-1)
                acc = (y_pred == y).sum().item() / len(y)

                losses.append(loss.item())
                accs.append(acc)
            print(f"Validation/loss: {np.mean(losses)}")
            print(f"Validation/accuracy: {np.mean(accs)}")

            # save state dict
            ckpt_save_path = os.path.join(ckpt_dir, "ckpt.pkl")
            torch.save(model.state_dict(), ckpt_save_path)
            print(f"Save checkpoint to {ckpt_save_path}.")

    print("Training finished.")

if __name__ == '__main__':
    main()






