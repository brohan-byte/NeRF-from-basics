import os
import torch
from data_load import load_nerf_data
from nerf import NeRF
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_images, train_poses, H, W, focal = load_nerf_data(split="train")
val_images, val_poses, _, _, _ = load_nerf_data(split="val")

train_images = train_images.to(device)
train_poses = train_poses.to(device)
val_images = val_images.to(device)
val_poses = val_poses.to(device)

model = NeRF(pos_dim=63).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

os.makedirs("checkpoints", exist_ok=True)

train(
    train_images,
    train_poses,
    H,
    W,
    focal,
    optimizer,
    model,
    val_images,
    val_poses
)

torch.save(model.state_dict(), "checkpoints/nerf_final.pt")
print("Saved checkpoints/nerf_final.pt")