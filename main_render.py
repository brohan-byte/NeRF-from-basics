import os
import torch
import matplotlib.pyplot as plt

from data_load import load_nerf_data
from nerf import NeRF
from render import render_image

def move_camera_radially(pose, scale):
    new_pose = pose.clone()
    new_pose[:3, 3] = new_pose[:3, 3] * scale
    return new_pose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_images, test_poses, H, W, focal = load_nerf_data(split="test")
test_poses = test_poses.to(device)

model = NeRF(pos_dim=63).to(device)
model.load_state_dict(torch.load("checkpoints/nerf_final.pt", map_location=device))
model.eval()

base_pose = test_poses[0]
novel_pose = move_camera_radially(base_pose, scale=1.15)

novel_img = render_image(
    model,
    novel_pose,
    H,
    W,
    focal,
    near=2.0,
    far=8.0,
    samples=64,
    chunk=4096
)

os.makedirs("outputs", exist_ok=True)

novel_img_np = novel_img.detach().cpu().numpy().clip(0, 1)

plt.imsave("outputs/novel_view.png", novel_img_np)

print("Saved outputs/novel_view.png")