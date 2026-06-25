from render import render_image
from train import train
import torch
from data_load import load_nerf_data
from nerf import NeRF
from eval import evaluate_split
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_images, train_poses, H, W, focal = load_nerf_data(split="train")
test_images, test_poses, _, _, _ = load_nerf_data(split="test")

train_images = train_images.to(device)
train_poses = train_poses.to(device)
test_images = test_images.to(device)
test_poses = test_poses.to(device)

model = NeRF(pos_dim=63).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

train(train_images, train_poses, H, W, focal, optimizer, model)

test_mse, test_psnr = evaluate_split(
    model,
    test_images,
    test_poses,
    H,
    W,
    focal,max_images=1
)

print("Test MSE:", test_mse)
print("Test PSNR:", test_psnr)



os.makedirs("outputs", exist_ok=True)
def move_camera_radially(pose, scale):
    new_pose = pose.clone()
    new_pose[:3, 3] = new_pose[:3, 3] * scale
    return new_pose


base_pose = test_poses[0]

novel_pose = move_camera_radially(
    base_pose,
    scale=1.15   
)

novel_img = render_image(
    model,
    novel_pose,
    H,
    W,
    focal,
    near=2.0,
    far=8.0,
    samples=64,
    chunk=512
)


novel_img_np = novel_img.detach().cpu().numpy()
novel_img_np = novel_img_np.clip(0, 1)

plt.imsave(
    "outputs/novel_view.png",
    novel_img_np
)

print("Saved novel view to outputs/novel_view.png")