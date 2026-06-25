import torch
from data_load import load_nerf_data
from nerf import NeRF
from eval import evaluate_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_images, test_poses, H, W, focal = load_nerf_data(split="test")

test_images = test_images.to(device)
test_poses = test_poses.to(device)

model = NeRF(pos_dim=63).to(device)
model.load_state_dict(torch.load("checkpoints/nerf_final.pt", map_location=device))
model.eval()
print(test_images.shape)
test_mse, test_psnr = evaluate_split(
    model,
    test_images,
    test_poses,
    H,
    W,
    focal,
    max_images=None
)

print("Test MSE:", test_mse)
print("Test PSNR:", test_psnr)