import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import random
from data_load import load_nerf_data

class NeRF(nn.Module):
    def __init__(self, pos_dim=63, dir_dim=27, hidden=256):
        super().__init__()

        self.position_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )

        self.sigma_head = nn.Linear(hidden, 1)
        self.feature_head = nn.Linear(hidden, hidden)

        self.color_mlp = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2),
            nn.ReLU(True),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, encoded_points, encoded_dirs):
        h = self.position_mlp(encoded_points)

        sigma = F.relu(self.sigma_head(h))
        features = self.feature_head(h)

        h_color = torch.cat([features, encoded_dirs], dim=-1)
        rgb = torch.sigmoid(self.color_mlp(h_color))

        return rgb, sigma
    




#applies inverse intrinisic matrix to convert pixel into a camera space ray (only direction no depth)
def camera_space_point(u,v, cx, cy, focal):
    x = (u.float()-cx)/focal
    y = -(v.float()-cy)/focal

    dirs_cam = torch.stack([x,y,-torch.ones_like(x)], dim=-1)
    return dirs_cam



def camera_to_world(ray_dirs_cam, pose):
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    rays_d = ray_dirs_cam @ rotation.T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rays_o = translation.expand_as(rays_d)

    return rays_o, rays_d

def sample_random_rays_one_image(images, poses, H, W , focal, batch_size):
    img_idx = random.randint(0, images.shape[0] - 1)
    img = images[img_idx]
    c2w = poses[img_idx]
    u = torch.randint(
        0,
        W,
        (batch_size,),
        device=images.device
    )

    v = torch.randint(
        0,
        H,
        (batch_size,),
        device=images.device
    )
    cx = W*0.5
    cy = H*0.5
    target_rgb = img[v,u]

    ray_dirs_cam = camera_space_point(u,v,cx,cy, focal)
    rays_o, rays_d = camera_to_world(ray_dirs_cam, c2w)
    return rays_o, rays_d, target_rgb



def sample_along_rays(rays_o, rays_d, near, far, samples):
    batch_size = rays_o.shape[0]
    z_vals = torch.linspace(near, far, samples, device=rays_o.device)
    z_vals = z_vals.expand(batch_size, samples)
    #equation for ray sampling
    points = (rays_o[:, None, :]+z_vals[:,:,None]*rays_d[:, None, :])
    return points, z_vals

def positional_encoding(x, L=10):
    out = [x]
    for i in range(L):
        out.append(
            torch.sin((2**i) * x)
        )
        out.append(
            torch.cos((2**i) * x)
        )
    return torch.cat(out, dim=-1)




def volume_render(rgb, sigma, z_vals):
   

    deltas = z_vals[:, 1:] - z_vals[:, :-1]

  
    infinity_delta = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, infinity_delta], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)

    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10],
            dim=-1
        ),
        dim=-1
    )[:, :-1]

    weights = transmittance * alpha

    pred_rgb = torch.sum(weights[..., None] * rgb, dim=1)

    return pred_rgb


