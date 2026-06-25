import os 
import torch 
import numpy as np 
import json
from PIL import Image
import random
import torch.nn as nn
import torch.functional as F


def load_nerf_data(scene_name="hotdog", split="train"):

    json_path = f"nerf_synthetic/{scene_name}/transforms_{split}.json"
    with open(json_path, "r") as f:
        meta = json.load(f)
    images = []
    poses = []
    for frame in meta["frames"]:
        image_path = f"nerf_synthetic/{scene_name}/{frame['file_path'][2:]}.png"
        image = Image.open(image_path)
        image = np.array(image).astype(np.float64)/255
        if(image.shape[-1]==4):
            rgb = image[..., :3]
            alpha = image[..., 3:4]
            image = rgb * alpha + (1.0 - alpha)

        images.append(image)
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        poses.append(pose)

    poses = torch.tensor(np.stack(poses))
    images = torch.tensor(np.stack(images))
    H = images.shape[1]
    W = images.shape[2]
    camera_angle_x = meta["camera_angle_x"]
    focal = (
        0.5 * W
        /
        np.tan(0.5 * camera_angle_x)
    )
    return images, poses, H, W, focal 

