import torch
from nerf import (
    sample_along_rays,
    sample_random_rays_one_image,
    volume_render,
    positional_encoding,
)
from eval import evaluate_split





def one_pass(images, poses, H, W, focal, optimizer, model):
    rays_o, rays_d, target_rgb = sample_random_rays_one_image(
        images, poses, H, W, focal, batch_size=1024
    )

    points, z_vals = sample_along_rays(
        rays_o, rays_d, near=2.0, far=6.0, samples=64
    )


    points_flat = points.reshape(-1, 3)
    encoded_points = positional_encoding(points_flat, L=10)

    dirs = rays_d[:, None, :].expand_as(points)
    dirs_flat = dirs.reshape(-1, 3)
    encoded_dirs = positional_encoding(dirs_flat, L=4)

    rgb, sigma = model(encoded_points, encoded_dirs)

    n_rays = rays_o.shape[0]
    n_samples = z_vals.shape[1]

    rgb = rgb.reshape(n_rays, n_samples, 3)
    sigma = sigma.reshape(n_rays, n_samples)
    pred_rgb = volume_render(rgb, sigma, z_vals)

    loss = ((pred_rgb - target_rgb) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(images, poses, H, W, focal, optimizer, model, val_images=None, val_poses=None):
    num_steps = 25000

    for step in range(num_steps):
        loss = one_pass(images, poses, H, W, focal, optimizer, model)

        if step % 100 == 0:
            print(step, "train loss:", loss)

        if val_images is not None and step % 5000 == 0:
            val_mse, val_psnr = evaluate_split(
                model,
                val_images,
                val_poses,
                H,
                W,
                focal,
                max_images=3
            )
            print("VAL:", val_mse, val_psnr)

            torch.save(model.state_dict(), "checkpoints/latest.pt")


