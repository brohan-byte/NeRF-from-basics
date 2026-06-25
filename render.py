import torch 
from nerf import camera_space_point, camera_to_world, sample_along_rays, volume_render, positional_encoding
def get_all_rays_for_image(H, W, focal, c2w):
    u, v = torch.meshgrid(
    torch.arange(
            W,
            dtype=torch.float32,
            device=c2w.device
        ),
        torch.arange(
            H,
            dtype=torch.float32,
            device=c2w.device
        ),
        indexing="xy"
    )

    # u, v are [H, W]
    cx = W * 0.5
    cy = H * 0.5

    ray_dirs_cam = camera_space_point(u, v, cx, cy, focal)
    rays_o, rays_d = camera_to_world(ray_dirs_cam, c2w)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def render_image(model, pose, H, W, focal, near=2.0, far=6.0, samples=64, chunk=4096):
    model.eval()

    rays_o, rays_d = get_all_rays_for_image(H, W, focal, pose)

    rendered_chunks = []

    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk):
            rays_o_chunk = rays_o[i:i+chunk]
            rays_d_chunk = rays_d[i:i+chunk]

            points, z_vals = sample_along_rays(
                rays_o_chunk,
                rays_d_chunk,
                near,
                far,
                samples
            )

            points_flat = points.reshape(-1, 3)
            encoded_points = positional_encoding(points_flat, L=10)

            dirs = rays_d_chunk[:, None, :].expand_as(points)
            dirs_flat = dirs.reshape(-1, 3)
            encoded_dirs = positional_encoding(dirs_flat, L=4)

            rgb, sigma = model(encoded_points, encoded_dirs)
            n_rays = rays_o_chunk.shape[0]
            rgb = rgb.reshape(n_rays, samples, 3)
            sigma = sigma.reshape(n_rays, samples)
            pred_rgb = volume_render(rgb, sigma, z_vals)

            rendered_chunks.append(pred_rgb)

    pred_img = torch.cat(rendered_chunks, dim=0)
    pred_img = pred_img.reshape(H, W, 3)

    model.train()

    return pred_img




