import torch
from render import render_image
from data_load import load_nerf_data


def psnr_from_mse(mse):
    return -10.0 * torch.log10(mse)


def evaluate_one_image(model, image, pose, H, W, focal):

    pred_img = render_image(
        model=model,
        pose=pose,
        H=H,
        W=W,
        focal=focal,
        near=2.0,
        far=6.0,
        samples=64,
        chunk=4096
    )

    mse = ((pred_img - image) ** 2).mean()
    psnr = psnr_from_mse(mse)

    return mse.item(), psnr.item(), pred_img


def evaluate_split(model, images, poses, H, W, focal, max_images=5):
    model.eval()

    mses = []
    psnrs = []

    if max_images is None:
        max_images = images.shape[0]

    with torch.no_grad():
        for i in range(max_images):
            mse, psnr, _ = evaluate_one_image(
                model,
                images[i],
                poses[i],
                H,
                W,
                focal
            )

            mses.append(mse)
            psnrs.append(psnr)

            print(f"image {i}: MSE={mse:.6f}, PSNR={psnr:.2f}")

    model.train()

    avg_mse = sum(mses) / len(mses)
    avg_psnr = sum(psnrs) / len(psnrs)

    return avg_mse, avg_psnr


