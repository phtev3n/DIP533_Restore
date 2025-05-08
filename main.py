#!/usr/bin/env python3
import os
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from image_processing import compress_jpeg, gaussian_filter, median_filter, bilateral_filter
from metrics import calculate_psnr, calculate_ssim, calculate_lpips
from analysis import plot_metrics, save_metrics_table
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_classifier,
    classifier_defaults,
)

# === CONFIGURATION ===
IMAGE_DIR         = 'data/source_images'
RESIZE_DIM        = (256, 256)
OUTPUT_DIR        = 'outputs'
RESIZED_DIR       = os.path.join(OUTPUT_DIR, 'resized_sources')
IMG_OUT_DIR       = os.path.join(OUTPUT_DIR, 'images')
JPEG_QUALITIES    = [10, 30, 50, 70, 90]
FILTER_METHODS    = {
    'No Filter': lambda x: x,
    'Gaussian':  gaussian_filter,
    'Median':    median_filter,
    'Bilateral': bilateral_filter,
}

GUIDED_MODEL_PATH  = 'guided-diffusion/models/256x256_diffusion.pt'
CLASSIFIER_PATH    = 'guided-diffusion/models/256x256_classifier.pt'
TIMESTEP_RESPACING = '250'
GUIDANCE_SCALE     = 4.0


# === UTILITIES ===
def convert_png_to_jpg(png_path, jpg_path):
    img = Image.open(png_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(jpg_path, 'JPEG')


def convert_directory_png_to_jpg(directory):
    for fn in os.listdir(directory):
        if fn.lower().endswith('.png'):
            base = os.path.splitext(fn)[0]
            convert_png_to_jpg(
                os.path.join(directory, fn),
                os.path.join(directory, base + '.jpg')
            )


def load_images(src_dir, resize_dim=None, resized_dir=None):
    convert_directory_png_to_jpg(src_dir)
    if resize_dim and resized_dir:
        os.makedirs(resized_dir, exist_ok=True)

    imgs = []
    for fn in sorted(os.listdir(src_dir)):
        if not fn.lower().endswith('.jpg'):
            continue
        path = os.path.join(src_dir, fn)
        pil = Image.open(path).convert('RGB')
        if resize_dim:
            pil = pil.resize(resize_dim, Image.LANCZOS)
            if resized_dir:
                base = os.path.splitext(fn)[0]
                pil.save(os.path.join(resized_dir, f"{base}.jpg"), format='JPEG')
        imgs.append(np.array(pil))
    return imgs


def save_comparison(orig, comp, rest, out_path):
    w, h = orig.size
    canvas = Image.new('RGB', (w*3, h))
    canvas.paste(orig,  (0, 0))
    canvas.paste(comp,  (w, 0))
    canvas.paste(rest,  (w*2, 0))
    canvas.save(out_path)


def make_guided_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])


def jpeg_degrade(x, quality):
    """
    x: Tensor in [-1,1], shape (B,3,H,W)
    """
    x01 = (x + 1) * 0.5
    out = []
    for img in x01:
        arr = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(arr).save(buf, 'JPEG', quality=quality)
        buf.seek(0)
        dec = Image.open(buf).convert('RGB')
        t = torch.from_numpy(np.array(dec)).permute(2,0,1).float().to(x.device) / 255.0
        out.append(t)
    return torch.stack(out) * 2 - 1  # back to [-1,1]


# === LOAD DIFFUSION + CLASSIFIER ===
def load_guided_model(device):
    cfg = model_and_diffusion_defaults()
    # cfg.update({
    #     'image_size':            RESIZE_DIM[0],
    #     'class_cond':            False,    # UNCONDITIONAL UNet
    #     'learn_sigma':           True,
    #     'num_channels':          256,
    #     'num_res_blocks':        2,
    #     'attention_resolutions': '32,16,8',
    #     'diffusion_steps':       1000,
    #     'noise_schedule':        'linear',
    #     'timestep_respacing':    '100',
    #     'rescale_timesteps':     True,
    #     'use_scale_shift_norm':  True,
    #     'use_fp16':              False,
    # })
    cfg.update({
        'image_size':            RESIZE_DIM[0],    # 256
        'class_cond':            False,            # unconditional model
        'learn_sigma':           True,
        'noise_schedule':        'linear',
        'timestep_respacing':    TIMESTEP_RESPACING,
        'num_res_blocks':        2,
        'attention_resolutions': '32,16,8',
        'num_channels':          256,
        'num_head_channels':     64,
        'resblock_updown':       True,
        # rest can stay at their defaults
    })
    model, diffusion = create_model_and_diffusion(**cfg)
    ckpt = torch.load(GUIDED_MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()

    # load the external ImageNet classifier
    clf_cfg    = classifier_defaults()
    clf_cfg['image_size'] = RESIZE_DIM[0]
    classifier = create_classifier(**clf_cfg)
    clf_ckpt   = torch.load(CLASSIFIER_PATH, map_location='cpu')
    classifier.load_state_dict(clf_ckpt, strict=False)
    classifier.to(device).eval()

    return model, diffusion, classifier


# === RESTORE FUNCTIONS ===
def restore_one(model, diffusion, y_t, quality, device):
    # forward to T
    T  = diffusion.num_timesteps - 1
    tT = torch.tensor([T], device=device)
    x  = diffusion.q_sample(y_t, tT)

    # reverse over respaced schedule
    n_steps = len(diffusion.use_timesteps)
    for i in reversed(range(n_steps)):
        ti = torch.tensor([i], device=device)
        x = diffusion.p_sample(
            model,
            x,
            ti,
            clip_denoised=True,
            model_kwargs=None
        )
        if isinstance(x, dict):
            for v in x.values():
                if torch.is_tensor(v):
                    x = v
                    break

    x0 = x
    y_hat  = jpeg_degrade(x0, quality)
    x_rest = x0 - (y_hat - y_t)
    return x_rest.clamp(-1,1)


def guided_restore(
    model,
    diffusion,
    classifier,
    gd_tf,
    device,
    img_np,
    quality,
    classifier_scale=GUIDANCE_SCALE
):
    # 1) preprocess into noisy start y_T
    pil = Image.fromarray(img_np)
    inp = gd_tf(pil).unsqueeze(0).to(device)
    y_T = jpeg_degrade(inp, quality)

    # define classifier-guidance function
    def cond_fn(x, t):
        # make x a leaf with grad enabled
        x = x.detach().clone().requires_grad_(True)
        # re-enable grad inside this block
        with torch.enable_grad():
            logits    = classifier(x, t)                         # (B,1000)
            log_probs = F.log_softmax(logits, dim=-1)            # (B,1000)
            B = x.shape[0]
            labels    = torch.full((B,), quality, dtype=torch.long, device=device)
            # gather the log-prob of our JPEG-quality “class”
            selected  = log_probs[torch.arange(B), labels]       # (B,)
            # backprop that log-prob w.r.t. x
            grads     = torch.autograd.grad(selected.sum(), x)[0]  # (B,3,256,256)
        # scale and return
        return grads * classifier_scale

    # 3) sample with guidance
    samples = diffusion.p_sample_loop(
        model,
        shape=y_T.shape,
        noise=y_T,
        clip_denoised=True,
        model_kwargs={},
        cond_fn=cond_fn,
        device=device,
    )
    x0_hat = samples if isinstance(samples, torch.Tensor) else samples['sample']

    # 4) artifact subtraction
    y_hat  = jpeg_degrade(x0_hat, quality)
    x_rest = x0_hat - (y_hat - y_T)

    # 5) to uint8 numpy
    out = ((x_rest + 1) * 0.5).clamp(0,1)[0]
    arr = (out.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return arr


def evaluate():
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Missing source dir: {IMAGE_DIR}")
    images = load_images(IMAGE_DIR, RESIZE_DIM, RESIZED_DIR)
    if not images:
        raise RuntimeError("No images loaded")
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    logger.configure()
    dist_util.setup_dist()
    device = dist_util.dev()

    model, diffusion, classifier = load_guided_model(device)
    gd_tf = make_guided_transform(RESIZE_DIM[0])

    methods = list(FILTER_METHODS.keys()) + ['Guided']
    metrics = {m: {'PSNR': [], 'SSIM': [], 'LPIPS': []} for m in methods}

    for q in JPEG_QUALITIES:
        tmp = {m: {'PSNR': [], 'SSIM': [], 'LPIPS': []} for m in methods}
        for idx, img in enumerate(images):
            comp     = compress_jpeg(img, q)
            orig_pil = Image.fromarray(img)
            comp_pil = Image.fromarray(comp)

            # traditional filters
            for name, fn in FILTER_METHODS.items():
                rest = fn(comp)
                tmp[name]['PSNR'].append(calculate_psnr(img, rest))
                tmp[name]['SSIM'].append(calculate_ssim(img, rest))
                tmp[name]['LPIPS'].append(calculate_lpips(img, rest))
                save_comparison(
                    orig_pil, comp_pil, Image.fromarray(rest),
                    os.path.join(IMG_OUT_DIR, f"img{idx}_q{q}_{name}.png")
                )

            # classifier-guided diffusion
            rest_g = guided_restore(
                model,
                diffusion,
                classifier,
                gd_tf,
                device,
                comp,
                q,
                GUIDANCE_SCALE
            )
            tmp['Guided']['PSNR'].append(calculate_psnr(img, rest_g))
            tmp['Guided']['SSIM'].append(calculate_ssim(img, rest_g))
            tmp['Guided']['LPIPS'].append(calculate_lpips(img, rest_g))
            save_comparison(
                orig_pil, comp_pil, Image.fromarray(rest_g),
                os.path.join(IMG_OUT_DIR, f"img{idx}_q{q}_Guided.png")
            )

        for m in methods:
            metrics[m]['PSNR'].append(np.mean(tmp[m]['PSNR']))
            metrics[m]['SSIM'].append(np.mean(tmp[m]['SSIM']))
            metrics[m]['LPIPS'].append(np.mean(tmp[m]['LPIPS']))

    qualities = JPEG_QUALITIES
    for metric_name in ['PSNR', 'SSIM', 'LPIPS']:
        data = {m: metrics[m][metric_name] for m in methods}
        plot_metrics(qualities, data, metric_name, OUTPUT_DIR)
        save_metrics_table(qualities, data, metric_name, OUTPUT_DIR)


if __name__ == '__main__':
    evaluate()
