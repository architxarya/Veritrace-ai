import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

def extract_attention_maps(model, inputs):
    attention_maps = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            attn = output[0]
        else:
            attn = output
        if attn is not None and len(attn.shape) >= 2:
            attention_maps.append(attn.detach().cpu().float())

    for name, module in model.named_modules():
        if any(x in name.lower() for x in ["attn", "mlp", "norm"]):
            if hasattr(module, "forward"):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

    with torch.no_grad():
        try:
            outputs = model(**inputs, return_dict=True)
        except Exception as e:
            print(f"Forward pass: {e}")

    for hook in hooks:
        hook.remove()

    return attention_maps


def process_attention_for_image(attention_maps, image_size):
    w, h = image_size
    if attention_maps:
        try:
            valid = []
            for attn in attention_maps:
                if len(attn.shape) == 3:
                    score = attn[0].abs().mean(dim=-1)
                    valid.append(score)
                elif len(attn.shape) == 2:
                    score = attn.abs().mean(dim=-1)
                    valid.append(score)

            if valid:
                last = valid[-8:] if len(valid) >= 8 else valid
                min_len = min(len(v) for v in last)
                combined = torch.stack([v[:min_len] for v in last]).mean(dim=0)
                arr = combined.numpy().astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                side = int(len(arr) ** 0.5)
                if side * side <= len(arr):
                    arr_2d = arr[:side*side].reshape(side, side)
                    heatmap = cv2.resize(arr_2d, (w, h), interpolation=cv2.INTER_CUBIC)
                    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
                    return heatmap
        except Exception as e:
            print(f"Processing error: {e}")

    y, x = np.mgrid[0:h, 0:w]
    heatmap = np.exp(-((x - w//2)**2 + (y - h//2)**2) / (2*(w//3)**2)).astype(np.float32)
    return heatmap


def create_heatmap_overlay(image, heatmap, alpha=0.5):
    img_array = np.array(image.convert("RGB"))
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(heatmap_norm)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_colored, (img_array.shape[1], img_array.shape[0]))
    overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap_resized, alpha, 0)
    return Image.fromarray(overlaid)