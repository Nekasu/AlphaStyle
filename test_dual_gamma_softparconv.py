
import argparse, os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from dual_gamma_softparconv import DualGammaSoftParConv
from Config import Config

def save_tensor(t, path):
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 4:
        t = t[0]
    if t.shape[0] == 1:
        arr = (t[0].numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode='L').save(path)
    else:
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

class DualChain(nn.Module):
    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = out_dir
        cfg = Config()
        args = dict(
            in_channels=cfg.input_nc, out_channels=cfg.input_nc,
            kernel_size=7, stride=1, padding=3,
            gamma_in=cfg.gamma_in, gamma_out=cfg.gamma_out,
            k_in=cfg.k_in, k_out=cfg.k_out,
            p_in=cfg.p_in, p_out=cfg.p_out,
            use_ratio=False
        )
        self.convs = nn.ModuleList([
            DualGammaSoftParConv(**args) for _ in range(5)
        ])

    def forward(self, x, alpha):
        out = x
        for i, conv in enumerate(self.convs, start=1):
            out, alpha = conv(out, alpha)
            save_tensor(alpha, f"{self.out_dir}/{i}.png")
        return out, alpha

def to_tensor(img_np):
    return transforms.ToTensor()(img_np)

def main(args):
    os.makedirs(args.out, exist_ok=True)
    rgba = Image.open(args.image).convert("RGBA")
    rgba_np = np.array(rgba).astype(np.float32) / 255.0

    rgb, a = rgba_np[..., :3], rgba_np[..., 3:4]
    x = to_tensor(rgb).unsqueeze(0)
    alpha = to_tensor(a).unsqueeze(0)

    dc = DualChain(out_dir=args.out).eval()
    with torch.no_grad():
        dc(x, alpha)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, default="./dual_gamma_outputs")
    args = p.parse_args()
    main(args)