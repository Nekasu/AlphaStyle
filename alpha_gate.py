
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian(k=5, sigma=1.0, device="cpu"):
    import math
    grid = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(grid ** 2) / (2.0 * (sigma ** 2) + 1e-8))
    g = g / (g.sum() + 1e-8)
    kernel = (g[:, None] @ g[None, :]).view(1, 1, k, k)
    return kernel

def gaussian_blur(alpha, k=5, sigma=1.0):
    if k <= 1:
        return alpha
    kernel = _gaussian(k, sigma, device=alpha.device)
    return F.conv2d(alpha, kernel, padding=k//2, groups=1)

class AlphaGate(nn.Module):
    """
    Lightweight visibility-aware gating/modulation for decoder features.
    Modes:
        - "mul": multiplicative gate using (blurred) alpha
        - "spade": SPADE-like modulation with gamma,beta conditioned on alpha
    """
    def __init__(self, mode: str = "mul", blur_k: int = 5, blur_sigma: float = 1.0, num_channels: int = None):
        super().__init__()
        assert mode in ("mul", "spade")
        self.mode = mode
        self.blur_k = blur_k
        self.blur_sigma = blur_sigma
        if mode == "spade":
            # tiny conv MLP to produce per-pixel gamma,beta
            hidden = 32
            self.mlp = nn.Sequential(
                nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 2, 3, padding=1)
            )
        self.num_channels = num_channels

    def forward(self, feat: torch.Tensor, alpha: torch.Tensor):
        """
        feat: (N,C,H,W), alpha: (N,1,H,W) in [0,1], same spatial size (will be resized by caller if needed)
        """
        if alpha is None:
            return feat
        a = gaussian_blur(alpha, self.blur_k, self.blur_sigma)
        if self.mode == "mul":
            return feat * a.clamp(0.0, 1.0)
        else:  # spade
            # instance norm then affine from alpha
            mean = feat.mean(dim=(2,3), keepdim=True)
            std  = feat.var(dim=(2,3), keepdim=True).add(1e-5).sqrt()
            norm = (feat - mean) / std
            gb = self.mlp(a)
            gamma, beta = gb[:, :1], gb[:, 1:]
            # broadcast to channels
            gamma = gamma.expand_as(norm)
            beta  = beta.expand_as(norm)
            return gamma * norm + beta
