import torch
import torch.nn as nn
import torch.nn.functional as F

def _safe_pow(x: torch.Tensor, gamma: torch.Tensor):
    # 只在 alpha 非零时执行幂操作
    x = torch.where(x == 0.0, x, torch.clamp(x, 1e-8, 1.0))  # 对非透明区域进行clamp
    return torch.pow(x, gamma)

class DualGammaSoftParConv(nn.Module):
    """
    Dual-Gamma Soft Partial Convolution (simplified version):
      - Removed alpha_floor.
      - Added minimal numerical protection for zero-sum regions.
      - Fully preserves the semantics of alpha=0 (completely transparent).

    Workflow:
        1. dynamic gamma_in/out from alpha.
        2. input gating:  X_g = X * alpha^{gamma_in_eff(alpha)}
        3. convolution:   Y_c = Conv(X_g)
        4. ratio (optional): normalization based on local alpha sum.
        5. output gating: Y = Y_c * alpha^{gamma_out_eff(alpha)}
        6. hard zero-field gating to prevent style leakage in fully transparent regions.
        7. mask update for next layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 gamma_in: float = 1.0,
                 gamma_out: float = 1.0,
                 stabilizer_lambda: float = 0.0,
                 use_ratio: bool = False,
                 cache_alpha: bool = False,
                 clamp_output=None,
                 layer_name=None,
                 dynamic_gamma: bool = True,
                 k_in: float = 2.5,
                 k_out: float = 1.5,
                 p_in: float = 1.0,
                 p_out: float = 1.0):
        super().__init__()
        self.layer_name = layer_name
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self._sum_conv = nn.Conv2d(1, 1, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=1, bias=False, padding_mode=padding_mode)
        with torch.no_grad():
            nn.init.constant_(self._sum_conv.weight, 1.0)
        for p in self._sum_conv.parameters():
            p.requires_grad = False

        self.gamma_in = float(gamma_in)
        self.gamma_out = float(gamma_out)
        self.stabilizer_lambda = float(stabilizer_lambda)
        self.use_ratio = bool(use_ratio)
        self.cache_alpha = bool(cache_alpha)
        self.clamp_output = clamp_output

        self.dynamic_gamma = bool(dynamic_gamma)
        self.k_in = float(k_in)
        self.k_out = float(k_out)
        self.p_in = float(p_in)
        self.p_out = float(p_out)

        self.register_buffer('_k2', torch.tensor(float(kernel_size*kernel_size)), persistent=False)

    def is_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f'nan found in {name}')
        if torch.isinf(tensor).any():
            print(f'inf found in {name}')

    def _compute_effective_gammas(self, alpha: torch.Tensor):
        """
        Map alpha -> gamma_in/out (monotonic differentiable mapping):
          gamma_in_eff  = gamma_in * (1 + k_in  * alpha^p_in)
          gamma_out_eff = gamma_out * (1 + k_out * (1 - alpha)^p_out)
        """
        if not self.dynamic_gamma:
            gin = torch.tensor(self.gamma_in, device=alpha.device, dtype=alpha.dtype).view(1,1,1,1)
            gout = torch.tensor(self.gamma_out, device=alpha.device, dtype=alpha.dtype).view(1,1,1,1)
            return gin, gout

        a = torch.clamp(alpha, 0.0, 1.0)  # Clamp alpha to avoid instability
        gin = self.gamma_in * (1.0 + self.k_in * torch.pow(a, self.p_in))
        gout = self.gamma_out * (1.0 + self.k_out * torch.pow(1.0 - a, self.p_out))

        # Add clamping to ensure stable gamma values
        gin = torch.clamp(gin, min=1e-2, max=10.0)  # Avoid extreme values for gin
        gout = torch.clamp(gout, min=1e-2, max=10.0)  # Avoid extreme values for gout

        self.is_nan_inf(gin, 'gin, line: 93')
        self.is_nan_inf(gout, 'gout, line: 95')

        return gin, gout

    def forward(self, in_x: torch.Tensor, in_mask: torch.Tensor):
        x = in_x
        alpha = in_mask
        alpha = torch.clamp(alpha, 1e-8, 1)
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(1)

        assert x.shape[0] == alpha.shape[0], "Batch size mismatch"
        assert alpha.shape[1] == 1, "Alpha/mask must have shape (B,1,H,W)"

        # Clamp alpha into [0,1] to ensure valid domain
        alpha = alpha.clamp(0.0, 1.0)
        alpha_nonzero = (alpha > 1e-8).float()

        # --- Compute dynamic gammas ---
        gin, gout = self._compute_effective_gammas(alpha)

        # --- Input gating ---
        a_in = _safe_pow(alpha, gin)
        self.is_nan_inf(a_in, 'a_in, line: 116')
        x_gated = x * a_in
        self.is_nan_inf(x_gated, 'x_gated, line: 118')

        # --- Convolution ---
        # print(f'min value before conv: {x_gated.min()}')
        # print(f'max value before conv: {x_gated.max()}')
        y = self.conv(x_gated)
        # print(f'min value after conv: {y.min()}')
        # print(f'max value after conv: {y.max()}')
        self.is_nan_inf(y, 'y, line: 121')


        # --- Local alpha-sum (with minimal numerical protection) ---
        with torch.no_grad():
            sum_m = self._sum_conv(alpha)
            # avoid divide-by-zero without changing alpha semantics
            sum_m = sum_m + (sum_m == 0).float() * 1e-8
            self.is_nan_inf(sum_m, 'sum_m, line: 128')

        if self.use_ratio:
            ratio = (self._k2 / (sum_m + self.stabilizer_lambda)).clamp(0.0, float(self._k2.item()))
            y = y * ratio

        # --- Output gating ---
        a_out = _safe_pow(alpha, gout)
        a_out = a_out * alpha_nonzero
        
        if a_out.shape[-2:] != y.shape[-2:]:
            a_out = F.interpolate(a_out, size=y.shape[-2:], mode='bilinear', align_corners=False)
            self.is_nan_inf(a_out, 'a_out, line: 140')
        y = y * a_out
        self.is_nan_inf(y, 'y, line: 142')

        # --- Hard zero-field gating (prevent style leak) ---
        with torch.no_grad():
            sum_m_true = self._sum_conv(alpha)
            zero_field = (sum_m_true <= 1e-12)
        if zero_field.any():
            y = y.masked_fill(zero_field.expand_as(y), 0.0)

        # --- Updated mask (using original alpha, no lower-bound) ---
        updated_mask = (sum_m / (self._k2 + 1e-8)).clamp(0.0, 1.0)
        self.is_nan_inf(updated_mask, 'updated_mask, line: 153')

        if self.clamp_output is not None:
            y = torch.clamp(y, self.clamp_output[0], self.clamp_output[1])
        self.is_nan_inf(y, 'y, line: 156')
        return y, updated_mask

# ======== Global Stabilizer Scheduler ========
class LambdaScheduler:
    """
    Global stabilizer scheduler shared by all DualGammaSoftParConv layers.
    Decays λ_stab smoothly from lambda_max → lambda_min within decay_steps iterations.
    """
    def __init__(self, lambda_max=2.0, lambda_min=0.5, decay_steps=200000, k=2.3):
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.decay_steps = decay_steps
        self.k = k
        self.global_step = 0

    def get_lambda(self):
        """Return current stabilizer value based on exponential decay."""
        progress = min(self.global_step / self.decay_steps, 1.0)
        return float(
            self.lambda_min
            + (self.lambda_max - self.lambda_min)
            * torch.exp(torch.tensor(-self.k * progress))
        )

    def step(self):
        """Advance one training step."""
        self.global_step += 1


# ======== Auto-update hook for all DualGammaSoftParConv layers ========
def auto_update_lambda(module, inputs):
    """
    This hook automatically updates stabilizer_lambda
    before each forward() if auto_stabilizer=True and a scheduler is attached.
    """
    if hasattr(module, "auto_stabilizer") and module.auto_stabilizer:
        if hasattr(module, "scheduler"):
            module.stabilizer_lambda = module.scheduler.get_lambda()



if __name__ == '__main__':
    dconv = DualGammaSoftParConv(in_channels=3,
                                out_channels=3,
                                kernel_size=3,
                                use_ratio=True)
    x_in = torch.randint(size=(3,3,256,256), low=0, high=255)
    mask_in = torch.rand(size=(3,1,256,256)).clamp(0.0,1.0)
    # print(x_in)
    dconv(x_in, mask_in)