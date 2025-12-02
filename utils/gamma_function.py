'''
this file is for generating curves in the paper
'''
import numpy as np
import matplotlib.pyplot as plt

# === 基本设置 ===
plt.rcParams.update({
    "font.size": 12,
    "font.family": "Arial",
    "axes.linewidth": 1.2,
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False
})

# === 定义变量 ===
alpha = np.linspace(0, 1, 400)

# 定义颜色方案（与前文统一：蓝-橙体系）
colors = {
    "gamma": ["#D64F38", "#76A2B9", "#8D2D2F"],  # 蓝, 橙, 红
    "beta": ["#D64F38", "#76A2B9", "#8D2D2F"],  # 蓝, 橙, 红
    "rho": ["#D64F38", "#76A2B9", "#8D2D2F"],  # 蓝, 橙, 红
    # "beta":  ["#3182CE", "#DD6B20", "#C53030"],
    # "rho":   ["#4299E1", "#F6AD55", "#F56565"]
}

# === 创建子图 ===
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# ---------- (a) γ ----------
gamma_vals = [0.5, 1.0, 2.0]
for g, c in zip(gamma_vals, colors["gamma"]):
    axes[0].plot(alpha, alpha ** g, color=c, lw=2.5, label=f"$\\gamma$={g}")
axes[0].set_title("(a) Effect of γ", weight="bold")
axes[0].set_xlabel("α (Transparency Level)")
axes[0].set_ylabel("α̂ (Modulated)")
axes[0].grid(alpha=0.3)
axes[0].legend(frameon=False, loc="upper left")

# ---------- (b) β ----------
beta_vals = [0.5, 1.0, 1.5]
for b, c in zip(beta_vals, colors["beta"]):
    y = np.clip(b * alpha, 0, 1)
    axes[1].plot(alpha, y, color=c, lw=2.5, label=f"$\\beta$={b}")
axes[1].set_title("(b) Effect of β", weight="bold")
axes[1].set_xlabel("α (Transparency Level)")
axes[1].set_ylabel("α̂ (Modulated)")
axes[1].grid(alpha=0.3)
axes[1].legend(frameon=False, loc="upper left")

# ---------- (c) ρ ----------
rho_vals = [0.5, 1.0, 2.0]
for r, c in zip(rho_vals, colors["rho"]):
    axes[2].plot(alpha, np.power(alpha, r), color=c, lw=2.5, label=f"$\\rho$={r}")
axes[2].set_title("(c) Effect of ρ", weight="bold")
axes[2].set_xlabel("α (Transparency Level)")
axes[2].set_ylabel("α̂ (Modulated)")
axes[2].grid(alpha=0.3)
axes[2].legend(frameon=False, loc="upper left")

# === 版面与导出 ===
plt.tight_layout()
plt.savefig("Fig4_ResponseCurves.svg", dpi=600, bbox_inches="tight")
plt.show()