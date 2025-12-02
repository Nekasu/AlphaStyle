'''
this file can be used to combine layers of PrismLayersPro Dataset, or a directory organized as PrismLayerPro.
'''
import os
import json
import argparse
import cv2
import numpy as np


def alpha_composite_over(dst: np.ndarray, src: np.ndarray, x: int, y: int):
    """
    将 src (RGBA, float32, 0~255) 以 "source over" 方式叠加到 dst 上，
    叠加到 dst 的 [y:y+h, x:x+w] 区域（in-place 修改 dst）。

    dst: (H, W, 4) float32
    src: (h, w, 4) float32
    """
    h, w = src.shape[:2]
    patch = dst[y:y + h, x:x + w]

    # 分离通道并归一化到 0~1
    src_rgb = src[..., :3] / 255.0
    src_a = src[..., 3:4] / 255.0

    dst_rgb = patch[..., :3] / 255.0
    dst_a = patch[..., 3:4] / 255.0

    # standard "over" 公式
    out_a = src_a + dst_a * (1.0 - src_a)
    eps = 1e-6
    out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / np.clip(out_a, eps, 1.0)

    patch[..., :3] = out_rgb * 255.0
    patch[..., 3:4] = out_a * 255.0

    dst[y:y + h, x:x + w] = patch


def find_info_json_for_layer(folder: str, layer_filename: str) -> str:
    """
    根据图层文件名寻找对应的 json 文件。

    规则：
    1. 先尝试 <stem>_info.json，例如:
       layer_01_stylized_xxx.png -> layer_01_stylized_xxx_info.json
    2. 若失败，再尝试利用前缀中的编号：
       若文件名形如 layer_01_xxx.png，则尝试 layer_01_info.json
    """
    stem = os.path.splitext(layer_filename)[0]
    candidate_1 = os.path.join(folder, f"{stem}_info.json")
    if os.path.exists(candidate_1):
        return candidate_1

    # 再尝试 layer_XX_info.json 的形式
    if stem.startswith("layer_"):
        # 提取从 layer_ 后面的连续数字（例如 layer_01_stylized_xxx）
        rest = stem[len("layer_"):]
        num = ""
        for ch in rest:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            candidate_2 = os.path.join(folder, f"layer_{num}_info.json")
            if os.path.exists(candidate_2):
                return candidate_2

    # 若仍未找到，返回第一个候选（用于报错信息）
    return candidate_1


def restore_image_from_layers(folder: str, output_path: str = "restored.png"):
    """
    在给定文件夹中读取:
      - base.png : 背景
      - 若干 layer_*.png 及其 *_info.json : 图层 + 位置信息
    使用 json 中的 box / width_dst / height_dst 将图层叠加到 base 上，
    输出 RGBA 图像。
    """

    base_path = os.path.join(folder, "base.png")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"base.png not found in {folder}")

    base = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        raise RuntimeError(f"Failed to load base.png from {base_path}")

    H, W = base.shape[:2]

    # 将 base 转为 RGBA
    if base.shape[2] == 3:
        alpha = np.ones((H, W, 1), dtype=np.uint8) * 255
        base_rgba = np.concatenate([base, alpha], axis=-1)
    elif base.shape[2] == 4:
        base_rgba = base
    else:
        raise ValueError(f"Unexpected channel count for base.png: {base.shape[2]}")

    # 使用 float32 做叠加
    canvas = base_rgba.astype(np.float32)

    # 遍历所有图层
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".png"):
            continue
        if not filename.startswith("layer_"):
            continue

        layer_path = os.path.join(folder, filename)
        layer = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
        if layer is None:
            print(f"[Warning] Failed to load {filename}, skipped.")
            continue

        # 寻找对应 json
        json_path = find_info_json_for_layer(folder, filename)
        if not os.path.exists(json_path):
            print(f"[Warning] JSON not found for {filename}, tried {json_path}, skipped.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        # 解析 box 与 width_dst / height_dst
        box = info.get("box", None)
        width_dst = info.get("width_dst", None)
        height_dst = info.get("height_dst", None)

        if box is None or width_dst is None or height_dst is None:
            print(f"[Warning] Invalid info json for {filename}, skipped.")
            continue

        x1, y1, x2, y2 = box
        w_from_box = x2 - x1
        h_from_box = y2 - y1

        # 这里根据你给的示例，可以确认:
        # width_dst == x2 - x1, height_dst == y2 - y1
        # 所以 width_dst / height_dst 是目标尺寸，与 box 的跨度一致。
        # 若出现不一致，我们以 width_dst/height_dst 为准，但可以打印警告。
        if w_from_box != width_dst or h_from_box != height_dst:
            print(
                f"[Warning] For {filename}, (x2-x1, y2-y1)=({w_from_box},{h_from_box}) "
                f"!= (width_dst,height_dst)=({width_dst},{height_dst}), "
                f"using width_dst/height_dst as target size."
            )

        w_dst = int(width_dst)
        h_dst = int(height_dst)

        # 边界检查
        if x1 < 0 or y1 < 0 or x1 + w_dst > W or y1 + h_dst > H:
            print(f"[Warning] Layer {filename} exceeds canvas boundary, skipped.")
            continue

        # 确保图层有 alpha
        if layer.shape[2] == 3:
            alpha = np.ones((layer.shape[0], layer.shape[1], 1), dtype=np.uint8) * 255
            layer = np.concatenate([layer, alpha], axis=-1)
        elif layer.shape[2] != 4:
            print(f"[Warning] Unexpected channel count for {filename}, skipped.")
            continue

        # 按目标尺寸缩放图层
        if (layer.shape[1], layer.shape[0]) != (w_dst, h_dst):
            layer = cv2.resize(layer, (w_dst, h_dst), interpolation=cv2.INTER_AREA)

        # 叠加到画布上
        alpha_composite_over(canvas, layer.astype(np.float32), x1, y1)
        print(f"[Info] Composited {filename} at ({x1}, {y1}) size=({w_dst}, {h_dst}).")

    # 保存结果 (RGBA)
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, out)
    print(f"[OK] Restored image saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore an image by compositing layer_*.png onto base.png "
                    "using their *_info.json files."
    )
    parser.add_argument(
        "--folder",
        '-i',
        type=str,
        required=True,
        help="Folder containing base.png, layer_*.png and their *_info.json."
    )
    parser.add_argument(
        "--output",
        '-o',
        type=str,
        default="restored.png",
        help="Output RGBA image path. Default: restored.png"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    restore_image_from_layers(args.folder, args.output)