'''
Use model.py->AesFA_test->forward function to generate stylized images.
'''
import argparse
import os
import torch
import numpy as np
import thop
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop

from Config_test import Config
from DataSplit import DataSplit
from model import AesFA_test
from blocks import test_model_load
import generate_results_html

import torch.nn.functional as F

def next_power_of_two(n):
    """找到大于等于n的最小的2的幂"""
    if n <= 0:
        return 1
    if (n & (n - 1)) == 0:  # 已经是2的幂
        return n
    power = 1
    while power < n:
        power <<= 1
    return power

def center_pad_to_power_of_two(img_tensor, target_height, target_width):
    """
    将图像张量填充到指定的2的幂次尺寸，以图像中心为基准。
    期望：RGB 通道已在 [-1,1]，Alpha 通道在 [0,1]。
    因此对 RGB 使用 pad value = -1（对应黑色），对 alpha 使用 0（透明）。
    """
    _, h, w = img_tensor.shape

    # 计算需要填充的上下左右边界
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left

    # 如果 RGBA
    if img_tensor.shape[0] == 4:  # RGBA图像
        rgb_channels = img_tensor[:3, :, :]
        alpha_channel = img_tensor[3:4, :, :]

        # 填充 RGB 通道，使用 -1 表示黑（在 [-1,1] 域）
        rgb_padded = F.pad(rgb_channels,
                           (pad_left, pad_right, pad_top, pad_bottom),
                           mode='constant',
                           value=-1.0)

        # 填充 Alpha 通道（使用 0 表示透明）
        alpha_padded = F.pad(alpha_channel,
                             (pad_left, pad_right, pad_top, pad_bottom),
                             mode='constant',
                             value=0.0)

        padded_img = torch.cat([rgb_padded, alpha_padded], dim=0)
    else:
        # 如果只有 RGB（或其他），默认也使用 -1 填充（黑色）
        padded_img = F.pad(img_tensor,
                           (pad_left, pad_right, pad_top, pad_bottom),
                           mode='constant',
                           value=-1.0)

    return padded_img

def load_img(img_name, img_size, device):
    # 加载图像并转换为RGBA格式
    img = Image.open(img_name).convert('RGBA')

    # 获取原始尺寸
    original_width, original_height = img.size

    # 计算目标尺寸（最接近的2的幂次）
    target_height = next_power_of_two(original_height)
    target_width = next_power_of_two(original_width)

    print(f"Original size: {original_height}x{original_width}")
    print(f"Target size: {target_height}x{target_width}")

    # 转换为Tensor（0..1）
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

    # 将 RGB 映射到 [-1,1]（模型通常期望），但保持 Alpha 在 [0,1]
    rgb = img_tensor[:3, :, :] * 2.0 - 1.0    # now in [-1,1]
    alpha = img_tensor[3:4, :, :]              # stay in [0,1]
    img_tensor = torch.cat([rgb, alpha], dim=0)

    # 中心填充到目标大小（如果需要）
    if original_height == target_height and original_width == target_width:
        padded_img = img_tensor
    else:
        padded_img = center_pad_to_power_of_two(img_tensor, target_height, target_width)

    padded_img = padded_img.to(device)

    # 提取掩膜（alpha）和内容（RGB）
    mask_img = padded_img[3:4, :, :]  # 保持通道维度
    if len(mask_img.shape) == 3:
        mask_img = mask_img.unsqueeze(0)

    true_img = padded_img[0:3, :, :]
    if len(true_img.shape) == 3:
        true_img = true_img.unsqueeze(0)

    print(f'Final mask_img.shape: {mask_img.shape}, true_img.shape: {true_img.shape}')

    return true_img, mask_img, (original_height, original_width)

# 如果你想要更灵活的版本，可以保留img_size参数用于指定最小尺寸：
def load_img_with_min_size(img_name, min_img_size, device):
    """
    版本2：确保图像至少达到某个最小尺寸，同时调整为2的幂次
    """
    img = Image.open(img_name).convert('RGBA')
    original_width, original_height = img.size
    
    # 确保尺寸至少达到min_img_size，然后调整为2的幂次
    target_height = next_power_of_two(max(original_height, min_img_size))
    target_width = next_power_of_two(max(original_width, min_img_size))
    
    print(f"Original: {original_height}x{original_width}")
    print(f"Target: {target_height}x{target_width}")
    
    # 转换为Tensor
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    
    if original_height == target_height and original_width == target_width:
        padded_img = img_tensor
    else:
        padded_img = center_pad_to_power_of_two(img_tensor, target_height, target_width)
    
    padded_img = padded_img.to(device)
    
    # 提取掩膜和内容图像
    mask_img = padded_img[3:4, :, :]
    true_img = padded_img[0:3, :, :]
    
    if len(mask_img.shape) == 3:
        mask_img = mask_img.unsqueeze(0)
    if len(true_img.shape) == 3:
        true_img = true_img.unsqueeze(0)
    
    # true_img = do_normalize_transform(true_img)
    
    return true_img, mask_img
# def next_power_of_two(n):
#     """
#     判断一个数是否是2的n次方，如果不是则提升到最近的2的n次方
#     """
#     # 检查是否是2的幂：2的幂的二进制表示只有一个1
#     if n <= 0:
#         return 1
    
#     # 判断是否是2的幂
#     if (n & (n - 1)) == 0:
#         return n  # 已经是2的幂
    
#     # 找到大于n的最小的2的幂
#     # 方法：将最高位的1左移1位
#     power = 1
#     while power < n:
#         power <<= 1  # 相当于 power *= 2
    
#     return power

# def load_img(img_name, img_size, device):
#     # print(img_name)
#     img = Image.open(img_name).convert('RGBA') # 加载图像, 并转换为 RGBA 格式
#     # print(img.size)
#     img_size = max(img.size)
#     img_size = next_power_of_two(img_size)
#     # print(img_size)
#     # img = ToTensor()(img).to(device)
#     img = do_base_transform(img, img_size).to(device)  # 进行剪切以及 ToTensor操作, 这个操作对掩膜部分与内容/风格部分都是必要的
    
#     # 从整个图像中提取掩膜图像
#     mask_img = img[3,:,:]
#     mask_img = mask_img.unsqueeze(0)
#     # mask_img = mask_img.repeat(3,1,1)
#     if len(mask_img.shape) == 3:
#         mask_img = mask_img.unsqueeze(0)

#     # 从整个图像中提取内容/风格图像
#     true_img = img[0:3,:,:]
#     if len(true_img) == 3:
#         true_img = true_img.unsqueeze(0)
#     true_img = do_normalize_transform(true_img) # 进行归一化等操作
#     # print(f'mask_img.shape: {mask_img.shape}, true_img.shape:{true_img.shape}')
    
#     # print(type(true_img), type(mask_img))
#     # print((true_img.shape),(mask_img.shape))
#     return true_img, mask_img
    

#     # img = do_transform(img, img_size).to(device)
#     # if len(img.shape) == 3:
#     #     img = img.unsqueeze(0)  # make batch dimension
#     # return img

def im_convert(image):
    image = image.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    # image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image * 0.5 + 0.5    # 修改为与输入归一化参数对应的值
    image = image.clip(0, 1)
    return image

def im_convert_alpha(stylized: torch.Tensor, mask: torch.Tensor):
    '''
    一个专门对png图像优化的im_convert方法. 由于在读取图像时没有对alpha通道进行归一化处理, 所以在此刻也不需要进行“反归一化”的处理. 故将alpha通道与RGB通道分开处理分开.
    '''
    stylized = stylized * 0.5 + 0.5
    mask_sliced = mask[:,0:1,:,:] # 经测试发现, mask的三个通道中的内容完全一致
    
    stylized_with_alpha = torch.cat([stylized, mask_sliced], dim=1)
    stylized_with_alpha = stylized_with_alpha.to("cpu").clone().detach().numpy()
    stylized_with_alpha = stylized_with_alpha.transpose(0,2,3,1)
    stylized_with_alpha = stylized_with_alpha.clip(0,1)

    return stylized_with_alpha
    # stylized_with_alpha = torch.cat([stylized, mask_sliced], dim=1)

    # true_image = image[:,0:3,:,:] # 分离RGB通道
    # mask_image = image[:,-1:,:] # 分离alpha通道, 并保持通道数一致
    # print(true_image.shape)
    # print(mask_image.shape)

    # true_image =  true_image * 0.5 + 0.5    # 处理RGB通道, 修改为与输入归一化参数对应的值
    # image = torch.cat([true_image, mask_image], dim=1) # 拼接RGB与alpha通道

    # image = image.to("cpu").clone().detach().numpy()
    # image = image.transpose(0, 2, 3, 1)
    # image = image.clip(0, 1)
    # print(image.shape)
    # return image

def do_base_transform(img, osize):
    transform = Compose([Resize(size=osize),
                        CenterCrop(size=osize),
                            ToTensor()])
    return transform(img)
    
def do_normalize_transform(img):
    # transform = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img)

# def do_transform(img, osize):
#     transform = Compose([Resize(size=osize),  # Resize to keep aspect ratio
#                         CenterCrop(size=osize),
#                         ToTensor(),
#                         Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#     return transform(img)
def center_crop_to_original_size(img_tensor, original_size):
    """
    将填充后的图像裁剪回原始尺寸
    """
    original_height, original_width = original_size
    _, h, w = img_tensor.shape
    
    # 计算裁剪区域
    start_y = (h - original_height) // 2
    start_x = (w - original_width) // 2
    end_y = start_y + original_height
    end_x = start_x + original_width
    
    # 进行裁剪
    cropped_img = img_tensor[:, start_y:end_y, start_x:end_x]
    return cropped_img

def save_img(config, cont_name, sty_name, content, style, stylized, 
             content_original_size, style_original_size, 
             content_mask=None, style_mask=None, freq=False, high=None, low=None):
    
    # 将张量转换回CPU并分离梯度
    content = content.cpu().detach()
    style = style.cpu().detach()
    stylized = stylized.cpu().detach()
    if content_mask is not None:
        content_mask = content_mask.cpu().detach()
    if style_mask is not None:
        style_mask = style_mask.cpu().detach()
    
    # 裁剪回原始尺寸
    content_cropped = center_crop_to_original_size(content[0], content_original_size)
    style_cropped = center_crop_to_original_size(style[0], style_original_size)
    stylized_cropped = center_crop_to_original_size(stylized[0], content_original_size)
    
    # 如果有掩膜，也需要裁剪
    if content_mask is not None:
        content_mask_cropped = center_crop_to_original_size(content_mask[0], content_original_size)
    else:
        content_mask_cropped = None
        
    if style_mask is not None:
        style_mask_cropped = center_crop_to_original_size(style_mask[0], style_original_size)
    else:
        style_mask_cropped = None
    
    # 转换回图像格式
    real_A = im_convert_alpha(content_cropped.unsqueeze(0), content_mask_cropped.unsqueeze(0) if content_mask_cropped is not None else None)
    real_B = im_convert_alpha(style_cropped.unsqueeze(0), style_mask_cropped.unsqueeze(0) if style_mask_cropped is not None else None)
    trs_AtoB_full = im_convert(stylized_cropped.unsqueeze(0))    # 保留完整风格化图像
    trs_AtoB = im_convert_alpha(stylized_cropped.unsqueeze(0), content_mask_cropped.unsqueeze(0) if content_mask_cropped is not None else None) # 将风格化图像使用内容图像掩膜裁剪
    
    A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
    B_image = Image.fromarray((real_B[0] * 255.0).astype(np.uint8))
    trs_image_full = Image.fromarray((trs_AtoB_full[0] * 255.0).astype(np.uint8))
    trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

    A_path = f"{config.img_dir}/{cont_name.stem}_content_{sty_name.stem}.png"
    B_path = f"{config.img_dir}/{cont_name.stem}_style_{sty_name.stem}.png"
    trs_full_path = f"{config.img_dir}/{cont_name.stem}_fullstylized_{sty_name.stem}.png"
    trs_path = f"{config.img_dir}/{cont_name.stem}_stylized_{sty_name.stem}.png"

    A_image.save(A_path)
    B_image.save(B_path)
    trs_image_full.save(trs_full_path)
    trs_image.save(trs_path)
    
    if freq:
        # 裁剪高频和低频图像
        high_cropped = center_crop_to_original_size(high[0], content_original_size)
        low_cropped = center_crop_to_original_size(low[0], content_original_size)
        
        trs_AtoB_high = im_convert(high_cropped.unsqueeze(0))
        trs_AtoB_low = im_convert(low_cropped.unsqueeze(0))

        trsh_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
        trsl_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))
        
        trsh_image.save('{}/{:s}_stylizing_high_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
        trsl_image.save('{}/{:s}_stylizing_low_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    
    return A_path, B_path, trs_full_path, trs_path  # 返回值: 内容图像路径、风格图像路径、风格化图像路径、添加α通道的风格化图像的路径
# def save_img(config, cont_name, sty_name, content, style, stylized, content_mask=None, style_mask=None, freq=False, high=None, low=None):
#     # real_A = im_convert(content)
#     real_A = im_convert_alpha(content, content_mask)
#     # real_B = im_convert(style)
#     real_B = im_convert_alpha(style, style_mask)
#     trs_AtoB_full = im_convert(stylized)    # 保留完整风格化图像
#     trs_AtoB = im_convert_alpha(stylized, content_mask) # 将风格化图像使用内容图像掩膜裁剪
#     # trs_AtoB = im_convert(stylized)
    
#     A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
#     B_image = Image.fromarray((real_B[0] * 255.0).astype(np.uint8))
#     trs_image_full = Image.fromarray((trs_AtoB_full[0] * 255.0).astype(np.uint8))
#     trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

#     A_path = f"{config.img_dir}/{cont_name.stem}_content_{sty_name.stem}.png"
#     B_path = f"{config.img_dir}/{cont_name.stem}_style_{sty_name.stem}.png"
#     trs_full_path = f"{config.img_dir}/{cont_name.stem}_fullstylized_{sty_name.stem}.jpg"
#     trs_path = f"{config.img_dir}/{cont_name.stem}_stylized_{sty_name.stem}.png"

#     A_path = '{}/{:s}_content_{:s}.png'.format(config.img_dir, cont_name.stem, sty_name.stem)
#     A_image.save(A_path)
#     B_path = '{}/{:s}_style_{:s}.png'.format(config.img_dir, cont_name.stem, sty_name.stem)
#     B_image.save(B_path)
#     trs_path ='{}/{:s}_stylized_{:s}.png'.format(config.img_dir, cont_name.stem, sty_name.stem) 
#     trs_image_full.save(trs_full_path)
#     trs_image.save(trs_path)
    
#     if freq:
#         trs_AtoB_high = im_convert(high)
#         trs_AtoB_low = im_convert(low)

#         trsh_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
#         trsl_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))
        
#         trsh_image.save('{}/{:s}_stylizing_high_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
#         trsl_image.save('{}/{:s}_stylizing_low_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    
#     return A_path, B_path, trs_full_path, trs_path  # 返回值: 内容图像路径、风格图像路径、风格化图像路径、添加α通道的风格化图像的路径

        
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', '-id', help='input dir')
    parser.add_argument('--output_dir', '-od', help='output dir')
    parser.add_argument('--input', '-i', help='input imgae path')
    parser.add_argument('--output', '-o', help='output image path')
    parser.add_argument('--pth_path', '-p', help='path of .pth file')
    arg = parser.parse_args()
    
    try:
        # 添加详细的错误日志
        import traceback
        import sys
        
        def exception_handler(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            print("Uncaught exception:", exc_type.__name__, exc_value)
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit(1)
        config = Config()

        device = torch.device(config.cuda_device if torch.cuda.is_available() else 'cpu')
        print('Version:', config.file_n)
        print(device)
        
        with torch.no_grad():
            ## Data Loader
            test_bs = 1
            test_data = DataSplit(config=config, phase='test')
            data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=test_bs, shuffle=False, num_workers=16, pin_memory=False)
            print("Test: ", test_data.__len__(), "images: ", len(data_loader_test), "x", test_bs, "(batch size) =", test_data.__len__())

            ## Model load
            ckpt = os.path.join(config.ckpt_dir, config.ckpt_name)    # ckpt files path&name is from Config.py
            print("checkpoint: ", ckpt)
            model = AesFA_test(config)
            model = test_model_load(checkpoint=ckpt, model=model)
            model.to(device)

            if not os.path.exists(config.img_dir):
                os.makedirs(config.img_dir)

            ## Start Testing
            freq = False                # whether save high, low frequency images or not
            count = 0
            t_during = 0

            A_path_list = []
            B_path_list = []
            trs_path_list = []
            trs_full_path_list = []

            contents = test_data.images # Load Content Images 一个列表, 里面存储了内容图像的名称.
            styles = test_data.style_images    # Load Style Images 一个列表, 里面存储了风格图像的名称.
            # masks = test_data.mask_images# Load Mask Images
            if config.multi_to_multi:   # one content image, N style image
                tot_imgs = len(contents) * len(styles)
                for idx in range(len(contents)):
                    cont_name = contents[idx]           # path of content image
                    content, content_mask, content_origin_size = load_img(cont_name, config.test_content_size, device)

                    for i in range(len(styles)):
                        sty_name = styles[i]            # path of style image
                        style, style_mask, style_origin_size = load_img(sty_name, config.test_style_size, device) # 想要将掩膜从风格图像中提取出来, 就必须改写 load_img 函数. 具体来说, 应该将 load_img 函数改写成类似于 DataSplit.py -> __getitem__函数 中, 从 sty_img 中分离 mask的形式.

                        # mask_name = masks[i]            # path of mask image
                        # mask = load_img(mask_name, config.test_style_size, device)
                        
                        if freq:
                            stylized, stylized_high, stylized_low, during = model(real_A=content, real_B=style, real_mask=style_mask, freq=freq) # Use `AesFA_test.forward` to generate styled images
                            A_path, B_path, trs_path = save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                            A_path_list.append(A_path)
                            B_path_list.append(B_path)
                            trs_path_list.append(trs_path)
                        else:
                            stylized, during = model(
                                real_A=content,
                                real_B=style,
                                real_mask=style_mask,
                                content_mask=content_mask,
                                freq=freq)  # Use `AesFA_test.forward` to generate styled images
                            A_path, B_path, trs_full_path, trs_path = save_img(config, cont_name, sty_name, content, style, stylized, content_original_size=content_origin_size, style_original_size=style_origin_size, content_mask=content_mask, style_mask=style_mask)
                            A_path_list.append(A_path)
                            B_path_list.append(B_path)
                            trs_path_list.append(trs_path)
                            trs_full_path_list.append(trs_full_path)

                        count += 1
                        print(count, idx+1, i+1, during)
                        t_during += during
                        flops, params = thop.profile(model, inputs=(content, style, style_mask, content_mask[:,:1,:,:], freq))
                        print("GFLOPS: %.4f, Params: %.4f"% (flops/1e9, params/1e6))
                        print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=config.cuda_device) / 1024. / 1024. / 1024.))

            else:
                tot_imgs = len(contents)
                for idx in range(len(contents)):
                    cont_name = contents[idx]
                    content = load_img(cont_name, config.test_content_size, device)

                    sty_name = styles[idx]
                    style = load_img(sty_name, config.test_style_size, device)

                    if freq:
                        stylized, stylized_high, stylized_low, during = model(content, style, freq)
                        save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                    else:
                        stylized, during = model(content, style, freq)
                        A_path, B_path, trs_path = save_img(config, cont_name, sty_name, content, style, stylized)
                        A_path_list.append(A_path)
                        B_path_list.append(B_path)
                        trs_path_list.append(trs_path)

                    t_during += during
                    flops, params = thop.profile(model, inputs=(content, style, freq))
                    print("GFLOPS: %.4f, Params: %.4f" % (flops / 1e9, params / 1e6))
                    print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=config.gpu) / 1024. / 1024. / 1024.))


            t_during = float(t_during / (len(contents) * len(styles)))
            print("[AesFA] Content size:", config.test_content_size, "Style size:", config.test_style_size,
                  " Total images:", tot_imgs, "Avg Testing time:", t_during)
            generate_results_html.generate_html(A_path_list, B_path_list, trs_full_path_list,trs_path_list, config)
    except Exception as e:
        print(f"Main function error: {e}")
        traceback.print_exc()
        sys.exit(1)
            
if __name__ == '__main__':
    main()

    # #### 数据读入
    # style_name = '/mnt/sda/Datasets/style_image/AlphaStyle/alpha_WikiArt_AllInOne2/Color_Field_Painting_anne-truitt_knight-s-heritage-1963.png'
    # content_name = '/mnt/sdb/zxt/3_code_area/code_develop/PartialConv_AesFA/input/contents/alpha/transparent_c1_main.png'
    # content, content_mask = load_img(img_name=content_name, img_size=256, device='cuda:1')
    # style, style_mask = load_img(img_name=style_name, img_size=256, device='cuda:1')
    
    # #### 参数设定
    # freq = False                # whether save high, low frequency images or not
    # config = Config()
    # ckpt = os.path.join(config.ckpt_dir, config.ckpt_name)    # ckpt files path&name is from Config.py
    # device = torch.device(config.cuda_device if torch.cuda.is_available() else 'cpu')

    # #### 模型导入
    # model = AesFA_test(config)
    # model = test_model_load(checkpoint=ckpt, model=model)
    # model.to(device)
    
    # #### 模型调用
    # stylized, during = model(
    #     real_A=content,
    #     real_B=style,
    #     real_mask=style_mask,
    #     freq=freq)  # Use `AesFA_test.forward` to generate styled images
    # print(type(stylized))
    # print(f'stylized shape: {stylized.shape}')
    
