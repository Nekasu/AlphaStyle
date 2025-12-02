from tkinter import W
from path import Path
import torch
import torch.nn as nn
# import pandas as pd
# import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomCrop
import random
from torch.utils.data import Dataset

from Config import Config

Image.MAX_IMAGE_PIXELS = 1000000000

class DataSplit(Dataset):
    def __init__(self, config, phase='train'):
        super(DataSplit, self).__init__()

        self.base_transform = Compose([Resize(size=[config.load_size, config.load_size]),
                                RandomCrop(size=(config.crop_size, config.crop_size)),
                                ToTensor(),
                                ])
        self.normalize_transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if phase == 'train':
            print('train mode')
            # Content image data
            img_dir = Path(config.content_dir)
            self.images = self.get_data(img_dir)
            if config.data_num < len(self.images):
                self.images = random.sample(self.images, config.data_num)

            # Style image data and Mask data
            sty_dir = Path(config.style_dir)
            self.style_images = self.get_data(sty_dir)
            
            # mask_dir = Path(config.mask_dir)
            # self.mask_images = self.get_data(mask_dir)
            
            # assert len(self.style_images) == len(self.mask_images)
            # self.style_and_mask = list(zip(self.style_images, self.mask_images))
            # print(self.style_and_mask)
            
            if len(self.images) < len(self.style_images):
                self.style_images = random.sample(self.style_images, len(self.images))
                # self.style_images = random.sample(self.style_images, len(self.images))
            elif len(self.images) > len(self.style_images):
                ratio = len(self.images) // len(self.style_images)
                bias = len(self.images) - ratio * len(self.style_images)
                # print(f'content image: {len(self.images)}, style image: {len(self.style_images)}') print(f'ratio is {ratio}', bias is {bias})
                # print("before unzip---------------------------------")
                # print(self.style_and_mask)
                
                self.style_images = self.style_images * ratio
                self.style_images += random.sample(self.style_images, bias) # 最后还不够的部分使用随机采样补全
                # self.mask_images = self.mask_images * ratio
                # print(f"bias is {bias}")
                # if bias > 0 : # 防止内容图像正好是风格图像的整数倍时, 无法进行unzip操作
                #     self.style_and_mask = random.sample(self.style_and_mask, bias)
                #     # print("after unzip---------------------------------")
                #     # print(self.style_and_mask)
                    
                #     self.style_images_bias, self.mask_images_bias = zip(*self.style_and_mask)
                    
                #     self.style_images += self.style_images_bias
                #     self.mask_images += self.mask_images_bias

            assert len(self.images) == len(self.style_images)
            
        elif phase == 'test':
            print('test mode')
            img_dir = Path(config.content_dir)
            self.images = self.get_data(img_dir)[:config.data_num]
            
            sty_dir = Path(config.style_dir)
            self.style_images = self.get_data(sty_dir)[:config.data_num]
            
            # mask_dir = Path(config.mask_dir)
            # self.mask_images = self.get_data(mask_dir)[:config.data_num]

            # assert len(self.style_images) == len(self.mask_images)
        
        print('content dir:', img_dir)
        print('style dir:', sty_dir)
        # print('mask dir:', mask_dir)
            
    def __len__(self):
        return len(self.images)
    
    def get_data(self, img_dir):
        file_type = ['*.jpg', '*.png', '*.jpeg', '*.tif']
        imgs = []
        for ft in file_type:
            imgs += sorted(img_dir.glob(ft))
        images = sorted(imgs)
        return images

    def __getitem__(self, index):
        # 如果对图像进行预处理, 请在此处进行
        # 内容图像与内容掩膜获取
        cont_img = self.images[index]   # 获取内容图像名称
        cont_img = Image.open(cont_img).convert('RGBA') # 读取内容图像, 并转换为RGBA格式
        cont_img = self.base_transform(cont_img)    # 对内容图像进行基础操作, 进行裁剪等操作

        cont_mask_img = cont_img[3,:,:] # 将alpha通道分离, 当作掩膜
        cont_mask_img = cont_mask_img.unsqueeze(0)
        # cont_mask_img = cont_mask_img.repeat(3,1,1) # 将内容图像掩膜通道数变为3, 从而与 RGB 图像匹配

        cont_rgb_img = cont_img[0:3,:,:] # 将RGB通道分离, 当作真正的内容图像
        cont_rgb_img = self.normalize_transform(cont_rgb_img)   # 进行归一化等操作

        # 风格图像与风格掩膜获取, 与对内容图像的处理完全一样.

        sty_img = self.style_images[index]  # 获取风格图像名称
        sty_img = Image.open(sty_img).convert('RGBA')   # 读取风格图像, 并转换为RGBA格式 
        sty_img = self.base_transform(sty_img)  # ToTensor 函数会自动将数据归一化到 0~1 之间, 所以无须手动归一化

        sty_mask_img = sty_img[3,:,:]   # 将 alpha 通道分离出来, 当作掩膜
        sty_mask_img = sty_mask_img.unsqueeze(0)  # 扩展第一维度, 为下一行代码做准备
        # sty_mask_img = sty_mask_img.repeat(3,1,1)   # 将掩膜图像的通道数变为3, 以与RGB图像匹配, 最终与 SoftPartialConv 代码匹配

        sty_rgb_img = sty_img[0:3,:,:]  # 将 RGB 通道分离出来, 当作真正的风格图像
        sty_rgb_img = self.normalize_transform(sty_rgb_img) # 进行归一化处理
        

        return {'content_img': cont_rgb_img, 'content_mask_img': cont_mask_img, 'style_img': sty_rgb_img, 'mask_img': sty_mask_img}
    
def test_get_data():
    img_dir = '/mnt/sda/Dataset/Detection/WikiArt/wikiart/train'
    img_dir = Path(img_dir)
    config = Config()
    ds = DataSplit(config=config, phase='train')
    
    img = ds.get_data(img_dir=img_dir)
    
    print(img[0:30])

def test_init():
    ### 类的实例化
    config = Config()
    ds = DataSplit(config=config, phase='train')
    
    ### 获取content_images数组
    content_img_dir = '/mnt/sda/Dataset/Detection/COCO/train2017'
    content_img_dir = Path(content_img_dir)
    content_images = ds.get_data(img_dir=content_img_dir)
    
    ### 获取 style_images数组
    style_img_dir = '/mnt/sda/Dataset/Detection/WikiArt/wikiart/train'
    style_img_dir = Path(style_img_dir)
    style_images = ds.get_data(img_dir=style_img_dir)  
    
    ### 打印content_images与sytle_images的长度
    print(f"len of content images: {len(content_images)}\n, len of origin style images: {len(style_images)}\n")
    
    ### 获取mask_images数组
    # 由于现在还没有mask_images的数据, 所以先创建一个mask_images数组模拟一下
    mask_images = []
    for i in range(len(style_images)):
        name_list = style_images[i].split('.')
        name = name_list[0:-1]
        name = ''.join(name)
        appendix = name_list[-1]
        # print(name, appendix)
        mask_images.append(name+'_mask'+'.'+appendix)
    # print(mask_images[0:30],)
    ####### 模拟完成 #########
    
    ### 测试 __init__函数中的逻辑是否正确
    assert len(style_images) == len(mask_images)
    style_and_mask = list(zip(style_images, mask_images))
    
    if len(content_images) < len(style_images):
        style_and_mask = random.sample(style_and_mask, len(content_images))
        style_images, mask_images = zip(*style_and_mask)
        # style_images = random.sample(style_images, len(images))
    elif len(content_images) > len(style_images):
        ratio = len(content_images) // len(style_images)
        bias = len(content_images) - ratio * len(style_images)
        print(ratio, bias)
        
        style_images = style_images * ratio
        print(len(style_images))
        mask_images = mask_images * ratio
        
        style_and_mask = random.sample(style_and_mask, bias)
        style_images_bias, mask_images_bias = zip(*style_and_mask)
        
        style_images += style_images_bias
        mask_images += mask_images_bias
    
    style_10 = '\n'.join(style_images[0:10])
    mask_10 = '\n'.join(mask_images[0:10])
    
    print(f"len of content image is {len(content_images)},\n len of style image is {len(style_images)},\n len of mask image is {len(mask_images)}")
    # print(f"1~10 image name of style image is {style_10}\n")
    # print(f"1~10 image name of mask image is {mask_10}")
        
    assert len(content_images) == len(style_images)
    
def test_phase_test():
    '''
    this function is used to test whether the class 'DataSplit' runs well when 'phase' is test
    '''
    config = Config()
    ds = DataSplit(config=config, phase='test')
    
if __name__ == '__main__':
    ds = DataSplit(config=Config, phase='train')
    c_img = ds[0]['content_img']
    s_img = ds[0]['style_img']
    m_img = ds[0]['mask_img']
    print(c_img.shape, s_img.shape, m_img.shape)
    print(torch.all(c_img == 1),torch.all(s_img == 1),torch.all(m_img == 1))
    torch.set_printoptions(threshold=torch.inf)
    # print(f'c_img: {c_img}')
    # print(f's_img: {s_img}')
    print(f'm_img: {m_img}')
    