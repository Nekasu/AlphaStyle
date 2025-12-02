import torch
from torch import nn
import torch.nn.functional as F

from blocks import *
# from SoftPartConv import PartialConv2d
from dual_gamma_softparconv import DualGammaSoftParConv
from alpha_gate import AlphaGate

from Config import Config

def define_network(net_type, config = None):
    net = None
    alpha_in = config.alpha_in
    alpha_out = config.alpha_out
    sk = config.style_kernel
    gamma_in = config.gamma_in
    gamma_out = config.gamma_out
    k_in = config.k_in
    k_out = config.k_out
    p_in = config.p_in
    p_out = config.p_out

    if net_type == 'StyleEncoder':
        net = StyleEncoder(in_dim=config.input_nc, nf=config.nf, style_kernel=[sk, sk], alpha_in=alpha_in, alpha_out=alpha_out, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
    elif net_type == 'ContentEncoder':
        net = ContentEncoder(in_dim=config.input_nc, nf=config.nf, style_kernel=[sk, sk], alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Generator':
        net = Decoder(nf=config.nf, out_dim=config.output_nc, style_channel=512, style_kernel=[sk, sk, 3], alpha_in=alpha_in, freq_ratio=config.freq_ratio, alpha_out=alpha_out)
    return net

class StyleEncoder(nn.Module):
    def __init__(self, in_dim, nf=64, style_kernel=[3, 3], alpha_in=0.5, alpha_out=0.5, gamma_in=0.9, gamma_out=1.1, k_in=1.5, k_out=1.2, p_in=1.0, p_out=1.0):
        '''
        在 Encoder 中, 对于内容图像而言, 实际上并不需要分前景与背景, 仅仅区分风格图像的前景与背景即可. 具体来说,
            1. 假设现有风格图像的前背景分离结果：背景风格信息与前景风格信息
            2. 利用风格图像的背景风格信息与前景风格信息分别对「整个内容图像」进行风格迁移
            3. 得到两种不同风格的「完成风格化图像」后, 再根据「原始内容图像」生成掩膜
            4. 利用内容图像的掩膜对这两种不同的风格图像处理、拼接, 得到最终的结果.
            5. 为了实现以上想法, 需要有如下步骤
                1. 为风格图像特别设立一个 Encoder 类, 命名为 StyleEcoder, 其中使用 PartialOctConv 进行卷积
                    - 即该类
                2. 将原始的 Encoder 更名为 ContentEncoder, 其中使用普通的 OctConv 进行卷积.
        '''
        super(StyleEncoder, self).__init__()
        
        # 替换为自己写的部分卷积
        self.conv =  DualGammaSoftParConv(in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3, gamma_in=gamma_in, gamma_out=gamma_out, use_ratio=False, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        # 注释掉原来的传统卷积
        # self.conv = nn.Conv2d(in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3)        
        
        self.PartialOctConv1_1 = PartialOctConv(in_channels=nf, out_channels=nf, kernel_size=3, stride=2, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="first", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)  
        self.PartialOctConv1_2 = PartialOctConv(in_channels=nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        self.PartialOctConv1_3 = PartialOctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        
        # self.PartialOctConv2_1 = PartialOctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")  
        # self.PartialOctConv2_2 = PartialOctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        # self.PartialOctConv2_3 = PartialOctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        # self.PartialOctConv3_1 = PartialOctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        # self.PartialOctConv3_2 = PartialOctConv(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        # self.PartialOctConv3_3 = PartialOctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        self.PartialOctConv2_1 = PartialOctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        self.PartialOctConv2_2 = PartialOctConv(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        self.PartialOctConv2_3 = PartialOctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)

        self.PartialOctConv3_1 = PartialOctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        self.PartialOctConv3_2 = PartialOctConv(in_channels=4*nf, out_channels=8*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)
        self.PartialOctConv3_3 = PartialOctConv(in_channels=8*nf, out_channels=8*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out)

        self.pool_h = nn.AdaptiveAvgPool2d(output_size=(style_kernel[0], style_kernel[0]))
        self.pool_l = nn.AdaptiveAvgPool2d(output_size=(style_kernel[1], style_kernel[1]))
        
        self.relu = Oct_conv_lreLU()

    def forward(self, x, mask):   
        # print('-------------------------- in styleencoder forward-------------------------- ')
        enc_feat = []
        # print(f'In StyleEncoder 的输入数据中,  数据最大、最小值为{x[0].max(), x[0].min()}')
        # print(f'conv')
        out, mask = self.conv(in_x=x, in_mask=mask) # o1
        # print(f'In StyleEncoder中, 经过conv处理后,  数据最大、最小值为{out[0].max(), out[0].min()}')
        
        #################### Encoder中的第一层 ####################
        # print(f'PartialOctConv1_1')
        out, mask = self.PartialOctConv1_1(x=out, mask=mask) # o2
        out = self.relu(out) # o3
        # print(f'PartialOctConv1_2')
        out, mask = self.PartialOctConv1_2(x=out, mask=mask) # o4
        # print(mask[0].max(),mask[0].min())
        # print(mask[1].max(),mask[1].min())
        out = self.relu(out) # o5
        # print(f'PartialOctConv1_3')
        # print(mask[0].max(),mask[0].min())
        # print(mask[1].max(),mask[1].min())
        out, mask = self.PartialOctConv1_3(x=out, mask=mask) # o6
        out = self.relu(out) # o7
        # print(f'In StyleEncoder 第一层,  数据最大、最小值为{out[0].max(), out[0].min()}')
        # print(f'out7[0] shape: {out[0].shape}, out7[1] shape: {out[1].shape}')
        ############################################################

        #################### Encoder中的第二层 ####################
        # print(f'PartialOctConv2_1')
        out, mask = self.PartialOctConv2_1(x=out, mask=mask) # o8
        out = self.relu(out) # o9
        # print(f'PartialOctConv2_2')
        out, mask = self.PartialOctConv2_2(x=out, mask=mask) # o10
        out = self.relu(out) # o11
        # print(f'PartialOctConv2_3')
        out, mask = self.PartialOctConv2_3(x=out, mask=mask) # o12
        out = self.relu(out) # o13
        # print(f'测试风格编码器的o13是否为nan:{torch.isnan(out[0]).any()},{torch.isnan(out[1]).any()}')
        enc_feat.append(out)
        # print(f'In StyleEncoder 第二层,  数据最大、最小值为{out[0].max(), out[0].min()}')
        # print(f'out13[0] shape: {out[0].shape}, out13[1] shape: {out[1].shape}')
        ############################################################
        
        #################### Encoder中的第三层 ####################
        # print(f'PartialOctConv3_1')
        out, mask = self.PartialOctConv3_1(x=out, mask=mask) # o14
        out = self.relu(out) # o15
        # print(f'PartialOctConv3_2')
        out, mask = self.PartialOctConv3_2(x=out, mask=mask) # o16
        out = self.relu(out) # o17
        # print(f'PartialOctConv3_3')
        out, mask = self.PartialOctConv3_3(x=out, mask=mask) # o18
        out = self.relu(out) # o19
        # print(f'In StyleEncoder 第三层,  数据最大、最小值为{out[0].max(), out[0].min()}')
        # print(f'测试风格编码器的o19是否为nan:{torch.isnan(out[0]).any()},{torch.isnan(out[1]).any()}')
        # print(f'out19[0] shape: {out[0].shape}, out19[1] shape: {out[1].shape}')
        enc_feat.append(out)
        ############################################################

        out_high, out_low = out
        out_sty_h = self.pool_h(out_high)
        out_sty_l = self.pool_l(out_low)
        out_sty = out_sty_h, out_sty_l
        # print('----------------------------------------------------------------------- ')

        return out, out_sty, enc_feat   # o19, downsampled o19, [o13,o19]
    
    def forward_test(self, x, mask, cond):
        # 使用与训练时相同的处理逻辑
        out, out_sty, enc_feat = self.forward(x, mask)
        
        if cond == 'style':
            out_high, out_low = out
            out_sty_h = self.pool_h(out_high)
            out_sty_l = self.pool_l(out_low)
            return out_sty_h, out_sty_l
        else:
            return out

class ContentEncoder(nn.Module):
    def __init__(self, in_dim, nf=64, style_kernel=[3, 3], alpha_in=0.5, alpha_out=0.5):
        '''
        在 Encoder 中, 对于内容图像而言, 实际上并不需要分前景与背景, 仅仅区分风格图像的前景与背景即可. 具体来说,
            1. 假设现有风格图像的前背景分离结果：背景风格信息与前景风格信息
            2. 利用风格图像的背景风格信息与前景风格信息分别对「整个内容图像」进行风格迁移
            3. 得到两种不同风格的「完成风格化图像」后, 再根据「原始内容图像」生成掩膜
            4. 利用内容图像的掩膜对这两种不同的风格图像处理、拼接, 得到最终的结果.
            5. 为了实现以上想法, 需要有如下步骤
                1. 为风格图像特别设立一个 Encoder 类, 命名为 StyleEcoder, 其中使用 PartialOctConv 进行卷积
                2. 将原始的 Encoder 更名为 ContentEncoder, 其中使用普通的 OctConv 进行卷积.
                    - 即该类
        '''
        super(ContentEncoder, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3)        
        
        self.OctConv1_1 = OctConv(in_channels=nf, out_channels=nf, kernel_size=3, stride=2, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="first")       
        self.OctConv1_2 = OctConv(in_channels=nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_3 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        # self.OctConv2_1 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")       
        # self.OctConv2_2 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        # self.OctConv2_3 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        # self.OctConv3_1 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        # self.OctConv3_2 = OctConv(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        # self.OctConv3_3 = OctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        self.OctConv2_1 = OctConv(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_3 = OctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        self.OctConv3_1 = OctConv(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_2 = OctConv(in_channels=4*nf, out_channels=8*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_3 = OctConv(in_channels=8*nf, out_channels=8*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        self.pool_h = nn.AdaptiveAvgPool2d((style_kernel[0], style_kernel[0]))
        self.pool_l = nn.AdaptiveAvgPool2d((style_kernel[1], style_kernel[1]))
        
        self.relu = Oct_conv_lreLU()

    def forward(self, x):   
        # print('-------------------------- in contentencoder forward-------------------------- ')
        enc_feat = []
        # print(f'测试ContentEncoder的输入是否为nan: {torch.isnan(x).any()}')
        out = self.conv(x)  #o1
        # print(f'测试ContentEncoder，o1：{type(out)}')
        
        out = self.OctConv1_1(out) #o2
        # print(f'len(o2):{len(out)}, o2[0].shape: {out[0].shape}, o2[1].shape: {out[1].shape}')
        out = self.relu(out)
        # print(f'len(o3):{len(out)}, o3[0].shape: {out[0].shape}, o3[1].shape: {out[1].shape}')
        out = self.OctConv1_2(out)
        # print(f'len(o4):{len(out)}, o4[0].shape: {out[0].shape}, o4[1].shape: {out[1].shape}')
        out = self.relu(out)
        # print(f'len(o5):{len(out)}, o5[0].shape: {out[0].shape}, o5[1].shape: {out[1].shape}')
        out = self.OctConv1_3(out)
        # print(f'len(o6):{len(out)}, o6[0].shape: {out[0].shape}, o6[1].shape: {out[1].shape}')
        out = self.relu(out) # o7
        #print(f'len(o7):{len(out)}, o7[0].shape: {out[0].shape}, o7[1].shape: {out[1].shape}')
        # print(f'out7[0] shape: {out[0].shape}, out7[1] shape: {out[1].shape}')
        
        out = self.OctConv2_1(out)   
        #print(f'len(o8):{len(out)}, o8[0].shape: {out[0].shape}, o8[1].shape: {out[1].shape}')
        out = self.relu(out)
        out = self.OctConv2_2(out)
        #print(f'len(o10):{len(out)}, o10[0].shape: {out[0].shape}, o10[1].shape: {out[1].shape}')
        out = self.relu(out)
        out = self.OctConv2_3(out)
        #print(f'len(o12):{len(out)}, o12[0].shape: {out[0].shape}, o12[1].shape: {out[1].shape}')
        out = self.relu(out)    # o13
        enc_feat.append(out)    # [o13]
        # print(f'In ContentEncoder 第二层,  数据最大、最小值为{out[0].max(), out[0].min()}')
        # print(f'测试内容编码器的o13是否为nan:{torch.isnan(out[0]).any()},{torch.isnan(out[1]).any()}')
        # print(f'out13[0] shape: {out[0].shape}, out13[1] shape: {out[1].shape}')
        
        out = self.OctConv3_1(out)
        #print(f'len(o14):{len(out)}, o14[0].shape: {out[0].shape}, o14[1].shape: {out[1].shape}')
        out = self.relu(out)
        # print(f'out15[0] shape: {out[0].shape}, out15[1] shape: {out[1].shape}')
        out = self.OctConv3_2(out)
        #print(f'len(o16):{len(out)}, o16[0].shape: {out[0].shape}, o16[1].shape: {out[1].shape}')
        out = self.relu(out)
        # print(f'out17[0] shape: {out[0].shape}, out19[1] shape: {out[1].shape}')
        out = self.OctConv3_3(out)
        #print(f'len(o18):{len(out)}, o18[0].shape: {out[0].shape}, o18[1].shape: {out[1].shape}')
        out = self.relu(out)    # o19
        # print(f'In ContentEncoder 第三层,  数据最大、最小值为{out[0].max(), out[0].min()}')
        # print(f'测试内容编码器的o19是否为nan:{torch.isnan(out[0]).any()},{torch.isnan(out[1]).any()}')
        # print(f'out19[0] shape: {out[0].shape}, out19[1] shape: {out[1].shape}')
        enc_feat.append(out)    # [o13, o19]

        out_high, out_low = out
        out_sty_h = self.pool_h(out_high)
        out_sty_l = self.pool_l(out_low)
        out_sty = out_sty_h, out_sty_l # downsampled o19
        # print('----------------------------------------------------------------------- ')

        return out, out_sty, enc_feat # o19, downsampled o19, [o13,o19]
    
    def forward_test(self, x, cond):
        # 使用与训练时相同的处理逻辑
        out, out_sty, enc_feat = self.forward(x)
        
        if cond == 'style':
            out_high, out_low = out
            out_sty_h = self.pool_h(out_high)
            out_sty_l = self.pool_l(out_low)
            return out_sty_h, out_sty_l
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, nf=64, out_dim=3, style_channel=512, style_kernel=[3, 3, 3], alpha_in=0.5, alpha_out=0.5, freq_ratio=[1,1], pad_type='reflect'):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        self.up_oct = Oct_conv_up(scale_factor=2)

        # print(f'style_channels is {style_channel}')
        self.AdaOctConv1_1 = AdaOctConv(in_channels=8*nf, out_channels=8*nf, group_div=group_div[0], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=4*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_2 = OctConv(in_channels=8*nf, out_channels=4*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_1 = Oct_Conv_aftup(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out)

        self.AdaOctConv2_1 = AdaOctConv(in_channels=4*nf, out_channels=4*nf, group_div=group_div[0], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=4*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=4*nf, out_channels=2*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_2 = Oct_Conv_aftup(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out)

        self.AdaOctConv3_1 = AdaOctConv(in_channels=2*nf, out_channels=2*nf, group_div=group_div[1], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=2*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_2 = OctConv(in_channels=2*nf, out_channels=nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_3 = Oct_Conv_aftup(nf, nf, 3, 1, 1, pad_type, alpha_in, alpha_out)

        self.AdaOctConv4_1 = AdaOctConv(in_channels=nf, out_channels=nf, group_div=group_div[2], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv4_2 = OctConv(in_channels=nf, out_channels=nf//2, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="last", freq_ratio=freq_ratio)
       
       # Decoder中不需要部分卷积
        # self.conv4 = PartialConv2d(in_channels=nf//2, out_channels=out_dim, kernel_size=1)
       
        # 使用传统的卷积核 
        self.conv5 = nn.Conv2d(in_channels=nf//2, out_channels=out_dim, kernel_size=1)

    def forward(self, content, style, content_alpha=None):        
        # print('-------------------------- in decoder forward-------------------------- ')
        # print(f'input nan test, content[0]: {torch.isnan(content[0]).any()}')
        # print(f'input nan test, content[1]: {torch.isnan(content[1]).any()}')
        # print(f'input nan test, style[0]: {torch.isnan(style[0]).any()}')
        # print(f'input nan test, style[1]: {torch.isnan(style[1]).any()}')
        # print(content[0].shape)
        # print(content[1].shape)
        # print(style[0].shape)
        # print(style[1].shape)
        # print(f'-------------------AdaOctConv1_1')
        out = self.AdaOctConv1_1(content, style) #o1, tuple
        # print(f'In Decoder, 经过AdaOctConv1_1后, 数据最大、最小值为{out[0].max(), out[0].min()}')
        # print('o1 start')
        # for i,t in enumerate(out):
        #     print(f'out1 nan test, out1[{i}]: {torch.isnan(out[i]).any()} ')
        # print('o1 end')
        out = self.OctConv1_2(out) #o2, tuple
        out = self.up_oct(out) #o3, tuple
        out = self.oct_conv_aftup_1(out) #o4, tuple

        # print(f'-------------------AdaOctConv2_1')
        out = self.AdaOctConv2_1(out, style) #o5, tuple
        # print(f'在decoder类中, AdaOctConv2_1 的输出是否为nan？{torch.isnan(out[0]).any()}, {torch.isnan(out[1]).any()}')
        # print(f'In Decoder, 经过AdaOctConv2_1后, 数据最大、最小值为{out[0].max(), out[0].min()}')
        out = self.OctConv2_2(out) #o6, tuple
        out = self.up_oct(out) #o7, tuple
        out = self.oct_conv_aftup_2(out) #o8, tuple
        # print(f'在decoder类中, oct_conv_aftup_2的输出是否为nan？{torch.isnan(out[0]).any()}, {torch.isnan(out[1]).any()}')

        # print(f'-------------------AdaOctConv3_1')
        out = self.AdaOctConv3_1(out, style) #o9, tuple
        # print(f'In Decoder, 经过AdaOctConv3_1后, 数据最大、最小值为{out[0].max(), out[0].min()}')
        out = self.OctConv3_2(out) #o10, tuple
        out = self.up_oct(out) #o11, tuple
        out = self.oct_conv_aftup_3(out) #o12, tuple

        # print(f'-------------------AdaOctConv4_1')
        out = self.AdaOctConv4_1(out, style) #o13, tuple
        # print(f'In Decoder, 经过AdaOctConv4_1后, 数据最大、最小值为{out[0].max(), out[0].min()}')
        out = self.OctConv4_2(out) #o14, tuple
        # print('o14 start')
        # for i,t in enumerate(out):
        #     print(f'out14 nan test, out1[{i}]: {torch.isnan(out[i]).any()} ')
        # print('o14 end')
        out, out_high, out_low = out #o15, tuple

        # visibility-aware gating (content alpha) before final rgb projection
        if content_alpha is not None:
            # resize alpha to match feature resolutions
            a_main = F.interpolate(content_alpha, size=out.shape[-2:], mode='bilinear', align_corners=False)
            a_high = F.interpolate(content_alpha, size=out_high.shape[-2:], mode='bilinear', align_corners=False)
            a_low  = F.interpolate(content_alpha, size=out_low.shape[-2:], mode='bilinear', align_corners=False)
            # initialize gates lazily if not present
        try:
            gate = self.alpha_gate
        except AttributeError:
            self.alpha_gate = AlphaGate(mode="mul", blur_k=5, blur_sigma=1.0)
            gate = self.alpha_gate
        out      = gate(out, a_main)
        out_high = gate(out_high, a_high)
        out_low  = gate(out_low, a_low)
        out = self.conv5(out) #o16, tensor
        out_high = self.conv5(out_high) #o17, tensor
        out_low = self.conv5(out_low) #o18, tensor
        
        # print(f'output of Decoder, is type_high a nan? {torch.isnan(out_high).any()}, is type_low a nan? {torch.isnan(out_low).any()}')
        # print('----------------------------------------------------------------------- ')

        return out, out_high, out_low
    
    # def forward_test(self, content, style):
    #     out = self.AdaOctConv1_1(content, style, 'test')
    #     out = self.OctConv1_2(out)
    #     out = self.up_oct(out)
    #     out = self.oct_conv_aftup_1(out)

    #     out = self.AdaOctConv2_1(out, style, 'test')
    #     out = self.OctConv2_2(out)
    #     out = self.up_oct(out)
    #     out = self.oct_conv_aftup_2(out)
     
    #     out = self.AdaOctConv3_1(out, style, 'test')
    #     out = self.OctConv3_2(out)
    #     out = self.up_oct(out)
    #     out = self.oct_conv_aftup_3(out)

    #     out = self.AdaOctConv4_1(out, style) #o13, tuple
    #     out = self.OctConv4_2(out) #o14, tuple
    #     # for t in out:
    #     #     print(f'nan test: {torch.isnan(t).any()}')
    #     #     print(f'inf test: {torch.isinf(t).any()}')
    #     out = self.conv5(out[0]) #o16, tensor

    #     return out


############## Contrastive Loss function ##############
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    mse_loss = nn.MSELoss()
    return mse_loss(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)

    loss = mse_loss(input_mean, target_mean) + \
            mse_loss(input_std, target_std)
    return loss

class EFDM_loss(nn.Module):
    # 
    def __init__(self):
        super(EFDM_loss, self).__init__()
        self.mse_loss = nn.MSELoss() # 一个类, 用于计算均方误差.
    
    def efdm_single(self, style, trans):
        B, C, W, H = style.size(0), style.size(1), style.size(2), style.size(3)
        
        value_style, index_style = torch.sort(style.view(B, C, -1)) # torch.view(B,C,-1)的作用是将 style 从 (B,C,W,H) 变为 (B,C,W*H)的多个一维向量
        value_trans, index_trans = torch.sort(trans.view(B, C, -1))
        inverse_index = index_trans.argsort(-1)
        
        return self.mse_loss(trans.view(B, C,-1), value_style.gather(-1, inverse_index))

    def forward(self, style_E, style_S, translate_E, translate_S, neg_idx):
        '''
        从调用该函数的地方来看, 这四个输入分别是:
        style_E: 是 风格编码器 编码 风格图像 后的 第三个输出, 即风格编码器中的 [o13, o19]
        style_S: 是 content_B_feat 加上 风格编码器 编码 风格图像后的第二个输出, 即 [o13,o19, downsampled_o19]
        translate_E: 是 内容编码器 编码 风格化图像 后的 第三个输出, 即 风格化图像的 [o13, o19]
        translate_S: 是 风格编码器 编码 风格化图像 后的 第三个输出, 即风格化图像的 [o13, o19]
        '''
        loss = 0.
        batch = style_E[0][0].shape[0]
        for b in range(batch):
            poss_loss = 0.
            neg_loss = 0.
        
            # Positive loss
            for i in range(len(style_E)): # len(style_E)为2, i=0,1
                poss_loss += self.efdm_single(style_E[i][0][b].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_E[i][1][b].unsqueeze(0), translate_E[i][1][b].unsqueeze(0)) 
                            # sytle_E[0] 为 o13, style_E[0][0] 为 o13 的高频部分,  style_E[0][1] 为 o13 的低频部分. style_E[0][0][b] 为 o13 高频部分的第b个特征图
                            # sytle_E[1] 为 o19, style_E[1][0] 为 o19 的高频部分,  style_E[1][1] 为 o19 的低频部分. style_E[1][0][b] 为 o19 高频部分的第b个特征图
            for i in range(len(style_S)): # len(style_S)为3, i=0,1,2
                poss_loss += self.efdm_single(style_S[i][0][b].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_S[i][1][b].unsqueeze(0), translate_S[i][1][b].unsqueeze(0))
                
            # Negative loss
            for nb in neg_idx[b]:
                for i in range(len(style_E)):
                    neg_loss += self.efdm_single(style_E[i][0][nb].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_E[i][1][nb].unsqueeze(0), translate_E[i][1][b].unsqueeze(0))
                for i in range(len(style_S)):
                    neg_loss += self.efdm_single(style_S[i][0][nb].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)) + \
                            self.efdm_single(style_S[i][1][nb].unsqueeze(0), translate_S[i][1][b].unsqueeze(0))

            loss += poss_loss / neg_loss

        return loss
    
def main_test_styleencoder():
    x = torch.rand(size=(1,3,128,128)).to(device="cuda:1")
    # print(x,x.shape)
    mask = torch.randint(low=0, high=2, size=x.shape).float().to(device="cuda:1")
    # print(mask.shape, mask)
    
    print("creating StyleEncoder........................")
    e = StyleEncoder(in_dim=3, nf=64, style_kernel=[3,3], alpha_in=0.5, alpha_out=0.5).to(device="cuda:1")
    print("StyleEncoder created sucessfully........................")
    out = e(x=x, mask=mask)
    print(len(out))
    print("StyleEncoder works well........................")
    
def main_test_contentencoder():
    x = torch.rand(size=(1,3,128,128)).to(device="cuda:1")
    # print(x,x.shape)
    
    print("creating ContentEncoder........................")
    e = ContentEncoder(in_dim=3, nf=64, style_kernel=[3,3], alpha_in=0.5, alpha_out=0.5).to(device="cuda:1")
    print("ContentEncoder created sucessfully........................")
    out = e(x=x)
    print(len(out))
    print("ContentEncoder works well........................")
    
def main_test_decoder():
    content_1 = torch.rand(size=(1,128,64,64)).to(device="cuda:1")
    content_2 = torch.rand(size=(1,128,32,32)).to(device="cuda:1")
    content = (content_1, content_2)
    # print(x,x.shape)
    style_1 = nn.AdaptiveAvgPool2d(output_size=(3,3))(content_1).to(device="cuda:1")
    style_2 = nn.AdaptiveAvgPool2d(output_size=(3,3))(content_2).to(device="cuda:1")
    style = (style_1, style_2)
    # print(mask.shape, mask)
    
    config = Config()
    print("creating Generator........................")
    g = define_network(net_type='Generator', config=config).to(device="cuda:1")
    print("Decoder created sucessfully........................")
    out_g = g(content, style)
    print(len(out_g))
    print("Decoder works well........................")
    

if __name__ == '__main__':
    main_test_contentencoder()
