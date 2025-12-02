import os
from path import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from blocks import *
# from SoftPartConv import PartialConv2d
from dual_gamma_softparconv import DualGammaSoftParConv

def model_save(ckpt_dir, model, optim_E, optim_S, optim_G, epoch, itr=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netE': model.netE.state_dict(),
                'netS': model.netS.state_dict(),
                'netG': model.netG.state_dict(),
                'optim_E': optim_E.state_dict(),
                'optim_S': optim_S.state_dict(),
                'optim_G': optim_G.state_dict()},
               '%s/model_iter_%d_epoch_%d.pth' % (ckpt_dir, itr+1, epoch+1))

def model_load(checkpoint, ckpt_dir, model, optim_E, optim_S, optim_G):
    if not os.path.exists(ckpt_dir):
        epoch = -1
        return model, optim_E, optim_S, optim_G, epoch
    
    ckpt_path = Path(ckpt_dir)
    if checkpoint:
        model_ckpt = ckpt_path + '/' + checkpoint
    else:
        ckpt_lst = ckpt_path.glob('model_iter_*')
        ckpt_lst.sort(key=lambda x: int(x.split('iter_')[1].split('_epoch')[0]))
        model_ckpt = ckpt_lst[-1]
    itr = int(model_ckpt.split('iter_')[1].split('_epoch_')[0])
    epoch = int(model_ckpt.split('iter_')[1].split('_epoch_')[1].split('.')[0])
    print(model_ckpt)

    dict_model = torch.load(model_ckpt)

    model.netE.load_state_dict(dict_model['netE'])
    model.netS.load_state_dict(dict_model['netS'])
    model.netG.load_state_dict(dict_model['netG'])
    optim_E.load_state_dict(dict_model['optim_E'])
    optim_S.load_state_dict(dict_model['optim_S'])
    optim_G.load_state_dict(dict_model['optim_G'])

    return model, optim_E, optim_S, optim_G, epoch, itr

def test_model_load(checkpoint, model):
    dict_model = torch.load(checkpoint)
    model.netE.load_state_dict(dict_model['netE'])
    model.netS.load_state_dict(dict_model['netS'])
    model.netG.load_state_dict(dict_model['netG'])
    return model

def get_scheduler(optimizer, config):
    if config.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.n_epoch - config.n_iter) / float(config.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

class Oct_Conv_aftup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type, alpha_in, alpha_out):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels*alpha_in)
        lf_out = int(out_channels*alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        # 将自己写的 PartConv 注释掉
        # self.conv_h = PartialConv2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        # self.conv_l = PartialConv2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        
        # 使用传统卷积
        self.conv_h = nn.Conv2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.conv_l = nn.Conv2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
    
    def forward(self, x):
        hf, lf = x
        # print(f"in Oct_Conv_aftup, input nan test: hf {torch.isnan(hf).any()}, lf {torch.isnan(lf).any()}" )
        # print(f'{hf.max()}, {hf.min()}')
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        # print(f"in Oct_Conv_aftup,  卷积后的nan test: hf {torch.isnan(hf).any()}, lf {torch.isnan(lf).any()}" )
        return hf, lf

class Oct_conv_reLU(nn.ReLU):# 不涉及卷积操作, 不用修改
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf
    
class Oct_conv_lreLU(nn.LeakyReLU): # 不涉及卷积操作, 不用修改
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf

class Oct_conv_up(nn.Upsample): # 不涉及卷积操作, 不用修改
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf


############## Encoder ##############
class PartialOctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = [1, 1], gamma_in=0.9, gamma_out=1.1, k_in=1.5, k_out=1.2, p_in=1.0, p_out=1.0):
        super(PartialOctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        if type == 'first':
            # 使用自己写的 PartConv代替传统卷积
            self.convh = DualGammaSoftParConv(in_channels=in_channels, out_channels=hf_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias = False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
            self.convl = DualGammaSoftParConv(in_channels=in_channels, out_channels=lf_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
        elif type == 'last':
            # 使用自己写的 PartConv代替传统卷积
            self.convh = DualGammaSoftParConv(in_channels=hf_ch_in, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
            self.convl = DualGammaSoftParConv(in_channels=lf_ch_in, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
            
            # 将传统卷积注释掉
            # self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
            # self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        else:
            # 使用自己写的 PartConv代替传统卷积
            self.L2L = DualGammaSoftParConv( in_channels=lf_ch_in, out_channels=lf_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
            # 将传统卷积注释掉
            # self.L2L = nn.Conv2d(
            #     lf_ch_in, lf_ch_out,
            #     kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False
            # )
            
            # 使用自己写的 PartConv代替传统卷积
            self.H2H = DualGammaSoftParConv( in_channels=hf_ch_in, out_channels=hf_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
            # 将传统卷积注释掉
            # self.H2H = nn.Conv2d(
            #     hf_ch_in, hf_ch_out,
            #     kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False
            # )
            if self.is_dw:
                self.L2H = None
                self.H2L = None
            else:
                # 使用自己写的 PartConv代替传统卷积
                self.L2H = DualGammaSoftParConv( in_channels=lf_ch_in, out_channels=hf_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
                self.H2L = DualGammaSoftParConv( in_channels=hf_ch_in, out_channels=lf_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False, gamma_in=gamma_in, gamma_out=gamma_out, k_in=k_in, k_out=k_out, p_in=p_in, p_out=p_out, use_ratio=False)
                # 将传统卷积注释掉
                # self.L2H = nn.Conv2d(
                #     lf_ch_in, hf_ch_out,
                #     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                # )
                # self.H2L = nn.Conv2d(
                #     hf_ch_in, lf_ch_out,
                #     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                # )
            # 使用自己写的 PartConv代替传统卷积
            
    def forward(self, x, mask):
        if self.type == 'first':
            hf, h_mask = self.convh(in_x=x, in_mask=mask)
            
            lf = self.avg_pool(x)
            lm = self.avg_pool(mask)
            lf, l_mask = self.convl(in_x=lf, in_mask=lm)
            
            return (hf, lf), (h_mask, l_mask)
        elif self.type == 'last':
            hf, lf = x
            hm, lm = mask
            out_h, out_h_mask = self.convh(in_x=hf, in_mask=hm)
            
            out_l, out_l_mask = self.convl(in_x=self.upsample(lf), in_mask=self.upsample(lm))
            output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
            return (output), (out_h, out_l)
        else:
            hf, lf = x
            hm, lm = mask
            if self.is_dw:
                hf, h_mask = self.H2H(in_x=hf, in_mask=hm)
                lf, l_mask = self.L2L(in_x=lf, in_mask=lm)
                # print(self.is_dw)
                # print(f'h_mask: {h_mask.max()}, {h_mask.max()}')
                # print(f'h_mask: {l_mask.max()}, {l_mask.max()}')
            else:
                a = self.H2H(in_x=hf, in_mask=hm)
                b = self.L2H(in_x = self.upsample(lf), in_mask = self.upsample(lm))
                
                c = self.L2L(in_x=lf, in_mask=lm)
                # print(f"hf shape is {hf.shape}, hm shape is {hm.shape}")
                # print(f"pool_hf shape is {self.avg_pool(hf).shape}, pool_hm shape is {self.avg_pool(hm).shape}")
                # print('---------------------------------------------------------------')
                d = self.H2L(in_x = self.avg_pool(hf), in_mask = self.avg_pool(hm))
                
                hf, _ = tuple(x + y for x, y in zip(a, b))
                lf, _ = tuple(x + y for x, y in zip(c, d))
                h_mask = a[1]
                l_mask = c[1]
                # print(self.is_dw)
                # print(f'h_mask: {h_mask.max()}, {h_mask.max()}')
                # print(f'h_mask: {l_mask.max()}, {l_mask.max()}')
            return (hf, lf), (h_mask, l_mask)
        

############## Decoder ##############
class AdaOctConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_div, style_channels, kernel_size,
                 stride, padding, oct_groups, alpha_in, alpha_out, type='normal'):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.type = type
        
        h_in = int(in_channels * (1 - self.alpha_in))
        l_in = in_channels - h_in

        n_groups_h = h_in // group_div
        n_groups_l = l_in // group_div
        
        style_channels_h = int(style_channels * (1 - self.alpha_in))
        style_channels_l = int(style_channels - style_channels_h)
        
        kernel_size_h = kernel_size[0]
        kernel_size_l = kernel_size[1]
        kernel_size_A = kernel_size[2]
        
        # print(f'in channels is {in_channels}')
        # print(h_in)
        self.kernelPredictor_h = KernelPredictor(in_channels=h_in,
                                              out_channels=h_in,
                                              n_groups=n_groups_h,
                                              style_channels=style_channels_h,
                                              kernel_size=kernel_size_h)
        self.kernelPredictor_l = KernelPredictor(in_channels=l_in,
                                               out_channels=l_in,
                                               n_groups=n_groups_l,
                                               style_channels=style_channels_l,
                                               kernel_size=kernel_size_l)
        
        self.AdaConv_h = AdaConv2d(in_channels=h_in, out_channels=h_in, n_groups=n_groups_h)
        self.AdaConv_l = AdaConv2d(in_channels=l_in, out_channels=l_in, n_groups=n_groups_l)
        
        self.OctConv = OctConv(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size_A, stride=stride, padding=padding, groups=oct_groups,
                            alpha_in=alpha_in, alpha_out=alpha_out, type=type)
        
        self.relu = Oct_conv_lreLU()

    def forward(self, content, style, cond='train'):
        c_hf, c_lf = content
        s_hf, s_lf = style
        # print(f'测试AdaoctConv的输入是否为nan, c_hf: {torch.isnan(content[0]).any()}')
        # print(f'测试AdaoctConv的输入是否为nan, c_lf: {torch.isnan(content[1]).any()}')
        # print(f'测试AdaoctConv的输入是否为nan, s_hf: {torch.isnan(style[0]).any()}')
        # print(f'测试AdaoctConv的输入是否为nan, s_lf: {torch.isnan(style[1]).any()}')
        # print(f's_hf shape: {s_hf.shape}')
        h_w_spatial, h_w_pointwise, h_bias = self.kernelPredictor_h(s_hf)
        l_w_spatial, l_w_pointwise, l_bias = self.kernelPredictor_l(s_lf)
        
        if cond == 'train':
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            # print(f'AdaConv 的计算中, output_h是否为nan？{torch.isnan(output_h).any()}')
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            # print(f'AdaConv 的计算中, output_l是否为nan？{torch.isnan(output_l).any()}')
            output = output_h, output_l

            output = self.relu(output)
            # print(f'relu(output)[0] nan test in AdaOctConv: {torch.isnan(output[0]).any()}')
            # print(f'relu(output)[1] nan test in AdaOctConv: {torch.isnan(output[1]).any()}')

            output = self.OctConv(output)
            # for i,tsr in enumerate(output):
            #     print(f'output[{i}] nan test: {torch.isnan(output[i]).any()}')
            if self.type != 'last':
                output = self.relu(output)
            return output
        
        if cond == 'test':
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l
            output = self.relu(output)
            output = self.OctConv(output)
            if self.type != 'last':
                output = self.relu(output)
            return output

class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super(KernelPredictor, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.w_channels = style_channels
        self.kernel_size = kernel_size

        # # Decoder中不需要使用部分卷积
        # padding = (kernel_size - 1) / 2
        # self.spatial = PartialConv2d(
        #     in_channels=style_channels,
        #     out_channels= in_channels * out_channels // n_groups,
        #     kernel_size= kernel_size,
        #     padding= (math.ceil(padding), math.ceil(padding)),
        #     padding_mode= 'reflect'
        # )
        # self.pointwise = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     PartialConv2d(in_channels= style_channels,
        #               put_channels= out_channels * out_channels // n_groups,
        #               kernel_size=1)
        # )
        # self.bias = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     PartialConv2d(in_channels= style_channels,
        #               put_channels= out_channels,
        #               kernel_size= 1)
        # )
        
        # Decoder中依旧使用传统卷积
        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(math.ceil(padding), math.ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        # print("--------------------------------------1 w_spatial working--------------------------------------")
        # print(self.spatial)
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)
        # print("--------------------------------------2 w_pointwise done--------------------------------------")

        # print("--------------------------------------3 bias working--------------------------------------")
        bias = self.bias(w)
        bias = bias.reshape(len(w), self.out_channels)
        # print("--------------------------------------3 bias done--------------------------------------")
        return w_spatial, w_pointwise, bias

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        
        # Decoder中不需要使用部分卷积
        # self.conv = PartialConv2d(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       kernel_size=(kernel_size, kernel_size),
        #                       padding=(math.ceil(padding), math.floor(padding)),
        #                       padding_mode='reflect')
        
        # Decoder中依旧使用传统卷积
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(math.ceil(padding), math.floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        ys = []
        for i in range(len(x)):
            y = self.forward_single(x[i:i+1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x
    
class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = [1, 1]):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        if type == 'first':
            self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, padding_mode=pad_type, bias = False)
            self.convl = nn.Conv2d(in_channels, lf_ch_out,
                                   kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        elif type == 'last':
            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        else:
            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False
            )
            if self.is_dw:
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d(
                    lf_ch_in, hf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
                self.H2L = nn.Conv2d(
                    hf_ch_in, lf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False
            )
            
    def forward(self, x):
        # print('---------------------------in OctConv---------------------------')
        if self.type == 'first':
            hf = self.convh(x)
            # print(f"在第一个OctConv中, hf nan test: {torch.isnan(hf).any()}")
            lf = self.avg_pool(x)
            lf = self.convl(lf)
            # print(f"在第一个OctConv中, lf nan test: {torch.isnan(lf).any()}")
            # print('---------------------------in OctConv---------------------------')
            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
            # print('---------------------------in OctConv---------------------------')
            return output, out_h, out_l
        else:
            hf, lf = x
            # print(f'input nan test. hf is nan? {torch.isnan(hf).any()}. lf is nan? {torch.isnan(lf).any()}')
            # print(f'is_dw : {self.is_dw}')
            if self.is_dw:
                hf, lf = self.H2H(hf), self.L2L(lf)
                # print(f'is_dw == True, nan test. hf is nan? {torch.isnan(hf).any()}. lf is nan? {torch.isnan(lf).any()}')
            else:
                hf, lf = self.H2H(hf) + self.L2H(self.upsample(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))
                # print(f'is_dw == False, nan test. hf is nan? {torch.isnan(hf).any()}. lf is nan? {torch.isnan(lf).any()}')
            # print('---------------------------in OctConv---------------------------')
            return hf, lf
    
    
def main():
    # ---------------------------------------- 测试OctConv, 一共三层, 分别表示初始、中间、最终层 ----------------------------------------
    x = torch.rand(size=(1,3,128,128)).to(device="cuda:1")
    # print(x,x.shape)
    mask = torch.randint(low=0, high=2, size=x.shape).float().to(device="cuda:1")
    # print(mask.shape, mask)
    o1 = OctConv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, type='first').to(device="cuda:1")
   
    # print(f"---------------------------layer e1---------------------------")
    out, mask = o1(x=x, mask=mask)
    # print(len(out))
    # print(len(mask))
    # hm, lm = mask
    # print(hm.shape)

    # up_hm = nn.Upsample(scale_factor=2)(hm)
    # print(up_hm.shape)
    # print(f"---------------------------layer e2---------------------------")
    o2 = OctConv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, groups=1, type='normal').to(device="cuda:1")
    out2, mask2 = o2(x=out, mask=mask)
    # print(len(out2))
    # print(len(mask2))
    
    # print(f"---------------------------layer e3---------------------------")
    o3 = OctConv(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=1, type='last').to(device="cuda:1")
    out3, mask3 = o3(x=out2, mask=mask2)
    # print(len(out2))
    # print(len(mask2))

if __name__ == '__main__':
    main()