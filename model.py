import torch
from torch import nn
import networks
import blocks
import time

from vgg19 import vgg, VGG_loss
from networks import EFDM_loss

class AesFA(nn.Module):
    def __init__(self, config):
        super(AesFA, self).__init__()

        self.config = config
        self.device = self.config.device

        self.lr = config.lr
        self.lambda_percept = config.lambda_percept
        self.lambda_const_style = config.lambda_const_style

        self.netE = networks.define_network(net_type='ContentEncoder', config = config)    # Content Encoder
        self.netS = networks.define_network(net_type='StyleEncoder', config = config)    # Style Encoder
        self.netG = networks.define_network(net_type='Generator', config = config)

        self.vgg_loss = VGG_loss(config, vgg)
        self.efdm_loss = EFDM_loss()

        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))

        self.E_scheduler = blocks.get_scheduler(self.optimizer_E, config)
        self.S_scheduler = blocks.get_scheduler(self.optimizer_S, config)
        self.G_scheduler = blocks.get_scheduler(self.optimizer_G, config)

        
    def forward(self, data):
        # data 是一个存放数据的字典, 其中具有三个键值对, 三个键分别为 content_img, style_img, mask_img
        ############################# 数据预处理: 将 RGB 通道与 alpha 通道分离, RGB 部分正常进入网络, alpha通道 当作掩膜, 归一化后进入网络. ############################# 
        ############################# 数据处理：转到 cuda 上 #############################
        self.real_A = data['content_img'].to(self.device)
        self.real_B = data['style_img'].to(self.device)
        self.real_C = data['mask_img'].to(self.device)
        self.content_alpha = data.get('content_mask_img', None)
        if self.content_alpha is not None:
            self.content_alpha = self.content_alpha.to(self.device)
        ##################################################################################
       
       ################################### 使用编码器 ###################################
        # print("内容编码器编码内容图像")
        # print('测试StyleEncoder的输入是否为nan:')
        # print('    1:')
        # for i, t in enumerate(self.real_A):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        self.content_A, _, _ = self.netE(x = self.real_A) # use Content Encoder
        # print('测试ContentEncoder的输出是否为nan：')
        # for i, t in enumerate(self.content_A):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())

        # print("风格编码器编码风格图像")
        # print('测试StyleEncoder的输入是否为nan:')
        # print('    1:')
        # for i, t in enumerate(self.real_B):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        # print('    2:')
        # for i, t in enumerate(self.real_C):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        _, self.style_B, self.content_B_feat = self.netS(x = self.real_B, mask = self.real_C)# use Style Encoder
        # print('测试StyleEncoder的输出是否为nan：')
        # print('    1:')
        # for i, t in enumerate(self.style_B):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        # print('    2:')
        # for i, t in enumerate(self.content_B_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
       ##################################################################################
            
        ######################### 损失函数计算准备工作：数据准备1 #########################
        self.style_B_feat = self.content_B_feat.copy() # 复制风格图像经过风格编码器后的第三个返回值, 并命名为 style_B_feat, 将用于损失函数计算
        self.style_B_feat.append(self.style_B)  #向 style_B_feat 中添加风格图像经过编码器后的第二个返回值:tuple, 将用于损失函数计算
       ##################################################################################
        
        ######################### 使用解码器 ###################################
        alpha_1ch = None
        if self.content_alpha is not None:
            alpha_1ch = self.content_alpha[:, :1, :, :]
        self.trs_AtoB, self.trs_AtoB_high, self.trs_AtoB_low = self.netG(self.content_A, self.style_B, alpha_1ch)
        # print('测试Decoder的输出是否为nan：')
        # print("     1:")
        # for i, t in enumerate(self.trs_AtoB):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        # print("     2:")
        # for i, t in enumerate(self.trs_AtoB_high):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        # print("     3:")
        # for i, t in enumerate(self.trs_AtoB_low):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
       ##################################################################################

        ######################### 损失函数计算准备工作：数据准备2 #########################
        # print("内容编码器编码风格化图像")
        self.trs_AtoB_content, _, self.content_trs_AtoB_feat = self.netE(self.trs_AtoB)
        # print('测试损失函数计算时, 内容编码器的输出是否为nan：')
        # print("     1:")
        # for i, t in enumerate(self.trs_AtoB_content):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        # print("     2:")
        # for i, t in enumerate(self.content_trs_AtoB_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())

        trs_AtoB_mask = torch.ones_like(self.trs_AtoB)[:,0:1,:,:]
        # print("风格编码器编码风格化图像")
        _, self.trs_AtoB_style, self.style_trs_AtoB_feat = self.netS(x=self.trs_AtoB, mask=trs_AtoB_mask)
        # print('测试损失函数计算时, 风格编码器的输出是否为nan：')
        # print("     1:")
        # for i, t in enumerate(self.trs_AtoB_style):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        # print("     2:")
        # for i, t in enumerate(self.style_trs_AtoB_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any())
        self.style_trs_AtoB_feat.append(self.trs_AtoB_style)
       ##################################################################################

        
    def calc_G_loss(self):
        # print(f'------------------------------G_percept问题检测------------------------------')
        ###################### 感知损失 G_percept 问题查询 ###################### 
            # 检查是否是输入数据有误
        # print(f"输入数据1 self.real_A 是否为 nan : {torch.isnan(self.real_A).any()}, {self.real_A.max()}, {self.real_A.min()}")
        # print(f"输入数据2 self.real_B 是否为 nan : {torch.isnan(self.real_B).any()}, {self.real_B.max()}, {self.real_B.min()}")
        # print(f"输入数据3 self.trs_AtoB 是否为 nan : {torch.isnan(self.trs_AtoB).any()}, {self.trs_AtoB.max()}, {self.trs_AtoB.min()}")
        # 下面的输出结果表明，不是inf的问题, 所以将下面的代码注释掉了
        # print(f"输入数据1 self.real_A 是否为 inf : {torch.isinf(self.real_A).any()}")
        # print(f"输入数据2 self.real_B 是否为 inf : {torch.isinf(self.real_B).any()}")
        # print(f"输入数据3 self.trs_AtoB 是否为 inf : {torch.isinf(self.trs_AtoB).any()}")
        self.G_percept, self.neg_idx = self.vgg_loss.perceptual_loss(self.real_A, self.real_B, self.trs_AtoB)
        # print(f"损失函数G_percept 是否为nan: {torch.isnan(self.G_percept).any()}")
        # print(f"损失函数G_percept 是否为inf: {torch.isinf(self.G_percept).any()}")
        ###################### ###################### ###################### ####

        self.G_percept *= self.lambda_percept
        # print(f'--------------------------------------------------------------------------')

        ###################### 对比损失 G_contrast 问题查询 ###################### 
        # print(f'------------------------------G_contrast问题检测------------------------------')
        # print(len(self.content_B_feat)) # 2
        # print(len(self.style_B_feat))# 3
        # print(len(self.content_trs_AtoB_feat))# 2
        # print(type(self.style_trs_AtoB_feat),len(self.style_trs_AtoB_feat))# 3
        # print(type(self.neg_idx))# 

        # print('input1 start')
        # for i, t in enumerate(self.content_B_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any(), {tsr.max()}, {tsr.min()})    
        # print('input1 end')

        # print('input2 start')
        # for i, t in enumerate(self.style_B_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any(), {tsr.max()}, {tsr.min()})    
        # print('input2 end')

        # print('input3 start')
        # for i, t in enumerate(self.content_trs_AtoB_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any(), {tsr.max()}, {tsr.min()})    
        # print('input3 end')

        # print('input4 start')
        # for i, t in enumerate(self.style_trs_AtoB_feat):
        #     for tsr in t:
        #         print(torch.isnan(tsr).any(), {tsr.max()}, {tsr.min()})    
        # print('input4 end')

        # print('input5 start')
        # for i, t in enumerate(self.neg_idx):
        #     print(type(t))
        #     # for tsr in t:
        #     #     print(torch.isnan(tsr).any())    
        # print('input5 end')

        self.G_contrast = self.efdm_loss(self.content_B_feat, self.style_B_feat, self.content_trs_AtoB_feat, self.style_trs_AtoB_feat, self.neg_idx) * self.lambda_const_style 
        # content_B_feat 是风格编码器编码风格图像后的第三个输出, 即风格编码器中的 [o13, o19]
        # style_B_feat 是 content_B_feat 加上风格编码器编码风格图像后的第二个输出, 即 [o13,o19, downsampled_o19]
        # content_trs_AtoB_feat 是 内容编码器 编码 风格化图像 后的 第三个输出, 即 风格化图像的 [o13, o19]
        # style_trs_AtoB_feat 是 风格编码器 编码 风格化图像 后的 第三个输出, 即风格化图像的 [o13, o19]

        # print(f"损失函数G_contrast 是否为nan: {torch.isnan(self.G_contrast).any()}")
        # print(f"损失函数G_contrast 是否为inf: {torch.isinf(self.G_contrast).any()}")
        # print(f'--------------------------------------------------------------------------')
        ###################### ###################### ###################### ####

        self.G_loss = self.G_percept + self.G_contrast

        
    def train_step(self, data):
        self.set_requires_grad([self.netE, self.netS, self.netG], True)

        self.forward(data)
        self.calc_G_loss()

        self.optimizer_E.zero_grad()
        self.optimizer_S.zero_grad()
        self.optimizer_G.zero_grad()
        self.G_loss.backward()
        self.optimizer_E.step()
        self.optimizer_S.step()
        self.optimizer_G.step()

        train_dict = {}
        train_dict['G_loss'] = self.G_loss
        train_dict['G_Percept'] = self.G_percept
        train_dict['G_Contrast'] = self.G_contrast
        
        train_dict['style_img'] = self.real_B
        train_dict['fake_AtoB'] = self.trs_AtoB
        train_dict['fake_AtoB_high'] = self.trs_AtoB_high
        train_dict['fake_AtoB_low'] = self.trs_AtoB_low
        
        return train_dict
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

                    
class AesFA_test(nn.Module):
    def __init__(self, config):
        super(AesFA_test, self).__init__()

        self.netE = networks.define_network(net_type='ContentEncoder', config = config)    # Content Encoder
        self.netS = networks.define_network(net_type='StyleEncoder', config = config)    # Style Encoder
        self.netG = networks.define_network(net_type='Generator', config = config)
        
    def forward(self, real_A, real_B, real_mask, content_mask, freq):
        with torch.no_grad():
            start = time.time() # sign the start time
            
            # 使用与训练时相同的处理流程
            content_A, _, _ = self.netE(x=real_A)
            _, style_B, _ = self.netS(x=real_B, mask=real_mask)
            
            if freq:
                trs_AtoB, trs_AtoB_high, trs_AtoB_low = self.netG(content_A, style_B)
                end = time.time()
                during = end - start
                return trs_AtoB, trs_AtoB_high, trs_AtoB_low, during
            else:
                trs_AtoB = self.netG(content_A, style_B, content_mask)[0]  # 只取第一个返回值
                end = time.time()
                during = end - start
                return trs_AtoB, during
    
    def style_blending(self, real_A, real_B_1, real_B_2):
        '''
        a function used only in test_video.py and style_blending.py
        '''
        with torch.no_grad():
            start = time.time()
            content_A = self.netE.forward_test(real_A, 'content')
            style_B1_h = self.netS.forward_test(real_B_1, 'style')[0]
            style_B2_l = self.netS.forward_test(real_B_2, 'style')[1]
            style_B = style_B1_h, style_B2_l

            trs_AtoB = self.netG.forward_test(content_A, style_B)
            end = time.time()
            during = end - start
            
        return trs_AtoB, during