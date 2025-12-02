from ast import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple

class PartialConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias = False,
        padding_mode: str = 'reflect',
        device = None,
        dtype = None
    ):
        super(PartialConv2d, self).__init__()
        ##################### 参数引入 #####################
        self.kernel_size = kernel_size
        # print(f"kernel_size is {kernel_size}")
        self.stride = stride
        self.bias: bool = bias
        self.in_channels = in_channels
        ####################################################
    
    #################### 创建卷积核 ####################
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,    # 在卷积之后手动添加偏置
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.conv_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,    # 在卷积之后手动添加偏置
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.conv_1.weight.requires_grad = False    # 设置梯度不自动更新
        self.relu = nn.ReLU()
        # print(f'显示初始化定义时, 卷积核conv的数据: {self.conv.weight}')
            # 设定一个其中参数全为1的卷积, 用于计算公式中的系数 sum(I)/sum(M)
        # print(f'显示初始化定义时, 卷积核conv_1的数据: {self.conv_1.weight}')
        nn.init.constant_(self.conv_1.weight, 1)
        # print(f'显示变1处理后, 卷积核conv的数据: {self.conv.weight}')
        # print(f'显示变1处理后, 卷积核conv_1的数据: {self.conv_1.weight}')
    ####################################################
    def forward(self, in_x, in_mask):
        # print(f'-------------in PartialConv2d-------------')
        # print(f'in_x 的最大最小值: {in_x.max()}, {in_x.min()}')
        # print(f"origin shape is {in_x.shape}")
        # print(f"origin mask shape: {in_mask.shape}")self.sum_mask
        # print(f"conv_1的大小为f{self.conv_1.weight.shape}")
        # print(f"conv_1 的最大最小值为{self.conv_1.weight.max(),self.conv_1.weight.min()}")
        # print(f"conv的大小为f{self.conv.weight.shape}")
        # print(f"in_mask 的大小为f{in_mask.shape}")
        ##################### 参数引入 #####################
        self.in_mask = in_mask
        self.in_x = in_x
        # print(f'显示输入数据in_x的最大最小值：{self.in_x.max()}, {self.in_x.min()}')
        ####################################################
        
        # print(f'in_mask 0/1 test: {torch.where(self.in_mask!=0.0 and self.in_mask!=1.0, False, True).any()}')
        # print(self.in_mask)
        ######################################### 部分卷积过程 #########################################
        # 1. 计算输入特征与掩膜的逐元素相乘 (X ⊙ M)
        # 确保掩膜值在0~1之间
        self.in_mask = torch.clamp(self.in_mask, min=0.0, max=1.0)
        self.calc_x = self.in_x * self.in_mask
        # print(f'显示第一步calc_x的最大最小值：{self.calc_x.max()}, {self.calc_x.min()}')
        
        # 2. 卷积操作 (W^T · (X ⊙ M))
        self.out: torch.Tensor = self.conv(self.calc_x) # out1
        # print(f'显示第二步中卷积核数据的最大最小值: {self.conv.weight.max()}, {self.conv.weight.min()}')
        # print(f'显示第二步out的最大最小值：{self.out.max()}, {self.out.min()}')
        
        # ------------------------------------------------------ 第二步到最终输出中, out值扩大了3~4个数量级, 这是恐怖的, 需要排查原因.
        # 3. 计算系数矩阵 sum(I)/sum(M)
        # print(f"sum mask shape: {self.sum_mask.shape}")
        with torch.no_grad():
            # print(f"self.conv_1 is {self.conv_1}")
            # print(f"self.conv is {self.conv}")
            # print(f'in_mask shape is {self.in_mask.shape}')
            
            self.sum_I: int = self.kernel_size * self.kernel_size * self.in_channels # sum(I)是一个固定值, 从数值上与卷积核的窗口大小相等
            # print(f'sum_I 的值: {self.sum_I}')
            # print(f'in_mask 的值:f{self.in_mask}')
            # print(f'self.in_mask.shape[1] is {self.in_mask.shape[1] } ')
            self.sum_mask: torch.Tensor = self.conv_1(self.in_mask) # 利用一个其中参数全为1的卷积核与输入掩膜相乘, 算出每个像素的sum(M)
            # 确保sum_mask非负
            self.sum_mask = torch.clamp(self.sum_mask, min=1e-8)  # 使用一个小的正数作为下限，避免除零

            # print(f"sum_mask is f{self.sum_mask}")
            # print(f'sum_mask 的最大最小值: {self.sum_mask.max()}, {self.sum_mask.min()}')
            # print(f'sum_mask 的最大、大于0的最小值: {pos_sum_mask.max()}, {pos_sum_mask.min()}')

            # print(f'out.shape is {self.out.shape}')
            # print(f'sum_mask.shape is {self.sum_mask.shape}')
            # print(f'self.out.shape: {self.out.shape}, self.sum_mask.shape: {self.sum_mask.shape}')
            if (self.out.shape == self.sum_mask.shape):
                # 计算ratio
                self.ratio: torch.Tensor = self.sum_I / self.sum_mask
                # 限制ratio的范围：[0, sum_I]
                # sum_I作为上限有明确的物理意义：表示窗口内仅有最小量的有效信息
                self.ratio = torch.clamp(self.ratio, min=0.0, max=float(self.sum_I))
            else:
                print("Error: Shape mismatch between output and mask!")
                return None

            # print(f'ratio.shape is {self.ratio.shape}')
        # print(f'ratio 的最大最小值：{self.ratio.max()}, {self.ratio.min()}')
        # 4. 第二步中的卷积操作结果与第三步中系数矩阵的乘积, 即进行(W^T · (X ⊙ M)) · sum(I)/sum(M)
        self.out: torch.Tensor = self.out * self.ratio # out2
        # print(f'ratio is {self.ratio}')
        # print(f'显示第四步out的最大最小值：{self.out.max()}, {self.out.min()}')
        
        if self.bias is True: #如果卷积核需要添加偏置：
            # 5. 创建一个偏置, 用于弥补卷积核中没有的偏置项, 即进行 (W^T · (X ⊙ M)) · sum(I)/sum(M) + b
            self.b = nn.Parameter(torch.zeros(list(self.out.shape)))  # 手动创建偏置参数
            self.out = self.out + self.b # 将偏置加入到结果中 # out3 # 返回值1
        # self.conv_1 = self.conv
        ################################################################################################
        
        ######################################### 掩膜更新过程 #########################################
        self.updated_mask: torch.Tensor = torch.where(
            (self.sum_mask > 1e-8),
            torch.clamp(self.sum_mask / (self.kernel_size * self.kernel_size * self.in_channels), min=0.0, max=1.0),
            torch.zeros_like(self.sum_mask)
        ) # tensor != 0：生成一个与原张量形状相同的布尔张量，标记出哪些元素不是0。torch.tensor(1)：如果条件为真（即元素不为0），则将其设置为 \frac{1}{K^2} \cdot \sum M。tensor：如果条件为假（即元素为0），则保持原样。# 返回值2
        # print(self.updated_mask)

        # print(f'显示返回值updated_mask的最大最小值：{self.updated_mask.max()}, {self.updated_mask.min()}')
        # print(f"updated mask shape: {self.updated_mask.shape}")
        # print(f"result shape is {self.out.shape}")
        # print(f"-------------------------------------------------------------")
        ################################################################################################
        
        # print(f'显示返回值out的最大最小值：{self.out.max()}, {self.out.min()}')
        # ------------------------------------------------------ 第二步到最终输出中, out值扩大了3~4个数量级, 这是恐怖的, 需要排查原因.
        # print(f'------------------------------------------')
        # 7. 确保输出值在合理范围内
        self.out = torch.clamp(self.out, min=-1.0, max=1.0)  # 假设我们期望输出在[-1,1]范围内
        return self.out, self.updated_mask
        
if __name__ == '__main__':
    pc = PartialConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
    pc2 = PartialConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
    pc3 = PartialConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
    pc4 = PartialConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
    in_x = torch.randint(low=0,high=256, size=(1,3,3,3)).float()
    in_mask = torch.randint(low=0, high=256, size=in_x.shape).float() / 255.0
    # in_mask = torch.zeros_like(in_x)
    # print(in_mask)

    print(f'-------------------------pc1------------------------')
    out_x, out_mask = pc(in_x, in_mask)
    print(f'-------------------------pc2------------------------')
    out_x, out_mask = pc2(out_x, out_mask)
    print(f'-------------------------pc3------------------------')
    out_x, out_mask = pc3(out_x, out_mask)
    print(f'-------------------------pc4------------------------')
    out_x, out_mask = pc4(out_x, out_mask)
    # print(f'mask is {out_mask}')
    # print(f'in_x is {in_x}')
    # print(f'out_x is {out_x}')
    # print((out_mask==1).any())