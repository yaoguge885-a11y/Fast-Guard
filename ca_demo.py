"""
Coordinate Attention (CA) + Depthwise Separable Convolution demo
-----------------------------------------------------------------
参考项琴论文 2.2 章节（坐标注意力、深度可分离卷积）。
本示例用于 PPT/技术讲解，不改动现有 ultralytics 推理流水线。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA_Block(nn.Module):
    """
    Coordinate Attention (CA)
    - 先分别在 H、W 方向做全局平均池化，保留单一空间坐标信息；
    - 拼接后经 1x1 Conv → 激活，再拆分为 H/W 分支；
    - 各自 1x1 Conv + Sigmoid 得到轴向注意力图；
    - 在通道维广播，增强目标的“位置感知”，抑制背景大楼等干扰。
    """

    def __init__(self, inp_channels: int, reduction: int = 32, activation: str = "relu"):
        super().__init__()
        assert inp_channels > 0, "inp_channels must be > 0"
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 按 W 压缩
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 按 H 压缩

        mid_channels = max(8, inp_channels // reduction)
        self.conv1 = nn.Conv2d(inp_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        if activation.lower() == "hswish":
            self.act = nn.Hardswish()
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, inp_channels, kernel_size=1, stride=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, inp_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # 1) 轴向全局池化
        x_h = self.pool_h(x)              # [B, C, H, 1]
        x_w = self.pool_w(x)              # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)     # [B, C, W, 1]，与论文保持一致

        # 2) 拼接后压缩通道
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3) 拆分回 H/W 分支
        x_h_att, x_w_att = torch.split(y, [h, w], dim=2)
        x_w_att = x_w_att.permute(0, 1, 3, 2)  # [B, C, 1, W]

        # 4) 生成轴向注意力权重
        a_h = torch.sigmoid(self.conv_h(x_h_att))  # [B, C, H, 1]
        a_w = torch.sigmoid(self.conv_w(x_w_att))  # [B, C, 1, W]

        # 5) 位置感知增强：对非机动车的小目标在其所在行/列加权，弱化大楼等背景
        out = x * a_h * a_w
        return out


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 = depthwise (逐通道) + pointwise (1x1) 两步，显著减少 FLOPs：
    - 常规 3x3 卷积：FLOPs ~ C_in * C_out * 3 * 3
    - 深度可分离：FLOPs ~ C_in * 3 * 3 + C_in * C_out * 1 * 1
    用于保持 1024 输入推理时的实时性。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TinyBackboneWithCA(nn.Module):
    """
    示例：在 backbone 中插入 CA 模块（可用于答辩 PPT 演示）。
    结构：Stem → 深度可分离卷积 → CA → 深度可分离卷积。
    """

    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.Hardswish(),
        )
        self.dw1 = DepthwiseSeparableConv(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)
        self.ca = CA_Block(base_ch * 2, reduction=16, activation="hswish")
        self.dw2 = DepthwiseSeparableConv(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dw1(x)
        x = self.ca(x)
        x = self.dw2(x)
        return x


if __name__ == "__main__":
    # 简单跑通一遍，供 PPT 截图/演示
    model = TinyBackboneWithCA(in_ch=3, base_ch=32)
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(dummy)
    print("TinyBackboneWithCA output shape:", y.shape)
    # 说明：在 YOLO 中可将 CA_Block 嵌入到早期/中期的特征提取层，增强位置感知，
    # 对非机动车小目标在长条状/遮挡背景下依然能被强调，同时深度可分离卷积控制计算量。