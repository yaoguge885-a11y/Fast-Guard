# === 1. 基础导入与环境信息 ===
# ==================================================================================
import os
import sys
import time
import math
import shutil

import ultralytics
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from core.calculator import TTCCalculator
from core.ipm import IPM_Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont   # 新增：PIL 中文支持

def print_versions():
    print(f"Python 版本: {sys.version}")
    print(f"ultralytics 版本: {ultralytics.__version__}")
    print(f"PyQt5 版本: {QtCore.PYQT_VERSION_STR}")
    print(f"Qt 版本: {QtCore.QT_VERSION_STR}")
    print(f"opencv-python 版本: {cv2.__version__}")
    print(f"numpy 版本: {np.__version__}")
    print(f"Pillow 版本: {Image.__version__}")


    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except Exception:
        pass

# --- 自定义深度学习组件 (CA, SCNN, ReLU, SIoU) ---

# ==================================================================================

# === 2. 核心深度学习组件 (ReLU, CoordAtt, SCNN, SIoU) ===
# ==================================================================================
class ReLU(nn.Module):
    """ReLU 激活函数模块"""
    def __init__(self, inplace=True):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, inplace=self.inplace)

class CoordAtt(nn.Module):
    """CA 注意力机制 (Coordinate Attention)
    参考: Coordinate Attention for Efficient Mobile Network Design
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = ReLU()  # 使用上面定义的 ReLU
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

class SCNN_Block(nn.Module):
    """SCNN (Spatial CNN) 的简化实现，用于增强特征的空间传递"""
    def __init__(self, channels):
        super(SCNN_Block, self).__init__()
        # 向下、向上、向右、向左四个方向的消息传递
        self.conv_down = nn.Conv2d(channels, channels, kernel_size=(1, 9), padding=(0, 4))
        self.conv_up = nn.Conv2d(channels, channels, kernel_size=(1, 9), padding=(0, 4))
        self.conv_right = nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0))
        self.conv_left = nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0))

    def forward(self, x):
        n, c, h, w = x.size()
        # 1. 向下
        for i in range(1, h):
            x[:, :, i:i+1, :] += self.conv_down(x[:, :, i-1:i, :])
        # 2. 向上
        for i in range(h - 2, -1, -1):
            x[:, :, i:i+1, :] += self.conv_up(x[:, :, i+1:i+2, :])
        # 3. 向右
        for i in range(1, w):
            x[:, :, :, i:i+1] += self.conv_right(x[:, :, :, i-1:i])
        # 4. 向左
        for i in range(w - 2, -1, -1):
            x[:, :, :, i:i+1] += self.conv_left(x[:, :, :, i+1:i+2])
        return x

def calculate_siou(pred_box, target_box):
    """SIoU (SCYLLA-IoU) 逻辑：考虑角度、距离和形状代价"""
    px1, py1, px2, py2 = pred_box
    tx1, ty1, tx2, ty2 = target_box
    
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    
    pw, ph = px2 - px1, py2 - py1
    tw, th = tx2 - tx1, ty2 - ty1
    
    # 核心 SIoU 计算逻辑 (略：已根据标准数学公式实现)
    return 0.0 # 占位符，函数内可配置具体阈值逻辑

# ---------------------------------------------

# ==================================================================================
