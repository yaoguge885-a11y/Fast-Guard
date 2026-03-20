# ==================================================================================
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

# 从 ui_main 导入程序的入口模块
from ui_main import main, print_versions, MainWindow, SplashScreen

# ==================================================================================
# === 7. 程序启动入口 ===
# ==================================================================================
if __name__ == "__main__":
    main()
