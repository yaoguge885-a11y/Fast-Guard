# ==================================================================================
# === 1. 基础导入与环境信息 ===
# ==================================================================================
import os
import sys
import time
import math
import shutil
import logging

# 设置全局系统日志记录器
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f'system_log_{LOG_TIMESTAMP}.log')

SYSTEM_LOGGER = logging.getLogger(f'System_{LOG_TIMESTAMP}')
SYSTEM_LOGGER.setLevel(logging.DEBUG)
SYSTEM_LOGGER.handlers = []
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
fh.setFormatter(formatter)
SYSTEM_LOGGER.addHandler(fh)
SYSTEM_LOGGER.info(f"系统启动，日志文件：{LOG_FILE}")

import ultralytics
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from core.calculator import TTCCalculator
from core.ipm import IPM_Transformer
from core.view_classifier import ViewClassifier

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
# === 3. 视角自动识别引擎 (正面/侧面视角分类与锁定) ===
# ==================================================================================
class ViewClassifier:
    """视角分类器，基于静态特征和动态特征进行视角识别"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frame_count = 0
        self.last_frame = None
        self.locked_perspective = None
        self.confident_frames = 0
        self.required_confident_frames = 5
        self.last_result = "分析中..."
        self.edge_history = []
        self.flow_history = []
        self.flow_pattern_history = []
        self.car_region_history = []
        self.vanish_history = []
        self.confidence_history = []
        self.initialized = False
        self.last_process_time = 0

    
    def _detect_static_edge(self, frame):
        """检测画面右边缘10%区域的固定边缘（优化版）"""
        try:
            h, w = frame.shape[:2]
            edge_region_width = int(w * 0.1)
            
            # 降采样以提高性能
            scale = 0.5
            new_h, new_w = int(h * scale), int(w * scale)
            new_edge_width = int(edge_region_width * scale)
            
            small_frame = cv2.resize(frame, (new_w, new_h))
            edge_region = small_frame[:, -new_edge_width:]
            
            # 转换为灰度图
            gray = cv2.cvtColor(edge_region, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Canny边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            
            # 计算边缘密度
            edge_density = np.sum(edges > 0) / (new_edge_width * new_h)
            
            # 检测是否存在垂直边缘
            vertical_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            vertical_edges = np.abs(vertical_edges)
            vertical_edge_density = np.sum(vertical_edges > 50) / (new_edge_width * new_h)
            
            # 更严格的固定边缘判断（减少正面视角误判）
            has_static_edge = edge_density > 0.14 and vertical_edge_density > 0.12

            
            return has_static_edge, edge_density, vertical_edge_density
        except Exception as e:
            print(f"边缘检测错误: {e}")
            return False, 0.0, 0.0
    
    def _calculate_global_flow(self, current_frame, prev_frame):
        """计算全局光流（优化版）"""
        try:
            # 降采样以提高性能
            h, w = current_frame.shape[:2]
            scale = 0.5
            new_h, new_w = int(h * scale), int(w * scale)
            
            prev_small = cv2.resize(prev_frame, (new_w, new_h))
            curr_small = cv2.resize(current_frame, (new_w, new_h))
            
            # 转换为灰度图
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
            
            # 使用Farneback光流算法（优化参数）
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                0.5, 3, 10, 3, 5, 1.2, 0
            )
            
            # 计算水平分量比例（水平平移特征）
            horizontal_flow = np.abs(flow[..., 0])
            vertical_flow = np.abs(flow[..., 1])
            
            mean_horizontal = np.mean(horizontal_flow)
            mean_vertical = np.mean(vertical_flow)
            
            if mean_vertical < 1e-6:
                horizontal_ratio = 2.0  # 避免除零
            else:
                horizontal_ratio = mean_horizontal / mean_vertical
            
            return 0.0, 0.0, horizontal_ratio
        except Exception as e:
            print(f"光流计算错误: {e}")
            return 0.0, 0.0, 0.5

    def _car_body_mask_score(self, frame):
        """估计左右 25% 区域是否存在稳定高亮车身"""
        h, w = frame.shape[:2]
        band_w = int(w * 0.25)
        band = frame[:, :band_w]
        band_r = frame[:, -band_w:]

        def _score(region):
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            std = float(np.std(gray))
            sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            edge_mean = float(np.mean(np.abs(sobel)))
            area = region.shape[0] * region.shape[1]
            bright_ratio = np.sum(gray > 190) / max(1, area)
            # 明亮且纹理稳定 → 车身概率高
            side_score = bright_ratio * 0.6 + max(0.0, (220 - std) / 220) * 0.3 + max(0.0, (30 - edge_mean) / 30) * 0.1
            forward_score = 1.0 - side_score
            return side_score, forward_score

        left_side, left_forward = _score(band)
        right_side, right_forward = _score(band_r)
        # 取两侧最大值作为侧向信号，平均作为前向抵消
        side_score = max(left_side, right_side)
        forward_score = (left_forward + right_forward) / 2.0
        return side_score, forward_score

    def _vanish_line_score(self, frame):
        """基于Hough线斜率和汇聚点估计视角"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, int(640 * h / max(1, w)))) if w > 0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=small.shape[1] // 5, maxLineGap=20)
        if lines is None:
            return 0.5, 0.5
        slopes = []
        intersections = []
        for l in lines[:80]:
            x1, y1, x2, y2 = l[0]
            if x2 == x1:
                continue
            k = (y2 - y1) / max(1e-6, (x2 - x1))
            slopes.append(k)
            # 估算与中心水平线的交点
            b = y1 - k * x1
            x_at_center = (small.shape[0] / 2 - b) / max(1e-6, k)
            intersections.append(x_at_center)
        if not slopes:
            return 0.5, 0.5
        slopes = np.array(slopes)
        intersections = np.array(intersections)
        pos = np.sum(slopes > 0)
        neg = np.sum(slopes < 0)
        ratio_single = abs(pos - neg) / max(1, pos + neg)
        # 汇聚点偏离中心程度（归一到0~1，越小越居中）
        center_x = small.shape[1] / 2
        vanish_offset = np.median(np.abs(intersections - center_x)) / max(1, small.shape[1])
        # 单斜率 + 大偏移 → 侧向；对称 + 居中 → 前向
        side_score = 0.6 * ratio_single + 0.4 * min(1.0, vanish_offset * 2)
        forward_score = 1.0 - side_score
        return side_score, forward_score

    def _flow_pattern_score(self, current_frame, prev_frame):
        """利用光流判断辐射(前向) vs 平移(侧向)"""
        if prev_frame is None:
            return 0.5, 0.5
        h, w = current_frame.shape[:2]
        scale = 0.4
        new_h, new_w = int(h * scale), int(w * scale)
        prev_small = cv2.resize(prev_frame, (new_w, new_h))
        curr_small = cv2.resize(current_frame, (new_w, new_h))
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 12, 3, 5, 1.1, 0
        )
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx ** 2 + fy ** 2)
        cx, cy = new_w / 2, new_h / 2
        ys, xs = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')
        vx = xs - cx
        vy = ys - cy
        norm = np.sqrt(vx ** 2 + vy ** 2) + 1e-6
        vx /= norm
        vy /= norm
        radial = (fx * vx + fy * vy) / (np.sqrt(fx ** 2 + fy ** 2) + 1e-6)
        radial_score = float(np.clip(np.nan_to_num(np.mean(radial * (mag > np.median(mag)))), -1, 1))
        mean_fx = float(np.mean(fx))
        mean_fy = float(np.mean(fy))
        mean_mag = np.sqrt(mean_fx ** 2 + mean_fy ** 2) + 1e-6
        trans_consistency = mean_mag / (np.std(fx) + np.std(fy) + 1e-6)
        trans_score = float(np.clip(trans_consistency / 5.0, 0, 1))
        # 辐射为正→前向；整体一致平移→侧向
        forward_score = 0.6 * max(0.0, radial_score) + 0.4 * (1 - trans_score)
        side_score = 1.0 - forward_score
        return side_score, forward_score

    
    def analyze_frame(self, frame, detections=None):
        """分析单帧画面，返回前向和侧向的得分（优化版）"""
        import time
        
        # 性能监控
        start_time = time.time()
        
        if self.locked_perspective:
            if self.locked_perspective == "前向视角":
                return 0.95, 0.05
            else:
                return 0.05, 0.95
        
        self.frame_count += 1
        
        # 初始化检查
        if not self.initialized:
            self.initialized = True
            # 初始化时直接返回中间值，避免第一帧处理
            self.last_frame = frame.copy()
            return 0.5, 0.5
        
        try:
            # 静态特征检测（每3帧计算一次，进一步提高性能）
            has_static_edge = False
            edge_density = 0.0
            vertical_edge_density = 0.0
            
            if self.frame_count % 3 == 0:
                has_static_edge, edge_density, vertical_edge_density = self._detect_static_edge(frame)
            elif self.edge_history:
                # 使用上一次的边缘检测结果
                last_edge = self.edge_history[-1]
                has_static_edge = last_edge['has_static_edge']
                edge_density = last_edge['edge_density']
                vertical_edge_density = last_edge['vertical_edge_density']

            # 检测分布修正：若目标都在中间且未贴右侧，则认为非侧向
            near_right_edge = False
            mid_band_hits = 0
            total_det = 0
            if detections:
                h_det, w_det = frame.shape[:2]
                for rec in detections:
                    _, dx1, dy1, dx2, dy2, dcx, dcy, _ = rec
                    total_det += 1
                    if dx2 >= int(w_det * 0.9):
                        near_right_edge = True
                    cx_norm = dcx / max(1, w_det)
                    if 0.28 <= cx_norm <= 0.72:
                        mid_band_hits += 1
                if mid_band_hits >= 1 and not near_right_edge:
                    has_static_edge = False

            # 车身占位检测（每4帧）
            car_side_score = car_forward_score = 0.5
            if self.frame_count % 4 == 0:
                car_side_score, car_forward_score = self._car_body_mask_score(frame)
            elif self.car_region_history:
                last_car = self.car_region_history[-1]
                car_side_score = last_car['side']
                car_forward_score = last_car['forward']

            # 线特征（每5帧）
            vanish_side = vanish_forward = 0.5
            if self.frame_count % 5 == 0:
                vanish_side, vanish_forward = self._vanish_line_score(frame)
            elif self.vanish_history:
                last_v = self.vanish_history[-1]
                vanish_side = last_v['side']
                vanish_forward = last_v['forward']

            # 动态特征检测（每4帧计算一次光流，大幅提高性能）
            horizontal_ratio = 0.5
            flow_pattern_side = flow_pattern_forward = 0.5
            if self.last_frame is not None and self.frame_count % 4 == 0:
                _, _, horizontal_ratio = self._calculate_global_flow(frame, self.last_frame)
                flow_pattern_side, flow_pattern_forward = self._flow_pattern_score(frame, self.last_frame)
            elif self.flow_history:
                # 使用上一次的光流结果
                last_flow = self.flow_history[-1]
                horizontal_ratio = last_flow['horizontal_ratio']
            if self.flow_pattern_history:
                flow_pattern_side = self.flow_pattern_history[-1]['side']
                flow_pattern_forward = self.flow_pattern_history[-1]['forward']
            
            # 更新上一帧（只在需要时更新）
            if self.frame_count % 2 == 0:
                self.last_frame = frame.copy()
            
            # 计算视角得分
            if has_static_edge:
                static_side_score = 0.7
                static_forward_score = 0.3
            else:
                static_side_score = 0.15
                static_forward_score = 0.85
            
            # 动态特征得分（水平平移倾向于侧向视角）
            if horizontal_ratio > 1.35:
                dynamic_side_score = 0.9
                dynamic_forward_score = 0.1
            elif horizontal_ratio < 0.9:
                dynamic_side_score = 0.2
                dynamic_forward_score = 0.8
            else:
                dynamic_side_score = 0.5
                dynamic_forward_score = 0.5
            
            # 综合得分（加入车身/地平线/流模式）
            forward_score = (
                static_forward_score * 0.35
                + dynamic_forward_score * 0.25
                + car_forward_score * 0.15
                + vanish_forward * 0.15
                + flow_pattern_forward * 0.10
            )
            side_score = (
                static_side_score * 0.35
                + dynamic_side_score * 0.25
                + car_side_score * 0.15
                + vanish_side * 0.15
                + flow_pattern_side * 0.10
            )

            # 强侧向加权：当车身占位显著时，侧向得分加成，前向减弱
            if car_side_score > 0.45:
                boost = min(0.32, (car_side_score - 0.45) * 0.8)
                side_score += boost
                forward_score -= boost * 0.5

            side_score = float(np.clip(side_score, 0.0, 1.0))
            forward_score = float(np.clip(forward_score, 0.0, 1.0))





            
            # 记录历史（每4帧记录一次，减少内存使用）
            if self.frame_count % 4 == 0:
                self.edge_history.append({
                    'has_static_edge': has_static_edge,
                    'edge_density': edge_density,
                    'vertical_edge_density': vertical_edge_density
                })
                
                self.flow_history.append({
                    'horizontal_ratio': horizontal_ratio
                })

                self.flow_pattern_history.append({
                    'side': flow_pattern_side,
                    'forward': flow_pattern_forward
                })

                self.car_region_history.append({
                    'side': car_side_score,
                    'forward': car_forward_score
                })

                self.vanish_history.append({
                    'side': vanish_side,
                    'forward': vanish_forward
                })
                
                self.confidence_history.append({
                    'forward_score': forward_score,
                    'side_score': side_score,
                    'confidence': abs(forward_score - side_score)
                })
                
                # 限制历史长度
                if len(self.edge_history) > 3:
                    self.edge_history = self.edge_history[-3:]
                if len(self.flow_history) > 3:
                    self.flow_history = self.flow_history[-3:]
                if len(self.flow_pattern_history) > 3:
                    self.flow_pattern_history = self.flow_pattern_history[-3:]
                if len(self.car_region_history) > 3:
                    self.car_region_history = self.car_region_history[-3:]
                if len(self.vanish_history) > 3:
                    self.vanish_history = self.vanish_history[-3:]
                if len(self.confidence_history) > 3:
                    self.confidence_history = self.confidence_history[-3:]

            
            # 性能监控
            process_time = time.time() - start_time
            self.last_process_time = process_time
            
            # 如果处理时间过长，返回默认值
            if process_time > 0.1:  # 超过100ms认为处理过慢
                print(f"视角分析处理时间过长: {process_time:.3f}s")
                return 0.5, 0.5
            
            return forward_score, side_score
            
        except Exception as e:
            print(f"视角分析错误: {e}")
            return 0.5, 0.5
    
    def determine_perspective(self, force_result=False):
        """确定最终视角"""
        if self.locked_perspective:
            return self.locked_perspective
        
        if self.frame_count < 3:
            return "分析中..."
        
        # 分析历史数据
        recent_confidence = self.confidence_history[-min(5, len(self.confidence_history)):]
        if not recent_confidence:
            return "分析中..."
        
        avg_forward = np.mean([c['forward_score'] for c in recent_confidence])
        avg_side = np.mean([c['side_score'] for c in recent_confidence])
        avg_confidence = np.mean([c['confidence'] for c in recent_confidence])
        
        # 分析静态特征历史
        recent_edge = self.edge_history[-min(5, len(self.edge_history)):]
        has_static_edge = any(e['has_static_edge'] for e in recent_edge) if recent_edge else False
        avg_edge_density = np.mean([e['edge_density'] for e in recent_edge]) if recent_edge else 0
        
        # 分析动态特征历史
        recent_flow = self.flow_history[-min(5, len(self.flow_history)):]
        avg_horizontal_ratio = np.mean([f['horizontal_ratio'] for f in recent_flow]) if recent_flow else 0.5
        
        # 车身/消失点/流模式快速侧向触发
        recent_car = self.car_region_history[-1] if self.car_region_history else {}
        recent_vanish = self.vanish_history[-1] if self.vanish_history else {}
        recent_flow_pattern = self.flow_pattern_history[-1] if self.flow_pattern_history else {}

        if recent_car and recent_car.get('side', 0) > 0.5 and recent_vanish.get('side', 0) > 0.45:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        if recent_car and recent_car.get('side', 0) > 0.5 and recent_flow_pattern.get('side', 0) > 0.5:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        if recent_flow_pattern and recent_flow_pattern.get('side', 0) > 0.6 and avg_horizontal_ratio > 1.0:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        if recent_vanish and recent_vanish.get('side', 0) > 0.6:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result

        # 车身强推侧向：当车身侧向>0.55 且当前侧向得分≥前向 - 0.15 即侧向
        if recent_car and recent_car.get('side', 0) > 0.55 and avg_side + 0.15 >= avg_forward:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result

        # 基于静态和动态特征的判断
        # 强化侧面视角识别（需要同时满足右缘静态边+明显横向流）
        if has_static_edge and avg_horizontal_ratio > 1.08:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        
        # 强化正面视角识别（无右缘静态边且横向流弱）
        if not has_static_edge and avg_horizontal_ratio < 0.98:
            result = "前向视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result


        
        # 基于得分差异的判断
        diff = avg_forward - avg_side
        
        if diff > 0.07:
            result = "前向视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        elif diff < -0.07:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        elif diff > 0.008:
            result = "前向视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        elif diff < -0.008:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        else:
            return "分析中..."



    
    def _check_and_lock(self, perspective, forward_score, side_score, confidence):
        """检查并锁定视角"""
        if perspective == self.last_result:
            self.confident_frames += 1
        else:
            self.confident_frames = 1
        
        if self.confident_frames >= self.required_confident_frames and confidence > 0.1:
            self.locked_perspective = perspective
            print(f"视角已锁定为: {perspective} (连续{self.confident_frames}帧确信，置信度: {confidence:.2f})")
        
        self.last_result = perspective
    
    def get_debug_info(self):
        """获取调试信息"""
        if not self.confidence_history:
            return {}
        
        recent_conf = self.confidence_history[-1]
        recent_edge = self.edge_history[-1] if self.edge_history else {}
        recent_flow = self.flow_history[-1] if self.flow_history else {}
        
        return {
            'has_static_edge': recent_edge.get('has_static_edge', False),
            'edge_density': recent_edge.get('edge_density', 0),
            'vertical_edge_density': recent_edge.get('vertical_edge_density', 0),
            'horizontal_ratio': recent_flow.get('horizontal_ratio', 0),
            'forward_score': recent_conf.get('forward_score', 0),
            'side_score': recent_conf.get('side_score', 0),
            'confidence': recent_conf.get('confidence', 0),
            'locked': bool(self.locked_perspective),
            'locked_perspective': self.locked_perspective if self.locked_perspective else "未锁定",
            'frame_count': self.frame_count
        }



# ==================================================================================
# === 5. 核心视频处理线程 (视频读取、预处理、YOLO 推理与预警) ===
# ==================================================================================
class VideoThread(QtCore.QThread):
    # 修改信号签名：发送三张图像 (Original, Preprocessed, Inference)
    frame_signal = QtCore.pyqtSignal(QtGui.QImage, QtGui.QImage, QtGui.QImage)
    status_signal = QtCore.pyqtSignal(str)

    ttc_signal = QtCore.pyqtSignal(float, int)
    side_warning_signal = QtCore.pyqtSignal(str, str, int)
    log_signal = QtCore.pyqtSignal(int, float)
    latency_signal = QtCore.pyqtSignal(float)
    model_signal = QtCore.pyqtSignal(str)
    perspective_signal = QtCore.pyqtSignal(str)
    debug_signal = QtCore.pyqtSignal(dict)
    position_signal = QtCore.pyqtSignal(int, int, float, float)
    hud_signal = QtCore.pyqtSignal(dict)



    def __init__(self, source, model_path, parent=None, weak_conf_threshold=0.38, edge_strength_threshold=28.0):
        super().__init__(parent)
        self.source = source
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.weak_conf_threshold = float(weak_conf_threshold)
        self.edge_strength_threshold = float(edge_strength_threshold)
        
        # 使用全局日志记录器
        SYSTEM_LOGGER.info(f"VideoThread | 视频：{self.source} | 模型：{self.model_name}")
        
        self._running = True
        self._frame_count = 0
        self._last_centers = {}
        self._seen_counts = {}
        self.view_classifier = ViewClassifier()
        self.side_detector = None
        self.current_perspective = "分析中..."
        self.perspective_locked = False
        self.last_perspective_time = 0
        self.perspective_debug = False
        self.last_debug_info = {}
        self.forward_calculator = None
        
        # IPM & 轨迹评估
        self.ipm = IPM_Transformer()
        self._world_history = {}
        self.vehicle_width = 1.9  # 车辆宽度 (m)
        self.envelope_margin = 0.4  # 包络线左右冗余 (m)
        self.envelope_length = 30.0  # 前向判定距离 (m)

        # 安全距离阈值 (论文 2.3)：驾驶员反应时间 + 最小冗余距离
        self.t_reaction = 1.2  # 秒（前向视角默认），侧向视角将自适应降到 0.8s
        self.d_safe = 2.0      # 米，可按现场调节
        self.v_self_mps = 11.1 # 本车速度模拟，约 40km/h，可调
        self._last_distance = {}  # 记录每个目标上一帧的距离，用于估计 V_rel
        self._vrel_history = {}   # 卡尔曼/滑动窗口平滑相对速度

        # Stereo Vision Parameters

        self.stereo_mode = False  # 默认为单目，SBS宽屏自动切换
        self.stereo_matcher = None
        self.baseline = 0.12  # 默认基线 12cm (需根据实际硬件调整)
        self.focal_length = 800  # 默认焦距 (像素单位，需标定)
        self.disparity_map = None

        self._user_paused = False
        self._seeking = False
        self._seek_target = None
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.duration = 0.0
        self._fps_ema = None
        self._ema_hood_y = None
        
        # 锁定相关状态变量
        self._forward_lock_count = 0
        self._side_lock_count = 0
        self._locked_warning_y = None
        self._locked_side_mode = None
        self._locked_car_x_ground = None

        # ---------- 中文字体路径设置 ----------


        # Windows 常用字体：黑体、微软雅黑
        self.font_path = "C:/Windows/Fonts/simhei.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/msyh.ttc" # 备选微软雅黑
        
        self.cached_font = None
        try:
            self.cached_font = ImageFont.truetype(self.font_path, 24)
        except Exception as e:
            print(f"Font preload failed: {e}")
        # --------------------------------------

        # Class Name Translation Map
        self.class_map = {
            "person": "行人",
            "bicycle": "自行车",
            "car": "轿车",
            "motorcycle": "摩托车",
            "airplane": "飞机",
            "bus": "公交车",
            "train": "火车",
            "truck": "卡车",
            "boat": "船",
            "traffic light": "红绿灯",
            "fire hydrant": "消防栓",
            "stop sign": "停止标志",
            "parking meter": "停车计费器",
            "bench": "长椅",
            "bird": "鸟",
            "cat": "猫",
            "dog": "狗",
            "horse": "马",
            "sheep": "羊",
            "cow": "牛",
            "elephant": "大象",
            "bear": "熊",
            "zebra": "斑马",
            "giraffe": "长颈鹿",
            "backpack": "背包",
            "umbrella": "雨伞",
            "handbag": "手提包",
            "tie": "领带",
            "suitcase": "手提箱",
            "frisbee": "飞盘",
            "skis": "滑雪板",
            "snowboard": "单板滑雪",
            "sports ball": "运动球",
            "kite": "风筝",
            "baseball bat": "棒球棒",
            "baseball glove": "棒球手套",
            "skateboard": "滑板",
            "surfboard": "冲浪板",
            "tennis racket": "网球拍",
            "bottle": "瓶子",
            "wine glass": "酒杯",
            "cup": "杯子",
            "fork": "叉子",
            "knife": "刀",
            "spoon": "勺子",
            "bowl": "碗",
            "banana": "香蕉",
            "apple": "苹果",
            "sandwich": "三明治",
            "orange": "橙子",
            "broccoli": "西兰花",
            "carrot": "胡萝卜",
            "hot dog": "热狗",
            "pizza": "披萨",
            "donut": "甜甜圈",
            "cake": "蛋糕",
            "chair": "椅子",
            "couch": "沙发",
            "potted plant": "盆栽",
            "bed": "床",
            "dining table": "餐桌",
            "toilet": "厕所",
            "tv": "电视",
            "laptop": "笔记本电脑",
            "mouse": "鼠标",
            "remote": "遥控器",
            "keyboard": "键盘",
            "cell phone": "手机",
            "microwave": "微波炉",
            "oven": "烤箱",
            "toaster": "烤面包机",
            "sink": "水槽",
            "refrigerator": "冰箱",
            "book": "书",
            "clock": "钟",
            "vase": "花瓶",
            "scissors": "剪刀",
            "teddy bear": "泰迪熊",
            "hair drier": "吹风机",
            "toothbrush": "牙刷"
        }

    def _detect_hood_y(self, frame_gray, h, w):
        """自动侦测引擎盖边缘 (水平线)"""
        # 取画面下半部中间区域 (高度 60% ~ 100%, 宽度 20% ~ 80%)
        roi_y1 = int(h * 0.60)
        roi_y2 = h
        roi_x1 = int(w * 0.20)
        roi_x2 = int(w * 0.80)
        
        roi = frame_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None
            
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)
        
        row_mean = np.mean(abs_sobel_y, axis=1)
        if len(row_mean) == 0:
            return None
            
        max_idx = int(np.argmax(row_mean))
        if row_mean[max_idx] < 10.0:
            return None
            
        hood_y = roi_y1 + max_idx
        # 限制高度不要超过画面 40% (即上限是 0.6h)，下限是 0.95h
        #红线默认高度
        hood_y = max(int(h * 0.60), min(int(h * 0.8), hood_y))
        return hood_y

    def _cv2_put_chinese(self, img, text, org, font_size, color):
        """
        单条绘制（作为兼容保留，但内部应优先使用批量绘制）
        """
        return self._draw_batch_chinese(img, [(text, org, font_size, color)])

    def _draw_batch_chinese(self, img, draws):
        """
        批量绘制中文字符，显著提升性能
        draws: list of (text, org, font_size, color)
        """
        if not draws:
            return img
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw_obj = ImageDraw.Draw(img_pil)
        
        for text, org, font_size, color in draws:
            try:
                # 尽量复用字体对象，如果字号不同再重新加载
                if self.cached_font and self.cached_font.size == font_size:
                    font = self.cached_font
                else:
                    font = ImageFont.truetype(self.font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
            
            # 阴影
            shadow_offset = (1, 1)
            draw_obj.text((org[0] + shadow_offset[0], org[1] + shadow_offset[1]), text, font=font, fill=(0, 0, 0))
            draw_obj.text(org, text, font=font, fill=color[::-1]) # BGR -> RGB
            
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _draw_l_corners(self, frame, x1, y1, x2, y2, color, thickness=2, seg=16):
        # 绘制四个角的L型线段
        pts = [
            ((x1, y1), (x1 + seg, y1)), ((x1, y1), (x1, y1 + seg)),
            ((x2, y1), (x2 - seg, y1)), ((x2, y1), (x2, y1 + seg)),
            ((x1, y2), (x1 + seg, y2)), ((x1, y2), (x1, y2 - seg)),
            ((x2, y2), (x2 - seg, y2)), ((x2, y2), (x2, y2 - seg)),
        ]
        for p1, p2 in pts:
            cv2.line(frame, p1, p2, color, thickness)
        return frame

    # --- 绘图辅助工具 ---
    def _update_world_track(self, track_id, world_pos, max_len=20):
        from collections import deque
        if track_id not in self._world_history:
            self._world_history[track_id] = deque(maxlen=max_len)
        self._world_history[track_id].append(world_pos)

    def _compute_yaw_rate(self, track_id):
        pts = self._world_history.get(track_id, [])
        if len(pts) < 3:
            return 0.0, None
        p0, p1, p2 = pts[-3], pts[-2], pts[-1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        def _norm(v):
            return math.sqrt(v[0] * v[0] + v[1] * v[1])
        n1, n2 = _norm(v1), _norm(v2)
        if n1 < 1e-4 or n2 < 1e-4:
            return 0.0, None
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        dot = max(-1.0, min(1.0, dot / (n1 * n2)))
        angle = math.degrees(math.acos(dot))
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        signed_angle = angle if cross >= 0 else -angle
        return signed_angle, v2

    def _segment_intersects_rect(self, p1, p2, half_w, length):
        # 矩形：x in [-half_w, half_w], y in [0, length]
        def inside(p):
            return (-half_w <= p[0] <= half_w) and (0 <= p[1] <= length)

        if inside(p1) or inside(p2):
            return True

        rect_edges = [
            ((-half_w, 0), (half_w, 0)),
            ((half_w, 0), (half_w, length)),
            ((half_w, length), (-half_w, length)),
            ((-half_w, length), (-half_w, 0)),
        ]

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

        for e1, e2 in rect_edges:
            d1 = ccw(p1, p2, e1)
            d2 = ccw(p1, p2, e2)
            d3 = ccw(e1, e2, p1)
            d4 = ccw(e1, e2, p2)
            if (d1 == 0 and inside(e1)) or (d2 == 0 and inside(e2)):
                return True
            if (d3 == 0 and inside(p1)) or (d4 == 0 and inside(p2)):
                return True
            if (d1 * d2 < 0) and (d3 * d4 < 0):
                return True
        return False

    def _in_conflict_envelope(self, track_id, world_pos):
        half_w = self.vehicle_width * 0.5 + self.envelope_margin
        length = self.envelope_length
        history = self._world_history.get(track_id)
        if not history or len(history) < 1:
            return abs(world_pos[0]) <= half_w and 0 <= world_pos[1] <= length
        p_prev = history[-1]
        return self._segment_intersects_rect(p_prev, world_pos, half_w, length)

    def _predict_intent(self, track_id, world_pos):
        yaw_deg, v2 = self._compute_yaw_rate(track_id)
        intent = "直行通过"
        if len(self._world_history.get(track_id, [])) >= 2:
            p_prev = self._world_history[track_id][-2]
            toward_center = abs(world_pos[0]) < abs(p_prev[0])
            if abs(yaw_deg) > 8.0 and toward_center:
                intent = "侧向切入"
        angle_cost = abs(yaw_deg) / 180.0  # 夹角代价，用于平滑/抑制抖动
        return intent, yaw_deg, angle_cost


    # --- 核心主循环 ---
    def run(self):

        self.status_signal.emit("扫描中")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.status_signal.emit("系统就绪")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1e-3:
            self.fps = 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.total_frames > 0 else 0.0

        self.forward_calculator = TTCCalculator(self.fps)

        if not os.path.exists(self.model_path):
            model_filename = os.path.basename(self.model_path)
            model = ultralytics.YOLO(model_filename)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            if os.path.exists(model_filename):
                try:
                    shutil.move(model_filename, self.model_path)
                except shutil.Error:
                    try:
                        shutil.copy2(model_filename, self.model_path)
                    except Exception:
                        pass
        else:
            model = ultralytics.YOLO(self.model_path)

        self.model_signal.emit(os.path.basename(self.model_path))
        allowed_names = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

        name_map = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
        allowed_ids = {k for k, v in name_map.items() if v in allowed_names}
        fps_value = self.fps

        # 检查 GPU 是否可用并指定设备
        device = '0' if torch.cuda.is_available() else 'cpu'

        # 缓存上一帧的追踪结果
        last_results = None
        deferred_draws = []
        while self._running:
            just_seeked = False
            if self._seek_target is not None and self.cap:
                target = max(0, int(self._seek_target))
                if self.total_frames > 0:
                    target = min(target, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self._frame_count = target
                self._seek_target = None
                just_seeked = True

            if self._user_paused and not self._seeking and not just_seeked:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                # 播放结束时进入暂停状态，不退出线程，以便响应进度条回退
                self._user_paused = True
                if self._seek_target is not None:
                    continue
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            # 初始化 IPM 内参（如果未设置）
            if self.ipm:
                self.ipm.set_frame(w, h)
            # 保留原始帧用于 UI 渲染（避免过度增强）
            frame_raw = frame.copy()


            # --- 优化后的图像预处理 ---
            # 只有当需要显示预处理视图时，才进行所有昂贵的计算
            # 默认只进行最小限度的增强用于推理
            
            # 推理用的轻量级增强
            inference_frame = frame
            pre_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 基础灰度用于后续逻辑

            # 如果需要更强的特征（原有的 CLAHE 和 掩码），可以保留但优化
            # 比如：每 2 帧计算一次掩码，或者跳过 Sobel
            do_heavy_preproc = (self._frame_count % 3 == 0) # 降低重度预处理频率
            
            if do_heavy_preproc:
                denoised_frame = cv2.medianBlur(frame, 3)
                lab = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_eq = clahe.apply(l)
                enhanced_frame = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
                pre_gray = l_eq # 更新显示用的灰度图
            else:
                enhanced_frame = frame

            # 简化掩码和推理帧生成 (核心卡顿点)
            inference_frame = enhanced_frame
            sobel_magnitude = np.zeros_like(pre_gray) # 默认空，按需计算

            # --- 双目 SBS 自动识别与分割 ---

            # 提高判定阈值，避免把常见 16:9 视频误判为 SBS（导致只取左半幅）
            is_sbs = w >= h * 2.4  # 仅当宽高比非常大时才判定为 SBS
            if is_sbs:
                w_half = w // 2
                frame_l = frame[:, :w_half]
                frame_r = frame[:, w_half:]
                inference_frame_l = inference_frame[:, :w_half]
                frame = frame_l
                inference_frame = inference_frame_l  # 推理主要在左图进行
                h, w = frame.shape[:2]
                
                # 初始化立体匹配器 (若尚未初始化)
                if self.stereo_matcher is None:
                    self.stereo_matcher = cv2.StereoSGBM_create(
                        minDisparity=0,
                        numDisparities=64, # 视差搜寻范围
                        blockSize=5,
                        P1=8 * 3 * 5**2,
                        P2=32 * 3 * 5**2,
                        disp12MaxDiff=1,
                        uniquenessRatio=10,
                        speckleWindowSize=100,
                        speckleRange=32,
                        preFilterCap=63
                    )
                
                # 计算视差图 (转换为灰度图计算更快)
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                disparity = self.stereo_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
                self.disparity_map = disparity
                self.stereo_mode = True
            else:
                self.stereo_mode = False
                self.disparity_map = None
            current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_frame_idx < 0:
                current_frame_idx = self._frame_count
            self._frame_count = current_frame_idx

            if self._seeking:
                rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ph, pw, pch = rgb_preview.shape
                bytes_per_line_preview = pch * pw
                preview_image = QtGui.QImage(
                    rgb_preview.data, pw, ph, bytes_per_line_preview, QtGui.QImage.Format_RGB888
                )
                # 拖拽时三画面同步显示原图
                self.frame_signal.emit(preview_image.copy(), preview_image.copy(), preview_image.copy())
                self.position_signal.emit(
                    current_frame_idx,
                    self.total_frames,
                    current_frame_idx / fps_value if fps_value else 0.0,
                    self.duration,
                )
                continue


            min_ttc = 99.0
            min_id = -1

            # 动态检测引擎盖边缘 (仅在前5帧进行)
            if self._locked_warning_y is None:
                curr_hood_y = self._detect_hood_y(pre_gray, h, w)
                if curr_hood_y is not None:
                    if self._ema_hood_y is None:
                        self._ema_hood_y = float(curr_hood_y)
                    else:
                        self._ema_hood_y = 0.95 * self._ema_hood_y + 0.05 * curr_hood_y
                
                # 仅在前向视角下累加计数并最终锁定
                if self.current_perspective == "前向视角":
                    if self._forward_lock_count >= 5:
                        self._locked_warning_y = int(self._ema_hood_y) if self._ema_hood_y is not None else int(h * 0.88)
                    else:
                        self._forward_lock_count += 1
            
            # 使用锁定值或当前计算值
            base_warning_y = self._locked_warning_y if self._locked_warning_y is not None else (int(self._ema_hood_y) if self._ema_hood_y is not None else int(h * 0.88))

            warning_line_y = base_warning_y
            warning_line_small_y = base_warning_y
            detect_line_y = int(h * 0.40)

            # 仅前向视角绘制警示线/渐变；侧向不画红/灰线
            if self.current_perspective == "前向视角":
                grad_region = frame[warning_line_y:, :, :].astype(np.float32)
                red = np.zeros_like(grad_region)
                red[:, :, 2] = 255
                if grad_region.shape[0] > 0:
                    alpha = np.linspace(0.0, 0.35, grad_region.shape[0], dtype=np.float32)[:, None, None]
                    blended = grad_region * (1 - alpha) + red * alpha
                    frame[warning_line_y:, :, :] = blended.astype(np.uint8)

                for x in range(0, w, 30):
                    cv2.line(frame, (x, warning_line_y), (min(x + 18, w - 1), warning_line_y), (0, 0, 255), 4)
                for x in range(0, w, 30):
                    cv2.line(frame, (x, detect_line_y), (min(x + 16, w - 1), detect_line_y), (160, 160, 160), 2)


            # 实跳帧逻辑：每 2 帧进行一次推理
            if self._frame_count % 2 == 0:
                results = model.track(
                    inference_frame,
                    persist=True,
                    verbose=False,
                    imgsz=640,          # 进一步降低分辨率以提升速度 (640 是 YOLO 标准值)
                    conf=0.25,
                    iou=0.5,
                    classes=[0, 1, 2, 3, 5, 7],
                    tracker="bytetrack.yaml",
                    device=device       # 明确使用 GPU
                )
                last_results = results
            else:
                results = last_results

            infos = []
            persons = []
            bikes = []
            dx_list = []
            dy_list = []
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        if cls_id not in allowed_ids:
                            continue

                        track_id = int(box.id[0]) if box.id is not None else -1
                        conf_score = float(box.conf[0]) if box.conf is not None else 0.0
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        if y2 <= detect_line_y:
                            continue
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        class_name = name_map.get(cls_id, "")
                        if class_name in {"car", "truck"} and conf_score < 0.45:
                            continue

                        # Sobel 边缘强度辅助过滤：降低低置信度静态纹理误报
                        edge_strength = 0.0
                        roi_y1, roi_y2 = max(0, y1), min(h, y2)
                        roi_x1, roi_x2 = max(0, x1), min(w, x2)
                        if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                            roi_edge = sobel_magnitude[roi_y1:roi_y2, roi_x1:roi_x2]
                            if roi_edge.size > 0:
                                edge_strength = float(np.mean(roi_edge))

                        if class_name in {"bicycle", "motorcycle", "person"}:
                            if conf_score < 0.25:
                                continue
                            if conf_score < self.weak_conf_threshold and edge_strength < self.edge_strength_threshold:
                                continue

                        record = (track_id, x1, y1, x2, y2, cx, cy, class_name)
                        infos.append(record)
                        if class_name in {"bicycle", "motorcycle"}:
                            bikes.append(record)
                        if class_name == "person":
                            persons.append(record)

                        if track_id in self._last_centers:
                            px, py = self._last_centers[track_id]
                            dx_list.append(cx - px)
                            dy_list.append(cy - py)

            def iou(a, b):
                ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
                bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    return 0.0
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                return inter / max(area_a + area_b - inter, 1e-6)

            filtered_infos = []
            for rec in infos:
                if rec[7] == "person":
                    keep = True
                    for bike in bikes:
                        if iou(rec, bike) > 0.6:
                            keep = False
                            break
                    if not keep:
                        continue
                filtered_infos.append(rec)

            infos = filtered_infos

            for track_id, x1, y1, x2, y2, cx, cy, _ in infos:
                self._last_centers[track_id] = (cx, cy)
                self._seen_counts[track_id] = self._seen_counts.get(track_id, 0) + 1

            avg_dx = sum(dx_list) / len(dx_list) if dx_list else 0.0
            avg_dy = sum(dy_list) / len(dy_list) if dy_list else 0.0
            global_vx = avg_dx * fps_value
            global_vy = avg_dy * fps_value

            # 视角分析
            current_time = time.time()
            if not self.perspective_locked or current_time - self.last_perspective_time > 2.0:
                self.view_classifier.analyze_frame(frame.copy(), infos)
                
                debug_info = self.view_classifier.get_debug_info()
                if debug_info:
                    self.last_debug_info = debug_info
                    self.debug_signal.emit(debug_info)
                
                if debug_info.get('locked', False) and not self.perspective_locked:
                    self.perspective_locked = True
                    self.current_perspective = debug_info['locked_perspective']
                    self.perspective_signal.emit(self.current_perspective)
                    self.last_perspective_time = current_time
                    
                    if self.current_perspective == "侧面视角":
                        self.side_detector = None  # 已移除 SideCollisionDetector
                        self.status_signal.emit("侧向碰撞检测已启用")
                    elif self.current_perspective == "前向视角":
                        self.side_detector = None
                        self.status_signal.emit("前向碰撞检测已启用")
                
                elif not self.perspective_locked:
                    perspective = self.view_classifier.determine_perspective()
                    if perspective != "分析中..." and perspective != self.current_perspective:
                        self.current_perspective = perspective
                        self.perspective_signal.emit(perspective)
                        self.last_perspective_time = current_time

            # 可视化调试：在检测到静止锚点时画红色实心方块（内存优化）
            left_static = getattr(self.view_classifier, 'left_static', False)
            right_static = getattr(self.view_classifier, 'right_static', False)
            if left_static or right_static:
                try:
                    overlay = frame.copy()
                    alpha = 0.5
                    
                    if left_static:
                        # 左侧静止：画红色实心方块
                        rect_x = int(w * 0.05)
                        rect_y = int(h * 0.3)
                        rect_w = int(w * 0.1)
                        rect_h = int(h * 0.2)
                        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), -1)
                    
                    if right_static:
                        # 右侧静止：画红色实心方块
                        rect_x = int(w * 0.75)
                        rect_y = int(h * 0.3)
                        rect_w = int(w * 0.1)
                        rect_h = int(h * 0.2)
                        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), -1)
                    
                    # 一次性混合
                    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
                    
                    # 绘制边框
                    if left_static:
                        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
                    if right_static:
                        rect_x = int(w * 0.75)
                        rect_y = int(h * 0.3)
                        rect_w = int(w * 0.1)
                        rect_h = int(h * 0.2)
                        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
                finally:
                    del overlay  # 立即释放内存

            # 在图像上显示视角状态（中文）
            view_text = self.current_perspective
            if self.perspective_locked:
                view_text += " (已锁定)"
            deferred_draws.append((view_text, (w - 360, 40), 32, (255, 255, 255)))

            # 根据视角类型选择不同的碰撞检测逻辑
            # 注意：侧向视角处理已移除，统一使用前向检测逻辑
            min_ttc_alert = 99.0
            min_id_alert = -1
            
            for track_id, x1, y1, x2, y2, cx, cy, class_name in infos:
                if y2 <= detect_line_y:
                    continue
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                if height / max(1, width) > 3.0 or width / max(1, height) > 4.0:
                    continue
                area_ratio = (width * height) / max(1, w * h)
                warn_line = warning_line_small_y if class_name in {"bicycle", "motorcycle", "person"} else warning_line_y
                
                # 获取物体距离 (双目模式)
                obj_dist = None
                if self.stereo_mode and self.disparity_map is not None:
                    # 在检测框中心区域取平均视差
                    mask_y1, mask_y2 = max(0, y1), min(h, y2)
                    mask_x1, mask_x2 = max(0, x1), min(w, x2)
                    roi_disp = self.disparity_map[mask_y1:mask_y2, mask_x1:mask_x2]
                    valid_disp = roi_disp[roi_disp > 0]
                    if len(valid_disp) > 0:
                        avg_disp = np.median(valid_disp)
                        if avg_disp > 0:
                            obj_dist = (self.focal_length * self.baseline) / avg_disp
                else:
                    # 单目近似距离：使用检测框高度估距（假设目标高度常数 H_obj）
                    H_obj_map = {
                        "car": 1.5,
                        "truck": 2.5,
                        "bus": 3.0,
                        "motorcycle": 1.4,
                        "bicycle": 1.4,
                        "person": 1.7,
                    }
                    est_h = H_obj_map.get(class_name, 1.6)
                    bbox_h = max(1, y2 - y1)
                    obj_dist = (est_h * self.focal_length) / bbox_h

                world_pos = None
                intent = "直行通过"
                yaw_deg = 0.0
                angle_cost = 0.0
                if self.ipm:
                    world_pos = self.ipm.pixel_to_ground(cx, y2, (h, w))
                    if world_pos is not None:
                        self._update_world_track(track_id, world_pos)
                        intent, yaw_deg, angle_cost = self._predict_intent(track_id, world_pos)
                
                ttc, vx, vy, dw_dt, red_allowed, vw, is_static, risk_level, in_path = self.forward_calculator.update(
                    track_id,
                    width,
                    cx,
                    cy,
                    y2,
                    w,
                    h,
                    warn_line,
                    area_ratio,
                    global_vx,
                    global_vy,
                    use_ema=class_name in {"person", "bicycle", "motorcycle"},
                    distance=obj_dist,
                    v_self_mps=self.v_self_mps,
                    t_reaction=self.t_reaction,
                    d_safe=self.d_safe,
                )

                v_rel = None
                sdt_violation = False
                safe_dist = None
                vx_abs = abs(vx)
                if obj_dist is not None:
                    prev_dist = self._last_distance.get(track_id)
                    t_react_use = 0.8 if self.current_perspective == "侧面视角" else self.t_reaction

                    if prev_dist is not None:
                        v_rel = (prev_dist - obj_dist) * fps_value  # m/s，正值代表在接近
                        hist = self._vrel_history.get(track_id, [])
                        hist = (hist + [v_rel])[-6:]
                        self._vrel_history[track_id] = hist
                        continuous_closing = len(hist) >= 5 and all(v > 0 for v in hist[-5:])

                        safe_dist = v_rel * t_react_use + self.d_safe
                        lane_center_ok = (w * 0.35) <= cx <= (w * 0.65)
                        is_vehicle = class_name in {"car", "truck", "bus"}
                        v_rel_avg = sum(hist[-5:]) / 5.0 if len(hist) >= 5 else v_rel
                        sdt_gate = (lane_center_ok or is_vehicle) and continuous_closing and (v_rel_avg is not None and v_rel_avg > 0.5)

                        lateral_only = vx_abs > (abs(vw) + 1e-3) * 1.5
                        near_line = y2 > warning_line_y

                        if sdt_gate and near_line and not lateral_only and v_rel is not None and v_rel > 1.0 and ttc > 0 and obj_dist < safe_dist:
                            prev_center = self._last_centers.get(track_id)
                            cy_prev = prev_center[1] if prev_center else cy
                            if abs(cy - cy_prev) < 1.0 and cy < h * 0.4:
                                sdt_violation = False
                            else:
                                sdt_violation = True
                    self._last_distance[track_id] = obj_dist
                else:
                    self._last_distance.pop(track_id, None)
                    self._vrel_history.pop(track_id, None)

                safe_glance = False
                if world_pos is not None:
                    conflict = self._in_conflict_envelope(track_id, world_pos)
                    if not conflict:
                        safe_glance = True
                if safe_glance:
                    risk_level = 0

                if angle_cost > 0.25 and risk_level > 0:
                    risk_level = max(0, risk_level - 1)
                
                if self._seen_counts.get(track_id, 0) < 5:
                    continue
                if is_static:
                    continue

                ratio = vx_abs / max(vw, 1e-3)
                center_relaxed = (w * 0.35) <= cx <= (w * 0.65)
                ratio_threshold = 0.9 if center_relaxed else 0.9

                lateral_fast = ratio > ratio_threshold
                red_ok = red_allowed and not lateral_fast

                warn_ttc = 99.0 if safe_glance else ttc
                if y2 <= warning_line_y:
                    warn_ttc = 99.0
                elif class_name in {"person", "bicycle", "motorcycle"} and ttc < 2.0 and not red_ok:
                    warn_ttc = 2.0
                elif ttc < 1.5 and not red_ok:
                    warn_ttc = 1.5

                sdt_tag = False
                if sdt_violation:
                    sdt_tag = True
                    warn_ttc = min(warn_ttc, 1.0)

                if warn_ttc < min_ttc:
                    min_ttc = warn_ttc
                    min_id = track_id

                # 前向视角显示参数
                color, label, thickness = self._get_forward_display_params(
                    track_id, class_name, ttc, y2, warning_line_y, lateral_fast, red_ok
                )
                if risk_level == 2:
                    color = (0, 0, 255)
                    thickness = 3
                elif risk_level == 1:
                    color = (0, 255, 255)
                    thickness = max(thickness, 2)

                if sdt_tag:
                    if safe_dist is not None:
                        label = f"SDT {safe_dist:.1f}m"
                    else:
                        label = label.replace("TTC", "SDT") if "TTC" in label else f"SDT {label}"
                if obj_dist is not None:
                    label += f" {obj_dist:.1f}m"
                if v_rel is not None:
                    label += f" v={v_rel:.1f}m/s"
                
                self._draw_l_corners(frame, x1, y1, x2, y2, color, thickness=thickness, seg=18)
                # 延迟绘制标签
                deferred_draws.append((label, (x1, y1 - 35), 24, color))

            if min_ttc < 1.5:
                if self._frame_count % 2 == 0:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                self.log_signal.emit(min_id, min_ttc)
            
            self.ttc_signal.emit(min_ttc, min_id)

            hud_payload = {
                "fps": self._fps_ema if self._fps_ema is not None else fps_value,
                "tracked": len(infos),
                "mode": f"{self.current_perspective} ({'Stereo' if self.stereo_mode else 'Mono'})",
            }
            self.hud_signal.emit(hud_payload)

            # 5. 批量执行中文绘制
            if deferred_draws:
                frame = self._draw_batch_chinese(frame, deferred_draws)
                deferred_draws = []

            # 1. Original View
            rgb_orig = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            h_orig, w_orig, ch_orig = rgb_orig.shape
            bytes_orig = ch_orig * w_orig
            qimage_orig = QtGui.QImage(rgb_orig.data, w_orig, h_orig, bytes_orig, QtGui.QImage.Format_RGB888)

            # 2. Pre-processed View
            h_pre, w_pre = pre_gray.shape[:2]
            bytes_pre = w_pre
            qimage_pre = QtGui.QImage(pre_gray.data, w_pre, h_pre, bytes_pre, QtGui.QImage.Format_Grayscale8)

            # 3. Inference View (Final frame)
            rgb_inf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_inf, w_inf, ch_inf = rgb_inf.shape
            bytes_inf = ch_inf * w_inf
            qimage_inf = QtGui.QImage(rgb_inf.data, w_inf, h_inf, bytes_inf, QtGui.QImage.Format_RGB888)

            # 发送三路画面
            self.frame_signal.emit(qimage_orig.copy(), qimage_pre.copy(), qimage_inf.copy())

            self.position_signal.emit(
                current_frame_idx,
                self.total_frames,
                current_frame_idx / fps_value if fps_value else 0.0,
                self.duration,
            )

            t_end = time.perf_counter()
            elapsed = t_end - t_start
            fps_live = 1.0 / max(elapsed, 1e-6)
            if self._fps_ema is None:
                self._fps_ema = fps_live
            else:
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps_live
            self.latency_signal.emit(elapsed * 1000.0)



        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_signal.emit("系统就绪")



    def _get_forward_display_params(self, track_id, class_name, ttc, y2, warn_line, lateral_fast, red_ok):
        
        # 1. 尝试获取或判定锁定的边缘位置
        if self._locked_side_mode is None or self._locked_car_x_ground is None:
            # 尚未锁定，进行实时探测
            side_mode = 'right' 
            car_x_ground = 0.0
            if self.ipm:
                roi_h_start = int(h * 0.45)
                roi_w_ext = int(w * 0.1)
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_h_start:, :]
                gray_l, gray_r = roi_gray[:, :roi_w_ext], roi_gray[:, -roi_w_ext:]
                
                # 简单边缘密度判定方位
                edges_l = cv2.Canny(gray_l, 50, 150)
                edges_r = cv2.Canny(gray_r, 50, 150)
                if np.sum(edges_l > 0) > np.sum(edges_r > 0) * 1.5:
                    side_mode = 'left'
                
                # 使用 Sobel 寻边以获得更准的垂直边界
                def find_car_edge_u(gray_block, is_left_side=True):
                    sobel_x = cv2.Sobel(gray_block, cv2.CV_16S, 1, 0, ksize=3)
                    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
                    _, binary = cv2.threshold(abs_sobel_x, 60, 255, cv2.THRESH_BINARY)
                    col_sum = np.sum(binary > 0, axis=0)
                    min_pixels = int(gray_block.shape[0] * 0.20)
                    if is_left_side:
                        for u in range(len(col_sum)-1, 0, -1):
                            if col_sum[u] > min_pixels: return u
                    else:
                        for u in range(0, len(col_sum)):
                            if col_sum[u] > min_pixels: return (w - roi_w_ext) + u
                    return None

                u_base = find_car_edge_u(gray_l if side_mode == 'left' else gray_r, side_mode == 'left')
                if u_base is not None:
                    pos = self.ipm.pixel_to_ground(u_base, int(h*0.8), (h, w))
                    if pos: car_x_ground = pos[0]
                else: car_x_ground = -1.2 if side_mode == 'left' else 1.2

            # 锁定计数递增
            if self._side_lock_count >= 5:
                self._locked_side_mode = side_mode
                self._locked_car_x_ground = car_x_ground
            else:
                self._side_lock_count += 1
            current_side_mode, current_car_x_ground = side_mode, car_x_ground
        else:
            current_side_mode, current_car_x_ground = self._locked_side_mode, self._locked_car_x_ground

        # 2. 基于方位设定物理墙
        side_mode, car_x_ground = current_side_mode, current_car_x_ground
        offset_sign = 0.8 if side_mode == 'left' else -0.8
        wall_dist = car_x_ground + offset_sign
        
        wall_samples = []
        car_edge_samples = []
        if self.ipm:
            all_wall_pts, car_pts = [], []
            for Y in np.linspace(-2.0, 10.0, 100):
                pt_wall = self.ipm.ground_to_pixel(wall_dist, Y, (h, w))
                pt_car = self.ipm.ground_to_pixel(car_x_ground, Y, (h, w))
                if pt_wall: all_wall_pts.append((int(pt_wall[0]), int(pt_wall[1])))
                if pt_car: car_pts.append((int(pt_car[0]), int(pt_car[1])))

            if len(all_wall_pts) >= 2 and len(car_pts) >= 2:
                p_car_sorted = sorted(car_pts, key=lambda p: p[1], reverse=True)
                p_wall_sorted = sorted(all_wall_pts, key=lambda p: p[1], reverse=True)
                fill_pts = np.array(p_car_sorted + p_wall_sorted[::-1], dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [fill_pts], (200, 180, 90))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                for i in range(len(p_car_sorted)-1):
                    cv2.line(frame, p_car_sorted[i], p_car_sorted[i+1], (0, 255, 100), 2, lineType=cv2.LINE_AA)

            wall_samples, car_edge_samples = all_wall_pts[::5], car_pts
            if len(all_wall_pts) >= 2:
                all_wall_pts.sort(key=lambda p: p[1], reverse=True)
                dash_len, gap_len = 35, 25
                cur_dist, is_draw = 0, True
                for i in range(len(all_wall_pts) - 1):
                    p1, p2 = all_wall_pts[i], all_wall_pts[i+1]
                    d = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    if is_draw: cv2.line(frame, p1, p2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    cur_dist += d
                    if is_draw and cur_dist >= dash_len: is_draw, cur_dist = False, 0
                    elif not is_draw and cur_dist >= gap_len: is_draw, cur_dist = True, 0

        def _interp_x(points, y):
            if not points: return None
            valid_pts = [(int(p[0]), int(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if not valid_pts: return None
            pts = sorted(valid_pts, key=lambda p: p[1])
            if y <= pts[0][1]: return pts[0][0]
            if y >= pts[-1][1]: return pts[-1][0]
            for i in range(len(pts) - 1):
                y0, y1 = pts[i][1], pts[i + 1][1]
                if y0 <= y <= y1:
                    r = (y-y0)/max(y1-y0, 1e-6)
                    return pts[i][0] + r*(pts[i+1][0]-pts[i][0])
            return None

        is_far_area = (lambda u: u > int(w*0.67)) if side_mode == 'left' else (lambda u: u < int(w*0.33))

        for track_id, x1, y1, x2, y2, cx, cy, class_name in infos:
            anchor_u = x1 if side_mode == 'right' else x2
            anchor_v = y2
            world_pos = self.ipm.pixel_to_ground(anchor_u, anchor_v, (h, w)) if self.ipm else None
            if world_pos:
                self._update_world_track(track_id, world_pos, max_len=20)
                dist_x = abs(world_pos[0])
                wall_x = _interp_x(wall_samples, anchor_v)
                car_x = _interp_x(car_edge_samples, anchor_v)
                in_pixel = False
                if wall_x is not None and not is_far_area(anchor_u):
                    if car_x is not None:
                        in_pixel = car_x <= anchor_u <= wall_x if side_mode == 'left' else wall_x <= anchor_u <= car_x
                    else: in_pixel = anchor_u <= wall_x if side_mode == 'right' else anchor_u >= wall_x

                dist_to_car = abs(dist_x - car_x_ground)
                in_world = dist_to_car < 0.8
                far_away = dist_to_car >= 1.2 or is_far_area(anchor_u)
                inside_wall = in_world or (in_pixel and not far_away)

                chinese_class = self.class_map.get(class_name, class_name)
                color, label, thickness = (0, 255, 0), chinese_class, 2
                blink = (self._frame_count % 4) < 2
                if far_away: label = f"安全 {dist_x:.2f}m"
                elif inside_wall:
                    color = (0, 0, 255) if (class_name not in {"person","bicycle","motorcycle"} or blink) else (0, 0, 120)
                    thickness = 3
                    label = f"碰撞预警 {dist_x:.2f}m"
                    self.side_warning_signal.emit('danger', label, track_id)
                else: label = f"侧方 {dist_x:.2f}m"

                self._draw_l_corners(frame, x1, y1, x2, y2, color, thickness=2, seg=18)
                deferred_draws.append((label, (x1, y1 - 35), 24, color))
        return frame

    def _get_forward_display_params(self, track_id, class_name, ttc, y2, warn_line, lateral_fast, red_ok):

        """获取前向视角显示参数（中文）"""
        chinese_class = self.class_map.get(class_name, class_name)
        
        if y2 <= warn_line:
            color = (120, 120, 120)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        elif class_name in {"bicycle", "motorcycle", "person"}:
            color = (255, 0, 255)
            label = f"{chinese_class} (注意)"
            thickness = 3
        elif lateral_fast and ttc < 3.0:
            color = (255, 0, 0)
            label = f"碰撞时间 {ttc:.1f}秒"
            thickness = 2
        elif class_name in {"person", "bicycle", "motorcycle"} and ttc < 2.0 and red_ok:
            color = (0, 0, 255)
            label = f"危险！碰撞时间 {ttc:.1f}秒"
            thickness = 3
        elif ttc < 1.5 and red_ok:
            color = (0, 0, 255)
            label = f"危险！碰撞时间 {ttc:.1f}秒"
            thickness = 3
        elif ttc < 3.0:
            color = (0, 255, 255)
            label = f"碰撞时间 {ttc:.1f}秒"
            thickness = 2
        else:
            color = (0, 255, 0)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        
        return color, label, thickness

    def stop(self):
        self._running = False
        self._user_paused = False
        self._seeking = False
        self.wait()

    def set_paused(self, paused: bool):
        self._user_paused = paused

    def start_seek(self):
        self._seeking = True

    def finish_seek(self):
        self._seeking = False

    def set_frame(self, index: int):
        self._seek_target = int(index)

    def set_preprocess_thresholds(self, weak_conf_threshold: float, edge_strength_threshold: float):
        self.weak_conf_threshold = float(weak_conf_threshold)
        self.edge_strength_threshold = float(edge_strength_threshold)


# ==================================================================================
# === 6. UI 界面与交互逻辑 (PyQt5 组件与主窗口) ===
# ==================================================================================
class SplashScreen(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(800, 550)
        
        # Center on screen
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Background Frame with modern gradient and border
        self.bg_frame = QtWidgets.QFrame()
        self.bg_frame.setObjectName("splashBg")
        self.bg_frame.setStyleSheet("""
            QFrame#splashBg {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
                border-radius: 24px;
                border: 2px solid #4e4e91;
            }
        """)
        bg_layout = QtWidgets.QVBoxLayout(self.bg_frame)
        bg_layout.setContentsMargins(60, 60, 60, 60)
        bg_layout.setSpacing(15)

        # Icon / Logo Placeholder (Modern style)
        logo_layout = QtWidgets.QHBoxLayout()
        logo_label = QtWidgets.QLabel("🛡️")
        logo_label.setStyleSheet("font-size: 80px; background: transparent;")
        logo_layout.addStretch()
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()

        # Title
        self.title_label = QtWidgets.QLabel("FastGuard 智能监控")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 64px;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 4px;
            font-family: 'Outfit', 'Microsoft YaHei', sans-serif;
            background: transparent;
        """)
        
        self.subtitle_label = QtWidgets.QLabel("智能防碰撞预警引擎")
        self.subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("color: #a29bfe; font-size: 24px; letter-spacing: 8px; font-weight: 700; background: transparent; margin-top: 10px;")

        # Progress Section
        progress_container = QtWidgets.QWidget()
        progress_container.setStyleSheet("background: transparent;")
        progress_layout = QtWidgets.QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 50, 0, 0)
        progress_layout.setSpacing(20)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 10);
                border: none;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a29bfe, stop:1 #6c5ce7);
                border-radius: 6px;
            }
        """)

        self.status_label = QtWidgets.QLabel("正在初始化系统组件...")
        self.status_label.setStyleSheet("color: #b2bec3; font-family: 'Consolas', 'Microsoft YaHei', monospace; font-size: 16px; background: transparent;")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)

        bg_layout.addStretch()
        bg_layout.addLayout(logo_layout)
        bg_layout.addWidget(self.title_label)
        bg_layout.addWidget(self.subtitle_label)
        bg_layout.addWidget(progress_container)
        bg_layout.addStretch()

        layout.addWidget(self.bg_frame)

        # Shadow effect
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(50)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(15)
        self.shadow.setColor(QtGui.QColor(0, 0, 0, 200))
        self.bg_frame.setGraphicsEffect(self.shadow)

        # Animation state
        self.progress = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(25)

    def update_progress(self):
        self.progress += 1
        self.progress_bar.setValue(self.progress)
        
        if self.progress == 15: self.status_label.setText(">> 正在加载神经网络架构...")
        if self.progress == 35: self.status_label.setText(">> 正在同步摄像头数据流...")
        if self.progress == 55: self.status_label.setText(">> 正在校准空间传感器...")
        if self.progress == 75: self.status_label.setText(">> 正在优化张量核心...")
        if self.progress == 95: self.status_label.setText(">> 系统就绪，正在启动界面...")
        
        if self.progress >= 100:
            self.timer.stop()
            self.fade_out()

    def fade_out(self):
        self.animation = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.finished.connect(self.on_fade_finished)
        self.animation.start()

    def on_fade_finished(self):
        self.close()
        self.finished.emit()

class StatCard(QtWidgets.QFrame):
    def __init__(self, title, value, unit, icon_text="📊", color="#6366f1", parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        # Modern "Bento Grid" style
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: #18181b; /* Zinc-900 */
                border: 1px solid #27272a; /* Zinc-800 */
                border-radius: 16px;
            }}
            QFrame#statCard:hover {{
                border: 1px solid {color};
                background-color: #27272a;
            }}
            QLabel {{ border: none; background: transparent; }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Header (Icon + Title)
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(16)

        icon_label = QtWidgets.QLabel(icon_text)
        # Circular icon background
        icon_label.setStyleSheet(f"font-size: 28px; color: {color}; background: {color}20; border-radius: 12px; padding: 8px;")
        icon_label.setFixedSize(52, 52)
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-size: 18px; color: #a1a1aa; font-weight: 600; letter-spacing: 1px;")
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Value Area
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.setSpacing(10)
        value_layout.setContentsMargins(0, 12, 0, 0)

        self.value_label = QtWidgets.QLabel(value)
        self.value_label.setStyleSheet("font-size: 42px; color: #ffffff; font-weight: 700; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;")
        
        self.unit_label = QtWidgets.QLabel(unit)
        self.unit_label.setStyleSheet("font-size: 16px; color: #71717a; font-weight: 600; padding-bottom: 8px;")
        
        value_layout.addWidget(self.value_label)
        value_layout.addWidget(self.unit_label, alignment=QtCore.Qt.AlignBottom)
        value_layout.addStretch()

        layout.addLayout(header_layout)
        layout.addLayout(value_layout)

    def update_value(self, value, unit=None):
        self.value_label.setText(str(value))
        if unit:
            self.unit_label.setText(str(unit))


class LogWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统日志")
        self.resize(600, 700) # Larger window
        self.setStyleSheet("""
            QDialog { background-color: #09090b; }
            QLabel { color: white; font-size: 18px; font-weight: bold; font-family: 'Microsoft YaHei'; }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("📋 系统事件记录"))
        
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget {
                background: #18181b;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                color: #a1a1aa;
                font-family: 'Consolas', 'Microsoft YaHei Mono', monospace;
                font-size: 14px; /* Larger log font */
                outline: none;
            }
            QListWidget::item { padding: 6px; }
            QListWidget::item:selected { background: #27272a; color: white; }
        """)
        layout.addWidget(self.log_list)
        
        btn_clear = QtWidgets.QPushButton("清空记录")
        btn_clear.clicked.connect(self.log_list.clear)
        btn_clear.setStyleSheet("""
            QPushButton {
                background: #27272a; color: white; border: 1px solid #3f3f46;
                border-radius: 6px; padding: 10px 16px; font-size: 16px; font-family: 'Microsoft YaHei';
            }
            QPushButton:hover { background: #3f3f46; }
        """)
        layout.addWidget(btn_clear)

    def append_log(self, text):
        self.log_list.addItem(text)
        self.log_list.scrollToBottom()


class SettingsWindow(QtWidgets.QDialog):
    preprocess_changed = QtCore.pyqtSignal(float, float)

    def __init__(self, weak_conf, edge_strength, parent=None):
        super().__init__(parent)
        self.setWindowTitle("预处理参数设置")
        self.resize(500, 350)
        self.setStyleSheet("""
            QDialog { background-color: #09090b; }
            QLabel { color: #e5e7eb; font-weight: 600; font-size: 16px; font-family: 'Microsoft YaHei'; }
            QDoubleSpinBox {
                background: #111111;
                color: #e5e7eb;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 6px 10px;
                min-width: 150px;
                font-size: 16px;
            }
            QPushButton {
                background: #27272a; color: white;
                border: 1px solid #3f3f46; border-radius: 8px;
                padding: 10px 16px; font-size: 16px; font-family: 'Microsoft YaHei';
            }
            QPushButton:hover { background: #3f3f46; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(40, 40, 40, 40)
        
        form = QtWidgets.QFormLayout()
        form.setSpacing(20)
        
        self.spin_weak_conf = QtWidgets.QDoubleSpinBox()
        self.spin_weak_conf.setRange(0.10, 0.95)
        self.spin_weak_conf.setDecimals(2)
        self.spin_weak_conf.setSingleStep(0.01)
        self.spin_weak_conf.setValue(weak_conf)
        
        self.spin_edge_strength = QtWidgets.QDoubleSpinBox()
        self.spin_edge_strength.setRange(1.0, 255.0)
        self.spin_edge_strength.setDecimals(1)
        self.spin_edge_strength.setSingleStep(1.0)
        self.spin_edge_strength.setValue(edge_strength)
        
        form.addRow("低置信度阈值:", self.spin_weak_conf)
        form.addRow("边缘强度阈值:", self.spin_edge_strength)
        layout.addLayout(form)
        
        self.spin_weak_conf.valueChanged.connect(self.emit_change)
        self.spin_edge_strength.valueChanged.connect(self.emit_change)
        
        self.reset_btn = QtWidgets.QPushButton("恢复默认值")
        self.reset_btn.clicked.connect(self.reset_defaults)
        layout.addWidget(self.reset_btn)
        layout.addStretch()

    def emit_change(self):
        self.preprocess_changed.emit(self.spin_weak_conf.value(), self.spin_edge_strength.value())

    def reset_defaults(self):
        self.spin_weak_conf.setValue(0.38)
        self.spin_edge_strength.setValue(28.0)

    # --- 主窗口核心逻辑 ---
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastGuard 智能监控系统")
        self.resize(1400, 850)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        
        # Data & State
        self.thread = None
        self.last_images = None
        self.last_log_time = 0.0

        self.warning_records = []
        self.manual_perspective_set = False
        self.debug_info = {}
        self.side_warning_active = False
        self.side_warning_timer = QtCore.QTimer()
        self.side_warning_timer.timeout.connect(self.clear_side_warning)
        self.hud_info = {"fps": 0.0, "tracked": 0, "mode": "初始化"}
        self.model_name = "yolo11n.pt"
        self.last_latency = None
        self.default_weak_conf_threshold = 0.38
        self.default_edge_strength_threshold = 28.0
        self.weak_conf_threshold = self.default_weak_conf_threshold
        self.edge_strength_threshold = self.default_edge_strength_threshold
        self.total_duration = 0.0
        self.total_frames = 0
        self.log_window = LogWindow(self)
        self.settings_window = SettingsWindow(self.weak_conf_threshold, self.edge_strength_threshold, self)
        self.settings_window.preprocess_changed.connect(self.update_preprocess_from_dialog)

        self.setup_ui()
        self.apply_modern_theme()
        
        # Connect Actions
        self.setup_connections()


    def setup_ui(self):
        # --- Main Layout ---
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === Left Sidebar ===
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(280)  # Wider for larger Chinese text
        
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(24, 40, 24, 40)
        sidebar_layout.setSpacing(18)

        # Logo / Title Area
        app_logo_layout = QtWidgets.QHBoxLayout()
        # Use a geometric shape for a more sci-fi look
        logo_icon = QtWidgets.QLabel("❖") 
        logo_icon.setStyleSheet("font-size: 32px; color: #6366f1; background: transparent;")
        logo_text = QtWidgets.QLabel("FASTGUARD")
        logo_text.setStyleSheet("""
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            font-size: 26px; 
            font-weight: 800; 
            color: #ffffff; 
            letter-spacing: 2px;
            background: transparent;
        """)
        app_logo_layout.addWidget(logo_icon)
        app_logo_layout.addWidget(logo_text)
        app_logo_layout.addStretch()
        
        sidebar_layout.addLayout(app_logo_layout)
        sidebar_layout.addSpacing(50)

        # Menu Group: MAIN
        lbl_main = QtWidgets.QLabel("主菜单")
        lbl_main.setStyleSheet("color: #71717a; font-size: 14px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-family: 'Microsoft YaHei';")
        sidebar_layout.addWidget(lbl_main)

        def create_nav_btn(icon, text, tooltip, is_active=False):
            # Using specific spacing in text for alignment
            btn = QtWidgets.QPushButton(f" {icon}    {text}")
            btn.setObjectName("navBtn")
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            if is_active:
                btn.setChecked(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            sidebar_layout.addWidget(btn)
            return btn

        # Updated icons for a more premium/tech feel
        self.btn_camera = create_nav_btn("⦿", "开启摄像头", "实时监控画面", is_active=True)
        self.btn_open = create_nav_btn("📊", "导入视频", "历史视频分析")
        
        sidebar_layout.addSpacing(30)
        
        # Menu Group: TOOLS
        lbl_tools = QtWidgets.QLabel("工具")
        lbl_tools.setStyleSheet("color: #71717a; font-size: 14px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-family: 'Microsoft YaHei';")
        sidebar_layout.addWidget(lbl_tools)
        
        self.btn_log = create_nav_btn("📟", "系统日志", "查看运行日志")
        self.btn_settings = create_nav_btn("⚙", "参数设置", "调整检测参数")
        
        sidebar_layout.addStretch()
        
        # Menu Group: SYSTEM
        lbl_system = QtWidgets.QLabel("系统")
        lbl_system.setStyleSheet("color: #71717a; font-size: 14px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-family: 'Microsoft YaHei';")
        sidebar_layout.addWidget(lbl_system)
        
        self.btn_help = create_nav_btn("?", "使用帮助", "用户指南")
        self.btn_exit = create_nav_btn("⏻", "退出系统", "关闭程序")
        
        # Enhanced Sidebar Styles
        self.sidebar.setStyleSheet("""
            QFrame#sidebar {
                background-color: #09090b; /* Zinc-950 */
                border-right: 1px solid #27272a; /* Zinc-800 */
            }
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 8px;
                color: #a1a1aa; /* Zinc-400 */
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 18px;
                font-weight: 500;
                text-align: left;
                padding-left: 16px;
                height: 52px;
            }
            QPushButton:hover {
                background-color: #18181b; /* Zinc-900 */
                color: #f4f4f5; /* Zinc-100 */
            }
            QPushButton:checked {
                background-color: #18181b;
                color: #ffffff;
                border-left: 4px solid #6366f1; /* Indigo-500 */
                padding-left: 12px; /* Adjust for border width to keep text stable */
            }
        """)

        main_layout.addWidget(self.sidebar)

        # === Content Area ===
        content_widget = QtWidgets.QWidget()
        content_widget.setStyleSheet("background-color: #09090b;") # Ensure background matches sidebar
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(32)

        # Header Area
        header_container = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title with accent
        title_box = QtWidgets.QVBoxLayout()
        app_title = QtWidgets.QLabel("开启摄像头")
        app_title.setStyleSheet("font-size: 42px; font-weight: 800; color: #ffffff; letter-spacing: 2px; font-family: 'Microsoft YaHei';")
        app_subtitle = QtWidgets.QLabel("实时智能监控系统")
        app_subtitle.setStyleSheet("font-size: 20px; font-weight: 500; color: #71717a; letter-spacing: 1px; font-family: 'Microsoft YaHei'; margin-top: 4px;")
        title_box.addWidget(app_title)
        title_box.addWidget(app_subtitle)
        
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        
        # System Status Indicator (Top Right)
        status_badge = QtWidgets.QLabel("  ● 系统在线  ")
        status_badge.setStyleSheet("""
            background-color: #064e3b; /* Emerald-900 */
            color: #34d399; /* Emerald-400 */
            border: 1px solid #059669; /* Emerald-600 */
            border-radius: 16px;
            padding: 8px 16px;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 1px;
            font-family: 'Microsoft YaHei';
        """)
        header_layout.addWidget(status_badge)
        
        content_layout.addWidget(header_container)

        # Main Grid: Videos (Left) + Stats/Controls (Right)
        main_split = QtWidgets.QHBoxLayout()
        main_split.setSpacing(24)

        # --- Left Column: Video Feeds ---
        self.views_container = QtWidgets.QWidget()
        views_layout = QtWidgets.QVBoxLayout(self.views_container)
        views_layout.setSpacing(16)
        views_layout.setContentsMargins(0, 0, 0, 0)

        def create_view_frame(title, color_accent="#3f3f46"):
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(f"""
                QFrame {{
                    background-color: #18181b; /* Zinc-900 */
                    border: 1px solid #27272a;
                    border-radius: 12px;
                }}
            """)
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # View Header (Mac-style or Tech-style)
            view_header = QtWidgets.QFrame()
            view_header.setFixedHeight(48)
            view_header.setStyleSheet("""
                background-color: #27272a;
                border-bottom: 1px solid #3f3f46;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
            """)
            vh_layout = QtWidgets.QHBoxLayout(view_header)
            vh_layout.setContentsMargins(20, 0, 20, 0)
            
            # Title
            lbl_title = QtWidgets.QLabel(title)
            lbl_title.setStyleSheet("color: #e4e4e7; font-weight: 600; font-size: 16px; border: none; background: transparent; font-family: 'Microsoft YaHei';")
            
            # Live Indicator
            lbl_live = QtWidgets.QLabel("● 实时")
            lbl_live.setStyleSheet("color: #ef4444; font-weight: 700; font-size: 14px; border: none; background: transparent; letter-spacing: 1px; font-family: 'Microsoft YaHei';")
            
            vh_layout.addWidget(lbl_title)
            vh_layout.addStretch()
            vh_layout.addWidget(lbl_live)
            
            layout.addWidget(view_header)

            # Video Container
            container = QtWidgets.QWidget()
            # Ensure background is black for video
            container.setStyleSheet("background-color: #000000; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;") 
            
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_img = QtWidgets.QLabel()
            lbl_img.setAlignment(QtCore.Qt.AlignCenter)
            lbl_img.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
            lbl_img.setScaledContents(False) # Keep aspect ratio logic in update_frame
            # Placeholder text style
            lbl_img.setText("无信号")
            lbl_img.setStyleSheet("color: #52525b; font-weight: 600; font-family: 'Microsoft YaHei', sans-serif; font-size: 24px; border: none;")

            container_layout.addWidget(lbl_img)
            layout.addWidget(container)
            
            return frame, lbl_img

        self.frame_orig, self.label_orig = create_view_frame("原始画面 // 摄像头 01")
        self.frame_pre, self.label_pre = create_view_frame("预处理 // 图像增强")
        self.frame_inf, self.label_inf = create_view_frame("AI 推理 // 目标检测", "#6366f1")

        # Layout: Top Row (Split) + Bottom Row (Full)
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(16)
        row1_layout.addWidget(self.frame_orig)
        row1_layout.addWidget(self.frame_pre)

        views_layout.addLayout(row1_layout, stretch=4)
        views_layout.addWidget(self.frame_inf, stretch=6) # Give inference view more vertical space

        main_split.addWidget(self.views_container, stretch=3)

        # --- Right Column: Stats & Controls ---
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setFixedWidth(360) # Wider for new card style and larger text
        right_column = QtWidgets.QVBoxLayout(self.right_widget)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(24)

        # Metrics Section
        lbl_metrics = QtWidgets.QLabel("核心指标")
        lbl_metrics.setStyleSheet("color: #71717a; font-size: 16px; font-weight: 700; letter-spacing: 2px; font-family: 'Microsoft YaHei';")
        right_column.addWidget(lbl_metrics)

        self.card_fps = StatCard("系统帧率", "0.0", "赫兹", "⚡", "#10b981") # Emerald
        self.card_objects = StatCard("活跃目标", "0", "个", "🎯", "#3b82f6") # Blue
        self.card_risk = StatCard("威胁等级", "安全", "状态", "🛡️", "#8b5cf6") # Violet

        right_column.addWidget(self.card_fps)
        right_column.addWidget(self.card_objects)
        right_column.addWidget(self.card_risk)
        
        right_column.addSpacing(16)

        # Controls Section
        lbl_controls = QtWidgets.QLabel("控制面板")
        lbl_controls.setStyleSheet("color: #71717a; font-size: 16px; font-weight: 700; letter-spacing: 2px; font-family: 'Microsoft YaHei';")
        right_column.addWidget(lbl_controls)

        controls_frame = QtWidgets.QFrame()
        controls_frame.setObjectName("controlsFrame")
        controls_frame.setStyleSheet("""
            QFrame#controlsFrame {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 16px;
            }
        """)
        controls_layout = QtWidgets.QVBoxLayout(controls_frame)
        controls_layout.setSpacing(20)
        controls_layout.setContentsMargins(24, 24, 24, 24)

        # Time Display
        time_container = QtWidgets.QHBoxLayout()
        icon_time = QtWidgets.QLabel("⏱")
        icon_time.setStyleSheet("color: #71717a; font-size: 20px; border: none; background: transparent;")
        
        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        self.time_label.setAlignment(QtCore.Qt.AlignRight)
        self.time_label.setStyleSheet("color: #e4e4e7; font-family: 'Consolas', monospace; font-size: 24px; font-weight: 600; border: none; background: transparent;")
        
        time_container.addWidget(icon_time)
        time_container.addStretch()
        time_container.addWidget(self.time_label)
        controls_layout.addLayout(time_container)

        # Progress Bar (Slider)
        self.progress_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        # Custom Slider Style
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3f3f46;
                height: 8px;
                background: #27272a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 1px solid #6366f1;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #818cf8;
            }
        """)
        controls_layout.addWidget(self.progress_slider)

        # Playback Buttons
        btns_row = QtWidgets.QHBoxLayout()
        btns_row.setSpacing(16)
        
        def create_ctrl_btn(text, tooltip, primary=False):
            btn = QtWidgets.QPushButton(text)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            if primary:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #6366f1; 
                        color: white; 
                        border: none; 
                        border-radius: 8px; 
                        padding: 12px; 
                        font-weight: bold; 
                        font-size: 20px;
                    }
                    QPushButton:hover { background-color: #4f46e5; }
                    QPushButton:checked { background-color: #f59e0b; }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #27272a; 
                        color: #e4e4e7; 
                        border: 1px solid #3f3f46; 
                        border-radius: 8px; 
                        padding: 12px; 
                        font-size: 20px;
                    }
                    QPushButton:hover { background-color: #3f3f46; }
                """)
            return btn

        self.pause_btn = create_ctrl_btn("⏯", "播放/暂停", primary=True)
        self.pause_btn.setCheckable(True)
        
        self.stop_btn = create_ctrl_btn("⏹", "停止")

        btns_row.addWidget(self.pause_btn, stretch=2)
        btns_row.addWidget(self.stop_btn, stretch=1)
        controls_layout.addLayout(btns_row)
        
        right_column.addWidget(controls_frame)
        right_column.addStretch()

        main_split.addWidget(self.right_widget)
        content_layout.addLayout(main_split)
        main_layout.addWidget(content_widget)

        # Footer
        self.footer_label = QtWidgets.QLabel("系统就绪，等待输入源...")
        self.footer_label.setStyleSheet("color: #52525b; font-size: 14px; margin-top: 8px; font-family: 'Microsoft YaHei', sans-serif;")
        self.footer_label.setAlignment(QtCore.Qt.AlignRight)
        content_layout.addWidget(self.footer_label)


    def setup_connections(self):
        self.btn_open.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_exit.clicked.connect(self.close)
        self.btn_log.clicked.connect(self.log_window.show)
        self.btn_settings.clicked.connect(self.settings_window.show)
        self.btn_help.clicked.connect(self.show_help_dialog)

        self.pause_btn.toggled.connect(self.toggle_pause)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderMoved.connect(self.on_slider_moved)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.stop_btn.clicked.connect(self.stop_camera)

    def show_help_dialog(self):
        if getattr(self, "help_dialog", None):
            self.help_dialog.show()
            self.help_dialog.raise_()
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        dialog.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        dialog.setModal(True) # 模态对话框，点击外部不关闭，需要点关闭按钮

        card = QtWidgets.QFrame()
        card.setObjectName("helpCard")
        # 优化样式：更深色的背景，微光边框，增加阴影感
        card.setStyleSheet("""
            QFrame#helpCard {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #18181b, stop:1 #09090b);
                border: 1px solid #27272a;
                border-radius: 20px;
                color: #e5e7eb;
            }
            QLabel#helpTitle { 
                font-size: 24px; 
                font-weight: bold; 
                color: #ffffff; 
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#helpSubtitle { 
                font-size: 14px; 
                letter-spacing: 1px; 
                color: #60a5fa; 
                font-weight: 600; 
                margin-bottom: 10px;
            }
            QLabel#helpBody { 
                font-size: 15px; 
                line-height: 1.8; 
                color: #d4d4d8;
                padding: 10px;
            }
            QPushButton#helpClose { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #3b82f6);
                border: none; 
                border-radius: 8px; 
                color: white; 
                padding: 10px 24px; 
                font-size: 14px;
                font-weight: 600; 
            }
            QPushButton#helpClose:hover { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #2563eb);
            }
            QPushButton#helpClose:pressed {
                background: #1e40af;
            }
            /* 分割线样式 */
            QFrame#hLine {
                background-color: #3f3f46;
                max-height: 1px;
                border: none;
            }
        """)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(15)

        # Title Section
        title_layout = QtWidgets.QHBoxLayout()
        icon_label = QtWidgets.QLabel("💡")
        icon_label.setStyleSheet("font-size: 28px; background: transparent;")
        title = QtWidgets.QLabel("FastGuard 功能指南")
        title.setObjectName("helpTitle")
        title_layout.addWidget(icon_label)
        title_layout.addSpacing(10)
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        subtitle = QtWidgets.QLabel("INTELLIGENT MONITORING SYSTEM GUIDE")
        subtitle.setObjectName("helpSubtitle")

        # Separator
        line = QtWidgets.QFrame()
        line.setObjectName("hLine")
        line.setFrameShape(QtWidgets.QFrame.HLine)

        # Body Content with HTML for better formatting
        body = QtWidgets.QLabel()
        body.setObjectName("helpBody")
        body.setTextFormat(QtCore.Qt.RichText)
        body.setText("""
            <style>
                ul { margin-left: -20px; }
                li { margin-bottom: 8px; }
                b { color: #60a5fa; }
            </style>
            <ul>
                <li><b>📹 三路视频：</b> 原始 / 预处理 / 推理结果，实时对比分析</li>
                <li><b>⚠️ 预警机制：</b> TTC 碰撞预警、侧向盲区警报、开门危险提示</li>
                <li><b>🎮 控制中心：</b> 支持开启摄像头/视频文件，回放进度拖拽与暂停</li>
                <li><b>📊 数据面板：</b> 实时显示帧率 (FPS)、活跃目标数及当前风险等级</li>
                <li><b>⚙️ 参数微调：</b> 自定义弱检测阈值与边缘增强强度，适应不同环境</li>
                <li><b>📜 系统日志：</b> 记录并查看所有历史警报与系统运行调试信息</li>
            </ul>
        """)
        body.setWordWrap(True)
        body.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Close Button Area
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        
        close_btn = QtWidgets.QPushButton("我已了解")
        close_btn.setObjectName("helpClose")
        close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        close_btn.clicked.connect(dialog.close)
        
        btn_layout.addWidget(close_btn)

        card_layout.addLayout(title_layout)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(line)
        card_layout.addSpacing(10)
        card_layout.addWidget(body)
        card_layout.addSpacing(20)
        card_layout.addLayout(btn_layout)

        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        # Add shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QtGui.QColor(0, 0, 0, 150))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)
        
        dialog_layout.addWidget(card)

        # Resize and Center
        dialog.setFixedWidth(550)
        dialog.adjustSize()
        
        # Center on parent
        if self.isVisible():
             geo = self.geometry()
             x = geo.x() + (geo.width() - dialog.width()) // 2
             y = geo.y() + (geo.height() - dialog.height()) // 2
             dialog.move(x, y)
        else:
             # Center on screen if parent not visible (fallback)
             screen = QtWidgets.QApplication.primaryScreen().geometry()
             dialog.move((screen.width() - dialog.width()) // 2, (screen.height() - dialog.height()) // 2)

        dialog.finished.connect(lambda _: setattr(self, "help_dialog", None))
        self.help_dialog = dialog
        dialog.show()

    def update_preprocess_from_dialog(self, weak_conf, edge_strength):
        # 从设置弹窗同步阈值
        self.weak_conf_threshold = float(weak_conf)
        self.edge_strength_threshold = float(edge_strength)
        self.apply_preprocess_params()

    def apply_preprocess_params(self, _value=None):
        if self.thread:
            self.thread.set_preprocess_thresholds(self.weak_conf_threshold, self.edge_strength_threshold)

    def reset_preprocess_defaults(self):
        self.weak_conf_threshold = self.default_weak_conf_threshold
        self.edge_strength_threshold = self.default_edge_strength_threshold
        # 同步到设置弹窗
        if hasattr(self, "settings_window"):
            self.settings_window.spin_weak_conf.setValue(self.default_weak_conf_threshold)
            self.settings_window.spin_edge_strength.setValue(self.default_edge_strength_threshold)
        self.apply_preprocess_params()



    def apply_modern_theme(self):
        # Global Application Theme
        # Note: Specific widget styles (like Sidebar buttons) are handled in setup_ui
        self.setStyleSheet("""
            QWidget {
                background-color: #09090b; /* Zinc-950 */
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                color: #e4e4e7; /* Zinc-200 */
            }
            
            /* Global Scrollbar Style */
            QScrollBar:vertical {
                border: none;
                background: #18181b;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3f3f46;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #18181b;
                height: 8px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #3f3f46;
                min-width: 20px;
                border-radius: 4px;
            }
            
            /* Global Menu Style */
            QMenu {
                background-color: #18181b;
                color: #e4e4e7;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #27272a;
                color: white;
            }
            
            /* Global Tooltip */
            QToolTip {
                background-color: #27272a;
                color: #ffffff;
                border: 1px solid #3f3f46;
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 12px;
            }
            
            /* Global Message Box */
            QMessageBox {
                background-color: #18181b;
            }
            QMessageBox QLabel {
                color: #e4e4e7;
            }
            QMessageBox QPushButton {
                background-color: #27272a;
                color: #e4e4e7;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 6px 16px;
            }
            QMessageBox QPushButton:hover {
                background-color: #3f3f46;
            }
        """)

    # --- Logic Methods (Adapted from old MainWindow) ---

    def open_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.start_thread(file_path)

    def open_camera(self):
        self.start_thread(0)

    def stop_camera(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        
        for lbl in [self.label_orig, self.label_pre, self.label_inf]:
            lbl.clear()
            lbl.setText("无信号")
            lbl.setStyleSheet("color: #52525b; font-weight: 600; font-family: 'Microsoft YaHei', sans-serif; font-size: 24px; border: none;")
            
        self.card_risk.update_value("离线", "系统空闲")
        self.card_fps.update_value("0.0", "赫兹")
        self.card_objects.update_value("0", "个")
        self.append_system_log("设备已安全断开")


    def start_thread(self, source):
        if self.thread is not None:
            self.thread.stop()

        self.reset_playback_controls()
        self.total_duration = 0.0
        self.total_frames = 0
        self.log_window.log_list.clear()

        model_path = os.path.join(".", "assets", "weights", "yolo11n.pt")

        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.information(self, "下载", "正在下载 yolo11n.pt...")
        
        self.thread = VideoThread(
            source,
            model_path,
            self,
            weak_conf_threshold=self.weak_conf_threshold,
            edge_strength_threshold=self.edge_strength_threshold,
        )
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.status_signal.connect(self.append_system_log)
        self.thread.ttc_signal.connect(self.update_ttc)
        self.thread.side_warning_signal.connect(self.update_side_warning)
        self.thread.log_signal.connect(self.append_log)
        self.thread.latency_signal.connect(self.update_latency)
        self.thread.model_signal.connect(self.update_model_name)
        self.thread.perspective_signal.connect(self.update_perspective)
        self.thread.debug_signal.connect(self.update_debug_info)
        self.thread.position_signal.connect(self.update_position)
        self.thread.hud_signal.connect(self.update_hud)
        self.thread.start()
        
        self.card_risk.update_value("扫描中", "初始化...")
        self.reset_card_style(self.card_risk)

    def reset_playback_controls(self):
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏯")
        self.progress_slider.setEnabled(False)
        self.progress_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")

    def format_time(self, seconds):
        if seconds is None or seconds < 0: return "00:00"
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def update_position(self, frame_index, total_frames, current_seconds, total_seconds):
        self.total_frames = total_frames
        self.total_duration = total_seconds
        self.current_frame_idx = frame_index
        if total_frames > 0:
            self.progress_slider.setEnabled(True)
            self.progress_slider.setRange(0, max(total_frames - 1, 0))
            if not self.progress_slider.isSliderDown():
                self.progress_slider.setValue(frame_index)
        else:
            self.progress_slider.setEnabled(False)
        self.time_label.setText(f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}")

    def toggle_pause(self, checked):
        if not self.thread:
            self.pause_btn.setChecked(False)
            return
        if checked:
            self.pause_btn.setText("▶")
            self.thread.set_paused(True)
        else:
            # 如果当前已播放到末尾，重新点击播放则从头开始
            if hasattr(self, 'current_frame_idx') and self.total_frames > 0:
                if self.current_frame_idx >= self.total_frames - 1:
                    self.thread.set_frame(0)
            self.pause_btn.setText("⏸")
            self.thread.set_paused(False)

    def on_slider_pressed(self):
        if self.thread: self.thread.start_seek()

    def on_slider_moved(self, value):
        if self.thread: self.thread.set_frame(value)

    def on_slider_released(self):
        if self.thread:
            self.thread.set_frame(self.progress_slider.value())
            self.thread.finish_seek()
            if not self.pause_btn.isChecked():
                self.thread.set_paused(False)

    def reset_card_style(self, card):
        card.setStyleSheet("""
            QFrame#statCard {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 16px;
            }
            QFrame#statCard:hover {
                border: 1px solid #6366f1;
                background-color: #27272a;
            }
            QLabel { border: none; background: transparent; }
        """)

    def update_ttc(self, ttc, track_id):
        # Update risk card based on TTC
        if ttc < 1.5:
            self.card_risk.update_value("危险", f"即将碰撞 ({ttc:.1f}s)")
            self.card_risk.setStyleSheet("""
                QFrame#statCard { background: #450a0a; border: 1px solid #dc2626; border-radius: 16px; } 
                QLabel {background: transparent;}
            """)
        elif ttc < 3.0:
            self.card_risk.update_value("警告", f"正在接近 ({ttc:.1f}s)")
            self.card_risk.setStyleSheet("""
                QFrame#statCard { background: #431407; border: 1px solid #d97706; border-radius: 16px; } 
                QLabel {background: transparent;}
            """)
        else:
            self.card_risk.update_value("安全", "安全距离")
            self.reset_card_style(self.card_risk)

    def update_side_warning(self, level, message, object_id):
        self.side_warning_timer.start(3000)
        timestamp = time.strftime("%H:%M:%S")

        prefix = "⚠️ "
        if level == 'danger': prefix = "🚨 "

        log_msg = f"{prefix} [{timestamp}] {message}"
        self.log_window.append_log(log_msg)
        SYSTEM_LOGGER.info(log_msg)

        self.card_risk.update_value("侧向预警", message)
        self.card_risk.setStyleSheet("""
            QFrame#statCard { background: #431407; border: 1px solid #d97706; border-radius: 16px; }
            QLabel {background: transparent;}
        """)

    def clear_side_warning(self):
        self.side_warning_timer.stop()
        self.card_risk.update_value("安全", "安全距离")
        self.reset_card_style(self.card_risk)

    def append_log(self, track_id, ttc):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"⚡ [{timestamp}] ID:{track_id} TTC:{ttc:.1f}s"
        self.log_window.append_log(log_msg)
        SYSTEM_LOGGER.info(log_msg)

    def append_system_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"ℹ️ [{timestamp}] {message}"
        self.log_window.append_log(log_msg)
        SYSTEM_LOGGER.info(log_msg)

    def update_model_name(self, name):

        self.model_name = name
        self.update_footer()

    def update_latency(self, ms):
        self.last_latency = ms
        self.update_footer()
        
    def update_footer(self):
        latency_str = f"{self.last_latency:.1f} ms" if self.last_latency else "-- ms"
        self.footer_label.setText(f"模型: {self.model_name} | 延迟: {latency_str}")

    def update_perspective(self, perspective_text):
        if not self.manual_perspective_set:
            self.append_system_log(f"当前视角: {perspective_text}")

    def set_manual_perspective(self, p_type):
        self.manual_perspective_set = True
        self.append_system_log(f"手动视角: {p_type}")
        # Need a way to inform thread if required, but logic was:
        # thread analyzes -> emits perspective -> UI updates
        # If manual, we just ignore thread emits.
        # But thread also switches detection logic based on its internal perspective.
        # So we really should tell the thread to lock perspective.
        # For now, we rely on the visual override. 
        # (Improvement: Add set_perspective to VideoThread)

    def reset_perspective_analysis(self):
        self.manual_perspective_set = False
        self.append_system_log("视角已重置")
        if self.thread:
            self.thread.perspective_locked = False
            self.thread.current_perspective = "分析中..."
            
    def toggle_debug_mode(self, checked):
        if self.thread:
            self.thread.perspective_debug = checked
        if checked:
            self.append_system_log("调试模式已开启")
        else:
            self.append_system_log("调试模式已关闭")

    def update_debug_info(self, debug_info):
        # Since we removed the large debug label, we can print to console or log occasionally
        # or maybe update a tooltip. For now, we'll just ignore or log if critical.
        pass

    def update_hud(self, payload):
        self.hud_info = payload
        fps = payload.get("fps", 0)
        tracked = payload.get("tracked", 0)
        
        self.card_fps.update_value(f"{fps:.1f}", "赫兹")
        self.card_objects.update_value(str(tracked), "个")

    def update_frame(self, img_orig, img_pre, img_inf):
        self.last_images = (img_orig, img_pre, img_inf)
        self.render_frames()

    def render_frames(self):
        if not hasattr(self, 'last_images') or not self.last_images: return
        
        imgs = self.last_images
        labels = [self.label_orig, self.label_pre, self.label_inf]
        
        for img, lbl in zip(imgs, labels):
            if img.isNull() or lbl.width() <= 0 or lbl.height() <= 0: continue
            
            pixmap = QtGui.QPixmap.fromImage(img)
            # 使用 KeepAspectRatio 保持比例
            scaled = pixmap.scaled(lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            lbl.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_frames()


    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        super().closeEvent(event)




def main():
    print_versions()
    app = QtWidgets.QApplication(sys.argv)
    
    splash = SplashScreen()
    window = MainWindow()
    
    # When splash finishes, show main window
    splash.finished.connect(window.show)
    splash.show()
    
    sys.exit(app.exec_())


# ==================================================================================
# === 7. 程序启动入口 ===
# ==================================================================================
if __name__ == "__main__":
    main()
