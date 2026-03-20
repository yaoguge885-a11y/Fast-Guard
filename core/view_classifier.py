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

# === 4. 视角自动识别引擎 (正面/侧面视角分类与锁定) ===
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
        """检测画面左右边缘10%区域的固定边缘（优化版）"""
        try:
            h, w = frame.shape[:2]
            edge_region_width = int(w * 0.1)
            
            # 降采样以提高性能
            scale = 0.5
            new_h, new_w = int(h * scale), int(w * scale)
            new_edge_width = int(edge_region_width * scale)
            
            small_frame = cv2.resize(frame, (new_w, new_h))
            left_region = small_frame[:, :new_edge_width]
            right_region = small_frame[:, -new_edge_width:]
            
            def calculate_region_edges(region):
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                edges = cv2.Canny(blurred, 50, 150)
                edge_density = np.sum(edges > 0) / (new_edge_width * new_h)
                
                vertical_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                vertical_edges = np.abs(vertical_edges)
                vertical_edge_density = np.sum(vertical_edges > 50) / (new_edge_width * new_h)
                
                has_edge = edge_density > 0.14 and vertical_edge_density > 0.12
                return has_edge, edge_density, vertical_edge_density

            left_has, left_ed, left_vd = calculate_region_edges(left_region)
            right_has, right_ed, right_vd = calculate_region_edges(right_region)
            
            has_static_edge = left_has or right_has
            edge_density = max(left_ed, right_ed)
            vertical_edge_density = max(left_vd, right_vd)

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

            # 检测分布修正：若目标都在中间且未贴左右两侧，则认为非侧向
            near_edge = False
            mid_band_hits = 0
            total_det = 0
            if detections:
                h_det, w_det = frame.shape[:2]
                for rec in detections:
                    _, dx1, dy1, dx2, dy2, dcx, dcy, _ = rec
                    total_det += 1
                    if dx2 >= int(w_det * 0.9) or dx1 <= int(w_det * 0.1):
                        near_edge = True
                    cx_norm = dcx / max(1, w_det)
                    if 0.28 <= cx_norm <= 0.72:
                        mid_band_hits += 1
                if mid_band_hits >= 1 and not near_edge:
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
        # 强化侧面视角识别（需要同时满足边缘静态边+明显横向流）
        if has_static_edge and avg_horizontal_ratio > 1.08:
            result = "侧面视角"
            self._check_and_lock(result, avg_forward, avg_side, avg_confidence)
            return result
        
        # 强化正面视角识别（无边缘静态边且横向流弱）
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
