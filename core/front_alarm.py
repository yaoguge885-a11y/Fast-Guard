"""
正面报警模块 (Front Alarm Module)

将正面视角的报警算法独立出来，实现与侧面视角完全分离的模块化设计。
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class FrontAlarm:
    """
    正面报警类，实现正面视角的报警逻辑
    """
    
    def __init__(self):
        """初始化正面报警模块"""
        self.warning_line_y = None
        self.detect_line_y = None
        self._locked_warning_y = None
        self._ema_hood_y = None
        self._forward_lock_count = 0
    
    def detect_hood_y(self, frame_gray, h, w):
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
        hood_y = max(int(h * 0.60), min(int(h * 0.8), hood_y))
        return hood_y
    
    def update_warning_lines(self, frame_gray, h, w):
        """更新警示线位置"""
        # 动态检测引擎盖边缘 (仅在前5帧进行)
        if self._locked_warning_y is None:
            curr_hood_y = self.detect_hood_y(frame_gray, h, w)
            if curr_hood_y is not None:
                if self._ema_hood_y is None:
                    self._ema_hood_y = float(curr_hood_y)
                else:
                    self._ema_hood_y = 0.95 * self._ema_hood_y + 0.05 * curr_hood_y
            
            if self._forward_lock_count >= 5:
                self._locked_warning_y = int(self._ema_hood_y) if self._ema_hood_y is not None else int(h * 0.88)
            else:
                self._forward_lock_count += 1
        
        # 使用锁定值或当前计算值
        base_warning_y = self._locked_warning_y if self._locked_warning_y is not None else (int(self._ema_hood_y) if self._ema_hood_y is not None else int(h * 0.88))
        
        self.warning_line_y = base_warning_y
        self.detect_line_y = int(h * 0.40)
        
        return self.warning_line_y, self.detect_line_y
    
    def draw_warning_lines(self, frame, h, w):
        """绘制警示线/渐变"""
        warning_line_y = self.warning_line_y
        detect_line_y = self.detect_line_y
        
        # 绘制渐变区域
        grad_region = frame[warning_line_y:, :, :].astype(np.float32)
        red = np.zeros_like(grad_region)
        red[:, :, 2] = 255
        if grad_region.shape[0] > 0:
            alpha = np.linspace(0.0, 0.35, grad_region.shape[0], dtype=np.float32)[:, None, None]
            blended = grad_region * (1 - alpha) + red * alpha
            frame[warning_line_y:, :, :] = blended.astype(np.uint8)
        
        # 绘制红色警示线
        for x in range(0, w, 30):
            cv2.line(frame, (x, warning_line_y), (min(x + 18, w - 1), warning_line_y), (0, 0, 255), 4)
        
        # 绘制灰色检测线
        for x in range(0, w, 30):
            cv2.line(frame, (x, detect_line_y), (min(x + 16, w - 1), detect_line_y), (160, 160, 160), 2)
        
        return frame
    
    def draw_l_corners(self, frame, x1, y1, x2, y2, color, thickness=2, seg=16):
        """绘制L型角框"""
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
    
    def get_display_params(self, track_id, class_name, ttc, y2, warning_line_y, lateral_fast, red_ok):
        """获取显示参数（中文）"""
        class_map = {
            "person": "行人",
            "bicycle": "自行车",
            "car": "轿车",
            "motorcycle": "摩托车",
            "truck": "卡车",
            "bus": "公交车"
        }
        chinese_class = class_map.get(class_name, class_name)
        
        if y2 <= warning_line_y:
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
