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

    def __init__(
        self,
        warn_ttc: float = 3.0,
        danger_ttc: float = 1.5,
        ped_danger_ttc: float = 2.0,
        lateral_danger_ttc: float = 3.0,
        confirm_frames: int = 2,
        hood_roi_y_start: float = 0.6,
        hood_roi_x_margin: float = 0.2,
        hood_gradient_min: float = 10.0,
        hood_y_max_ratio: float = 0.8,
    ):
        """
        初始化正面报警模块

        Args:
            warn_ttc: 一级预警（黄色）TTC阈值（秒）
            danger_ttc: 二级危险（红色）TTC阈值（秒）
            ped_danger_ttc: 行人/非机动车危险TTC阈值（秒）
            lateral_danger_ttc: 快速横移时的危险TTC阈值（秒）
            confirm_frames: 连续满足条件才输出报警的帧数（去抖动）
            hood_roi_y_start: 引擎盖检测 ROI 起始高度比例
            hood_roi_x_margin: 引擎盖检测 ROI 水平边距比例
            hood_gradient_min: 引擎盖边缘最低 Sobel 梯度
            hood_y_max_ratio: 引擎盖位置最大高度比例
        """
        self.warn_ttc = warn_ttc
        self.danger_ttc = danger_ttc
        self.ped_danger_ttc = ped_danger_ttc
        self.lateral_danger_ttc = lateral_danger_ttc
        self._confirm_frames = confirm_frames

        # 引擎盖检测参数（可配置）
        self.hood_roi_y_start = hood_roi_y_start
        self.hood_roi_x_margin = hood_roi_x_margin
        self.hood_gradient_min = hood_gradient_min
        self.hood_y_max_ratio = hood_y_max_ratio

        # 去抖动计数器：track_id → 连续触发帧数
        self._alarm_counters: Dict[int, int] = {}
        self._debounce_frame_counter: int = 0
        self._debounce_last_seen: Dict[int, int] = {}

        self.warning_line_y = None
        self.detect_line_y = None
        self._locked_warning_y = None
        self._ema_hood_y = None
        self._forward_lock_count = 0

    def check_debounce(self, track_id: int, raw_risk: int) -> int:
        """
        去抖动：连续 confirm_frames 帧满足触发条件才真正输出报警。

        Args:
            track_id: 跟踪 ID
            raw_risk: 当前帧原始风险等级（0/1/2）

        Returns:
            去抖后的风险等级
        """
        self._debounce_frame_counter += 1
        self._debounce_last_seen[track_id] = self._debounce_frame_counter
        if raw_risk >= 2:
            # 高危立即输出，同时拉满计数器防止后续降级抖动
            self._alarm_counters[track_id] = self._confirm_frames
            return raw_risk
        if raw_risk > 0:
            cnt = self._alarm_counters.get(track_id, 0) + 1
            self._alarm_counters[track_id] = cnt
            return raw_risk if cnt >= self._confirm_frames else 0
        # raw_risk == 0 → 清零，避免旧计数影响下次触发
        self._alarm_counters.pop(track_id, None)
        return 0

    def clear_debounce(self, track_id: int) -> None:
        """显式清除某目标的去抖计数（目标消失时调用）"""
        self._alarm_counters.pop(track_id, None)
        self._debounce_last_seen.pop(track_id, None)

    def cleanup_debounce(self, max_age: int = 60) -> None:
        """清理超过 max_age 帧未活跃的去抖计数器，防止内存泄漏"""
        stale = [tid for tid, last in self._debounce_last_seen.items()
                 if self._debounce_frame_counter - last > max_age]
        for tid in stale:
            self._alarm_counters.pop(tid, None)
            self._debounce_last_seen.pop(tid, None)

    def detect_hood_y(self, frame_gray, h, w):
        """自动侦测引擎盖边缘 (水平线)"""
        roi_y1 = int(h * self.hood_roi_y_start)
        roi_y2 = h
        roi_x1 = int(w * self.hood_roi_x_margin)
        roi_x2 = int(w * (1.0 - self.hood_roi_x_margin))
        
        roi = frame_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None
            
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)
        
        row_mean = np.mean(abs_sobel_y, axis=1)
        if len(row_mean) == 0:
            return None
            
        max_idx = int(np.argmax(row_mean))
        if row_mean[max_idx] < self.hood_gradient_min:
            return None

        hood_y = roi_y1 + max_idx
        hood_y = max(int(h * self.hood_roi_y_start), min(int(h * self.hood_y_max_ratio), hood_y))
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
        """
        获取显示参数（中文）。

        阈值使用 __init__ 中配置的值，保持向后兼容。
        """
        class_map = {
            "person": "行人",
            "bicycle": "自行车",
            "car": "轿车",
            "motorcycle": "摩托车",
            "truck": "卡车",
            "bus": "公交车"
        }
        chinese_class = class_map.get(class_name, class_name)

        ped_classes = {"bicycle", "motorcycle", "person"}
        # 针对行人/非机动车使用独立的危险阈值
        danger_t = self.ped_danger_ttc if class_name in ped_classes else self.danger_ttc
        warn_t = self.warn_ttc

        if y2 <= warning_line_y:
            color = (120, 120, 120)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        elif lateral_fast and ttc < self.lateral_danger_ttc:
            color = (255, 0, 0)
            label = f"碰撞时间 {ttc:.1f}秒"
            thickness = 2
        elif ttc < danger_t and red_ok:
            color = (0, 0, 255)
            label = f"危险！碰撞时间 {ttc:.1f}秒"
            thickness = 3
        elif ttc < warn_t:
            color = (0, 255, 255)
            label = f"碰撞时间 {ttc:.1f}秒"
            thickness = 2
        elif class_name in ped_classes:
            color = (255, 0, 255)
            label = f"{chinese_class} (注意)"
            thickness = 3
        else:
            color = (0, 255, 0)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2

        return color, label, thickness
