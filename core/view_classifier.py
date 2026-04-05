"""
视角分类器 (View Classifier) - V8 优化版 (连续置信度+自适应补偿)

核心逻辑优化：
1. 动态自适应 MSE 阈值：基于全局图像帧间 MSE 校准判定边缘静止的标准，抵抗光照突跳。
2. 历史滑动窗口平滑：记录多帧 ROI MSE 及对消短期高频抖动。
3. 矢量置信度驱动 (fw_score / sw_score)：光流进行精细降级融合 (dx_ratio / dy_ratio)。
4. 自适应双向滞回机制：基于得分置信水平动态调整 confirm 和 unlock 的锁定帧数。
"""

import cv2
import numpy as np
import logging
from datetime import datetime
import os
import collections


class ViewClassifier:
    """视角分类器：具备连续反馈和多维时延感知的判决状态机"""

    def __init__(self):
        self.reset()
        self.setup_logger()

    def setup_logger(self):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'view_classifier_{timestamp}.log')

        self.logger = logging.getLogger(f'ViewClassifier_{timestamp}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"视角分类器 (Continuous&Adaptive) 启动，日志文件：{log_file}")

    def reset(self):
        self.frame_count = 0
        self.current_view = "分析中..."

        self.prev_left_roi = None
        self.prev_right_roi = None
        self.prev_center_roi = None
        self.prev_gray = None

        # ---- 参数配置 (问题4) ----
        self.check_interval = 5          
        self.mse_base_threshold = 15.0   # 基础环境阈值
        self.center_motion_base = 20.0   

        self.confirm_frames = 6          # 初始确认帧数要求
        self.unlock_frames = 15          # 解锁恢复前向所需的帧数

        self.locked = False
        self.locked_perspective = None

        self.side_anchor = None          
        self.left_static = False
        self.right_static = False
        self.center_moving = False

        self.left_mse = 0.0
        self.right_mse = 0.0
        self.center_mse = 0.0

        # ---- 新增优化：队列与连续得分 (问题2, 5) ----
        self.left_mse_history = collections.deque(maxlen=3)
        self.right_mse_history = collections.deque(maxlen=3)
        self.global_mse_history = collections.deque(maxlen=5)

        self.history_centers = {}
        
        self.latest_fw_score = 0.5
        self.latest_sw_score = 0.5
        self.current_forward_score = 0.5
        self.current_side_score = 0.5
        
        self.side_lock_counter = 0

        self.target_width = 160
        self.target_height = 120
        self.max_frame_count = 10000

    def _preprocess(self, frame):
        try:
            small = cv2.resize(frame, (self.target_width, self.target_height))
            if len(small.shape) == 3:
                return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return small.copy()
        except Exception:
            return None

    def _extract_rois(self, gray):
        h, w = gray.shape[:2]
        edge_w = int(w * 0.2)
        left_roi = gray[:, :edge_w]
        right_roi = gray[:, -edge_w:]
        center_roi = gray[:, edge_w:-edge_w]
        return left_roi, right_roi, center_roi

    @staticmethod
    def _compute_mse(roi1, roi2):
        if roi1 is None or roi2 is None or roi1.shape != roi2.shape:
            return 999.0
        diff = cv2.absdiff(roi1, roi2)
        return float(np.mean(diff.astype(np.float32) ** 2))

    def _static_buffer_check(self, gray, flow_data):
        """基于自适应与光流方向的全面得分评估"""
        left_roi, right_roi, center_roi = self._extract_rois(gray)

        if self.prev_left_roi is None:
            self.prev_left_roi = left_roi.copy()
            self.prev_right_roi = right_roi.copy()
            self.prev_center_roi = center_roi.copy()
            self.prev_gray = gray.copy()
            return self.latest_fw_score, self.latest_sw_score

        if self.frame_count % self.check_interval != 0:
            return self.latest_fw_score, self.latest_sw_score

        # --- 问题1: 动态自适应阈值 ---
        global_diff = cv2.absdiff(gray, self.prev_gray)
        global_mse = float(np.mean(global_diff.astype(np.float32) ** 2))
        self.prev_gray = gray.copy()
        
        self.global_mse_history.append(global_mse)
        # 以全局抖动的平均值向上放宽，防止颠簸造成的全局变亮/模糊
        dynamic_thresh = self.mse_base_threshold + 0.4 * np.mean(self.global_mse_history)
        
        self.left_mse = self._compute_mse(left_roi, self.prev_left_roi)
        self.right_mse = self._compute_mse(right_roi, self.prev_right_roi)
        self.center_mse = self._compute_mse(center_roi, self.prev_center_roi)

        self.prev_left_roi = left_roi.copy()
        self.prev_right_roi = right_roi.copy()
        self.prev_center_roi = center_roi.copy()

        # --- 问题2: 滑动均值缓解闪烁 ---
        self.left_mse_history.append(self.left_mse)
        self.right_mse_history.append(self.right_mse)
        
        avg_left_mse = np.mean(self.left_mse_history)
        avg_right_mse = np.mean(self.right_mse_history)

        self.left_static = avg_left_mse < dynamic_thresh
        self.right_static = avg_right_mse < dynamic_thresh
        self.center_moving = self.center_mse > self.center_motion_base
        
        if self.left_static and self.right_static:
            self.side_anchor = "left" if avg_left_mse < avg_right_mse else "right"
        elif self.left_static:
            self.side_anchor = "left"
        elif self.right_static:
            self.side_anchor = "right"

        # --- 问题3 & 5: 分区光流方向与特征加分 ---
        flow_dx_abs = []
        flow_dy_abs = []
        for (cx, cy, dx, dy) in flow_data:
            flow_dx_abs.append(abs(dx))
            flow_dy_abs.append(abs(dy))
            
        dx_mean = np.mean(flow_dx_abs) if flow_dx_abs else 0.0
        dy_mean = np.mean(flow_dy_abs) if flow_dy_abs else 0.0
        total_flow = dx_mean + dy_mean + 1e-5
        
        dx_ratio = dx_mean / total_flow
        dy_ratio = dy_mean / total_flow
        
        side_evidence = 0.0
        front_evidence = 0.0
        
        # 1) 边缘静止提供原生证据
        if (self.left_static or self.right_static) and self.center_moving:
            side_evidence += 0.4
        if not self.left_static and not self.right_static:
            front_evidence += 0.3
            
        # 2) 光流一致性得分提权
        if dx_ratio > 0.65 and dx_mean > 2.0:
            side_evidence += 0.6 * ((dx_ratio - 0.5) * 2)
        if dy_ratio > 0.6 and dy_mean > 1.0:
            front_evidence += 0.7 * ((dy_ratio - 0.5) * 2)
            
        self.latest_fw_score = min(1.0, max(0.0, front_evidence))
        self.latest_sw_score = min(1.0, max(0.0, side_evidence))
        
        return self.latest_fw_score, self.latest_sw_score

    def _state_locker(self, fw_score, sw_score):
        """基于双向自适应置信度滞回机制的状态机 (问题4)"""
        if self.locked:
            return self.locked_perspective

        # EMA 更新整体打分趋势
        self.current_forward_score = 0.8 * self.current_forward_score + 0.2 * fw_score
        self.current_side_score = 0.8 * self.current_side_score + 0.2 * sw_score
        
        # 平滑后再进行一轮 softmax 感的归一化，使得相加为 1.0 (不强求也可)
        total = self.current_forward_score + self.current_side_score + 1e-5
        fw_conf = self.current_forward_score / total
        sw_conf = self.current_side_score / total

        if self.current_view == "分析中...":
            self.current_view = "前向视角"

        confidence_gap = abs(sw_conf - fw_conf)
        # 高置信度可以极大缩短确认所需的时间帧数
        scale_factor = max(0.3, 1.0 - 0.7 * confidence_gap)  
        
        req_confirm = max(2, int(self.confirm_frames * scale_factor))
        req_unlock = max(4, int(self.unlock_frames * scale_factor))

        if sw_conf > fw_conf and sw_conf > 0.55:
            # 倾向侧面
            if self.current_view == "侧面视角":
                self.side_lock_counter = 0
            else:
                self.side_lock_counter += 1
                if self.side_lock_counter >= req_confirm:
                    self.current_view = "侧面视角"
                    self.logger.warning(f"切换至侧向视角 (短切所需帧: {req_confirm}, gap: {confidence_gap:.2f})")
        elif fw_conf > sw_conf and fw_conf > 0.55:
            # 倾向前方
            if self.current_view == "侧面视角":
                self.side_lock_counter += 1 # 反向借用计数器做退回累积
                if self.side_lock_counter >= req_unlock:
                    self.current_view = "前向视角"
                    self.side_lock_counter = 0
                    self.logger.info(f"切回前向视角 (解锁所需帧: {req_unlock})")
            else:
                self.current_view = "前向视角"
                self.side_lock_counter = 0

    def analyze_frame(self, frame, detections=None):
        if self.locked:
            fw = 0.95 if self.locked_perspective == "前向视角" else 0.05
            sw = 0.95 if self.locked_perspective == "侧面视角" else 0.05
            return fw, sw

        if self.frame_count >= self.max_frame_count:
            # 简单周期重启
            self.frame_count = 0
            self.prev_left_roi = None
            self.prev_right_roi = None
            self.prev_center_roi = None

        self.frame_count += 1

        gray = self._preprocess(frame)
        if gray is None:
            return 0.5, 0.5

        flow_data = [] 
        if detections:
            current_ids = set()
            for rec in detections:
                tid, cx, cy = rec[0], rec[5], rec[6]
                current_ids.add(tid)
                if tid in self.history_centers:
                    px, py = self.history_centers[tid]
                    flow_data.append((cx, cy, cx - px, cy - py))
                self.history_centers[tid] = (cx, cy)
            
            for tid in list(self.history_centers.keys()):
                if tid not in current_ids:
                    del self.history_centers[tid]

        fw_score, sw_score = self._static_buffer_check(gray, flow_data)
        self._state_locker(fw_score, sw_score)

        # 发送连续值！由当前时延平滑的结果为准。
        total = self.current_forward_score + self.current_side_score + 1e-5
        return self.current_forward_score / total, self.current_side_score / total

    def determine_perspective(self):
        if self.locked:
            return self.locked_perspective
        return self.current_view

    def get_debug_info(self):
        total = self.current_forward_score + self.current_side_score + 1e-5
        fw = self.current_forward_score / total
        sw = self.current_side_score / total
        
        return {
            'current_view': self.current_view,
            'side_anchor': self.side_anchor,
            'left_mse': self.left_mse,
            'right_mse': self.right_mse,
            'center_mse': self.center_mse,
            'left_static': getattr(self, "left_static", False),
            'right_static': getattr(self, "right_static", False),
            'center_moving': getattr(self, "center_moving", False),
            'side_lock_counter': self.side_lock_counter,
            'frame_count': self.frame_count,
            'locked': self.locked,
            'locked_perspective': self.locked_perspective or "未锁定",
            'has_static_edge': getattr(self, "left_static", False) or getattr(self, "right_static", False),
            'forward_score': fw,
            'side_score': sw,
            'confidence': abs(fw - sw),
        }

    # -- 后续绘图函数保持 --
    def is_side_view(self): return self.current_view == "侧面视角"
    def is_front_view(self): return self.current_view == "前向视角"
    def get_perspective(self): return self.current_view
    def get_anchor_position(self): return self.side_anchor
    def should_enable_side_detection(self): return self.is_side_view()
    def get_detection_offset(self):
        if self.is_side_view():
            if self.side_anchor == "left": return -1
            elif self.side_anchor == "right": return 1
        return 0

    def get_debug_rects(self, original_frame):
        if original_frame is None: return []
        h, w = original_frame.shape[:2]
        rects = []
        if getattr(self, 'left_static', False):
            rects.append((int(w * 0.05), int(h * 0.3), int(w * 0.1), int(h * 0.2), (0, 0, 255)))
        if getattr(self, 'right_static', False):
            rects.append((int(w * 0.75), int(h * 0.3), int(w * 0.1), int(h * 0.2), (0, 0, 255)))
        return rects

    def draw_debug_info(self, frame):
        if frame is None: return frame
        try:
            h, w = frame.shape[:2]
            y_offset = int(h * 0.8)
            x_offset = int(w * 0.05)
            line_height = 25
            
            info = self.get_debug_info()
            color = (0, 255, 0) if info['current_view'] == "侧面视角" else (0, 165, 255)
            
            texts = [
                f"View: {info['current_view']}",
                f"Fw_Score: {info['forward_score']:.2f} | Sw_Score: {info['side_score']:.2f}",
                f"Lock Delay: {info['side_lock_counter']} frames"
            ]
            
            for i, text in enumerate(texts):
                cv2.putText(frame, text, (x_offset, y_offset + i * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            pass
        return frame
