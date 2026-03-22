"""
视角分类器 (View Classifier) - 物理锚点一票否决制 V7
核心逻辑：
1. 死区像素对比 (Static Buffer)：左侧 20% 和右侧 20% ROI 对比
2. 斜率一致性 (Slope Consistency)：几何辅助检测
3. 视角状态锁 (State Locker)：30 帧计数器防止跳变
4. 可视化调试：红色方块标记静止锚点
"""

import cv2
import numpy as np
import logging
from collections import deque
from datetime import datetime
import os


class ViewClassifier:
    """视角分类器：物理锚点一票否决制 V7"""
    
    def __init__(self):
        self.reset()
        self.setup_logger()
    
    def setup_logger(self):
        """设置日志记录器"""
        # 创建 logs 目录
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名（使用日期时间）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'view_classifier_{timestamp}.log')
        
        # 配置日志
        self.logger = logging.getLogger('ViewClassifier')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的 handler
        self.logger.handlers = []
        
        # 文件 handler（使用缓冲，减少 I/O 次数）
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"视角分类器启动，日志文件：{log_file}")
    
    def reset(self):
        """重置分类器状态"""
        self.frame_count = 0
        self.current_view = "分析中..."
        
        # 静态缓冲区（简化为单帧缓存）
        self.prev_left_roi = None
        self.prev_right_roi = None
        
        # 参数配置
        self.check_interval = 5  # 每 5 帧检测一次
        self.mse_threshold = 25.0  # 均方误差阈值（低于此值认为静止）- 大幅提高以减少误判
        self.slope_threshold = 0.3  # 斜率一致性阈值
        self.vanish_x_deviation = 0.4  # 消失点 X 偏离中心 40%
        
        # 视角状态锁
        self.side_lock_counter = 0
        self.required_unlock_frames = 30  # 连续 30 帧剧烈运动才能解锁
        
        # 前向视角保护
        self.front_lock_counter = 0
        self.required_front_frames = 20  # 连续 20 帧才能锁定前向视角
        
        # 侧向锚点
        self.side_anchor = None  # "left" 或 "right"
        self.left_static = False
        self.right_static = False
        
        # 前向视角默认锁定（初始状态）
        self.default_front_view = True
        
        # 调试信息
        self.left_mse = 0.0
        self.right_mse = 0.0
        self.debug_rects = []
        
        # 性能优化：降采样尺寸
        self.target_width = 160
        self.target_height = 120
        
        # 防止计数器溢出
        self.max_frame_count = 10000
        
        # 清理日志 handler（防止重复）
        if hasattr(self, 'logger') and self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
    
    def _preprocess(self, frame):
        """预处理：降采样 + 转灰度图（性能优化）"""
        try:
            # 降采样到 160x120 以提高性能
            small = cv2.resize(frame, (self.target_width, self.target_height))
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small.copy()
            return gray
        except Exception as e:
            print(f"[ViewClassifier] 预处理错误：{e}")
            return None
    
    def _extract_rois(self, gray):
        """提取左右两侧 20% ROI"""
        h, w = gray.shape[:2]
        
        # 左侧 20%
        left_w = int(w * 0.2)
        left_roi = gray[:, :left_w]
        
        # 右侧 20%
        right_w = int(w * 0.2)
        right_roi = gray[:, -right_w:]
        
        return left_roi, right_roi, left_w, right_w
    
    def _compute_mse(self, roi1, roi2):
        """计算两个 ROI 的均方误差 (MSE)"""
        if roi1 is None or roi2 is None:
            return 999.0
        if roi1.shape != roi2.shape:
            return 999.0
        
        # 绝对差值
        diff = cv2.absdiff(roi1, roi2)
        
        # 均方误差
        mse = float(np.mean(diff ** 2))
        
        return mse
    
    def _static_buffer_check(self, gray):
        """
        死区像素对比（核心算法）- 内存优化版
        
        Returns:
            is_side_view: 是否为侧向视角
            side_anchor: "left" 或 "right" 或 None
            left_static: 左侧是否静止
            right_static: 右侧是否静止
            left_mse: 左侧 MSE
            right_mse: 右侧 MSE
        """
        left_roi, right_roi, left_w, right_w = self._extract_rois(gray)
        
        # 初始化缓冲区
        if self.prev_left_roi is None:
            self.prev_left_roi = left_roi.copy()
            self.prev_right_roi = right_roi.copy()
            return False, None, False, False, 0.0, 0.0
        
        # 每隔 5 帧检测一次
        if self.frame_count % self.check_interval != 0:
            return (self.left_static or self.right_static), \
                   self.side_anchor, \
                   self.left_static, \
                   self.right_static, \
                   self.left_mse, \
                   self.right_mse
        
        # 计算 MSE（使用视图而非 copy，减少内存）
        self.left_mse = self._compute_mse(left_roi, self.prev_left_roi)
        self.right_mse = self._compute_mse(right_roi, self.prev_right_roi)
        
        # 物理判定：MSE 低于阈值认为静止
        self.left_static = self.left_mse < self.mse_threshold
        self.right_static = self.right_mse < self.mse_threshold
        
        # 更新缓冲区（直接赋值视图，避免 copy）
        self.prev_left_roi = left_roi.copy()
        self.prev_right_roi = right_roi.copy()
        
        # 一票否决制优化：要求两侧都静止才判定为侧向视角（减少误判）
        # 前向视角中，车头引擎盖也会导致一侧静止，所以必须两侧都静止才是侧向
        is_side_view = self.left_static and self.right_static
        
        # 确定侧向锚点
        side_anchor = None
        if is_side_view:
            # 两侧都静止时，选择 MSE 更低的一侧作为锚点
            side_anchor = "left" if self.left_mse < self.right_mse else "right"
        
        self.side_anchor = side_anchor
        
        # 前向视角保护计数器
        if not is_side_view:
            self.front_lock_counter += 1
        else:
            self.front_lock_counter = 0
        
        # 减少日志频率（每 10 次检测记录一次）
        if self.frame_count % (self.check_interval * 10) == 0:
            self.logger.debug(f"左侧 MSE: {self.left_mse:.2f} {'(静止)' if self.left_static else ''} | 右侧 MSE: {self.right_mse:.2f} {'(静止)' if self.right_static else ''} | 侧向={is_side_view} | 前向计数：{self.front_lock_counter}")
        
        return is_side_view, side_anchor, self.left_static, self.right_static, self.left_mse, self.right_mse
    
    def _slope_consistency_check(self, gray):
        """
        斜率一致性检测（几何辅助）- 已禁用以提高性能
        
        在像素对比不明显时（如夜晚），使用直线斜率辅助判断
        注意：由于 HoughLinesP 计算量过大，此功能已禁用
        
        Returns:
            is_side_view: 是否为侧向视角
            vanish_x: 消失点 X 坐标（归一化）
        """
        # 已禁用：HoughLinesP 计算量过大导致卡死
        return False, None
    
    def _state_locker(self, is_side_view):
        """
        视角状态锁（防止跳变）- 保守版
        
        核心原则：默认前向视角，除非有充分证据证明是侧向
        侧向视角判定：需要连续 10 帧 + 两侧 MSE 都很低（<10）
        前向视角解锁：连续 30 帧检测到运动
        """
        # 如果初始状态，保持前向视角
        if self.current_view == "分析中...":
            self.current_view = "前向视角"
            return self.current_view
        
        if is_side_view:
            # 重置前向计数器
            self.front_lock_counter = 0
            
            # 如果已经是侧向视角，保持
            if self.current_view == "侧面视角":
                self.side_lock_counter = 0  # 重置解锁计数器
            else:
                # 首次检测到侧向，需要连续 10 帧确认（更严格）
                self.side_lock_counter += 1
                # 额外检查：两侧 MSE 都必须非常低
                both_very_static = self.left_mse < 10.0 and self.right_mse < 10.0
                if self.side_lock_counter >= 10 and both_very_static:
                    self.logger.warning(f"确认侧向视角，连续{self.side_lock_counter}帧，MSE: L={self.left_mse:.1f}, R={self.right_mse:.1f}")
                    self.current_view = "侧面视角"
                else:
                    # 保持前向视角
                    pass
        else:
            # 前向视角检测
            if self.current_view == "侧面视角":
                # 需要连续 30 帧前向才能解锁
                self.side_lock_counter += 1
                if self.side_lock_counter >= self.required_unlock_frames:
                    self.logger.info(f"连续{self.side_lock_counter}帧前向，解锁切回前向视角")
                    self.current_view = "前向视角"
                    self.side_lock_counter = 0
                else:
                    # 保持侧向视角
                    pass
            else:
                # 已经是前向视角，保持稳定
                self.current_view = "前向视角"
        
        return self.current_view
    
    def analyze_frame(self, frame, detections=None):
        """
        分析单帧画面（内存优化版）
        
        Args:
            frame: 输入帧
            detections: 检测结果（未使用）
            
        Returns:
            forward_score: 前向视角得分
            side_score: 侧向视角得分
        """
        # 防止计数器溢出
        if self.frame_count >= self.max_frame_count:
            self.frame_count = 0
            self.prev_left_roi = None
            self.prev_right_roi = None
        
        self.frame_count += 1
        
        gray = self._preprocess(frame)
        if gray is None:
            return 0.5, 0.5
        
        # 1. 死区像素对比（核心）
        is_side_view, side_anchor, left_static, right_static, left_mse, right_mse = \
            self._static_buffer_check(gray)
        
        # 2. 斜率一致性检测已禁用（性能优化）
        # 3. 视角状态锁
        self._state_locker(is_side_view)
        
        # 4. 返回得分
        if self.current_view == "侧面视角":
            return 0.05, 0.95
        elif self.current_view == "前向视角":
            return 0.95, 0.05
        else:
            return 0.5, 0.5
    
    def get_debug_rects(self, original_frame):
        """
        获取调试用的矩形框（红色方块标记静止锚点）
        
        Args:
            original_frame: 原始帧
            
        Returns:
            rects: 矩形框列表 [(x, y, w, h, color), ...]
        """
        if original_frame is None:
            return []
        
        h, w = original_frame.shape[:2]
        rects = []
        
        # 左侧静止：画红色实心方块
        if self.left_static:
            left_w = int(w * 0.2)
            rect_x = int(w * 0.05)
            rect_y = int(h * 0.3)
            rect_w = int(w * 0.1)
            rect_h = int(h * 0.2)
            rects.append((rect_x, rect_y, rect_w, rect_h, (0, 0, 255)))
        
        # 右侧静止：画红色实心方块
        if self.right_static:
            right_x = int(w * 0.75)
            rect_y = int(h * 0.3)
            rect_w = int(w * 0.1)
            rect_h = int(h * 0.2)
            rects.append((right_x, rect_y, rect_w, rect_h, (0, 0, 255)))
        
        return rects
    
    def draw_debug_info(self, frame):
        """
        在画面上绘制调试信息
        
        Args:
            frame: 输入帧
            
        Returns:
            frame: 绘制调试信息后的帧
        """
        if frame is None:
            return frame
        
        # 绘制红色方块标记静止锚点
        rects = self.get_debug_rects(frame)
        for (x, y, w, h, color) in rects:
            # 实心方块（半透明）
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # 边框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # 绘制视角状态文字
        view_text = f"视角：{self.current_view}"
        if self.side_anchor:
            view_text += f" (锚点：{self.side_anchor})"
        
        cv2.putText(frame, view_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制 MSE 值
        mse_text = f"左 MSE: {self.left_mse:.1f} | 右 MSE: {self.right_mse:.1f}"
        cv2.putText(frame, mse_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 绘制解锁计数器
        if self.current_view == "侧面视角":
            lock_text = f"解锁计数：{self.side_lock_counter}/{self.required_unlock_frames}"
            cv2.putText(frame, lock_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return frame
    
    def determine_perspective(self):
        """确定最终视角"""
        return self.current_view
    
    def get_debug_info(self):
        """获取调试信息"""
        return {
            'current_view': self.current_view,
            'side_anchor': self.side_anchor,
            'left_mse': self.left_mse,
            'right_mse': self.right_mse,
            'left_static': self.left_static,
            'right_static': self.right_static,
            'side_lock_counter': self.side_lock_counter,
            'frame_count': self.frame_count
        }
    
    def is_side_view(self):
        """判断是否为侧向视角"""
        return self.current_view == "侧面视角"
    
    def is_front_view(self):
        """判断是否为前向视角"""
        return self.current_view == "前向视角"
    
    def get_perspective(self):
        """获取当前视角"""
        return self.current_view
    
    def get_anchor_position(self):
        """获取车身锚点位置"""
        return self.side_anchor
    
    def should_enable_side_detection(self):
        """是否启用侧向检测"""
        return self.is_side_view()
    
    def get_detection_offset(self):
        """获取检测框偏移量"""
        if self.is_side_view():
            if self.side_anchor == "left":
                return -1
            elif self.side_anchor == "right":
                return 1
        return 0
