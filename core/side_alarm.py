"""
侧面报警模块 (Side Alarm Module)

功能：
1. 侧面视角警示线绘制（红线 + 黄线）
2. 0.8m 物理壁垒 (Side Wall) 绘制 — 颜色随距离动态变化
3. L型角框绘制
4. 去抖动报警 + 视角联动静默
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class SideAlarm:
    """侧面报警类"""

    def __init__(self, camera_side: str = "left", confirm_frames: int = 2,
                 approach_speed_threshold: float = 2.0, safe_distance: float = 2.0):
        """
        Args:
            camera_side: 相机位置，"left" 或 "right"
            confirm_frames: 连续满足触发条件的帧数才输出报警（去抖动）
            approach_speed_threshold: 侧向靠近速度报警阈值（m/s）
            safe_distance: 安全距离阈值（米），用于壁垒颜色和标签判定
        """
        self.camera_side = camera_side
        self.warning_line_x = None
        self.yellow_line_x = None

        # ---- 物理壁垒参数（用户可调） ----
        self.wall_distance = 0.8          # 壁垒到车身边缘的距离（米）
        self.wall_pixel_x = None          # 壁垒在画面中的 X 像素位置（由 IPM 计算）
        self.wall_pixel_x_fallback = None # 无 IPM 时的后备位置

        # 可配置报警参数
        self.approach_speed_threshold = approach_speed_threshold
        self.safe_distance = safe_distance

        # ---- 壁垒颜色状态 ----
        self.closest_object_distance = 99.0   # 最近目标到车身的距离（米），每帧更新
        self._flash_counter = 0               # 闪烁计数器

        # ---- 视角联动 ----
        self._current_perspective: str = "分析中..."  # 由主程序调用 set_perspective() 更新

        # ---- 去抖动 ----
        self._confirm_frames: int = max(1, confirm_frames)
        self._alarm_counters: Dict[int, int] = {}
        self._debounce_frame_counter: int = 0
        self._debounce_last_seen: Dict[int, int] = {}

    def set_perspective(self, perspective: str) -> None:
        """
        接收 ViewClassifier 的当前视角状态。
        当视角为前向时，get_display_params 将降低灵敏度。

        Args:
            perspective: "前向视角" / "侧面视角" / "分析中..."
        """
        self._current_perspective = perspective

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
            self._alarm_counters[track_id] = self._confirm_frames
            return raw_risk
        if raw_risk > 0:
            cnt = self._alarm_counters.get(track_id, 0) + 1
            self._alarm_counters[track_id] = cnt
            return raw_risk if cnt >= self._confirm_frames else 0
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

    def set_wall_distance(self, distance: float):
        """设置物理壁垒距离（用户可调）"""
        self.wall_distance = max(0.1, float(distance))

    def update_closest_distance(self, distance: float):
        """更新最近目标距离（每帧由主循环调用）"""
        self.closest_object_distance = float(distance)

    def update_wall_position(self, side_ipm, frame_shape):
        """
        使用 IPM 计算壁垒像素位置

        Args:
            side_ipm: SideIPM 实例
            frame_shape: (h, w)
        """
        if side_ipm is None:
            return

        pixel_x = side_ipm.get_wall_pixel_x(self.wall_distance, frame_shape)
        if pixel_x is not None:
            self.wall_pixel_x = pixel_x
        else:
            # IPM 计算失败，使用后备位置
            h, w = frame_shape
            if self.camera_side == "left":
                self.wall_pixel_x = int(w * 0.65)
            else:
                self.wall_pixel_x = int(w * 0.35)

    def set_camera_side(self, side):
        """设置相机视角"""
        if side in ["left", "right"]:
            self.camera_side = side
            self.wall_pixel_x = None  # 重置，等待下次 IPM 计算
            print(f"[侧向视角] 已切换到: {side}")

    def update_warning_lines(self, frame_gray, h, w):
        """更新警示线位置"""
        if self.camera_side == "left":
            self.warning_line_x = int(w * 0.45)
            self.yellow_line_x = int(w * 0.75)
        else:
            self.warning_line_x = int(w * 0.55)
            self.yellow_line_x = int(w * 0.25)
        return self.warning_line_x, self.yellow_line_x

    def draw_warning_lines(self, frame, h, w):
        """绘制警示线 / 渐变"""
        warning_line_x = self.warning_line_x
        current_side = self.camera_side

        # 红色渐变区域
        if current_side == "left":
            grad_region = frame[:, :warning_line_x, :].astype(np.float32)
        else:
            grad_region = frame[:, warning_line_x:, :].astype(np.float32)

        red = np.zeros_like(grad_region)
        red[:, :, 2] = 255
        if grad_region.shape[0] > 0 and grad_region.shape[1] > 0:
            alpha = np.linspace(0.35, 0.0, grad_region.shape[1], dtype=np.float32)[None, :, None]
            blended = grad_region * (1 - alpha) + red * alpha
            if current_side == "left":
                frame[:, :warning_line_x, :] = blended.astype(np.uint8)
            else:
                frame[:, warning_line_x:, :] = blended.astype(np.uint8)

        # 红色垂直分割线
        for y in range(0, h, 30):
            cv2.line(frame, (warning_line_x, y),
                     (warning_line_x, min(y + 18, h - 1)), (0, 0, 255), 4)

        # 黄色垂直分割线
        yellow_line_x = self.yellow_line_x
        for y in range(0, h, 30):
            cv2.line(frame, (yellow_line_x, y),
                     (yellow_line_x, min(y + 18, h - 1)), (0, 255, 255), 3)

        return frame

    def draw_side_wall(self, frame, h, w, frame_count: int = 0):
        """
        绘制 0.8m 物理壁垒 (Side Wall)

        颜色逻辑：
        - 绿色：所有目标 > 2.0m（安全）
        - 黄色：最近目标 0.8m ~ 2.0m（注意）
        - 闪烁红色：最近目标 < 0.8m（突破壁垒！）

        Args:
            frame: 当前帧
            h, w: 帧高宽
            frame_count: 当前帧号（用于闪烁效果）
        """
        wall_x = self.wall_pixel_x
        if wall_x is None:
            return frame

        # 确保 wall_x 在画面范围内
        wall_x = max(10, min(w - 10, wall_x))

        dist = self.closest_object_distance
        self._flash_counter += 1

        # ---------- 颜色判定 ----------
        if dist < self.wall_distance:
            # 突破壁垒！闪烁红色
            if self._flash_counter % 6 < 3:  # 每 6 帧闪烁一次
                wall_color = (0, 0, 255)      # 红色
                alpha_val = 0.4
            else:
                wall_color = (0, 0, 180)      # 暗红
                alpha_val = 0.2
            line_thickness = 4
        elif dist < self.safe_distance:
            # 注意区域
            wall_color = (0, 255, 255)        # 黄色
            alpha_val = 0.15
            line_thickness = 3
        else:
            # 安全
            wall_color = (0, 255, 0)          # 绿色
            alpha_val = 0.08
            line_thickness = 2

        # ---------- 绘制半透明带（ROI 优化：仅拷贝窄条区域而非整帧） ----------
        band_half_w = 12  # 半透明带宽度（像素）
        x_left = max(0, wall_x - band_half_w)
        x_right = min(w, wall_x + band_half_w)

        if x_right > x_left:
            roi = frame[0:h, x_left:x_right].copy()
            color_band = np.full_like(roi, wall_color, dtype=np.uint8)
            cv2.addWeighted(color_band, alpha_val, roi, 1.0 - alpha_val, 0, roi)
            frame[0:h, x_left:x_right] = roi

        # ---------- 绘制中心实线 ----------
        cv2.line(frame, (wall_x, 0), (wall_x, h), wall_color, line_thickness)

        # ---------- 绘制刻度标记 ----------
        # 每隔 80 像素画一个短横线
        for y_pos in range(40, h, 80):
            cv2.line(frame, (wall_x - 8, y_pos), (wall_x + 8, y_pos),
                     wall_color, max(1, line_thickness - 1))

        # ---------- 绘制标签 ----------
        label = f"{self.wall_distance:.1f}m"
        if dist < self.wall_distance:
            label = f"!! {self.wall_distance:.1f}m !!"
        elif dist < self.safe_distance:
            label = f"! {self.wall_distance:.1f}m"

        # 标签位置：壁垒线上方
        label_x = wall_x - 30
        label_y = 25
        # 背景框
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame,
                      (label_x - 4, label_y - th - 6),
                      (label_x + tw + 4, label_y + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, wall_color, 2)

        return frame

    def draw_l_corners(self, frame, x1, y1, x2, y2, color, thickness=2, seg=16):
        """绘制 L 型角框"""
        pts = [
            ((x1, y1), (x1 + seg, y1)), ((x1, y1), (x1, y1 + seg)),
            ((x2, y1), (x2 - seg, y1)), ((x2, y1), (x2, y1 + seg)),
            ((x1, y2), (x1 + seg, y2)), ((x1, y2), (x1, y2 - seg)),
            ((x2, y2), (x2 - seg, y2)), ((x2, y2), (x2, y2 - seg)),
        ]
        for p1, p2 in pts:
            cv2.line(frame, p1, p2, color, thickness)
        return frame

    def get_display_params(self, track_id, class_name, ttc, x1, x2, vx,
                           warning_line_x, yellow_line_x):
        """
        获取显示参数（中文）。

        改进：
        - 以目标框横向速度 vx（m/s，来自 side_detector 世界坐标）为主要危险判据，
          阈值从原 30.0（≈108 km/h，几乎不触发）改为 2.0 m/s（≈7.2 km/h）。
        - 当视角为前向时（set_perspective 已更新），降低灵敏度：仅显示绿色安全标签。
        """
        class_map = {
            "person": "行人", "bicycle": "自行车", "car": "轿车",
            "motorcycle": "摩托车", "truck": "卡车", "bus": "公交车"
        }
        chinese_class = class_map.get(class_name, class_name)
        id_suffix = f" {track_id}" if track_id >= 0 else ""
        current_side = self.camera_side

        # ---- 前向视角时静默：侧向报警降低灵敏度 ----
        if self._current_perspective == "前向视角":
            return (0, 200, 0), f"{chinese_class}{id_suffix}", 1

        in_red_zone = (x2 <= warning_line_x) if current_side == "left" else (x1 >= warning_line_x)

        # 横向靠近速度（m/s）：正值表示向本车靠近
        lateral_speed = float(vx)
        if current_side == "right":
            lateral_speed = -lateral_speed  # 右侧摄像头：vx 符号反转
        # 主要危险判据：横向靠近速度 >= 阈值（可配置，默认 2.0 m/s）
        approach_fast = lateral_speed >= self.approach_speed_threshold

        if in_red_zone and approach_fast:
            color = (0, 0, 255)
            label = f"危险！{chinese_class} vx={lateral_speed:.1f}m/s"
            thickness = 3
        elif in_red_zone:
            color = (0, 255, 255)
            label = f"注意：{chinese_class} 并行中"
            thickness = 2
        else:
            color = (0, 255, 0)
            label = f"{chinese_class}{id_suffix}"
            thickness = 2

        return color, label, thickness
