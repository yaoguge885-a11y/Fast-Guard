"""
侧面报警模块 (Side Alarm Module)

功能：
1. 侧面视角警示线绘制（红线 + 黄线）
2. 0.8m 物理壁垒 (Side Wall) 绘制 — 颜色随距离动态变化
3. L型角框绘制
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class SideAlarm:
    """侧面报警类"""

    def __init__(self, camera_side="left"):
        """
        Args:
            camera_side: 相机位置，"left" 或 "right"
        """
        self.camera_side = camera_side
        self.warning_line_x = None
        self.yellow_line_x = None

        # ---- 物理壁垒参数（用户可调） ----
        self.wall_distance = 0.8          # 壁垒到车身边缘的距离（米）
        self.wall_pixel_x = None          # 壁垒在画面中的 X 像素位置（由 IPM 计算）
        self.wall_pixel_x_fallback = None # 无 IPM 时的后备位置

        # ---- 壁垒颜色状态 ----
        self.closest_object_distance = 99.0   # 最近目标到车身的距离（米），每帧更新
        self._flash_counter = 0               # 闪烁计数器

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
        elif dist < 2.0:
            # 注意区域
            wall_color = (0, 255, 255)        # 黄色
            alpha_val = 0.15
            line_thickness = 3
        else:
            # 安全
            wall_color = (0, 255, 0)          # 绿色
            alpha_val = 0.08
            line_thickness = 2

        # ---------- 绘制半透明带 ----------
        band_half_w = 12  # 半透明带宽度（像素）
        x_left = max(0, wall_x - band_half_w)
        x_right = min(w, wall_x + band_half_w)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_left, 0), (x_right, h), wall_color, -1)
        cv2.addWeighted(overlay, alpha_val, frame, 1.0 - alpha_val, 0, frame)

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
        elif dist < 2.0:
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
        """获取显示参数（中文）"""
        class_map = {
            "person": "行人", "bicycle": "自行车", "car": "轿车",
            "motorcycle": "摩托车", "truck": "卡车", "bus": "公交车"
        }
        chinese_class = class_map.get(class_name, class_name)
        current_side = self.camera_side

        in_red_zone = False
        if current_side == "left":
            in_red_zone = x2 <= warning_line_x
        else:
            in_red_zone = x1 >= warning_line_x

        if in_red_zone:
            color = (0, 0, 255)
            label = f"危险！{chinese_class} 进入报警区域"
            thickness = 3
        else:
            color = (0, 255, 0)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2

        return color, label, thickness
