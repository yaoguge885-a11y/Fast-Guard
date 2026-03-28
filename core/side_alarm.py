"""
侧面报警模块 (Side Alarm Module)

将侧面视角的报警算法独立出来，实现与正面视角完全相同的报警逻辑，
包括红框绘制和灰线显示。
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class SideAlarm:
    """
    侧面报警类，实现与正面视角相同的报警逻辑
    """
    
    def __init__(self, camera_side="left"):
        """初始化侧面报警模块
        
        Args:
            camera_side: 相机位置，"left" 或 "right"（手动选择）
        """
        self.camera_side = camera_side
        self.warning_line_x = None
        self.yellow_line_x = None
    
    def update_warning_lines(self, frame_gray, h, w):
        """更新警示线位置"""
        # 直接使用camera_side设置
        current_side = self.camera_side
        
        # 设置红线位置（画在车身稍微外侧）
        # 左侧视角：车身在右侧，红线在画面偏左一点（车身外侧）
        # 右侧视角：车身在左侧，红线在画面偏右一点（车身外侧）
        if current_side == "left":
            # 红色区域在左侧，红线稍微偏左（车身外侧）
            self.warning_line_x = int(w * 0.45)
        else:
            # 红色区域在右侧，红线稍微偏右（车身外侧）
            self.warning_line_x = int(w * 0.55)
        
        # 设置黄线位置（红线外侧，距离红线 1/4 画面宽度）
        # 黄线始终在红色区域的外侧（远离车身的一侧）
        if current_side == "left":
            # 红色区域在左侧 → 黄线在红线右侧（外侧）
            self.yellow_line_x = int(w * 0.75)
        else:
            # 红色区域在右侧 → 黄线在红线左侧（外侧）
            self.yellow_line_x = int(w * 0.25)
        
        return self.warning_line_x, self.yellow_line_x
    
    def set_camera_side(self, side):
        """设置相机视角
        
        Args:
            side: "left" 或 "right"
        """
        if side in ["left", "right"]:
            self.camera_side = side
            print(f"[侧向视角] 已切换到: {side}")
    
    def draw_warning_lines(self, frame, h, w):
        """绘制警示线/渐变"""
        warning_line_x = self.warning_line_x
        yellow_line_x = self.yellow_line_x
        
        # 使用camera_side设置
        current_side = self.camera_side
        
        # 根据视角绘制报警区域
        # current_side 直接表示红色区域应该在的位置
        if current_side == "left":
            # 红色区域在左侧（红线稍微偏左），渐变区域在红线以内
            grad_region = frame[:, :warning_line_x, :].astype(np.float32)
        else:
            # 红色区域在右侧（红线稍微偏右），渐变区域在红线以内
            grad_region = frame[:, warning_line_x:, :].astype(np.float32)
        
        # 绘制红色渐变区域
        red = np.zeros_like(grad_region)
        red[:, :, 2] = 255
        if grad_region.shape[0] > 0 and grad_region.shape[1] > 0:
            if current_side == "left":
                # 红色区域在左侧，渐变从左向右递减
                alpha = np.linspace(0.35, 0.0, grad_region.shape[1], dtype=np.float32)[None, :, None]
            else:
                # 红色区域在右侧，渐变从右向左递减
                alpha = np.linspace(0.35, 0.0, grad_region.shape[1], dtype=np.float32)[None, :, None]
            blended = grad_region * (1 - alpha) + red * alpha
            # 将混合后的区域绘制回原位置
            if current_side == "left":
                # 红色区域在左侧，绘制到左侧
                frame[:, :warning_line_x, :] = blended.astype(np.uint8)
            else:
                # 红色区域在右侧，绘制到右侧
                frame[:, warning_line_x:, :] = blended.astype(np.uint8)
        
        # 绘制红色垂直分割线
        for y in range(0, h, 30):
            cv2.line(frame, (warning_line_x, y), (warning_line_x, min(y + 18, h - 1)), (0, 0, 255), 4)
        
        # 绘制黄色垂直分割线
        for y in range(0, h, 30):
            cv2.line(frame, (yellow_line_x, y), (yellow_line_x, min(y + 18, h - 1)), (0, 255, 255), 3)
        
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
    
    def get_display_params(self, track_id, class_name, ttc, x1, x2, vx, warning_line_x, yellow_line_x):
        """获取显示参数（中文）
        
        Args:
            track_id: 跟踪 ID
            class_name: 类别名称
            ttc: 碰撞时间
            x1: 目标左边界 x 坐标
            x2: 目标右边界 x 坐标
            vx: 横向速度（像素/帧，正值向右）
            warning_line_x: 红线位置
            yellow_line_x: 黄线位置
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
        
        # 使用camera_side设置
        current_side = self.camera_side
        
        # 判断目标所在区域（使用边界框判断）
        in_red_zone = False
        in_yellow_zone = False
        partially_in_yellow = False
        
        # current_side 直接表示红色区域应该在的位置
        if current_side == "left":
            # 红色区域在左侧（靠近车身）
            # 整个目标在红色区域内
            in_red_zone = x2 <= warning_line_x
            # 整个目标在黄色区域内
            in_yellow_zone = x1 >= warning_line_x and x2 < yellow_line_x
            # 目标有一部分在黄色区域内（跨越黄线或红线）
            partially_in_yellow = (x1 < yellow_line_x and x2 > yellow_line_x) or \
                                  (x1 <= warning_line_x and x2 > warning_line_x)
        else:
            # 红色区域在右侧（靠近车身）
            # 整个目标在红色区域内
            in_red_zone = x1 >= warning_line_x
            # 整个目标在黄色区域内
            in_yellow_zone = x1 > yellow_line_x and x2 < warning_line_x
            # 目标有一部分在黄色区域内（跨越黄线或红线）
            partially_in_yellow = (x1 < yellow_line_x and x2 > yellow_line_x) or \
                                  (x1 < warning_line_x and x2 >= warning_line_x)
        
        # 计算相对于红色区域中心的速度
        # 正值表示靠近红色区域，负值表示远离红色区域
        if current_side == "left":
            # 红色区域在左侧，vx < 0 表示向左移动（靠近红色区域）
            approaching_speed = -vx  # 正值表示靠近
        else:
            # 红色区域在右侧，vx > 0 表示向右移动（靠近红色区域）
            approaching_speed = vx  # 正值表示靠近
        
        # 报警逻辑
        if in_red_zone:
            # 红色区域内直接报警
            color = (0, 0, 255)
            label = f"危险！{chinese_class} 进入报警区域"
            thickness = 3
        elif in_yellow_zone or partially_in_yellow:
            # 黄线区域内或部分在黄线区域内，只追踪不报警
            color = (0, 255, 0)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        else:
            # 不在任何报警区域
            color = (0, 255, 0)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        
        return color, label, thickness
