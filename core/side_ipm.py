"""
侧面视角 IPM 模块 (Side View IPM Module)

专门为侧向相机设计的逆透视变换模块，支持左右视角切换。
相机安装位置：后视镜（高度约 1.35m）
"""

import math
from typing import Optional, Tuple
from core.ipm import IPM_Transformer


class SideIPM:
    """
    侧面视角 IPM 变换器

    坐标系定义：
    - X: 横向距离（米），X=0 是车身边缘，X>0 是远离车身
    - Y: 纵向距离（米），Y>0 是前方，Y<0 是后方
    """

    def __init__(self, camera_side: str = "left"):
        """
        初始化侧面 IPM

        Args:
            camera_side: "left" 或 "right"
        """
        self.camera_side = camera_side
        self.vehicle_half_width = 1.0  # 车身半宽（米）

        # 根据相机侧设置 IPM 参数
        # pitch_deg > 0 = 向下看（IPM 约定）
        # 后视镜高度 1.35m，向下看 12° 覆盖近距离区域
        # 修正：盲区后视摄像头一般不仅朝侧面，同时也朝后倾斜（例如 165度）
        if camera_side == "left":
            self.ipm = IPM_Transformer(
                cam_height=1.35,
                pitch_deg=12.0,
                yaw_deg=165.0,       # 朝左后方
                fov_deg=90.0
            )
        else:
            self.ipm = IPM_Transformer(
                cam_height=1.35,
                pitch_deg=12.0,
                yaw_deg=-165.0,      # 朝右后方
                fov_deg=90.0
            )

    def set_camera_side(self, camera_side: str):
        """切换相机侧"""
        if camera_side != self.camera_side:
            self.camera_side = camera_side
            yaw = 165.0 if camera_side == "left" else -165.0
            self.ipm = IPM_Transformer(
                cam_height=1.35,
                pitch_deg=12.0,
                yaw_deg=yaw,
                fov_deg=90.0
            )

    def pixel_to_ground(self, u: float, v: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        像素坐标转地面坐标

        Returns:
            (X, Y): X 是到车身边缘的距离（米），Y 是纵向距离（米）
        """
        result = self.ipm.pixel_to_ground(u, v, frame_shape)
        if result is None:
            return None

        x_raw, y = result

        # 坐标转换：让 X=0 代表车身边缘
        if self.camera_side == "left":
            x = abs(x_raw) - self.vehicle_half_width
        else:
            x = x_raw - self.vehicle_half_width

        x = max(0.0, x)
        return x, y

    def ground_to_pixel(self, x_body: float, y: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        地面坐标（车身坐标系）转像素坐标

        Args:
            x_body: 到车身边缘的距离（米），X=0 是车身边缘
            y: 纵向距离（米），Y>0 是前方
            frame_shape: (h, w)

        Returns:
            (u, v): 像素坐标，或 None
        """
        # 反向转换：从车身边缘距离 → 原始地面坐标
        if self.camera_side == "left":
            x_raw = -(x_body + self.vehicle_half_width)
        else:
            x_raw = x_body + self.vehicle_half_width

        return self.ipm.ground_to_pixel(x_raw, y, frame_shape)

    def get_wall_pixel_x(self, wall_distance: float, frame_shape: Tuple[int, int]) -> Optional[int]:
        """
        计算物理壁垒在画面中的平均 X 像素位置

        Args:
            wall_distance: 壁垒到车身边缘的距离（米）
            frame_shape: (h, w)

        Returns:
            int: 壁垒的 X 像素坐标，或 None
        """
        h, w = frame_shape
        # 在不同 Y 值上采样，由于相机看后方，应该采样负数的 Y (车后)
        sample_ys = [-1.0, -2.0, -4.0, -6.0, -8.0, -10.0]
        pixel_xs = []
        for yv in sample_ys:
            result = self.ground_to_pixel(wall_distance, yv, frame_shape)
            if result is not None:
                px, py = result
                if 0 <= px < w and 0 <= py < h:
                    pixel_xs.append(px)

        if not pixel_xs:
            return None

        return int(sorted(pixel_xs)[len(pixel_xs) // 2])

    def get_side_corner_point(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        """
        获取检测框靠近车身一侧的底角点

        Returns:
            (u, v): 像素坐标
        """
        bottom_y = max(y1, y2)
        if self.camera_side == "left":
            return x2, bottom_y   # 左相机：车身在右侧，取右底角
        else:
            return x1, bottom_y   # 右相机：车身在左侧，取左底角


# ---- 全局单例 ----
_side_ipm_instance: Optional[SideIPM] = None


def get_side_ipm(camera_side: str = "left") -> SideIPM:
    """获取侧面 IPM 实例（单例模式）"""
    global _side_ipm_instance
    if _side_ipm_instance is None:
        _side_ipm_instance = SideIPM(camera_side)
    else:
        _side_ipm_instance.set_camera_side(camera_side)
    return _side_ipm_instance


def switch_side_ipm(camera_side: str) -> SideIPM:
    """切换侧面 IPM 的相机侧"""
    return get_side_ipm(camera_side)
