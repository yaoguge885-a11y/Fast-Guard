"""
侧面视角 IPM 模块 (Side View IPM Module)

专门为侧向相机设计的逆透视变换模块，支持左右视角切换。
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
        self.vehicle_half_width = 0.95  # 车身半宽（米）
        
        # 根据相机侧设置 IPM 参数
        # 注意：pitch 为负值表示相机向下看，对于侧面视角需要更大的向下角度
        # 以覆盖近距离的车辆（避免 IPM_FAIL）
        if camera_side == "left":
            # 左相机：朝向左侧
            self.ipm = IPM_Transformer(
                cam_height=1.2,      # 侧后视镜高度
                pitch_deg=-15.0,     # 向下15度，确保覆盖近距离区域
                yaw_deg=90.0,        # 朝向左侧
                fov_deg=100.0        # 广角
            )
        else:
            # 右相机：朝向右侧
            self.ipm = IPM_Transformer(
                cam_height=1.2,
                pitch_deg=-15.0,     # 向下15度
                yaw_deg=-90.0,       # 朝向右侧
                fov_deg=100.0
            )
    
    def set_camera_side(self, camera_side: str):
        """切换相机侧"""
        if camera_side != self.camera_side:
            self.camera_side = camera_side
            # 重新初始化 IPM（使用相同的 pitch 参数）
            if camera_side == "left":
                self.ipm = IPM_Transformer(
                    cam_height=1.2,
                    pitch_deg=-15.0,
                    yaw_deg=90.0,
                    fov_deg=100.0
                )
            else:
                self.ipm = IPM_Transformer(
                    cam_height=1.2,
                    pitch_deg=-15.0,
                    yaw_deg=-90.0,
                    fov_deg=100.0
                )
    
    def pixel_to_ground(self, u: float, v: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        像素坐标转地面坐标
        
        Args:
            u, v: 像素坐标
            frame_shape: (h, w)
        
        Returns:
            (X, Y): X是到车身边缘的距离，Y是纵向距离
        """
        result = self.ipm.pixel_to_ground(u, v, frame_shape)
        if result is None:
            return None
        
        x_raw, y = result
        
        # 坐标转换：让 X=0 代表车身边缘
        # 对于左相机，x_raw 为负表示在左侧
        # 对于右相机，x_raw 为正表示在右侧
        if self.camera_side == "left":
            # 左相机：目标在左侧，x_raw < 0
            # 到左侧车身边缘的距离 = |x_raw| - vehicle_half_width
            x = abs(x_raw) - self.vehicle_half_width
        else:
            # 右相机：目标在右侧，x_raw > 0
            # 到右侧车身边缘的距离 = x_raw - vehicle_half_width
            x = x_raw - self.vehicle_half_width
        
        # 确保距离不为负
        x = max(0.0, x)
        
        return x, y
    
    def get_side_corner_point(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        """
        获取检测框靠近车身一侧的底角点
        
        Args:
            x1, y1: 左上角
            x2, y2: 右下角
        
        Returns:
            (u, v): 像素坐标
        """
        # 使用检测框的底部y坐标
        bottom_y = max(y1, y2)
        
        if self.camera_side == "left":
            # 左相机：车身在右侧，取右侧底角（x2）
            return x2, bottom_y
        else:
            # 右相机：车身在左侧，取左侧底角（x1）
            return x1, bottom_y


# 全局实例（便于切换）
_side_ipm_instance: Optional[SideIPM] = None


def get_side_ipm(camera_side: str = "left") -> SideIPM:
    """
    获取侧面 IPM 实例（单例模式）
    
    Args:
        camera_side: "left" 或 "right"
    
    Returns:
        SideIPM 实例
    """
    global _side_ipm_instance
    if _side_ipm_instance is None:
        _side_ipm_instance = SideIPM(camera_side)
    else:
        _side_ipm_instance.set_camera_side(camera_side)
    return _side_ipm_instance


def switch_side_ipm(camera_side: str) -> SideIPM:
    """
    切换侧面 IPM 的相机侧
    
    Args:
        camera_side: "left" 或 "right"
    
    Returns:
        SideIPM 实例
    """
    return get_side_ipm(camera_side)
