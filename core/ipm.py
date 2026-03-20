import math
from typing import Optional, Tuple


class IPM_Transformer:
    """
    单目逆透视变换工具：将像素点映射到地平面 (X:横向, Y:纵向/前方)，单位米。
    - 默认坐标系：X 右为正，Y 前为正，Z 竖直向上为正；相机位于 (0, H, 0)。
    - 相机绕 X 轴俯仰 (pitch) 向下为正；绕 Y 轴偏航 (yaw) 左转为正。
    """

    def __init__(
        self,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        cam_height: float = 1.4,
        pitch_deg: float = 5.0,
        yaw_deg: float = 0.0,
        fov_deg: float = 90.0,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.cam_height = cam_height
        self.pitch = math.radians(pitch_deg)
        self.yaw = math.radians(yaw_deg)
        self.fov_deg = fov_deg
        self.frame_w = None
        self.frame_h = None

    def set_frame(self, w: int, h: int):
        self.frame_w, self.frame_h = w, h
        if self.cx is None:
            self.cx = w * 0.5
        if self.cy is None:
            self.cy = h * 0.5
        if self.fx is None or self.fy is None:
            # 由水平 FOV 估算焦距
            f = (w * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)
            self.fx = self.fx or f
            self.fy = self.fy or f

    def set_pitch_yaw(self, pitch_deg: float, yaw_deg: float = 0.0):
        self.pitch = math.radians(pitch_deg)
        self.yaw = math.radians(yaw_deg)

    def pixel_to_ground(self, u: float, v: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        将像素点 (u,v) 映射到地平面坐标 (X, Y)，单位米。
        返回 None 表示射线未与地面相交（例如指向天空）。
        """
        h, w = frame_shape
        if self.frame_w != w or self.frame_h != h:
            self.set_frame(w, h)

        if any(x is None for x in [self.fx, self.fy, self.cx, self.cy]):
            return None

        # 像素 -> 相机坐标归一化方向
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        z_cam = 1.0

        # 俯仰、偏航旋转
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cyaw, syaw = math.cos(self.yaw), math.sin(self.yaw)

        # 先绕 X（俯仰），再绕 Y（偏航）
        # Rx
        y1 = cp * y_cam - sp * z_cam
        z1 = sp * y_cam + cp * z_cam
        x1 = x_cam
        # Ry
        x2 = cyaw * x1 + syaw * z1
        z2 = -syaw * x1 + cyaw * z1
        y2 = y1

        # 直线 P = (0,H,0) + t * d，与地面 y=0 相交
        if y2 >= -1e-6:
            return None  # 指向地面以上
        t = self.cam_height / -y2
        X = x2 * t
        Y = z2 * t
        return X, Y

    def ground_to_pixel(self, X: float, Y: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        将地平面坐标 (X, Y) 映射到像素坐标 (u, v)。
        返回 None 表示无法映射。
        """
        h, w = frame_shape
        if self.frame_w != w or self.frame_h != h:
            self.set_frame(w, h)

        if any(x is None for x in [self.fx, self.fy, self.cx, self.cy]):
            return None

        # 地平面点到相机坐标的射线方向
        # 相机位置为 (0, self.cam_height, 0)，地平面点为 (X, 0, Y)
        x_dir = X
        y_dir = -self.cam_height  # 相机到地面的垂直距离
        z_dir = Y

        # 偏航和俯仰的逆旋转
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cyaw, syaw = math.cos(self.yaw), math.sin(self.yaw)

        # 先逆绕 Y（偏航）
        x1 = cyaw * x_dir - syaw * z_dir
        z1 = syaw * x_dir + cyaw * z_dir
        y1 = y_dir
        # 再逆绕 X（俯仰）
        y2 = cp * y1 + sp * z1
        z2 = -sp * y1 + cp * z1
        x2 = x1

        # 归一化相机坐标
        if z2 <= 1e-6:
            return None  # 射线与相机平面平行或指向后方

        x_cam = x2 / z2
        y_cam = y2 / z2

        # 相机坐标到像素坐标
        u = self.fx * x_cam + self.cx
        v = self.fy * y_cam + self.cy

        return u, v
