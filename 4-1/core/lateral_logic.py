"""
侧向碰撞检测模块 (Lateral Collision Detection Module)

核心思想：
- 前向碰撞：基于目标宽度变化率 (dw/dt) 计算 TTC
- 侧向碰撞：待重构

坐标系定义：
- X 轴：横向距离，车身侧边为 X=0（正值=右侧，负值=左侧），单位：米
- Y 轴：纵向距离（正值=前方，负值=后方），单位：米
- 相机位于原点 (0, 0)
"""

import math
from collections import deque
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging


@dataclass
class LateralConfig:
    """侧向碰撞检测配置参数"""
    
    # 监控范围参数
    monitor_y_min: float = -2.0      # 纵向监控范围最小值（米），负值表示车尾后方
    monitor_y_max: float = 5.0       # 纵向监控范围最大值（米），正值表示车头前方
    monitor_x_max: float = 6.0       # 横向监控范围（单侧，米）
    
    # BSD（盲区检测）参数
    bsd_distance: float = 3.5        # 盲区范围（米）
    bsd_warning_distance: float = 1.5 # 盲区警告距离（米）
    
    # TTL（侧向侵入时间）参数
    ttl_danger_threshold: float = 1.0 # TTL 危险阈值（秒）
    ttl_warning_threshold: float = 2.0 # TTL 警告阈值（秒）
    
    # 速度阈值参数
    min_approach_speed: float = 0.02  # 最小靠近速度（米/秒）
    max_safe_speed: float = 0.1       # 安全速度阈值（米/秒）
    
    # 历史数据参数
    history_length: int = 8           # 历史轨迹长度（帧）
    velocity_smooth_frames: int = 4   # 速度平滑帧数
    
    # EMA 平滑参数
    use_ema: bool = True              # 是否使用 EMA 平滑
    ema_alpha: float = 0.4            # EMA 系数
    
    # 静态目标过滤
    static_threshold: float = 0.3     # 静态判定阈值（米）
    static_frames: int = 8            # 静态判定帧数


@dataclass
class LateralTargetState:
    """侧向目标状态"""
    track_id: int                    # 跟踪 ID
    class_name: str                  # 类别名称
    
    # 位置信息（地面坐标系，单位：米）
    x: float                         # 横向距离（正值=右侧）
    y: float                         # 纵向距离（正值=前方）
    
    # 速度信息（单位：米/秒）
    vx: float                        # 横向速度（正值=向右移动）
    vy: float                        # 纵向速度（正值=向前移动）
    
    # TTC 信息
    ttc_lateral: float               # 侧向碰撞时间（秒）
    
    # 风险等级
    risk_level: int                  # 0=安全, 1=注意, 2=危险
    
    # 辅助信息
    distance_x: float                # 横向距离绝对值
    approach_speed: float            # 靠近速度（绝对值）
    is_approaching: bool             # 是否正在靠近
    is_static: bool                  # 是否静止
    in_monitor_zone: bool            # 是否在监控区域内
    
    # 预警标签
    warning_label: str               # 预警标签文本
    warning_color: Tuple[int, int, int]  # 预警颜色 (BGR)


class LateralCollisionDetector:
    """
    侧向碰撞检测器 - 待重构
    
    核心逻辑：
    1. BSD（盲区检测）：待实现
    2. TTL（侧向侵入时间）：待实现
    3. 支持左右视角坐标镜像
    
    使用示例：
        detector = LateralCollisionDetector(fps=30.0, camera_side="left")
        
        # 每帧更新
        for track_id, world_pos, class_name in targets:
            state = detector.update(track_id, world_pos, class_name)
            if state and state.risk_level > 0:
                print(f"目标 {track_id}: {state.warning_label}")
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        config: Optional[LateralConfig] = None,
        camera_side: str = "left",  # "left" 或 "right"
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化侧向碰撞检测器
        
        Args:
            fps: 视频帧率
            config: 配置参数，若为 None 则使用默认配置
            camera_side: 相机位置，"left" 或 "right"
            logger: 日志记录器
        """
        self.fps = fps if fps and fps > 0 else 30.0
        self.config = config or LateralConfig()
        self.camera_side = camera_side
        
        # 设置日志
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('LateralCollisionDetector')
            self.logger.setLevel(logging.DEBUG)
        
        # 历史数据存储
        self._x_history: Dict[int, deque] = {}      # 横向位置历史
        self._y_history: Dict[int, deque] = {}      # 纵向位置历史
        self._vx_history: Dict[int, deque] = {}     # 横向速度历史
        self._vy_history: Dict[int, deque] = {}     # 纵向速度历史
        
        # EMA 平滑值
        self._x_ema: Dict[int, float] = {}
        self._vx_ema: Dict[int, float] = {}
        
        # 目标类别记录
        self._class_names: Dict[int, str] = {}
        
        # 目标出现帧数
        self._seen_frames: Dict[int, int] = {}
        
        # 静态目标标记
        self._static_targets: set = set()
        
        self.logger.info(f"侧向碰撞检测器初始化完成，FPS={self.fps}，相机位置={camera_side}")
    
    def update(
        self,
        track_id: int,
        world_pos: Tuple[float, float],
        class_name: str = "unknown",
        self_speed: float = 0.0
    ) -> Optional[LateralTargetState]:
        """
        更新目标状态并计算侧向碰撞风险
        
        Args:
            track_id: 跟踪 ID
            world_pos: 地面坐标 (X, Y)，单位：米
            class_name: 目标类别名称
            self_speed: 本车速度（米/秒），用于相对速度修正
        
        Returns:
            LateralTargetState: 目标状态，若目标不在监控区域或数据不足则返回 None
        """
        if world_pos is None:
            return None
        
        x, y = world_pos
        cfg = self.config
        
        # 记录类别
        self._class_names[track_id] = class_name
        
        # 更新出现帧数
        self._seen_frames[track_id] = self._seen_frames.get(track_id, 0) + 1
        
        # 初始化历史队列
        if track_id not in self._x_history:
            self._x_history[track_id] = deque(maxlen=cfg.history_length)
            self._y_history[track_id] = deque(maxlen=cfg.history_length)
            self._vx_history[track_id] = deque(maxlen=cfg.velocity_smooth_frames)
            self._vy_history[track_id] = deque(maxlen=cfg.velocity_smooth_frames)
            self._x_ema[track_id] = x
            self._vx_ema[track_id] = 0.0
        
        # 更新位置历史
        self._x_history[track_id].append(x)
        self._y_history[track_id].append(y)
        
        # EMA 平滑位置
        if cfg.use_ema:
            x_smooth = cfg.ema_alpha * x + (1 - cfg.ema_alpha) * self._x_ema[track_id]
            self._x_ema[track_id] = x_smooth
        else:
            x_smooth = x
        
        # 计算速度
        vx, vy = self._calculate_velocity(track_id, self_speed)
        
        # 更新速度历史
        self._vx_history[track_id].append(vx)
        self._vy_history[track_id].append(vy)
        
        # EMA 平滑速度
        if cfg.use_ema:
            vx_smooth = cfg.ema_alpha * vx + (1 - cfg.ema_alpha) * self._vx_ema[track_id]
            self._vx_ema[track_id] = vx_smooth
        else:
            vx_smooth = vx
        
        # 判断是否在监控区域
        in_monitor_zone = self._is_in_monitor_zone(x_smooth, y)
        
        if not in_monitor_zone:
            return None
        
        # 判断是否静止
        is_static = self._is_static(track_id)
        if is_static:
            self._static_targets.add(track_id)
        else:
            self._static_targets.discard(track_id)
        
        # 计算横向距离（目标到车身的距离）
        distance_x = abs(x_smooth)
        
        # 调试日志：打印 raw_x 和 vx_smooth
        self.logger.debug(f"目标 {track_id} - raw_x: {x:.2f}, x_smooth: {x_smooth:.2f}, vx_smooth: {vx_smooth:.2f}, distance_x: {distance_x:.2f}")
        
        # 判定风险等级（待重构）
        risk_level = 0
        is_approaching = False
        approach_speed = 0.0
        ttl = 99.0
        is_in_bsd = False
        
        # 生成预警标签和颜色
        warning_label, warning_color = self._generate_warning(
            track_id=track_id,
            class_name=class_name,
            distance_x=distance_x,
            approach_speed=approach_speed,
            ttc_lateral=ttl,
            risk_level=risk_level,
            vx=vx_smooth,
            is_in_bsd=is_in_bsd
        )
        
        # 构建状态对象
        state = LateralTargetState(
            track_id=track_id,
            class_name=class_name,
            x=x_smooth,
            y=y,
            vx=vx_smooth,
            vy=vy,
            ttc_lateral=ttl,
            risk_level=risk_level,
            distance_x=distance_x,
            approach_speed=approach_speed,
            is_approaching=is_approaching,
            is_static=is_static,
            in_monitor_zone=in_monitor_zone,
            warning_label=warning_label,
            warning_color=warning_color
        )
        
        return state
    
    def _calculate_velocity(
        self,
        track_id: int,
        self_speed: float = 0.0
    ) -> Tuple[float, float]:
        """
        计算目标速度
        
        Args:
            track_id: 跟踪 ID
            self_speed: 本车速度（米/秒）
        
        Returns:
            (vx, vy): 横向速度和纵向速度（米/秒）
        """
        x_hist = self._x_history[track_id]
        y_hist = self._y_history[track_id]
        
        if len(x_hist) < 2:
            return 0.0, 0.0
        
        # 使用最近两帧计算瞬时速度
        x_curr, x_prev = x_hist[-1], x_hist[-2]
        y_curr, y_prev = y_hist[-1], y_hist[-2]
        
        dt = 1.0 / self.fps
        
        vx = (x_curr - x_prev) / dt
        vy = (y_curr - y_prev) / dt
        
        return vx, vy
    
    def _is_in_monitor_zone(self, x: float, y: float) -> bool:
        """
        判断目标是否在监控区域内
        
        Args:
            x: 横向距离（米）
            y: 纵向距离（米）
        
        Returns:
            是否在监控区域
        """
        cfg = self.config
        
        # 检查纵向范围
        if not (cfg.monitor_y_min <= y <= cfg.monitor_y_max):
            return False
        
        # 检查横向范围
        if abs(x) > cfg.monitor_x_max:
            return False
        
        return True
    
    def _is_static(self, track_id: int) -> bool:
        """
        判断目标是否静止
        
        Args:
            track_id: 跟踪 ID
        
        Returns:
            是否静止
        """
        x_hist = self._x_history.get(track_id, [])
        y_hist = self._y_history.get(track_id, [])
        
        if len(x_hist) < self.config.static_frames:
            return False
        
        recent_x = list(x_hist)[-self.config.static_frames:]
        recent_y = list(y_hist)[-self.config.static_frames:]
        
        x_range = max(recent_x) - min(recent_x)
        y_range = max(recent_y) - min(recent_y)
        
        return (x_range < self.config.static_threshold and 
                y_range < self.config.static_threshold)
    
    def _generate_warning(
        self,
        track_id: int,
        class_name: str,
        distance_x: float,
        approach_speed: float,
        ttc_lateral: float,
        risk_level: int,
        vx: float = None,
        is_in_bsd: bool = False
    ) -> Tuple[str, Tuple[int, int, int]]:
        """
        生成预警标签和颜色
        
        Args:
            track_id: 跟踪 ID
            class_name: 类别名称
            distance_x: 横向距离
            approach_speed: 靠近速度
            ttc_lateral: 侧向 TTC
            risk_level: 风险等级
            vx: 横向速度
            is_in_bsd: 是否在盲区范围内
        
        Returns:
            (预警标签, BGR 颜色)
        """
        # 类别中文名映射
        class_map = {
            "person": "行人",
            "bicycle": "自行车",
            "car": "轿车",
            "motorcycle": "摩托车",
            "truck": "卡车",
            "bus": "公交车"
        }
        chinese_name = class_map.get(class_name, class_name)
        
        # 基础标签信息
        base_info = f"{chinese_name} 横距{distance_x:.1f}m"
        if vx is not None:
            base_info += f" vx={vx:.2f}m/s"
        
        if risk_level == 2:
            label = f"危险！{base_info} TTC{ttc_lateral:.1f}s"
            color = (0, 0, 255)  # 红色
        elif risk_level == 1:
            label = f"注意 {base_info}"
            color = (0, 255, 255)  # 黄色
        else:
            label = base_info
            color = (0, 255, 0)  # 绿色
        
        return label, color
    
    def get_all_targets(self) -> List[int]:
        """获取所有跟踪目标的 ID 列表"""
        return list(self._x_history.keys())
    
    def get_target_state(self, track_id: int) -> Optional[Dict]:
        """
        获取目标的详细状态信息（用于调试）
        
        Args:
            track_id: 跟踪 ID
        
        Returns:
            状态字典
        """
        if track_id not in self._x_history:
            return None
        
        return {
            'track_id': track_id,
            'class_name': self._class_names.get(track_id, 'unknown'),
            'seen_frames': self._seen_frames.get(track_id, 0),
            'x_history': list(self._x_history[track_id]),
            'y_history': list(self._y_history[track_id]),
            'vx_history': list(self._vx_history[track_id]),
            'vy_history': list(self._vy_history[track_id]),
            'x_ema': self._x_ema.get(track_id, 0),
            'vx_ema': self._vx_ema.get(track_id, 0),
            'is_static': track_id in self._static_targets
        }
    
    def remove_target(self, track_id: int):
        """
        移除目标（当目标消失时调用）
        
        Args:
            track_id: 跟踪 ID
        """
        self._x_history.pop(track_id, None)
        self._y_history.pop(track_id, None)
        self._vx_history.pop(track_id, None)
        self._vy_history.pop(track_id, None)
        self._x_ema.pop(track_id, None)
        self._vx_ema.pop(track_id, None)
        self._class_names.pop(track_id, None)
        self._seen_frames.pop(track_id, None)
        self._static_targets.discard(track_id)
    
    def clear_all(self):
        """清空所有目标数据"""
        self._x_history.clear()
        self._y_history.clear()
        self._vx_history.clear()
        self._vy_history.clear()
        self._x_ema.clear()
        self._vx_ema.clear()
        self._class_names.clear()
        self._seen_frames.clear()
        self._static_targets.clear()
    
    def cleanup_stale_targets(self, max_inactive_frames: int = 30):
        """
        清理长时间未更新的目标
        
        Args:
            max_inactive_frames: 最大不活跃帧数
        """
        active_ids = set(self._seen_frames.keys())
        for track_id in list(active_ids):
            if self._seen_frames[track_id] < max_inactive_frames:
                continue
            self.remove_target(track_id)


class LateralIPMConfig:
    """
    侧向相机 IPM 配置
    
    侧向相机的透视畸变与前向相机完全不同，需要单独配置参数。
    假设侧向相机安装在车辆侧面，朝向侧面方向。
    """
    
    def __init__(
        self,
        cam_height: float = 1.35,      # 相机安装高度（米）- 根据照片调整为后视镜高度
        pitch_deg: float = 5.0,        # 俯仰角（度），向下为正 - 向下俯视
        yaw_deg: float = 95.0,         # 偏航角（度），左侧相机≈95°（向后方旋转）
        roll_deg: float = 0.0,         # 滚转角（度），顺时针为正
        fov_deg: float = 85.0,         # 水平视场角（度）
        fx: Optional[float] = None,    # 焦距 X（像素）
        fy: Optional[float] = None,    # 焦距 Y（像素）
    ):
        self.cam_height = cam_height
        self.pitch = math.radians(pitch_deg)
        self.yaw = math.radians(yaw_deg)
        self.roll = math.radians(roll_deg)
        self.fov_deg = fov_deg
        self.fx = fx
        self.fy = fy
        
        # 标记相机方向
        if yaw_deg > 0:
            self.camera_side = "left"   # 左侧相机
        else:
            self.camera_side = "right"  # 右侧相机
    
    def get_ipm_params(self) -> Dict:
        """获取 IPM 初始化参数"""
        return {
            'cam_height': self.cam_height,
            'pitch_deg': math.degrees(self.pitch),
            'yaw_deg': math.degrees(self.yaw),
            'fov_deg': self.fov_deg,
            'fx': self.fx,
            'fy': self.fy
        }


def create_lateral_ipm(
    camera_side: str = "left",
    config: Optional[LateralIPMConfig] = None
):
    """
    创建侧向相机的 IPM 变换器
    
    Args:
        camera_side: 相机位置，"left" 或 "right"
        config: IPM 配置，若为 None 则使用默认配置
    
    Returns:
        IPM_Transformer 实例
    """
    from core.ipm import IPM_Transformer
    
    if config is None:
        # 默认配置 - 根据左侧摄像头调整
        yaw_deg = 95.0 if camera_side == "left" else -95.0  # 调整为 95°（向后方旋转）
        config = LateralIPMConfig(
            cam_height=1.35,      # 后视镜高度
            pitch_deg=5.0,        # 向下俯视
            yaw_deg=yaw_deg,
            fov_deg=85.0          # 水平视场角
        )
    
    params = config.get_ipm_params()
    ipm = IPM_Transformer(
        cam_height=params['cam_height'],
        pitch_deg=params['pitch_deg'],
        yaw_deg=params['yaw_deg'],
        fov_deg=params['fov_deg']
    )
    
    return ipm
