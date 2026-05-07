"""
侧面碰撞检测器模块 (Side Collision Detector)

将侧面视角的碰撞检测算法独立出来，实现与正面视角完全分离的模块化设计。
核心算法：BSD（盲区检测）和TTL（侧向侵入时间）
"""

from collections import deque
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import math

from core.reasoning_logger import ReasoningLogger


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
    
    # TTL 信息
    ttl_lateral: float               # 侧向碰撞时间（秒）
    
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


@dataclass
class SideDetectorConfig:
    """侧面碰撞检测配置参数"""
    
    # 监控范围参数
    monitor_y_min: float = -2.0      # 纵向监控范围最小值（米），负值表示车尾后方
    monitor_y_max: float = 5.0       # 纵向监控范围最大值（米），正值表示车头前方
    monitor_x_max: float = 6.0       # 横向监控范围（单侧，米）
    
    # BSD（盲区检测）参数
    bsd_distance: float = 5.0        # 盲区范围（米），增加到5米
    bsd_warning_distance: float = 2.0 # 盲区警告距离（米），增加到2米
    
    # TTL（侧向侵入时间）参数
    ttl_danger_threshold: float = 1.3 # TTL 危险阈值（秒）
    ttl_warning_threshold: float = 2.5 # TTL 警告阈值（秒）
    
    # 速度阈值参数
    min_approach_speed: float = 0.2   # 最小靠近速度（米/秒）
    max_safe_speed: float = 0.1       # 安全速度阈值（米/秒）

    
    # 历史数据参数
    history_length: int = 8           # 历史轨迹长度（帧）
    velocity_smooth_frames: int = 4   # 速度平滑帧数
    
    # EMA 平滑参数
    use_ema: bool = True              # 是否使用 EMA 平滑
    ema_alpha: float = 0.4            # EMA 系数
    
    # 方向一致性门控
    approach_consistency_frames: int = 4   # 方向一致性窗口
    approach_consistency_ratio: float = 0.5 # 一致性比例阈值

    
    # 静态目标过滤

    static_threshold: float = 0.3     # 静态判定阈值（米）
    static_frames: int = 8            # 静态判定帧数
    
    # 报警连续帧阈值
    min_alarm_frames: int = 3         # 连续报警帧数阈值，只有连续满足报警条件达到此帧数才输出报警
    
    # 车体偏移补偿
    body_offset: float = 1.0         # 车体宽度偏移（米），让 X=0 代表车身边缘而非镜头中心线
    
    # 极近距离参数
    critical_distance: float = 1.2  # 极近距离阈值（米），低于此距离无视速度判定
    
    # 贴身车辆检测参数
    immediate_alarm_distance: float = 1.5  # 小于此距离立即报警，无视连续帧阈值
    monitor_y_min_close: float = -5.0      # 贴身车辆的Y轴监控范围（更宽松）
    monitor_y_max_close: float = 10.0

    # 风险判定距离阈值（可配置，替代原硬编码常量）
    risk_critical_distance: float = 0.8    # 极近区阈值（米），低于此距离无条件红警
    risk_near_distance: float = 2.0        # 近区阈值（米）
    risk_mid_distance: float = 4.0         # 中区阈值（米）
    fast_retreat_threshold: float = -1.0   # 快速后撤纵向速度阈值（m/s）
    anchor_zone_expansion: float = 1.2     # 锚点侧检测范围扩展倍数
    vy_dead_zone: float = 0.5             # 纵向速度死区（m/s），低于此值忽略纵向判断
    velocity_clamp: float = 10.0           # 速度回归输出最大绝对值（m/s）


class SideCollisionDetector:
    """
    侧面碰撞检测器，基于BSD（盲区检测）和TTL（侧向侵入时间）算法
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        config: Optional[SideDetectorConfig] = None,
        camera_side: str = "left",  # "left" 或 "right"
        logger: Optional[ReasoningLogger] = None
    ):
        """
        初始化侧面碰撞检测器
        
        Args:
            fps: 视频帧率
            config: 配置参数，若为 None 则使用默认配置
            camera_side: 相机位置，"left" 或 "right"
            logger: ReasoningLogger 实例，若为 None 则创建新实例
        """
        self.fps = fps if fps and fps > 0 else 30.0
        self.config = config or SideDetectorConfig()
        self.camera_side = camera_side
        self.logger = logger or ReasoningLogger()
        
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
        
        # 报警连续帧计数器
        self._alarm_counters: Dict[int, int] = {}
        
        # 距离变化历史（用于判定 is_approaching）
        self._distance_history: Dict[int, deque] = {}
        
        # 横向靠近方向一致性历史
        self._approach_dir_history: Dict[int, deque] = {}
        
        # 车体偏移补偿
        self._body_offset = self.config.body_offset

        # 锚点联动（由 ViewClassifier 实时更新，None 表示未确定）
        self._anchor_side: Optional[str] = None

        # 轨迹生命周期管理（基于帧计数器，修复原 cleanup_stale_targets 逻辑错误）
        self._frame_counter: int = 0
        self._last_seen: Dict[int, int] = {}

    
    def set_anchor_side(self, anchor: Optional[str]) -> None:
        """
        联动 ViewClassifier 的侧向锚点，动态调整优先检测区域。

        Args:
            anchor: "left"、"right" 或 None
        """
        self._anchor_side = anchor

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

        # 轨迹生命周期：记录当前帧编号
        self._frame_counter += 1
        self._last_seen[track_id] = self._frame_counter

        x, y = world_pos
        cfg = self.config

        # 记录类别
        self._class_names[track_id] = class_name

        # 更新出现帧数
        self._seen_frames[track_id] = self._seen_frames.get(track_id, 0) + 1
        
        # 最少观察 3 帧才参与风险判定（减少闪现误报）
        min_observe_frames = 3
        
        # 初始化历史队列
        if track_id not in self._x_history:
            self._x_history[track_id] = deque(maxlen=cfg.history_length)
            self._y_history[track_id] = deque(maxlen=cfg.history_length)
            self._vx_history[track_id] = deque(maxlen=cfg.velocity_smooth_frames)
            self._vy_history[track_id] = deque(maxlen=cfg.velocity_smooth_frames)
            self._distance_history[track_id] = deque(maxlen=cfg.history_length)
            self._approach_dir_history[track_id] = deque(maxlen=cfg.approach_consistency_frames)
            self._x_ema[track_id] = x
            self._vx_ema[track_id] = 0.0

        
        # 更新位置历史
        self._x_history[track_id].append(x)
        self._y_history[track_id].append(y)
        
        # 计算当前距离并更新距离历史（考虑车体偏移补偿）
        # IPM 输出的是相对于镜头中心线的距离，需要减去 body_offset 才能得到相对于车身边缘的距离
        distance_x = abs(x) - self._body_offset
        distance_x = max(0.0, distance_x)  # 确保距离不为负
        self._distance_history[track_id].append(distance_x)
        
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
        
        # 记录横向靠近方向一致性（正值表示靠近）
        x_raw = self._x_history[track_id][-1]
        approaching_speed = (-vx_smooth) if x_raw > 0 else vx_smooth
        if approaching_speed > 1e-6:
            self._approach_dir_history[track_id].append(1)
        elif approaching_speed < -1e-6:
            self._approach_dir_history[track_id].append(-1)
        else:
            self._approach_dir_history[track_id].append(0)
        
        # 判断是否在监控区域（传入distance_x用于贴身车辆检测）
        in_monitor_zone = self._is_in_monitor_zone(x_smooth, y, distance_x)

        
        if not in_monitor_zone:
            return None
        
        # 判断是否静止
        is_static = self._is_static(track_id)
        if is_static:
            self._static_targets.add(track_id)
            # 记录静态目标过滤日志
            self.logger.add_log(f"[ID:{track_id}] 判定为静止目标，过滤")
        else:
            self._static_targets.discard(track_id)
        
        # 判断是否正在靠近（只要距离连续减少就判定为靠近）
        is_approaching = self._is_approaching(track_id)
        approach_speed = abs(vx_smooth) if is_approaching else 0.0
        
        # 速度过滤日志
        if abs(vx_smooth) < self.config.max_safe_speed:
            self.logger.add_log(f"[ID:{track_id}] 横向速度 {abs(vx_smooth):.2f}m/s < 阈值，判定为安全平行")
        
        # 计算 TTL（侧向侵入时间）
        ttl = self._calculate_ttl(distance_x, vx_smooth, is_approaching)
        
        # BSD（盲区检测）
        is_in_bsd = distance_x < cfg.bsd_distance
        
        # 判定风险等级
        raw_risk_level = self._calculate_risk_level(
            distance_x, ttl, is_approaching, approach_speed, is_static, is_in_bsd, vy
        )
        
        # 记录风险判定日志
        if raw_risk_level > 0:
            self.logger.add_log(f"[ID:{track_id}] 原始风险等级: {raw_risk_level}, TTL: {ttl:.2f}s, 距离: {distance_x:.1f}m")
        
        # 报警连续帧计数逻辑
        # 紧急情况（risk_level == 2）跳过计数器，立即输出报警
        # 只有 temp_risk == 1 时才需要连续帧过滤，防止误报
        if raw_risk_level == 2:
            # 极度危险情况，立即报警，不等待连续帧计数
            risk_level = 2
            # 保持计数器累积，但不依赖它
            self._alarm_counters[track_id] = self._alarm_counters.get(track_id, 0) + 1
            # 记录危险报警日志
            self.logger.add_log(f"[ID:{track_id}] TTL: {ttl:.2f}s < {cfg.ttl_danger_threshold}s，触发二级预警")
        elif raw_risk_level > 0:
            # 满足报警条件（黄色预警），计数器加1
            current_count = self._alarm_counters.get(track_id, 0) + 1
            self._alarm_counters[track_id] = current_count
            
            # 记录连续帧计数日志
            self.logger.add_log(f"[ID:{track_id}] 检测到侵入趋势，计数器: {current_count}/{self.config.min_alarm_frames}")
            
            # 最少观察帧数过滤
            if self._seen_frames.get(track_id, 0) < min_observe_frames:
                risk_level = 0
            # 贴身车辆（distance_x < immediate_alarm_distance）立即报警
            elif distance_x < cfg.immediate_alarm_distance:
                risk_level = raw_risk_level
                self.logger.add_log(f"[ID:{track_id}] 距离 {distance_x:.1f}m < 阈值，立即报警")
            # 只有连续帧数达到阈值才输出真正的报警
            elif current_count >= self.config.min_alarm_frames:
                risk_level = raw_risk_level
                self.logger.add_log(f"[ID:{track_id}] 连续 {current_count} 帧确认，触发预警")
            else:
                risk_level = 0
        else:
            # 不满足报警条件，立即清零
            self._alarm_counters[track_id] = 0
            risk_level = 0
        
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
            ttl_lateral=ttl,
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
        """问题5修复：使用最近 4 帧线性回归替代相邻帧差分，抑制 IPM 抖动引入的速度尖峰"""
        x_hist = self._x_history[track_id]
        y_hist = self._y_history[track_id]

        if len(x_hist) < 2:
            return 0.0, 0.0

        dt = 1.0 / self.fps

        # 当有 4 帧以上时，用最小二乘法计算速度
        if len(x_hist) >= 4:
            n = min(4, len(x_hist))
            xs = list(x_hist)[-n:]
            ys_list = list(y_hist)[-n:]
            ts = [i * dt for i in range(n)]
            mean_t = sum(ts) / n
            mean_x = sum(xs) / n
            mean_y = sum(ys_list) / n
            denom = sum((t - mean_t) ** 2 for t in ts)
            if denom > 1e-9:
                vx = sum((t - mean_t) * (x - mean_x) for t, x in zip(ts, xs)) / denom
                vy = sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys_list)) / denom
            else:
                vx = (xs[-1] - xs[-2]) / dt
                vy = (ys_list[-1] - ys_list[-2]) / dt
        else:
            # 少于 4 帧时用相邻帧差分
            x_curr, x_prev = x_hist[-1], x_hist[-2]
            y_curr, y_prev = y_hist[-1], y_hist[-2]
            vx = (x_curr - x_prev) / dt
            vy = (y_curr - y_prev) / dt

        # 钳制速度输出，防止小样本回归噪声放大
        clamp = self.config.velocity_clamp
        vx = max(-clamp, min(clamp, vx))
        vy = max(-clamp, min(clamp, vy))

        return vx, vy
    
    def _is_in_monitor_zone(self, x: float, y: float, distance_x: float = None) -> bool:
        """
        判断目标是否在监控区域内
        
        Args:
            x: 横向距离（米，原始坐标）
            y: 纵向距离（米）
            distance_x: 到车身边缘的距离（米），如果提供则用于贴身车辆判断
        
        Returns:
            是否在监控区域
        """
        cfg = self.config

        # 对于贴身车辆（distance_x < immediate_alarm_distance），使用更宽松的Y轴范围
        if distance_x is not None and distance_x < cfg.immediate_alarm_distance:
            y_min = cfg.monitor_y_min_close
            y_max = cfg.monitor_y_max_close
        else:
            y_min = cfg.monitor_y_min
            y_max = cfg.monitor_y_max

        # 检查纵向范围
        if not (y_min <= y <= y_max):
            return False

        # 盲区补偿：锚点侧横向检测范围扩展（加强对本车侧的感知）
        if self._anchor_side == "left" and x < 0:
            x_max = cfg.monitor_x_max * cfg.anchor_zone_expansion
        elif self._anchor_side == "right" and x > 0:
            x_max = cfg.monitor_x_max * cfg.anchor_zone_expansion
        else:
            x_max = cfg.monitor_x_max

        if abs(x) > x_max:
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
    
    def _is_approaching(self, track_id: int) -> bool:
        """
        判断目标是否正在靠近（EMA + 方向一致性）
        同时考虑纵向速度，对于后方目标，只有当它向前行驶时才认为在靠近
        
        Args:
            track_id: 跟踪 ID

        Returns:
            是否正在靠近
        """
        cfg = self.config
        vx_ema = self._vx_ema.get(track_id, 0.0)

        # 使用原始 x_history 中的值来判断目标在左侧还是右侧（不受 body_offset 影响）
        x_hist = self._x_history.get(track_id, deque())
        y_hist = self._y_history.get(track_id, deque())
        if len(x_hist) < 2 or len(y_hist) < 2:
            return False

        x_raw = x_hist[-1]  # 使用原始 x 坐标判断方向
        y_raw = y_hist[-1]  # 纵向位置

        # 获取纵向速度 vy
        vy_hist = self._vy_history.get(track_id, deque())
        vy_ema = 0.0
        if len(vy_hist) > 0:
            vy_ema = vy_hist[-1]  # 使用最新速度

        # 纵向速度死区：当 |vy| 很小时忽略纵向方向判断（避免噪声干扰）
        dz = self.config.vy_dead_zone
        # 对于后方目标（y < 0），需要它正在向前行驶（vy > dz）才算在靠近
        if y_raw < 0 and vy_ema < -dz:
            return False

        # 对于前方目标（y > 0），需要它正在向后行驶（vy < -dz）才算在靠近
        if y_raw > 0 and vy_ema > dz:
            return False

        # 计算横向靠近速度（正值表示靠近）
        if x_raw > 0:
            approaching_speed = -vx_ema
        else:
            approaching_speed = vx_ema

        # 速度幅度门控
        if approaching_speed <= cfg.min_approach_speed:
            return False

        # 方向一致性门控（最近窗口内多数为“靠近”）
        dir_hist = self._approach_dir_history.get(track_id, deque())
        if len(dir_hist) == 0:
            return False
        window = list(dir_hist)[-cfg.approach_consistency_frames:]
        positives = sum(1 for v in window if v > 0)
        required = max(2, int(math.ceil(len(window) * cfg.approach_consistency_ratio)))
        if positives < required:
            return False

        return True

    
    def _calculate_ttl(self, distance_x: float, vx: float, is_approaching: bool) -> float:
        """
        计算侧向侵入时间 (TTL)
        
        Args:
            distance_x: 横向距离（米）
            vx: 横向速度（米/秒）
            is_approaching: 是否正在靠近
        
        Returns:
            TTL（秒）
        """
        if not is_approaching or abs(vx) < max(self.config.min_approach_speed, 1e-3):
            return 99.0

        # TTL = 距离 / 靠近速度
        ttl = distance_x / abs(vx)
        return max(0.0, ttl)
    
    def _calculate_risk_level(
        self,
        distance_x: float,
        ttl: float,
        is_approaching: bool,
        approach_speed: float,
        is_static: bool,
        is_in_bsd: bool,
        vy: float = 0.0
    ) -> int:
        """
        问题4与问题7修复：基于距离×靠近速度组合判定，并过滤超车场景误报
        """
        cfg = self.config
        # 静止目标全面过滤
        if is_static:
            return 0

        # ========== 极近区（X < risk_critical_distance）：无条件红警，已物理越线 ==========
        if distance_x < cfg.risk_critical_distance:
            return 2

        # ========== 近区（risk_critical ≤ X < risk_near）：需要有明确的靠近趋势才报警 ==========
        if distance_x < cfg.risk_near_distance:
            # 超车场景过滤 — 目标在快速向后掉队
            if vy < cfg.fast_retreat_threshold and not is_approaching:
                return 0
            # 有明确的横向靠近趋势才控车
            if is_approaching and approach_speed > cfg.min_approach_speed:
                return 2 if ttl < cfg.ttl_warning_threshold else 1
            # 在极近区但未明显靠近，保守性黄色预警
            return 1

        # ========== 中区（risk_near ≤ X < risk_mid）：有靠近趋势才黄警 ==========
        if distance_x < cfg.risk_mid_distance:
            if vy < cfg.fast_retreat_threshold and not is_approaching:
                return 0
            if is_approaching and approach_speed > cfg.min_approach_speed * 2:
                return 1
            return 0

        # ========== 安全区（X ≥ risk_mid_distance）==========
        return 0
    
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
            label = f"危险！{base_info} TTL{ttc_lateral:.1f}s"
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
        self._distance_history.pop(track_id, None)
        self._approach_dir_history.pop(track_id, None)
        self._x_ema.pop(track_id, None)
        self._vx_ema.pop(track_id, None)

        self._class_names.pop(track_id, None)
        self._seen_frames.pop(track_id, None)
        self._static_targets.discard(track_id)
        self._alarm_counters.pop(track_id, None)
        self._last_seen.pop(track_id, None)
    
    def clear_all(self):
        """清空所有目标数据"""
        self._x_history.clear()
        self._y_history.clear()
        self._vx_history.clear()
        self._vy_history.clear()
        self._distance_history.clear()
        self._approach_dir_history.clear()
        self._x_ema.clear()
        self._vx_ema.clear()

        self._class_names.clear()
        self._seen_frames.clear()
        self._static_targets.clear()
        self._alarm_counters.clear()
        self._last_seen.clear()
    
    def cleanup_stale_targets(self, max_inactive_frames: int = 30) -> List[int]:
        """
        清理超过 max_inactive_frames 帧未更新的轨迹，防止 ID 积累和内存泄漏。

        修复：原实现使用 _seen_frames（累计观测计数）判断活跃度，会误删新目标。
        现改用 _frame_counter / _last_seen 时间戳对比，与 front_detector 一致。

        Args:
            max_inactive_frames: 允许最大未更新帧数

        Returns:
            已清理的 track_id 列表
        """
        stale = [
            tid for tid, last in self._last_seen.items()
            if self._frame_counter - last > max_inactive_frames
        ]
        for tid in stale:
            self.remove_target(tid)
        return stale
