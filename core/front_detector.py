"""
正面碰撞检测器模块 (Front Collision Detector)

将正面视角的碰撞检测算法独立出来，实现与侧面视角完全分离的模块化设计。
核心算法：基于目标宽度变化率 (dw/dt) 计算 TTC
"""

from collections import deque
import math
from typing import Tuple, Optional, Dict, List


class FrontCollisionDetector:
    """
    正面碰撞检测器，基于目标宽度变化率计算 TTC
    """

    def __init__(self, fps: float = 30.0,
                 ttc_base: float = 1.8, ttc_max: float = 3.5,
                 ttc_min: float = 2.2, ttc_speed_factor: float = 0.06,
                 width_growth_threshold: float = 1.03,
                 vy_gate_start: float = 2.0, vy_gate_end: float = 5.0,
                 auto_cleanup_interval: int = 120):
        """初始化正面碰撞检测器

        Args:
            fps: 视频帧率
            ttc_base: TTC 阈值基础值（秒）
            ttc_max: TTC 阈值上限（秒）
            ttc_min: TTC 阈值下限（秒）
            ttc_speed_factor: 本车速度对 TTC 阈值的缩放因子
            width_growth_threshold: 标准 TTC 最小宽度增长比例（如 1.03 = 3%）
            vy_gate_start: vy 软门控起始速度（px/s），低于此值 TTC 不受影响
            vy_gate_end: vy 软门控终止速度（px/s），高于此值 TTC 完全取消
            auto_cleanup_interval: 自动清理陈旧轨迹的帧间隔
        """
        self.fps = fps if fps and fps > 0 else 30.0
        self.safe_ttc = 99.0
        self.width_history_len = 10

        # TTC 阈值可配置参数
        self.ttc_base = ttc_base
        self.ttc_max = ttc_max
        self.ttc_min = ttc_min
        self.ttc_speed_factor = ttc_speed_factor
        self.width_growth_threshold = width_growth_threshold
        self.vy_gate_start = vy_gate_start
        self.vy_gate_end = vy_gate_end
        self._auto_cleanup_interval = auto_cleanup_interval

        # 历史数据存储
        self.width_history: Dict[int, deque] = {}
        self.width_ema: Dict[int, float] = {}
        self.x_history: Dict[int, deque] = {}
        self.y_history: Dict[int, deque] = {}
        self.ground_dist_history: Dict[int, deque] = {}

        # 轨迹生命周期管理
        self._frame_counter: int = 0          # 全局帧计数
        self._last_seen: Dict[int, int] = {}  # track_id → 最后一次更新的帧编号
    
    def update(
        self,
        track_id: int,
        width: float,
        center_x: float,
        center_y: float,
        bottom_y: float,
        frame_w: int,
        frame_h: int,
        warning_line_y: float,
        area_ratio: float,
        global_vx: float = 0.0,
        global_vy: float = 0.0,
        min_area_ratio: float = 0.015,  # 提高面积阈值 0.005 -> 0.015
        use_ema: bool = True,  # 默认启用EMA
        ema_alpha: float = 0.3,
        distance: Optional[float] = None,
        v_self_mps: float = 0.0,
        t_reaction: float = 1.2,
        d_safe: float = 2.0,
        ipm: Optional[object] = None,
    ) -> Tuple[float, float, float, float, bool, float, bool, int, bool]:
        """
        更新目标状态并计算正面碰撞风险
        
        Args:
            track_id: 跟踪 ID
            width: 检测框宽度（像素）
            center_x: 目标中心 X 坐标
            center_y: 目标中心 Y 坐标
            bottom_y: 检测框底部 Y 坐标
            frame_w: 帧宽度
            frame_h: 帧高度
            warning_line_y: 警示线 Y 坐标
            area_ratio: 检测框面积占比
            global_vx: 全局水平速度（像素/秒）
            global_vy: 全局垂直速度（像素/秒）
            min_area_ratio: 最小面积阈值
            use_ema: 是否使用 EMA 平滑
            ema_alpha: EMA 平滑系数
            distance: 外部提供的距离估计（米）
            v_self_mps: 本车速度（米/秒）
            t_reaction: 反应时间（秒）
            d_safe: 最小安全距离（米）
            ipm: IPM_Transformer 实例（可选）
        
        Returns:
            (ttc, vx, vy, dw_dt, red_allowed, vw, is_static, risk_level, in_path)
        """
        # 轨迹生命周期：记录当前帧编号
        self._frame_counter += 1
        self._last_seen[track_id] = self._frame_counter

        # 自清理：定期清除陈旧轨迹
        if self._auto_cleanup_interval > 0 and self._frame_counter % self._auto_cleanup_interval == 0:
            self.cleanup_stale_tracks(max_age=90)

        # 初始化历史数据
        if track_id not in self.width_history:
            self.width_history[track_id] = deque(maxlen=self.width_history_len)
        if track_id not in self.x_history:
            self.x_history[track_id] = deque(maxlen=self.width_history_len)
        if track_id not in self.y_history:
            self.y_history[track_id] = deque(maxlen=self.width_history_len)
        if track_id not in self.width_ema:
            self.width_ema[track_id] = float(width)
        if track_id not in self.ground_dist_history:
            self.ground_dist_history[track_id] = deque(maxlen=2)
        
        widths = self.width_history[track_id]
        xs = self.x_history[track_id]
        ys = self.y_history[track_id]
        
        # EMA 平滑宽度（主要用于人/非机动车减小抖动）
        if use_ema:
            prev = self.width_ema[track_id]
            smooth_w = ema_alpha * float(width) + (1 - ema_alpha) * prev
            self.width_ema[track_id] = smooth_w
            width_for_calc = smooth_w
        else:
            width_for_calc = float(width)
            self.width_ema[track_id] = width_for_calc
        
        # 更新历史数据
        widths.append(width_for_calc)
        xs.append(float(center_x))
        ys.append(float(center_y))
        
        # IPM 物理距离估计（模型B）
        ground_distance = None
        if ipm is not None:
            # 边界裁剪：防止边缘检测框坐标越界导致 IPM 反投影失败
            ipm_cx = max(0.0, min(float(center_x), float(frame_w) - 1))
            ipm_by = max(0.0, min(float(bottom_y), float(frame_h) - 1))
            world_pos = ipm.pixel_to_ground(ipm_cx, ipm_by, (frame_h, frame_w))
            if world_pos is not None:
                gx, gy = world_pos
                if gy > 0:
                    ground_distance = math.hypot(gx, gy)
                    self.ground_dist_history[track_id].append(ground_distance)
        
        # 面积阈值判断 - 提高阈值减少小目标误报
        area_ok = area_ratio >= float(min_area_ratio)
        red_allowed = area_ok
        
        # 横向安全走廊（收窄车道范围，过滤路侧/偏离目标）
        # 收窄中心走廊 0.35-0.65 -> 0.40-0.60
        in_path = (frame_w * 0.40) <= center_x <= (frame_w * 0.60)
        
        # 计算 TTC（基于宽度变化率）
        ttc = self.safe_ttc
        dw_dt = 0.0
        
        # 近距离紧急 TTC：目标面积占比 > 15% 时使用 2 帧起算
        if area_ratio > 0.15 and len(widths) >= 2:
            w_n = widths[-1]
            w_prev = widths[-2]
            if w_n > w_prev * 1.02:  # 增加最小变化阈值 2%
                dw_dt = (w_n - w_prev) * self.fps
                if dw_dt > 0:
                    ttc = w_n / dw_dt
                    if ttc <= 0:
                        ttc = self.safe_ttc
        # 标准 TTC：4 帧窗口（从 6 帧降低，更快响应）
        elif len(widths) >= 4:
            w_n = widths[-1]
            w_prev = widths[-4]
            # 最小宽度变化阈值（可配置，默认 3%，原 5%，提早检测慢速接近）
            if w_n <= w_prev * self.width_growth_threshold:
                ttc = self.safe_ttc
                dw_dt = 0.0
            else:
                dw_dt = (w_n - w_prev) / (3.0 / self.fps)
                if dw_dt <= 0:
                    ttc = self.safe_ttc
                else:
                    ttc = w_n / dw_dt
                    if ttc <= 0:
                        ttc = self.safe_ttc
        
        # 距离兜底（当提供距离且有接近速度时）
        provided_distance = None if distance is None else float(distance)
        if provided_distance is not None and len(xs) >= 2:
            vy_world = (ys[-1] - ys[-2]) * self.fps - float(global_vy)
            if vy_world < -1e-3:  # 向前移动（靠近相机）
                ttc_by_dist = abs(provided_distance / vy_world)
                if ttc_by_dist > 0:
                    ttc = min(ttc, ttc_by_dist)
        
        # 模型B TTC：IPM 物理距离差分
        ttc_b = self.safe_ttc
        dist_hist = self.ground_dist_history[track_id]
        if ground_distance is not None and len(dist_hist) >= 2:
            d_prev = dist_hist[-2]
            d_cur = dist_hist[-1]
            v_rel = (d_prev - d_cur) * self.fps  # m/s，正值表示接近
            if v_rel > 1e-3 and d_cur > 1e-3:
                ttc_b = d_cur / v_rel
        
        # 双模型 TTC 取更小值，防止单模型失效
        ttc = min(ttc, ttc_b)
        
        # 计算速度（提前到静止判断之前，为 vy 门控提供数据）
        vx = 0.0
        vy = 0.0
        vw = 0.0
        if len(xs) >= 2:
            vx = (xs[-1] - xs[-2]) * self.fps - float(global_vx)
            vy = (ys[-1] - ys[-2]) * self.fps - float(global_vy)
            vw = (widths[-1] - widths[-2]) * self.fps

        # 问题3修复：纵向方向门控 — 横穿目标（vy向下）渐进衰减 TTC
        # 使用平滑衰减替代硬截断：vy_gate_start 以下无影响，vy_gate_end 以上完全取消
        if vy > self.vy_gate_start and ttc < self.safe_ttc:
            gate_range = max(self.vy_gate_end - self.vy_gate_start, 0.1)
            decay = max(0.0, min(1.0, 1.0 - (vy - self.vy_gate_start) / gate_range))
            if decay < 0.01:
                ttc = self.safe_ttc
                dw_dt = 0.0
            else:
                ttc = min(ttc / max(decay, 0.01), self.safe_ttc)

        # 判断是否静止 — 分辨率自适应阈值（原固定 5px，现按帧宽缩放）
        is_static = False
        static_thresh = max(3.0, frame_w * 0.008)
        if len(xs) >= 10:
            recent_x = list(xs)[-10:]
            recent_y = list(ys)[-10:]
            recent_w = list(widths)[-10:]
            dx_range = max(recent_x) - min(recent_x)
            dy_range = max(recent_y) - min(recent_y)
            dw_range = max(recent_w) - min(recent_w)
            if dx_range <= static_thresh and dy_range <= static_thresh and abs(dw_range) <= static_thresh:
                is_static = True
        
        # TTC 直接使用（已通过宽度EMA与双模型取最小值抑制抖动）
        ttc_effective = min(float(ttc), self.safe_ttc)
        
        # 风险分级（0 安全 / 1 注意 / 2 危险）
        risk_level = 0
        v_rel_eff = -v_self_mps
        
        if provided_distance is not None and len(xs) >= 2:
            v_rel_eff = ((ys[-1] - ys[-2]) * self.fps - float(global_vy)) - v_self_mps
        
        phys_distance = ground_distance if ground_distance is not None else provided_distance
        
        # 动态 SDT 计算
        sdt_dyn = (max(0.0, v_self_mps) + max(0.0, -vy)) * t_reaction + d_safe
        
        # 动态 TTC 阈值（可配置参数）
        ttc_threshold = min(self.ttc_max, max(self.ttc_min, self.ttc_base + float(v_self_mps) * self.ttc_speed_factor))
        
        # 视觉冲突过滤（背景流）
        bg_flow = False
        if len(xs) >= 2:
            sign_flow = (center_x - frame_w * 0.5) * vx
            if sign_flow > 0 and abs(v_rel_eff) < 0.5:
                bg_flow = True
        
        # 基于 SDT + TTC 的风险判定
        if in_path and red_allowed and not bg_flow:
            close_enough = phys_distance is not None and phys_distance < sdt_dyn
            # 增加TTC上限限制，避免极远目标误报
            if close_enough and ttc_effective < ttc_threshold and ttc_effective > 0.1:
                risk_level = 2
            elif ttc_effective < ttc_threshold and ttc_effective > 0.1:
                risk_level = 1
        
        # 横移且偏离中心线时强制降级，避免路侧快速横移误报
        # 提高横向速度阈值 6.0 -> 10.0
        lateral_far = abs(vx) > 10.0 and not in_path
        if lateral_far:
            risk_level = 0
        
        # 静止目标强制安全
        if is_static:
            risk_level = 0
        
        # 物理距离近场硬触发：D < 3m 直接危险
        if phys_distance is not None and phys_distance < 3.0:
            risk_level = 2
        
        final_risk = risk_level
        
        return min(ttc_effective, self.safe_ttc), vx, vy, dw_dt, red_allowed, vw, is_static, final_risk, in_path
    
    def remove_target(self, track_id: int):
        """移除目标（当目标消失时调用）"""
        self.width_history.pop(track_id, None)
        self.width_ema.pop(track_id, None)
        self.x_history.pop(track_id, None)
        self.y_history.pop(track_id, None)
        self.ground_dist_history.pop(track_id, None)
        self._last_seen.pop(track_id, None)

    def clear_all(self):
        """清空所有目标数据"""
        self.width_history.clear()
        self.width_ema.clear()
        self.x_history.clear()
        self.y_history.clear()
        self.ground_dist_history.clear()
        self._last_seen.clear()

    def cleanup_stale_tracks(self, max_age: int = 60) -> List[int]:
        """
        清理超过 max_age 帧未更新的轨迹，防止 ID 积累和内存泄漏。

        Args:
            max_age: 允许最大未更新帧数（默认 60 帧，约 2 秒 @ 30fps）

        Returns:
            已清理的 track_id 列表
        """
        stale = [
            tid for tid, last in self._last_seen.items()
            if self._frame_counter - last > max_age
        ]
        for tid in stale:
            self.remove_target(tid)
        return stale
