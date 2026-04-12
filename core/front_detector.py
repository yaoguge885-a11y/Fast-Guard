"""
正面碰撞检测器模块 (Front Collision Detector)

将正面视角的碰撞检测算法独立出来，实现与侧面视角完全分离的模块化设计。
核心算法：基于目标宽度变化率 (dw/dt) 计算 TTC
"""

from collections import deque
import math
from typing import Tuple, Optional, Dict


class FrontCollisionDetector:
    """
    正面碰撞检测器，基于目标宽度变化率计算 TTC
    """
    
    def __init__(self, fps: float = 30.0):
        """初始化正面碰撞检测器"""
        self.fps = fps if fps and fps > 0 else 30.0
        self.safe_ttc = 99.0
        self.width_history_len = 10
        
        # 历史数据存储
        self.width_history = {}
        self.width_ema = {}
        self.x_history = {}
        self.y_history = {}
        self.ground_dist_history = {}
    
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
            world_pos = ipm.pixel_to_ground(center_x, bottom_y, (frame_h, frame_w))
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
            # 增加最小宽度变化阈值，减少抖动误报
            if w_n <= w_prev * 1.05:  # 至少5%变化才计算
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

        # 问题3修复：纵向方向门控 — 横穿目标（vy向下）误到 TTC 报警，取消其 TTC
        if vy > 3.0 and ttc < self.safe_ttc:
            ttc = self.safe_ttc
            dw_dt = 0.0

        # 判断是否静止 - 提高阈值减少误判
        is_static = False
        if len(xs) >= 10:
            recent_x = list(xs)[-10:]
            recent_y = list(ys)[-10:]
            recent_w = list(widths)[-10:]
            dx_range = max(recent_x) - min(recent_x)
            dy_range = max(recent_y) - min(recent_y)
            dw_range = max(recent_w) - min(recent_w)
            if dx_range <= 5.0 and dy_range <= 5.0 and abs(dw_range) <= 5.0:
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
        
        # 更灵敏的 TTC 阈值（废除连续帧强制限制）
        ttc_threshold = min(3.5, max(2.2, 1.8 + float(v_self_mps) * 0.06))
        
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
    
    def clear_all(self):
        """清空所有目标数据"""
        self.width_history.clear()
        self.width_ema.clear()
        self.x_history.clear()
        self.y_history.clear()
        self.ground_dist_history.clear()
