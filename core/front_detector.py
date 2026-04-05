"""
正面碰撞检测器模块 (Front Collision Detector)

将正面视角的碰撞检测算法独立出来，实现与侧面视角完全分离的模块化设计。
核心算法：基于目标宽度变化率 (dw/dt) 计算 TTC
"""

from collections import deque
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
        self.ttc_history_len = 5
        
        # 历史数据存储
        self.width_history = {}
        self.width_ema = {}
        self.ttc_history = {}
        self.x_history = {}
        self.y_history = {}
        
        # 新增：连续报警确认计数器
        self.alert_frames = {}
        self.min_alert_frames = 3  # 至少3帧才报警
        
        # 新增：历史风险等级记录
        self.risk_history = {}
        self.risk_history_len = 5
    
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
        
        Returns:
            (ttc, vx, vy, dw_dt, red_allowed, vw, is_static, risk_level, in_path)
        """
        # 初始化历史数据
        if track_id not in self.width_history:
            self.width_history[track_id] = deque(maxlen=self.width_history_len)
        if track_id not in self.ttc_history:
            self.ttc_history[track_id] = deque(maxlen=self.ttc_history_len)
        if track_id not in self.x_history:
            self.x_history[track_id] = deque(maxlen=self.width_history_len)
        if track_id not in self.y_history:
            self.y_history[track_id] = deque(maxlen=self.width_history_len)
        if track_id not in self.width_ema:
            self.width_ema[track_id] = float(width)
        if track_id not in self.alert_frames:
            self.alert_frames[track_id] = 0
        if track_id not in self.risk_history:
            self.risk_history[track_id] = deque(maxlen=self.risk_history_len)
        
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
        
        # 计算速度
        vx = 0.0
        vy = 0.0
        vw = 0.0
        if len(xs) >= 2:
            vx = (xs[-1] - xs[-2]) * self.fps - float(global_vx)
            vy = (ys[-1] - ys[-2]) * self.fps - float(global_vy)
            vw = (widths[-1] - widths[-2]) * self.fps
        
        # 判断是否静止 - 提高阈值减少误判
        is_static = False
        if len(xs) >= 10:
            recent_x = list(xs)[-10:]
            recent_y = list(ys)[-10:]
            recent_w = list(widths)[-10:]
            dx_range = max(recent_x) - min(recent_x)
            dy_range = max(recent_y) - min(recent_y)
            dw_range = max(recent_w) - min(recent_w)
            # 提高静止判断阈值 2.0 -> 5.0 像素
            if dx_range <= 5.0 and dy_range <= 5.0 and abs(dw_range) <= 5.0:
                is_static = True
        
        # TTC 历史平滑
        ttc_hist = self.ttc_history[track_id]
        ttc_hist.append(min(float(ttc), self.safe_ttc))
        avg_ttc = sum(ttc_hist) / len(ttc_hist)
        
        # 风险分级（0 安全 / 1 注意 / 2 危险）
        risk_level = 0
        v_rel_eff = -v_self_mps
        
        if provided_distance is not None and len(xs) >= 2:
            v_rel_eff = ((ys[-1] - ys[-2]) * self.fps - float(global_vy)) - v_self_mps
        
        # 动态 SDT 计算
        sdt_dyn = (max(0.0, v_self_mps) + max(0.0, -vy)) * t_reaction + d_safe
        # 提高TTC阈值 2.0 -> 2.5秒
        ttc_threshold = 2.5
        
        # 视觉冲突过滤（背景流）
        bg_flow = False
        if len(xs) >= 2:
            sign_flow = (center_x - frame_w * 0.5) * vx
            if sign_flow > 0 and abs(v_rel_eff) < 0.5:
                bg_flow = True
        
        # 基于 SDT + TTC 的风险判定
        if in_path and red_allowed and not bg_flow:
            close_enough = provided_distance is not None and provided_distance < sdt_dyn
            # 增加TTC上限限制，避免极远目标误报
            if close_enough and avg_ttc < ttc_threshold and avg_ttc > 0.1:
                risk_level = 2
            elif avg_ttc < ttc_threshold and avg_ttc > 0.1:
                risk_level = 1
        
        # 横移且偏离中心线时强制降级，避免路侧快速横移误报
        # 提高横向速度阈值 6.0 -> 10.0
        lateral_far = abs(vx) > 10.0 and not in_path
        if lateral_far:
            risk_level = 0
        
        # 静止目标强制安全
        if is_static:
            risk_level = 0
        
        # 记录风险历史
        self.risk_history[track_id].append(risk_level)
        
        # 连续帧确认机制 - 减少瞬时抖动导致的误报
        if risk_level >= 2:
            self.alert_frames[track_id] += 1
        else:
            self.alert_frames[track_id] = max(0, self.alert_frames[track_id] - 1)
        
        # 只有连续多帧高风险的才最终确认
        final_risk = risk_level
        if risk_level >= 2 and self.alert_frames[track_id] < self.min_alert_frames:
            # 降级为注意级别
            final_risk = 1 if risk_level >= 2 else risk_level
        
        # 历史风险平滑 - 如果最近几帧都是低风险，当前帧也不应突然高风险
        if len(self.risk_history[track_id]) >= 3:
            recent_risks = list(self.risk_history[track_id])[-3:]
            avg_risk = sum(recent_risks) / len(recent_risks)
            # 如果历史平均风险低，当前突然高风险，可能是误报
            if avg_risk < 1.0 and final_risk >= 2:
                final_risk = 1
        
        return min(avg_ttc, self.safe_ttc), vx, vy, dw_dt, red_allowed, vw, is_static, final_risk, in_path
    
    def remove_target(self, track_id: int):
        """移除目标（当目标消失时调用）"""
        self.width_history.pop(track_id, None)
        self.width_ema.pop(track_id, None)
        self.ttc_history.pop(track_id, None)
        self.x_history.pop(track_id, None)
        self.y_history.pop(track_id, None)
        self.alert_frames.pop(track_id, None)
        self.risk_history.pop(track_id, None)
    
    def clear_all(self):
        """清空所有目标数据"""
        self.width_history.clear()
        self.width_ema.clear()
        self.ttc_history.clear()
        self.x_history.clear()
        self.y_history.clear()
        self.alert_frames.clear()
        self.risk_history.clear()
