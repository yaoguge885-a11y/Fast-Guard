from collections import deque


class TTCCalculator:
    def __init__(self, fps, width_history_len=10, ttc_history_len=5, safe_ttc=99.0):
        self.fps = fps if fps and fps > 0 else 30.0
        self.safe_ttc = float(safe_ttc)
        self.width_history_len = width_history_len
        self.ttc_history_len = ttc_history_len
        self.width_history = {}
        self.width_ema = {}
        self.ttc_history = {}
        self.x_history = {}
        self.y_history = {}

    def update(
        self,
        track_id,
        width,
        center_x,
        center_y,
        bottom_y,
        frame_w,
        frame_h,
        warning_line_y,
        area_ratio,
        global_vx=0.0,
        global_vy=0.0,
        min_area_ratio=0.005,
        use_ema=False,
        ema_alpha=0.3,
        distance=None,  # 可选：外部提供的距离估计（如双目视差或单目估距）
        v_self_mps=0.0,  # 本车速度（m/s），用于流场矫正
        t_reaction=1.2,  # 反应时间，用于 SDT 计算
        d_safe=2.0,      # 最小安全冗余距离
    ):

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

        widths.append(width_for_calc)
        xs.append(float(center_x))
        ys.append(float(center_y))

        area_ok = area_ratio >= float(min_area_ratio)
        redline_cross = bottom_y > warning_line_y
        red_allowed = area_ok and redline_cross
        # 横向安全走廊（收窄车道范围，过滤路侧/偏离目标）
        in_path = (frame_w * 0.35) <= center_x <= (frame_w * 0.65)

        ttc = self.safe_ttc

        dw_dt = 0.0
        # 若外部提供了距离，可用距离/速度进一步约束 TTC（可选增强；当前仅做安全兜底）
        provided_distance = None if distance is None else float(distance)

        if len(widths) >= 6:
            w_n = widths[-1]
            w_prev = widths[-6]
            if w_n <= w_prev:
                ttc = self.safe_ttc
                dw_dt = 0.0
            else:
                dw_dt = (w_n - w_prev) / (5.0 / self.fps)
                if dw_dt <= 0:
                    ttc = self.safe_ttc
                else:
                    ttc = w_n / dw_dt
                    if ttc <= 0:
                        ttc = self.safe_ttc

        # 距离兜底（当提供距离且有接近速度时，可给出更保守/更真实的 TTC）
        if provided_distance is not None and len(xs) >= 2:
            # 视距 / 纵向速度，防止 0 分母
            vx_dummy = 0.0  # 不用于纵向 TTC
            vy_world = (ys[-1] - ys[-2]) * self.fps - float(global_vy)
            # 这里 vy_world 仅作近似，若无可靠缩放可忽略；仅当速度向前（负向）才用
            if vy_world < -1e-3:
                ttc_by_dist = abs(provided_distance / vy_world)
                if ttc_by_dist > 0:
                    ttc = min(ttc, ttc_by_dist)


        vx = 0.0
        vy = 0.0
        vw = 0.0
        if len(xs) >= 2:
            vx = (xs[-1] - xs[-2]) * self.fps - float(global_vx)
            vy = (ys[-1] - ys[-2]) * self.fps - float(global_vy)
            vw = (widths[-1] - widths[-2]) * self.fps

        is_static = False
        if len(xs) >= 10:
            recent_x = list(xs)[-10:]
            recent_y = list(ys)[-10:]
            recent_w = list(widths)[-10:]
            dx_range = max(recent_x) - min(recent_x)
            dy_range = max(recent_y) - min(recent_y)
            dw_range = max(recent_w) - min(recent_w)
            if dx_range <= 2.0 and dy_range <= 2.0 and abs(dw_range) <= 2.0:
                is_static = True

        ttc_hist = self.ttc_history[track_id]
        ttc_hist.append(min(float(ttc), self.safe_ttc))
        avg_ttc = sum(ttc_hist) / len(ttc_hist)

        # === 风险分级（0 安全 / 1 注意 / 2 危险）===
        risk_level = 0
        # 相对速度：观测速度 - 本车流场速度
        v_rel_eff = -v_self_mps
        if provided_distance is not None and len(xs) >= 2:
            # vy <0 表示靠近相机；这里简化用距离帧差推算 vx/纵向相对速度
            v_rel_eff = ( (ys[-1] - ys[-2]) * self.fps - float(global_vy) ) - v_self_mps
        # 动态 SDT： (V_self + V_obj) * t_reaction + d_safe
        sdt_dyn = (max(0.0, v_self_mps) + max(0.0, -vy)) * t_reaction + d_safe
        ttc_threshold = 2.0

        # 视觉冲突过滤（背景流）：若 vx 与 (x-中心) 同向且 v_rel_eff 近 0，视为背景
        bg_flow = False
        if len(xs) >= 2:
            sign_flow = (center_x - frame_w * 0.5) * vx
            if sign_flow > 0 and abs(v_rel_eff) < 0.5:
                bg_flow = True

        # 基于 SDT + TTC 的风险判定
        if in_path and red_allowed and not bg_flow:
            close_enough = provided_distance is not None and provided_distance < sdt_dyn
            if close_enough and avg_ttc < ttc_threshold:
                risk_level = 2
            elif avg_ttc < ttc_threshold:
                risk_level = 1

        # 横移且偏离中心线时强制降级，避免路侧快速横移误报
        lateral_far = abs(vx) > 6.0 and not in_path
        if lateral_far:
            risk_level = 0

        return min(avg_ttc, self.safe_ttc), vx, vy, dw_dt, red_allowed, vw, is_static, risk_level, in_path


