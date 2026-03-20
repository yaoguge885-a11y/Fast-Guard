# === 1. 基础导入与环境信息 ===
# ==================================================================================
import os
import sys
import time
import math
import shutil

import ultralytics
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from core.calculator import TTCCalculator
from core.ipm import IPM_Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont   # 新增：PIL 中文支持

def print_versions():
    print(f"Python 版本: {sys.version}")
    print(f"ultralytics 版本: {ultralytics.__version__}")
    print(f"PyQt5 版本: {QtCore.PYQT_VERSION_STR}")
    print(f"Qt 版本: {QtCore.QT_VERSION_STR}")
    print(f"opencv-python 版本: {cv2.__version__}")
    print(f"numpy 版本: {np.__version__}")
    print(f"Pillow 版本: {Image.__version__}")


    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except Exception:
        pass

# --- 自定义深度学习组件 (CA, SCNN, ReLU, SIoU) ---

# ==================================================================================

from core.dl_components import *
from core.side_collision import *
from core.view_classifier import *
# === 5. 核心视频处理线程 (视频读取、预处理、YOLO 推理与预警) ===
# ==================================================================================
class VideoThread(QtCore.QThread):
    # 修改信号签名：发送三张图像 (Original, Preprocessed, Inference)
    frame_signal = QtCore.pyqtSignal(QtGui.QImage, QtGui.QImage, QtGui.QImage)
    status_signal = QtCore.pyqtSignal(str)

    ttc_signal = QtCore.pyqtSignal(float, int)
    side_warning_signal = QtCore.pyqtSignal(str, str, int)
    log_signal = QtCore.pyqtSignal(int, float)
    latency_signal = QtCore.pyqtSignal(float)
    model_signal = QtCore.pyqtSignal(str)
    perspective_signal = QtCore.pyqtSignal(str)
    debug_signal = QtCore.pyqtSignal(dict)
    position_signal = QtCore.pyqtSignal(int, int, float, float)
    hud_signal = QtCore.pyqtSignal(dict)



    def __init__(self, source, model_path, parent=None, weak_conf_threshold=0.38, edge_strength_threshold=28.0):
        super().__init__(parent)
        self.source = source
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.weak_conf_threshold = float(weak_conf_threshold)
        self.edge_strength_threshold = float(edge_strength_threshold)
        self._running = True
        self._frame_count = 0
        self._last_centers = {}
        self._seen_counts = {}
        self.view_classifier = ViewClassifier()
        self.side_detector = None
        self.current_perspective = "分析中..."
        self.perspective_locked = False
        self.last_perspective_time = 0
        self.perspective_debug = False
        self.last_debug_info = {}
        self.forward_calculator = None
        
        # IPM & 轨迹评估
        self.ipm = IPM_Transformer()
        self._world_history = {}
        self.vehicle_width = 1.9  # 车辆宽度 (m)
        self.envelope_margin = 0.4  # 包络线左右冗余 (m)
        self.envelope_length = 30.0  # 前向判定距离 (m)

        # 安全距离阈值 (论文 2.3)：驾驶员反应时间 + 最小冗余距离
        self.t_reaction = 1.2  # 秒（前向视角默认），侧向视角将自适应降到 0.8s
        self.d_safe = 2.0      # 米，可按现场调节
        self.v_self_mps = 11.1 # 本车速度模拟，约 40km/h，可调
        self._last_distance = {}  # 记录每个目标上一帧的距离，用于估计 V_rel
        self._vrel_history = {}   # 卡尔曼/滑动窗口平滑相对速度

        # Stereo Vision Parameters

        self.stereo_mode = False  # 默认为单目，SBS宽屏自动切换
        self.stereo_matcher = None
        self.baseline = 0.12  # 默认基线 12cm (需根据实际硬件调整)
        self.focal_length = 800  # 默认焦距 (像素单位，需标定)
        self.disparity_map = None

        self._user_paused = False
        self._seeking = False
        self._seek_target = None
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.duration = 0.0
        self._fps_ema = None
        self._ema_hood_y = None
        
        # 锁定相关状态变量
        self._forward_lock_count = 0
        self._side_lock_count = 0
        self._locked_warning_y = None
        self._locked_side_mode = None
        self._locked_car_x_ground = None

        # ---------- 中文字体路径设置 ----------


        # Windows 常用字体：黑体、微软雅黑
        self.font_path = "C:/Windows/Fonts/simhei.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/msyh.ttc" # 备选微软雅黑
        
        self.cached_font = None
        try:
            self.cached_font = ImageFont.truetype(self.font_path, 24)
        except Exception as e:
            print(f"Font preload failed: {e}")
        # --------------------------------------

        # Class Name Translation Map
        self.class_map = {
            "person": "行人",
            "bicycle": "自行车",
            "car": "轿车",
            "motorcycle": "摩托车",
            "airplane": "飞机",
            "bus": "公交车",
            "train": "火车",
            "truck": "卡车",
            "boat": "船",
            "traffic light": "红绿灯",
            "fire hydrant": "消防栓",
            "stop sign": "停止标志",
            "parking meter": "停车计费器",
            "bench": "长椅",
            "bird": "鸟",
            "cat": "猫",
            "dog": "狗",
            "horse": "马",
            "sheep": "羊",
            "cow": "牛",
            "elephant": "大象",
            "bear": "熊",
            "zebra": "斑马",
            "giraffe": "长颈鹿",
            "backpack": "背包",
            "umbrella": "雨伞",
            "handbag": "手提包",
            "tie": "领带",
            "suitcase": "手提箱",
            "frisbee": "飞盘",
            "skis": "滑雪板",
            "snowboard": "单板滑雪",
            "sports ball": "运动球",
            "kite": "风筝",
            "baseball bat": "棒球棒",
            "baseball glove": "棒球手套",
            "skateboard": "滑板",
            "surfboard": "冲浪板",
            "tennis racket": "网球拍",
            "bottle": "瓶子",
            "wine glass": "酒杯",
            "cup": "杯子",
            "fork": "叉子",
            "knife": "刀",
            "spoon": "勺子",
            "bowl": "碗",
            "banana": "香蕉",
            "apple": "苹果",
            "sandwich": "三明治",
            "orange": "橙子",
            "broccoli": "西兰花",
            "carrot": "胡萝卜",
            "hot dog": "热狗",
            "pizza": "披萨",
            "donut": "甜甜圈",
            "cake": "蛋糕",
            "chair": "椅子",
            "couch": "沙发",
            "potted plant": "盆栽",
            "bed": "床",
            "dining table": "餐桌",
            "toilet": "厕所",
            "tv": "电视",
            "laptop": "笔记本电脑",
            "mouse": "鼠标",
            "remote": "遥控器",
            "keyboard": "键盘",
            "cell phone": "手机",
            "microwave": "微波炉",
            "oven": "烤箱",
            "toaster": "烤面包机",
            "sink": "水槽",
            "refrigerator": "冰箱",
            "book": "书",
            "clock": "钟",
            "vase": "花瓶",
            "scissors": "剪刀",
            "teddy bear": "泰迪熊",
            "hair drier": "吹风机",
            "toothbrush": "牙刷"
        }

    def _detect_hood_y(self, frame_gray, h, w):
        """自动侦测引擎盖边缘 (水平线)"""
        # 取画面下半部中间区域 (高度 60% ~ 100%, 宽度 20% ~ 80%)
        roi_y1 = int(h * 0.60)
        roi_y2 = h
        roi_x1 = int(w * 0.20)
        roi_x2 = int(w * 0.80)
        
        roi = frame_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None
            
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)
        
        row_mean = np.mean(abs_sobel_y, axis=1)
        if len(row_mean) == 0:
            return None
            
        max_idx = int(np.argmax(row_mean))
        if row_mean[max_idx] < 10.0:
            return None
            
        hood_y = roi_y1 + max_idx
        # 限制高度不要超过画面 40% (即上限是 0.6h)，下限是 0.95h
        #红线默认高度
        hood_y = max(int(h * 0.60), min(int(h * 0.8), hood_y))
        return hood_y

    def _cv2_put_chinese(self, img, text, org, font_size, color):
        """
        单条绘制（作为兼容保留，但内部应优先使用批量绘制）
        """
        return self._draw_batch_chinese(img, [(text, org, font_size, color)])

    def _draw_batch_chinese(self, img, draws):
        """
        批量绘制中文字符，显著提升性能
        draws: list of (text, org, font_size, color)
        """
        if not draws:
            return img
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw_obj = ImageDraw.Draw(img_pil)
        
        for text, org, font_size, color in draws:
            try:
                # 尽量复用字体对象，如果字号不同再重新加载
                if self.cached_font and self.cached_font.size == font_size:
                    font = self.cached_font
                else:
                    font = ImageFont.truetype(self.font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
            
            # 阴影
            shadow_offset = (1, 1)
            draw_obj.text((org[0] + shadow_offset[0], org[1] + shadow_offset[1]), text, font=font, fill=(0, 0, 0))
            draw_obj.text(org, text, font=font, fill=color[::-1]) # BGR -> RGB
            
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _draw_l_corners(self, frame, x1, y1, x2, y2, color, thickness=2, seg=16):
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

    # --- 绘图辅助工具 ---
    def _update_world_track(self, track_id, world_pos, max_len=20):
        from collections import deque
        if track_id not in self._world_history:
            self._world_history[track_id] = deque(maxlen=max_len)
        self._world_history[track_id].append(world_pos)

    def _compute_yaw_rate(self, track_id):
        pts = self._world_history.get(track_id, [])
        if len(pts) < 3:
            return 0.0, None
        p0, p1, p2 = pts[-3], pts[-2], pts[-1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        def _norm(v):
            return math.sqrt(v[0] * v[0] + v[1] * v[1])
        n1, n2 = _norm(v1), _norm(v2)
        if n1 < 1e-4 or n2 < 1e-4:
            return 0.0, None
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        dot = max(-1.0, min(1.0, dot / (n1 * n2)))
        angle = math.degrees(math.acos(dot))
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        signed_angle = angle if cross >= 0 else -angle
        return signed_angle, v2

    def _segment_intersects_rect(self, p1, p2, half_w, length):
        # 矩形：x in [-half_w, half_w], y in [0, length]
        def inside(p):
            return (-half_w <= p[0] <= half_w) and (0 <= p[1] <= length)

        if inside(p1) or inside(p2):
            return True

        rect_edges = [
            ((-half_w, 0), (half_w, 0)),
            ((half_w, 0), (half_w, length)),
            ((half_w, length), (-half_w, length)),
            ((-half_w, length), (-half_w, 0)),
        ]

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

        for e1, e2 in rect_edges:
            d1 = ccw(p1, p2, e1)
            d2 = ccw(p1, p2, e2)
            d3 = ccw(e1, e2, p1)
            d4 = ccw(e1, e2, p2)
            if (d1 == 0 and inside(e1)) or (d2 == 0 and inside(e2)):
                return True
            if (d3 == 0 and inside(p1)) or (d4 == 0 and inside(p2)):
                return True
            if (d1 * d2 < 0) and (d3 * d4 < 0):
                return True
        return False

    def _in_conflict_envelope(self, track_id, world_pos):
        half_w = self.vehicle_width * 0.5 + self.envelope_margin
        length = self.envelope_length
        history = self._world_history.get(track_id)
        if not history or len(history) < 1:
            return abs(world_pos[0]) <= half_w and 0 <= world_pos[1] <= length
        p_prev = history[-1]
        return self._segment_intersects_rect(p_prev, world_pos, half_w, length)

    def _predict_intent(self, track_id, world_pos):
        yaw_deg, v2 = self._compute_yaw_rate(track_id)
        intent = "直行通过"
        if len(self._world_history.get(track_id, [])) >= 2:
            p_prev = self._world_history[track_id][-2]
            toward_center = abs(world_pos[0]) < abs(p_prev[0])
            if abs(yaw_deg) > 8.0 and toward_center:
                intent = "侧向切入"
        angle_cost = abs(yaw_deg) / 180.0  # 夹角代价，用于平滑/抑制抖动
        return intent, yaw_deg, angle_cost


    # --- 核心主循环 ---
    def run(self):

        self.status_signal.emit("扫描中")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.status_signal.emit("系统就绪")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1e-3:
            self.fps = 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.total_frames > 0 else 0.0

        self.forward_calculator = TTCCalculator(self.fps)

        if not os.path.exists(self.model_path):
            model_filename = os.path.basename(self.model_path)
            model = ultralytics.YOLO(model_filename)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            if os.path.exists(model_filename):
                try:
                    shutil.move(model_filename, self.model_path)
                except shutil.Error:
                    try:
                        shutil.copy2(model_filename, self.model_path)
                    except Exception:
                        pass
        else:
            model = ultralytics.YOLO(self.model_path)

        self.model_signal.emit(os.path.basename(self.model_path))
        allowed_names = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

        name_map = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
        allowed_ids = {k for k, v in name_map.items() if v in allowed_names}
        fps_value = self.fps

        # 检查 GPU 是否可用并指定设备
        device = '0' if torch.cuda.is_available() else 'cpu'

        # 缓存上一帧的追踪结果
        last_results = None
        deferred_draws = []
        while self._running:
            just_seeked = False
            if self._seek_target is not None and self.cap:
                target = max(0, int(self._seek_target))
                if self.total_frames > 0:
                    target = min(target, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self._frame_count = target
                self._seek_target = None
                just_seeked = True

            if self._user_paused and not self._seeking and not just_seeked:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                # 播放结束时进入暂停状态，不退出线程，以便响应进度条回退
                self._user_paused = True
                if self._seek_target is not None:
                    continue
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            # 初始化 IPM 内参（如果未设置）
            if self.ipm:
                self.ipm.set_frame(w, h)
            # 保留原始帧用于 UI 渲染（避免过度增强）
            frame_raw = frame.copy()


            # --- 优化后的图像预处理 ---
            # 只有当需要显示预处理视图时，才进行所有昂贵的计算
            # 默认只进行最小限度的增强用于推理
            
            # 推理用的轻量级增强
            inference_frame = frame
            pre_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 基础灰度用于后续逻辑

            # 如果需要更强的特征（原有的 CLAHE 和 掩码），可以保留但优化
            # 比如：每 2 帧计算一次掩码，或者跳过 Sobel
            do_heavy_preproc = (self._frame_count % 3 == 0) # 降低重度预处理频率
            
            if do_heavy_preproc:
                denoised_frame = cv2.medianBlur(frame, 3)
                lab = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_eq = clahe.apply(l)
                enhanced_frame = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
                pre_gray = l_eq # 更新显示用的灰度图
            else:
                enhanced_frame = frame

            # 简化掩码和推理帧生成 (核心卡顿点)
            inference_frame = enhanced_frame
            sobel_magnitude = np.zeros_like(pre_gray) # 默认空，按需计算

            # --- 双目 SBS 自动识别与分割 ---

            # 提高判定阈值，避免把常见 16:9 视频误判为 SBS（导致只取左半幅）
            is_sbs = w >= h * 2.4  # 仅当宽高比非常大时才判定为 SBS
            if is_sbs:
                w_half = w // 2
                frame_l = frame[:, :w_half]
                frame_r = frame[:, w_half:]
                inference_frame_l = inference_frame[:, :w_half]
                frame = frame_l
                inference_frame = inference_frame_l  # 推理主要在左图进行
                h, w = frame.shape[:2]
                
                # 初始化立体匹配器 (若尚未初始化)
                if self.stereo_matcher is None:
                    self.stereo_matcher = cv2.StereoSGBM_create(
                        minDisparity=0,
                        numDisparities=64, # 视差搜寻范围
                        blockSize=5,
                        P1=8 * 3 * 5**2,
                        P2=32 * 3 * 5**2,
                        disp12MaxDiff=1,
                        uniquenessRatio=10,
                        speckleWindowSize=100,
                        speckleRange=32,
                        preFilterCap=63
                    )
                
                # 计算视差图 (转换为灰度图计算更快)
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                disparity = self.stereo_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
                self.disparity_map = disparity
                self.stereo_mode = True
            else:
                self.stereo_mode = False
                self.disparity_map = None
            current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_frame_idx < 0:
                current_frame_idx = self._frame_count
            self._frame_count = current_frame_idx

            if self._seeking:
                rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ph, pw, pch = rgb_preview.shape
                bytes_per_line_preview = pch * pw
                preview_image = QtGui.QImage(
                    rgb_preview.data, pw, ph, bytes_per_line_preview, QtGui.QImage.Format_RGB888
                )
                # 拖拽时三画面同步显示原图
                self.frame_signal.emit(preview_image.copy(), preview_image.copy(), preview_image.copy())
                self.position_signal.emit(
                    current_frame_idx,
                    self.total_frames,
                    current_frame_idx / fps_value if fps_value else 0.0,
                    self.duration,
                )
                continue


            min_ttc = 99.0
            min_id = -1

            # 动态检测引擎盖边缘 (仅在前5帧进行)
            if self._locked_warning_y is None:
                curr_hood_y = self._detect_hood_y(pre_gray, h, w)
                if curr_hood_y is not None:
                    if self._ema_hood_y is None:
                        self._ema_hood_y = float(curr_hood_y)
                    else:
                        self._ema_hood_y = 0.95 * self._ema_hood_y + 0.05 * curr_hood_y
                
                # 仅在前向视角下累加计数并最终锁定
                if self.current_perspective == "前向视角":
                    if self._forward_lock_count >= 5:
                        self._locked_warning_y = int(self._ema_hood_y) if self._ema_hood_y is not None else int(h * 0.88)
                    else:
                        self._forward_lock_count += 1
            
            # 使用锁定值或当前计算值
            base_warning_y = self._locked_warning_y if self._locked_warning_y is not None else (int(self._ema_hood_y) if self._ema_hood_y is not None else int(h * 0.88))

            warning_line_y = base_warning_y
            warning_line_small_y = base_warning_y
            detect_line_y = int(h * 0.40)

            # 仅前向视角绘制警示线/渐变；侧向不画红/灰线
            if self.current_perspective == "前向视角":
                grad_region = frame[warning_line_y:, :, :].astype(np.float32)
                red = np.zeros_like(grad_region)
                red[:, :, 2] = 255
                if grad_region.shape[0] > 0:
                    alpha = np.linspace(0.0, 0.35, grad_region.shape[0], dtype=np.float32)[:, None, None]
                    blended = grad_region * (1 - alpha) + red * alpha
                    frame[warning_line_y:, :, :] = blended.astype(np.uint8)

                for x in range(0, w, 30):
                    cv2.line(frame, (x, warning_line_y), (min(x + 18, w - 1), warning_line_y), (0, 0, 255), 4)
                for x in range(0, w, 30):
                    cv2.line(frame, (x, detect_line_y), (min(x + 16, w - 1), detect_line_y), (160, 160, 160), 2)


            # 实跳帧逻辑：每 2 帧进行一次推理
            if self._frame_count % 2 == 0:
                results = model.track(
                    inference_frame,
                    persist=True,
                    verbose=False,
                    imgsz=640,          # 进一步降低分辨率以提升速度 (640 是 YOLO 标准值)
                    conf=0.25,
                    iou=0.5,
                    classes=[0, 1, 2, 3, 5, 7],
                    tracker="bytetrack.yaml",
                    device=device       # 明确使用 GPU
                )
                last_results = results
            else:
                results = last_results

            infos = []
            persons = []
            bikes = []
            dx_list = []
            dy_list = []
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        if cls_id not in allowed_ids:
                            continue

                        track_id = int(box.id[0]) if box.id is not None else -1
                        conf_score = float(box.conf[0]) if box.conf is not None else 0.0
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        if y2 <= detect_line_y:
                            continue
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        class_name = name_map.get(cls_id, "")
                        if class_name in {"car", "truck"} and conf_score < 0.45:
                            continue

                        # Sobel 边缘强度辅助过滤：降低低置信度静态纹理误报
                        edge_strength = 0.0
                        roi_y1, roi_y2 = max(0, y1), min(h, y2)
                        roi_x1, roi_x2 = max(0, x1), min(w, x2)
                        if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                            roi_edge = sobel_magnitude[roi_y1:roi_y2, roi_x1:roi_x2]
                            if roi_edge.size > 0:
                                edge_strength = float(np.mean(roi_edge))

                        if class_name in {"bicycle", "motorcycle", "person"}:
                            if conf_score < 0.25:
                                continue
                            if conf_score < self.weak_conf_threshold and edge_strength < self.edge_strength_threshold:
                                continue

                        record = (track_id, x1, y1, x2, y2, cx, cy, class_name)
                        infos.append(record)
                        if class_name in {"bicycle", "motorcycle"}:
                            bikes.append(record)
                        if class_name == "person":
                            persons.append(record)

                        if track_id in self._last_centers:
                            px, py = self._last_centers[track_id]
                            dx_list.append(cx - px)
                            dy_list.append(cy - py)

            def iou(a, b):
                ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
                bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    return 0.0
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                return inter / max(area_a + area_b - inter, 1e-6)

            filtered_infos = []
            for rec in infos:
                if rec[7] == "person":
                    keep = True
                    for bike in bikes:
                        if iou(rec, bike) > 0.6:
                            keep = False
                            break
                    if not keep:
                        continue
                filtered_infos.append(rec)

            infos = filtered_infos

            for track_id, x1, y1, x2, y2, cx, cy, _ in infos:
                self._last_centers[track_id] = (cx, cy)
                self._seen_counts[track_id] = self._seen_counts.get(track_id, 0) + 1

            avg_dx = sum(dx_list) / len(dx_list) if dx_list else 0.0
            avg_dy = sum(dy_list) / len(dy_list) if dy_list else 0.0
            global_vx = avg_dx * fps_value
            global_vy = avg_dy * fps_value

            # 视角分析
            current_time = time.time()
            if not self.perspective_locked or current_time - self.last_perspective_time > 2.0:
                self.view_classifier.analyze_frame(frame.copy(), infos)
                
                debug_info = self.view_classifier.get_debug_info()
                if debug_info:
                    self.last_debug_info = debug_info
                    self.debug_signal.emit(debug_info)
                
                if debug_info.get('locked', False) and not self.perspective_locked:
                    self.perspective_locked = True
                    self.current_perspective = debug_info['locked_perspective']
                    self.perspective_signal.emit(self.current_perspective)
                    self.last_perspective_time = current_time
                    
                    if self.current_perspective == "侧面视角":
                        self.side_detector = SideCollisionDetector(w, h, fps_value)
                        self.status_signal.emit("侧向碰撞检测已启用")
                    elif self.current_perspective == "前向视角":
                        self.side_detector = None
                        self.status_signal.emit("前向碰撞检测已启用")
                
                elif not self.perspective_locked:
                    perspective = self.view_classifier.determine_perspective()
                    if perspective != "分析中..." and perspective != self.current_perspective:
                        self.current_perspective = perspective
                        self.perspective_signal.emit(perspective)
                        self.last_perspective_time = current_time

            # 在图像上显示视角状态（中文）
            view_text = self.current_perspective
            if self.perspective_locked:
                view_text += " (已锁定)"
            deferred_draws.append((view_text, (w - 360, 40), 32, (255, 255, 255)))

            # 根据视角类型选择不同的碰撞检测逻辑
            if self.current_perspective == "侧面视角":
                frame = self._process_side_ipm(frame, infos, w, h, fps_value, deferred_draws)
            elif self.current_perspective == "前向视角":

                min_ttc_alert = 99.0
                min_id_alert = -1
                
                for track_id, x1, y1, x2, y2, cx, cy, class_name in infos:
                    if y2 <= detect_line_y:
                        continue
                    width = max(1, x2 - x1)
                    height = max(1, y2 - y1)
                    if height / max(1, width) > 3.0 or width / max(1, height) > 4.0:
                        continue
                    area_ratio = (width * height) / max(1, w * h)
                    warn_line = warning_line_small_y if class_name in {"bicycle", "motorcycle", "person"} else warning_line_y
                    
                    # 获取物体距离 (双目模式)
                    obj_dist = None
                    if self.stereo_mode and self.disparity_map is not None:
                        # 在检测框中心区域取平均视差
                        mask_y1, mask_y2 = max(0, y1), min(h, y2)
                        mask_x1, mask_x2 = max(0, x1), min(w, x2)
                        roi_disp = self.disparity_map[mask_y1:mask_y2, mask_x1:mask_x2]
                        valid_disp = roi_disp[roi_disp > 0]
                        if len(valid_disp) > 0:
                            avg_disp = np.median(valid_disp)
                            if avg_disp > 0:
                                obj_dist = (self.focal_length * self.baseline) / avg_disp
                    else:
                        # 单目近似距离：使用检测框高度估距（假设目标高度常数 H_obj）
                        H_obj_map = {
                            "car": 1.5,
                            "truck": 2.5,
                            "bus": 3.0,
                            "motorcycle": 1.4,
                            "bicycle": 1.4,
                            "person": 1.7,
                        }
                        est_h = H_obj_map.get(class_name, 1.6)
                        bbox_h = max(1, y2 - y1)
                        obj_dist = (est_h * self.focal_length) / bbox_h

                    world_pos = None
                    intent = "直行通过"
                    yaw_deg = 0.0
                    angle_cost = 0.0
                    if self.ipm:
                        world_pos = self.ipm.pixel_to_ground(cx, y2, (h, w))
                        if world_pos is not None:
                            self._update_world_track(track_id, world_pos)
                            intent, yaw_deg, angle_cost = self._predict_intent(track_id, world_pos)
                    
                    ttc, vx, vy, dw_dt, red_allowed, vw, is_static, risk_level, in_path = self.forward_calculator.update(

                        track_id,
                        width,
                        cx,
                        cy,
                        y2,
                        w,
                        h,
                        warn_line,
                        area_ratio,
                        global_vx,
                        global_vy,
                        use_ema=class_name in {"person", "bicycle", "motorcycle"},
                        distance=obj_dist,
                        v_self_mps=self.v_self_mps,
                        t_reaction=self.t_reaction,
                        d_safe=self.d_safe,
                    )

                    v_rel = None
                    sdt_violation = False
                    safe_dist = None
                    vx_abs = abs(vx)
                    if obj_dist is not None:

                        prev_dist = self._last_distance.get(track_id)
                        t_react_use = 0.8 if self.current_perspective == "侧面视角" else self.t_reaction

                        if prev_dist is not None:
                            v_rel = (prev_dist - obj_dist) * fps_value  # m/s，正值代表在接近
                            hist = self._vrel_history.get(track_id, [])
                            hist = (hist + [v_rel])[-6:]
                            self._vrel_history[track_id] = hist
                            continuous_closing = len(hist) >= 5 and all(v > 0 for v in hist[-5:])

                            safe_dist = v_rel * t_react_use + self.d_safe
                            lane_center_ok = (w * 0.35) <= cx <= (w * 0.65)
                            is_vehicle = class_name in {"car", "truck", "bus"}
                            v_rel_avg = sum(hist[-5:]) / 5.0 if len(hist) >= 5 else v_rel
                            sdt_gate = (lane_center_ok or is_vehicle) and continuous_closing and (v_rel_avg is not None and v_rel_avg > 0.5)

                            lateral_only = vx_abs > (abs(vw) + 1e-3) * 1.5
                            near_line = y2 > warning_line_y

                            if sdt_gate and near_line and not lateral_only and v_rel is not None and v_rel > 1.0 and ttc > 0 and obj_dist < safe_dist:
                                prev_center = self._last_centers.get(track_id)
                                cy_prev = prev_center[1] if prev_center else cy
                                if abs(cy - cy_prev) < 1.0 and cy < h * 0.4:
                                    sdt_violation = False
                                else:
                                    sdt_violation = True
                        self._last_distance[track_id] = obj_dist
                    else:
                        self._last_distance.pop(track_id, None)
                        self._vrel_history.pop(track_id, None)

                    safe_glance = False
                    if world_pos is not None:
                        conflict = self._in_conflict_envelope(track_id, world_pos)
                        if not conflict:
                            safe_glance = True
                    if safe_glance:
                        risk_level = 0

                    if angle_cost > 0.25 and risk_level > 0:
                        risk_level = max(0, risk_level - 1)
                    
                    if self._seen_counts.get(track_id, 0) < 5:
                        continue
                    if is_static:
                        continue

                    ratio = vx_abs / max(vw, 1e-3)
                    center_relaxed = (w * 0.35) <= cx <= (w * 0.65)
                    ratio_threshold = 0.9 if center_relaxed else 0.9

                    lateral_fast = ratio > ratio_threshold
                    red_ok = red_allowed and not lateral_fast

                    warn_ttc = 99.0 if safe_glance else ttc
                    if y2 <= warning_line_y:
                        warn_ttc = 99.0
                    elif class_name in {"person", "bicycle", "motorcycle"} and ttc < 2.0 and not red_ok:
                        warn_ttc = 2.0
                    elif ttc < 1.5 and not red_ok:
                        warn_ttc = 1.5

                    sdt_tag = False
                    if sdt_violation:
                        sdt_tag = True
                        warn_ttc = min(warn_ttc, 1.0)

                    if warn_ttc < min_ttc:
                        min_ttc = warn_ttc
                        min_id = track_id

                    # 前向视角显示参数
                    color, label, thickness = self._get_forward_display_params(
                        track_id, class_name, ttc, y2, warning_line_y, lateral_fast, red_ok
                    )
                    if risk_level == 2:
                        color = (0, 0, 255)
                        thickness = 3
                    elif risk_level == 1:
                        color = (0, 255, 255)
                        thickness = max(thickness, 2)

                    if sdt_tag:
                        if safe_dist is not None:
                            label = f"SDT {safe_dist:.1f}m"
                        else:
                            label = label.replace("TTC", "SDT") if "TTC" in label else f"SDT {label}"
                    if obj_dist is not None:
                        label += f" {obj_dist:.1f}m"
                    if v_rel is not None:
                        label += f" v={v_rel:.1f}m/s"
                    
                    self._draw_l_corners(frame, x1, y1, x2, y2, color, thickness=thickness, seg=18)
                    # 延迟绘制标签
                    deferred_draws.append((label, (x1, y1 - 35), 24, color))

                if min_ttc < 1.5:
                    if self._frame_count % 2 == 0:
                        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                    self.log_signal.emit(min_id, min_ttc)
                
                self.ttc_signal.emit(min_ttc, min_id)

            hud_payload = {
                "fps": self._fps_ema if self._fps_ema is not None else fps_value,
                "tracked": len(infos),
                "mode": f"{self.current_perspective} ({'Stereo' if self.stereo_mode else 'Mono'})",
            }
            self.hud_signal.emit(hud_payload)

            # 5. 批量执行中文绘制
            if deferred_draws:
                frame = self._draw_batch_chinese(frame, deferred_draws)
                deferred_draws = []

            # 1. Original View
            rgb_orig = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            h_orig, w_orig, ch_orig = rgb_orig.shape
            bytes_orig = ch_orig * w_orig
            qimage_orig = QtGui.QImage(rgb_orig.data, w_orig, h_orig, bytes_orig, QtGui.QImage.Format_RGB888)

            # 2. Pre-processed View
            h_pre, w_pre = pre_gray.shape[:2]
            bytes_pre = w_pre
            qimage_pre = QtGui.QImage(pre_gray.data, w_pre, h_pre, bytes_pre, QtGui.QImage.Format_Grayscale8)

            # 3. Inference View (Final frame)
            rgb_inf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_inf, w_inf, ch_inf = rgb_inf.shape
            bytes_inf = ch_inf * w_inf
            qimage_inf = QtGui.QImage(rgb_inf.data, w_inf, h_inf, bytes_inf, QtGui.QImage.Format_RGB888)

            # 发送三路画面
            self.frame_signal.emit(qimage_orig.copy(), qimage_pre.copy(), qimage_inf.copy())

            self.position_signal.emit(
                current_frame_idx,
                self.total_frames,
                current_frame_idx / fps_value if fps_value else 0.0,
                self.duration,
            )

            t_end = time.perf_counter()
            elapsed = t_end - t_start
            fps_live = 1.0 / max(elapsed, 1e-6)
            if self._fps_ema is None:
                self._fps_ema = fps_live
            else:
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps_live
            self.latency_signal.emit(elapsed * 1000.0)



        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_signal.emit("系统就绪")


    # --- 视角专用处理逻辑 (侧向与正向显示) ---
    def _process_side_ipm(self, frame, infos, w, h, fps_value, deferred_draws):
        """侧向视角：基于锁定后的车身边缘 + 0.8m 碰撞壁垒的报警处理"""
        
        # 1. 尝试获取或判定锁定的边缘位置
        if self._locked_side_mode is None or self._locked_car_x_ground is None:
            # 尚未锁定，进行实时探测
            side_mode = 'right' 
            car_x_ground = 0.0
            if self.ipm:
                roi_h_start = int(h * 0.45)
                roi_w_ext = int(w * 0.1)
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_h_start:, :]
                gray_l, gray_r = roi_gray[:, :roi_w_ext], roi_gray[:, -roi_w_ext:]
                
                # 简单边缘密度判定方位
                edges_l = cv2.Canny(gray_l, 50, 150)
                edges_r = cv2.Canny(gray_r, 50, 150)
                if np.sum(edges_l > 0) > np.sum(edges_r > 0) * 1.5:
                    side_mode = 'left'
                
                # 使用 Sobel 寻边以获得更准的垂直边界
                def find_car_edge_u(gray_block, is_left_side=True):
                    sobel_x = cv2.Sobel(gray_block, cv2.CV_16S, 1, 0, ksize=3)
                    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
                    _, binary = cv2.threshold(abs_sobel_x, 60, 255, cv2.THRESH_BINARY)
                    col_sum = np.sum(binary > 0, axis=0)
                    min_pixels = int(gray_block.shape[0] * 0.20)
                    if is_left_side:
                        for u in range(len(col_sum)-1, 0, -1):
                            if col_sum[u] > min_pixels: return u
                    else:
                        for u in range(0, len(col_sum)):
                            if col_sum[u] > min_pixels: return (w - roi_w_ext) + u
                    return None

                u_base = find_car_edge_u(gray_l if side_mode == 'left' else gray_r, side_mode == 'left')
                if u_base is not None:
                    pos = self.ipm.pixel_to_ground(u_base, int(h*0.8), (h, w))
                    if pos: car_x_ground = pos[0]
                else: car_x_ground = -1.2 if side_mode == 'left' else 1.2

            # 锁定计数递增
            if self._side_lock_count >= 5:
                self._locked_side_mode = side_mode
                self._locked_car_x_ground = car_x_ground
            else:
                self._side_lock_count += 1
            current_side_mode, current_car_x_ground = side_mode, car_x_ground
        else:
            current_side_mode, current_car_x_ground = self._locked_side_mode, self._locked_car_x_ground

        # 2. 基于方位设定物理墙
        side_mode, car_x_ground = current_side_mode, current_car_x_ground
        offset_sign = 0.8 if side_mode == 'left' else -0.8
        wall_dist = car_x_ground + offset_sign
        
        wall_samples = []
        car_edge_samples = []
        if self.ipm:
            all_wall_pts, car_pts = [], []
            for Y in np.linspace(-2.0, 10.0, 100):
                pt_wall = self.ipm.ground_to_pixel(wall_dist, Y, (h, w))
                pt_car = self.ipm.ground_to_pixel(car_x_ground, Y, (h, w))
                if pt_wall: all_wall_pts.append((int(pt_wall[0]), int(pt_wall[1])))
                if pt_car: car_pts.append((int(pt_car[0]), int(pt_car[1])))

            if len(all_wall_pts) >= 2 and len(car_pts) >= 2:
                p_car_sorted = sorted(car_pts, key=lambda p: p[1], reverse=True)
                p_wall_sorted = sorted(all_wall_pts, key=lambda p: p[1], reverse=True)
                fill_pts = np.array(p_car_sorted + p_wall_sorted[::-1], dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [fill_pts], (200, 180, 90))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                for i in range(len(p_car_sorted)-1):
                    cv2.line(frame, p_car_sorted[i], p_car_sorted[i+1], (0, 255, 100), 2, lineType=cv2.LINE_AA)

            wall_samples, car_edge_samples = all_wall_pts[::5], car_pts
            if len(all_wall_pts) >= 2:
                all_wall_pts.sort(key=lambda p: p[1], reverse=True)
                dash_len, gap_len = 35, 25
                cur_dist, is_draw = 0, True
                for i in range(len(all_wall_pts) - 1):
                    p1, p2 = all_wall_pts[i], all_wall_pts[i+1]
                    d = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    if is_draw: cv2.line(frame, p1, p2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    cur_dist += d
                    if is_draw and cur_dist >= dash_len: is_draw, cur_dist = False, 0
                    elif not is_draw and cur_dist >= gap_len: is_draw, cur_dist = True, 0

        def _interp_x(points, y):
            if not points: return None
            valid_pts = [(int(p[0]), int(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if not valid_pts: return None
            pts = sorted(valid_pts, key=lambda p: p[1])
            if y <= pts[0][1]: return pts[0][0]
            if y >= pts[-1][1]: return pts[-1][0]
            for i in range(len(pts) - 1):
                y0, y1 = pts[i][1], pts[i + 1][1]
                if y0 <= y <= y1:
                    r = (y-y0)/max(y1-y0, 1e-6)
                    return pts[i][0] + r*(pts[i+1][0]-pts[i][0])
            return None

        is_far_area = (lambda u: u > int(w*0.67)) if side_mode == 'left' else (lambda u: u < int(w*0.33))

        for track_id, x1, y1, x2, y2, cx, cy, class_name in infos:
            anchor_u = x1 if side_mode == 'right' else x2
            anchor_v = y2
            world_pos = self.ipm.pixel_to_ground(anchor_u, anchor_v, (h, w)) if self.ipm else None
            if world_pos:
                self._update_world_track(track_id, world_pos, max_len=20)
                dist_x = abs(world_pos[0])
                wall_x = _interp_x(wall_samples, anchor_v)
                car_x = _interp_x(car_edge_samples, anchor_v)
                in_pixel = False
                if wall_x is not None and not is_far_area(anchor_u):
                    if car_x is not None:
                        in_pixel = car_x <= anchor_u <= wall_x if side_mode == 'left' else wall_x <= anchor_u <= car_x
                    else: in_pixel = anchor_u <= wall_x if side_mode == 'right' else anchor_u >= wall_x

                dist_to_car = abs(dist_x - car_x_ground)
                in_world = dist_to_car < 0.8
                far_away = dist_to_car >= 1.2 or is_far_area(anchor_u)
                inside_wall = in_world or (in_pixel and not far_away)

                chinese_class = self.class_map.get(class_name, class_name)
                color, label, thickness = (0, 255, 0), chinese_class, 2
                blink = (self._frame_count % 4) < 2
                if far_away: label = f"安全 {dist_x:.2f}m"
                elif inside_wall:
                    color = (0, 0, 255) if (class_name not in {"person","bicycle","motorcycle"} or blink) else (0, 0, 120)
                    thickness = 3
                    label = f"碰撞预警 {dist_x:.2f}m"
                    self.side_warning_signal.emit('danger', label, track_id)
                else: label = f"侧方 {dist_x:.2f}m"

                self._draw_l_corners(frame, x1, y1, x2, y2, color, thickness=2, seg=18)
                deferred_draws.append((label, (x1, y1 - 35), 24, color))
        return frame

    def _get_forward_display_params(self, track_id, class_name, ttc, y2, warn_line, lateral_fast, red_ok):

        """获取前向视角显示参数（中文）"""
        chinese_class = self.class_map.get(class_name, class_name)
        
        if y2 <= warn_line:
            color = (120, 120, 120)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        elif class_name in {"bicycle", "motorcycle", "person"}:
            color = (255, 0, 255)
            label = f"{chinese_class} (注意)"
            thickness = 3
        elif lateral_fast and ttc < 3.0:
            color = (255, 0, 0)
            label = f"碰撞时间 {ttc:.1f}秒"
            thickness = 2
        elif class_name in {"person", "bicycle", "motorcycle"} and ttc < 2.0 and red_ok:
            color = (0, 0, 255)
            label = f"危险！碰撞时间 {ttc:.1f}秒"
            thickness = 3
        elif ttc < 1.5 and red_ok:
            color = (0, 0, 255)
            label = f"危险！碰撞时间 {ttc:.1f}秒"
            thickness = 3
        elif ttc < 3.0:
            color = (0, 255, 255)
            label = f"碰撞时间 {ttc:.1f}秒"
            thickness = 2
        else:
            color = (0, 255, 0)
            label = f"{chinese_class} {track_id}" if track_id >= 0 else chinese_class
            thickness = 2
        
        return color, label, thickness

    def stop(self):
        self._running = False
        self._user_paused = False
        self._seeking = False
        self.wait()

    def set_paused(self, paused: bool):
        self._user_paused = paused

    def start_seek(self):
        self._seeking = True

    def finish_seek(self):
        self._seeking = False

    def set_frame(self, index: int):
        self._seek_target = int(index)

    def set_preprocess_thresholds(self, weak_conf_threshold: float, edge_strength_threshold: float):
        self.weak_conf_threshold = float(weak_conf_threshold)
        self.edge_strength_threshold = float(edge_strength_threshold)


# ==================================================================================
