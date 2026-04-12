# === 1. 基础导入与环境信息 ===
# ==================================================================================
import os
import sys
import time
import math
import shutil
import logging

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

# 设置系统日志记录器
def setup_logger():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'ui_log_{timestamp}.log')
    
    logger = logging.getLogger(f'UI_{timestamp}')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, log_file

system_logger, system_log_file = setup_logger()
system_logger.info(f"UI系统启动，日志文件：{system_log_file}")

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

from core.video_thread import *
# === 6. UI 界面与交互逻辑 (PyQt5 组件与主窗口) ===
# ==================================================================================
class SplashScreen(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(800, 550)
        
        # Center on screen
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Background Frame with modern gradient and border
        self.bg_frame = QtWidgets.QFrame()
        self.bg_frame.setObjectName("splashBg")
        self.bg_frame.setStyleSheet("""
            QFrame#splashBg {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
                border-radius: 24px;
                border: 2px solid #4e4e91;
            }
        """)
        bg_layout = QtWidgets.QVBoxLayout(self.bg_frame)
        bg_layout.setContentsMargins(60, 60, 60, 60)
        bg_layout.setSpacing(15)

        # Icon / Logo Placeholder (Modern style)
        logo_layout = QtWidgets.QHBoxLayout()
        logo_label = QtWidgets.QLabel("🛡️")
        logo_label.setStyleSheet("font-size: 80px; background: transparent;")
        logo_layout.addStretch()
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()

        # Title
        self.title_label = QtWidgets.QLabel("FastGuard 智能监控")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 64px;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 4px;
            font-family: 'Outfit', 'Microsoft YaHei', sans-serif;
            background: transparent;
        """)
        
        self.subtitle_label = QtWidgets.QLabel("智能防碰撞预警引擎")
        self.subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("color: #a29bfe; font-size: 24px; letter-spacing: 8px; font-weight: 700; background: transparent; margin-top: 10px;")

        # Progress Section
        progress_container = QtWidgets.QWidget()
        progress_container.setStyleSheet("background: transparent;")
        progress_layout = QtWidgets.QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 50, 0, 0)
        progress_layout.setSpacing(20)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 10);
                border: none;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a29bfe, stop:1 #6c5ce7);
                border-radius: 6px;
            }
        """)

        self.status_label = QtWidgets.QLabel("正在初始化系统组件...")
        self.status_label.setStyleSheet("color: #b2bec3; font-family: 'Consolas', 'Microsoft YaHei', monospace; font-size: 16px; background: transparent;")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)

        bg_layout.addStretch()
        bg_layout.addLayout(logo_layout)
        bg_layout.addWidget(self.title_label)
        bg_layout.addWidget(self.subtitle_label)
        bg_layout.addWidget(progress_container)
        bg_layout.addStretch()

        layout.addWidget(self.bg_frame)

        # Shadow effect
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(50)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(15)
        self.shadow.setColor(QtGui.QColor(0, 0, 0, 200))
        self.bg_frame.setGraphicsEffect(self.shadow)

        # Animation state
        self.progress = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(25)

    def update_progress(self):
        self.progress += 1
        self.progress_bar.setValue(self.progress)
        
        if self.progress == 15: self.status_label.setText(">> 正在加载神经网络架构...")
        if self.progress == 35: self.status_label.setText(">> 正在同步摄像头数据流...")
        if self.progress == 55: self.status_label.setText(">> 正在校准空间传感器...")
        if self.progress == 75: self.status_label.setText(">> 正在优化张量核心...")
        if self.progress == 95: self.status_label.setText(">> 系统就绪，正在启动界面...")
        
        if self.progress >= 100:
            self.timer.stop()
            self.fade_out()

    def fade_out(self):
        self.animation = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.finished.connect(self.on_fade_finished)
        self.animation.start()

    def on_fade_finished(self):
        self.close()
        self.finished.emit()

class StatCard(QtWidgets.QFrame):
    def __init__(self, title, value, unit, icon_text="📊", color="#6366f1", parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        # Modern "Bento Grid" style
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: #18181b; /* Zinc-900 */
                border: 1px solid #27272a; /* Zinc-800 */
                border-radius: 16px;
            }}
            QFrame#statCard:hover {{
                border: 1px solid {color};
                background-color: #27272a;
            }}
            QLabel {{ border: none; background: transparent; }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Header (Icon + Title)
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(16)

        icon_label = QtWidgets.QLabel(icon_text)
        # Circular icon background
        icon_label.setStyleSheet(f"font-size: 42px; color: {color}; background: {color}20; border-radius: 12px; padding: 8px;")
        icon_label.setFixedSize(52, 52)
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-size: 27px; color: #a1a1aa; font-weight: 600; letter-spacing: 1px;")
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Value Area
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.setSpacing(10)
        value_layout.setContentsMargins(0, 12, 0, 0)

        self.value_label = QtWidgets.QLabel(value)
        self.value_label.setStyleSheet("font-size: 63px; color: #ffffff; font-weight: 700; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;")
        
        self.unit_label = QtWidgets.QLabel(unit)
        self.unit_label.setStyleSheet("font-size: 24px; color: #71717a; font-weight: 600; padding-bottom: 8px;")
        
        value_layout.addWidget(self.value_label)
        value_layout.addWidget(self.unit_label, alignment=QtCore.Qt.AlignBottom)
        value_layout.addStretch()

        layout.addLayout(header_layout)
        layout.addLayout(value_layout)

    def update_value(self, value, unit=None):
        self.value_label.setText(str(value))
        if unit:
            self.unit_label.setText(str(unit))


class LogWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统日志")
        self.resize(600, 700) # Larger window
        self.setStyleSheet("""
            QDialog { background-color: #09090b; }
            QLabel { color: white; font-size: 27px; font-weight: bold; font-family: 'Microsoft YaHei'; }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("📋 系统事件记录"))
        
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget {
                background: #18181b;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                color: #a1a1aa;
                font-family: 'Consolas', 'Microsoft YaHei Mono', monospace;
                font-size: 21px; /* Larger log font */
                outline: none;
            }
            QListWidget::item { padding: 6px; }
            QListWidget::item:selected { background: #27272a; color: white; }
        """)
        layout.addWidget(self.log_list)
        
        btn_clear = QtWidgets.QPushButton("清空记录")
        btn_clear.clicked.connect(self.log_list.clear)
        btn_clear.setStyleSheet("""
            QPushButton {
                background: #27272a; color: white; border: 1px solid #3f3f46;
                border-radius: 6px; padding: 10px 16px; font-size: 24px; font-family: 'Microsoft YaHei';
            }
            QPushButton:hover { background: #3f3f46; }
        """)
        layout.addWidget(btn_clear)

    def append_log(self, text):
        self.log_list.addItem(text)
        self.log_list.scrollToBottom()


class SettingsWindow(QtWidgets.QDialog):
    preprocess_changed = QtCore.pyqtSignal(float, float)

    def __init__(self, weak_conf, edge_strength, parent=None):
        super().__init__(parent)
        self.setWindowTitle("预处理参数设置")
        self.resize(500, 350)
        self.setStyleSheet("""
            QDialog { background-color: #09090b; }
            QLabel { color: #e5e7eb; font-weight: 600; font-size: 24px; font-family: 'Microsoft YaHei'; }
            QDoubleSpinBox {
                background: #111111;
                color: #e5e7eb;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 6px 10px;
                min-width: 150px;
                font-size: 36px;
            }
            QPushButton {
                background: #27272a; color: white;
                border: 1px solid #3f3f46; border-radius: 8px;
                padding: 10px 16px; font-size: 24px; font-family: 'Microsoft YaHei';
            }
            QPushButton:hover { background: #3f3f46; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(40, 40, 40, 40)
        
        form = QtWidgets.QFormLayout()
        form.setSpacing(20)
        
        self.spin_weak_conf = QtWidgets.QDoubleSpinBox()
        self.spin_weak_conf.setRange(0.10, 0.95)
        self.spin_weak_conf.setDecimals(2)
        self.spin_weak_conf.setSingleStep(0.01)
        self.spin_weak_conf.setValue(weak_conf)
        
        self.spin_edge_strength = QtWidgets.QDoubleSpinBox()
        self.spin_edge_strength.setRange(1.0, 255.0)
        self.spin_edge_strength.setDecimals(1)
        self.spin_edge_strength.setSingleStep(1.0)
        self.spin_edge_strength.setValue(edge_strength)
        
        form.addRow("低置信度阈值:", self.spin_weak_conf)
        form.addRow("边缘强度阈值:", self.spin_edge_strength)
        layout.addLayout(form)
        
        self.spin_weak_conf.valueChanged.connect(self.emit_change)
        self.spin_edge_strength.valueChanged.connect(self.emit_change)
        
        self.reset_btn = QtWidgets.QPushButton("恢复默认值")
        self.reset_btn.clicked.connect(self.reset_defaults)
        layout.addWidget(self.reset_btn)
        layout.addStretch()

    def emit_change(self):
        self.preprocess_changed.emit(self.spin_weak_conf.value(), self.spin_edge_strength.value())

    def reset_defaults(self):
        self.spin_weak_conf.setValue(0.38)
        self.spin_edge_strength.setValue(28.0)

    # --- 主窗口核心逻辑 ---
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastGuard 智能监控系统")
        self.resize(1400, 850)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        
        # Data & State
        self.thread = None
        self.last_images = None
        self.last_log_time = 0.0

        self.warning_records = []
        self.manual_perspective_set = False
        self.debug_info = {}
        self.side_warning_active = False
        self.side_warning_timer = QtCore.QTimer()
        self.side_warning_timer.timeout.connect(self.clear_side_warning)
        self.hud_info = {"fps": 0.0, "tracked": 0, "mode": "初始化"}
        self.model_name = "yolo11n.pt"
        self.last_latency = None
        self.default_weak_conf_threshold = 0.38
        self.default_edge_strength_threshold = 28.0
        self.weak_conf_threshold = self.default_weak_conf_threshold
        self.edge_strength_threshold = self.default_edge_strength_threshold
        self.total_duration = 0.0
        self.total_frames = 0
        self.log_window = LogWindow(self)
        self.settings_window = SettingsWindow(self.weak_conf_threshold, self.edge_strength_threshold, self)
        self.settings_window.preprocess_changed.connect(self.update_preprocess_from_dialog)

        self.setup_ui()
        self.apply_modern_theme()
        
        # Connect Actions
        self.setup_connections()


    def setup_ui(self):
        # --- Main Layout ---
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === Left Sidebar ===
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(280)  # Wider for larger Chinese text
        
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(24, 40, 24, 40)
        sidebar_layout.setSpacing(18)

        # Logo / Title Area
        app_logo_layout = QtWidgets.QHBoxLayout()
        # Use a geometric shape for a more sci-fi look
        logo_icon = QtWidgets.QLabel("❖") 
        logo_icon.setStyleSheet("font-size: 48px; color: #6366f1; background: transparent;")
        logo_text = QtWidgets.QLabel("FASTGUARD")
        logo_text.setStyleSheet("""
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            font-size: 39px; 
            font-weight: 800; 
            color: #ffffff; 
            letter-spacing: 2px;
            background: transparent;
        """)
        app_logo_layout.addWidget(logo_icon)
        app_logo_layout.addWidget(logo_text)
        app_logo_layout.addStretch()
        
        sidebar_layout.addLayout(app_logo_layout)
        sidebar_layout.addSpacing(50)

        # Menu Group: MAIN
        lbl_main = QtWidgets.QLabel("主菜单")
        lbl_main.setStyleSheet("color: #71717a; font-size: 21px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-family: 'Microsoft YaHei';")
        sidebar_layout.addWidget(lbl_main)

        def create_nav_btn(icon, text, tooltip, is_active=False):
            # Using specific spacing in text for alignment
            btn = QtWidgets.QPushButton(f" {icon}    {text}")
            btn.setObjectName("navBtn")
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            if is_active:
                btn.setChecked(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            sidebar_layout.addWidget(btn)
            return btn

        # Updated icons for a more premium/tech feel
        self.btn_camera = create_nav_btn("⦿", "开启摄像头", "实时监控画面", is_active=True)
        self.btn_open = create_nav_btn("📊", "导入视频", "历史视频分析")
        
        sidebar_layout.addSpacing(30)
        
        # Menu Group: TOOLS
        lbl_tools = QtWidgets.QLabel("工具")
        lbl_tools.setStyleSheet("color: #71717a; font-size: 21px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-family: 'Microsoft YaHei';")
        sidebar_layout.addWidget(lbl_tools)
        
        self.btn_log = create_nav_btn("📟", "系统日志", "查看运行日志")
        self.btn_settings = create_nav_btn("⚙", "参数设置", "调整检测参数")
        
        sidebar_layout.addStretch()
        
        # Menu Group: SYSTEM
        lbl_system = QtWidgets.QLabel("系统")
        lbl_system.setStyleSheet("color: #71717a; font-size: 21px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-family: 'Microsoft YaHei';")
        sidebar_layout.addWidget(lbl_system)
        
        self.btn_help = create_nav_btn("?", "使用帮助", "用户指南")
        self.btn_exit = create_nav_btn("⏻", "退出系统", "关闭程序")
        
        # Enhanced Sidebar Styles
        self.sidebar.setStyleSheet("""
            QFrame#sidebar {
                background-color: #09090b; /* Zinc-950 */
                border-right: 1px solid #27272a; /* Zinc-800 */
            }
            QPushButton#navBtn {
                background: transparent;
                border: none;
                border-radius: 8px;
                color: #a1a1aa; /* Zinc-400 */
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 36px;
                font-weight: 500;
                text-align: left;
                padding-left: 20px;
                padding-top: 15px;
                padding-bottom: 15px;
                height: 80px;
            }
            QPushButton#navBtn:hover {
                background-color: #18181b; /* Zinc-900 */
                color: #f4f4f5; /* Zinc-100 */
            }
            QPushButton#navBtn:checked {
                background-color: #18181b;
                color: #ffffff;
                border-left: 4px solid #6366f1; /* Indigo-500 */
                padding-left: 16px; /* Adjust for border width to keep text stable */
            }
        """)

        main_layout.addWidget(self.sidebar)

        # === Content Area ===
        content_widget = QtWidgets.QWidget()
        content_widget.setStyleSheet("background-color: #09090b;") # Ensure background matches sidebar
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(32)

        # Header Area
        header_container = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title with accent
        title_box = QtWidgets.QVBoxLayout()
        app_title = QtWidgets.QLabel("开启摄像头")
        app_title.setStyleSheet("font-size: 63px; font-weight: 800; color: #ffffff; letter-spacing: 2px; font-family: 'Microsoft YaHei';")
        app_subtitle = QtWidgets.QLabel("实时智能监控系统")
        app_subtitle.setStyleSheet("font-size: 30px; font-weight: 500; color: #71717a; letter-spacing: 1px; font-family: 'Microsoft YaHei'; margin-top: 4px;")
        title_box.addWidget(app_title)
        title_box.addWidget(app_subtitle)
        
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        
        # System Status Indicator (Top Right)
        status_badge = QtWidgets.QLabel("  ● 系统在线  ")
        status_badge.setStyleSheet("""
            background-color: #064e3b; /* Emerald-900 */
            color: #34d399; /* Emerald-400 */
            border: 1px solid #059669; /* Emerald-600 */
            border-radius: 16px;
            padding: 8px 16px;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 1px;
            font-family: 'Microsoft YaHei';
        """)
        header_layout.addWidget(status_badge)
        
        content_layout.addWidget(header_container)

        # Main Grid: Videos (Left) + Stats/Controls (Right)
        main_split = QtWidgets.QHBoxLayout()
        main_split.setSpacing(24)

        # --- Left Column: Video Feeds ---
        self.views_container = QtWidgets.QWidget()
        views_layout = QtWidgets.QVBoxLayout(self.views_container)
        views_layout.setSpacing(16)
        views_layout.setContentsMargins(0, 0, 0, 0)

        def create_view_frame(title, color_accent="#3f3f46"):
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(f"""
                QFrame {{
                    background-color: #18181b; /* Zinc-900 */
                    border: 1px solid #27272a;
                    border-radius: 12px;
                }}
            """)
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # View Header (Mac-style or Tech-style)
            view_header = QtWidgets.QFrame()
            view_header.setFixedHeight(48)
            view_header.setStyleSheet("""
                background-color: #27272a;
                border-bottom: 1px solid #3f3f46;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
            """)
            vh_layout = QtWidgets.QHBoxLayout(view_header)
            vh_layout.setContentsMargins(20, 0, 20, 0)
            
            # Title
            lbl_title = QtWidgets.QLabel(title)
            lbl_title.setStyleSheet("color: #e4e4e7; font-weight: 600; font-size: 24px; border: none; background: transparent; font-family: 'Microsoft YaHei';")
            
            # Live Indicator
            lbl_live = QtWidgets.QLabel("● 实时")
            lbl_live.setStyleSheet("color: #ef4444; font-weight: 700; font-size: 21px; border: none; background: transparent; letter-spacing: 1px; font-family: 'Microsoft YaHei';")
            
            vh_layout.addWidget(lbl_title)
            vh_layout.addStretch()
            vh_layout.addWidget(lbl_live)
            
            layout.addWidget(view_header)

            # Video Container
            container = QtWidgets.QWidget()
            # Ensure background is black for video
            container.setStyleSheet("background-color: #000000; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;") 
            
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_img = QtWidgets.QLabel()
            lbl_img.setAlignment(QtCore.Qt.AlignCenter)
            lbl_img.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
            lbl_img.setScaledContents(False) # Keep aspect ratio logic in update_frame
            # Placeholder text style
            lbl_img.setText("无信号")
            lbl_img.setStyleSheet("color: #52525b; font-weight: 600; font-family: 'Microsoft YaHei', sans-serif; font-size: 36px; border: none;")

            container_layout.addWidget(lbl_img)
            layout.addWidget(container)
            
            return frame, lbl_img

        self.frame_orig, self.label_orig = create_view_frame("原始画面 // 摄像头 01")
        self.frame_pre, self.label_pre = create_view_frame("预处理 // 图像增强")
        self.frame_inf, self.label_inf = create_view_frame("AI 推理 // 目标检测", "#6366f1")
        self.frame_bev, self.label_bev = create_view_frame("IPM 俯视图 // BEV", "#10b981") # 绿色强调 BEV

        # Layout: Top Row (Split) + Bottom Row (Split: Inference + BEV)
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(16)
        row1_layout.addWidget(self.frame_orig)
        row1_layout.addWidget(self.frame_pre)

        row2_layout = QtWidgets.QHBoxLayout()
        row2_layout.setSpacing(16)
        row2_layout.addWidget(self.frame_inf, stretch=7)
        row2_layout.addWidget(self.frame_bev, stretch=3) # BEV 占比较窄

        views_layout.addLayout(row1_layout, stretch=4)
        views_layout.addLayout(row2_layout, stretch=6)

        main_split.addWidget(self.views_container, stretch=3)

        # --- Right Column: Stats & Controls ---
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setFixedWidth(360) # Wider for new card style and larger text
        right_column = QtWidgets.QVBoxLayout(self.right_widget)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(24)

        # Metrics Section
        lbl_metrics = QtWidgets.QLabel("核心指标")
        lbl_metrics.setStyleSheet("color: #71717a; font-size: 24px; font-weight: 700; letter-spacing: 2px; font-family: 'Microsoft YaHei';")
        right_column.addWidget(lbl_metrics)

        self.card_fps = StatCard("系统帧率", "0.0", "赫兹", "⚡", "#10b981") # Emerald
        self.card_objects = StatCard("活跃目标", "0", "个", "🎯", "#3b82f6") # Blue
        self.card_risk = StatCard("威胁等级", "安全", "状态", "🛡️", "#8b5cf6") # Violet

        right_column.addWidget(self.card_fps)
        right_column.addWidget(self.card_objects)
        right_column.addWidget(self.card_risk)
        
        right_column.addSpacing(16)

        # Controls Section
        lbl_controls = QtWidgets.QLabel("控制面板")
        lbl_controls.setStyleSheet("color: #71717a; font-size: 24px; font-weight: 700; letter-spacing: 2px; font-family: 'Microsoft YaHei';")
        right_column.addWidget(lbl_controls)

        controls_frame = QtWidgets.QFrame()
        controls_frame.setObjectName("controlsFrame")
        controls_frame.setStyleSheet("""
            QFrame#controlsFrame {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 16px;
            }
        """)
        controls_layout = QtWidgets.QVBoxLayout(controls_frame)
        controls_layout.setSpacing(20)
        controls_layout.setContentsMargins(24, 24, 24, 24)

        # Time Display
        time_container = QtWidgets.QHBoxLayout()
        icon_time = QtWidgets.QLabel("⏱")
        icon_time.setStyleSheet("color: #71717a; font-size: 30px; border: none; background: transparent;")
        
        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        self.time_label.setAlignment(QtCore.Qt.AlignRight)
        self.time_label.setStyleSheet("color: #e4e4e7; font-family: 'Consolas', monospace; font-size: 36px; font-weight: 600; border: none; background: transparent;")
        
        time_container.addWidget(icon_time)
        time_container.addStretch()
        time_container.addWidget(self.time_label)
        controls_layout.addLayout(time_container)

        # Progress Bar (Slider)
        self.progress_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        # Custom Slider Style
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3f3f46;
                height: 8px;
                background: #27272a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 1px solid #6366f1;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #818cf8;
            }
        """)
        controls_layout.addWidget(self.progress_slider)

        # Playback Buttons
        btns_row = QtWidgets.QHBoxLayout()
        btns_row.setSpacing(16)
        
        def create_ctrl_btn(text, tooltip, primary=False):
            btn = QtWidgets.QPushButton(text)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            if primary:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #6366f1; 
                        color: white; 
                        border: none; 
                        border-radius: 8px; 
                        padding: 12px; 
                        font-weight: bold; 
                        font-size: 30px;
                    }
                    QPushButton:hover { background-color: #4f46e5; }
                    QPushButton:checked { background-color: #f59e0b; }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #27272a; 
                        color: #e4e4e7; 
                        border: 1px solid #3f3f46; 
                        border-radius: 8px; 
                        padding: 12px; 
                        font-size: 30px;
                    }
                    QPushButton:hover { background-color: #3f3f46; }
                """)
            return btn

        self.pause_btn = create_ctrl_btn("⏯", "播放/暂停", primary=True)
        self.pause_btn.setCheckable(True)
        
        self.stop_btn = create_ctrl_btn("⏹", "停止")

        btns_row.addWidget(self.pause_btn, stretch=2)
        btns_row.addWidget(self.stop_btn, stretch=1)
        controls_layout.addLayout(btns_row)
        
        right_column.addWidget(controls_frame)
        right_column.addStretch()

        main_split.addWidget(self.right_widget)
        content_layout.addLayout(main_split)
        main_layout.addWidget(content_widget)

        # Footer
        self.footer_label = QtWidgets.QLabel("系统就绪，等待输入源...")
        self.footer_label.setStyleSheet("color: #52525b; font-size: 21px; margin-top: 8px; font-family: 'Microsoft YaHei', sans-serif;")
        self.footer_label.setAlignment(QtCore.Qt.AlignRight)
        content_layout.addWidget(self.footer_label)


    def setup_connections(self):
        self.btn_open.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_exit.clicked.connect(self.close)
        self.btn_log.clicked.connect(self.log_window.show)
        self.btn_settings.clicked.connect(self.settings_window.show)
        self.btn_help.clicked.connect(self.show_help_dialog)

        self.pause_btn.toggled.connect(self.toggle_pause)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderMoved.connect(self.on_slider_moved)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.stop_btn.clicked.connect(self.stop_camera)

    def show_help_dialog(self):
        if getattr(self, "help_dialog", None):
            self.help_dialog.show()
            self.help_dialog.raise_()
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        dialog.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        dialog.setModal(True) # 模态对话框，点击外部不关闭，需要点关闭按钮

        card = QtWidgets.QFrame()
        card.setObjectName("helpCard")
        # 优化样式：更深色的背景，微光边框，增加阴影感
        card.setStyleSheet("""
            QFrame#helpCard {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #18181b, stop:1 #09090b);
                border: 1px solid #27272a;
                border-radius: 20px;
                color: #e5e7eb;
            }
            QLabel#helpTitle { 
                font-size: 36px; 
                font-weight: bold; 
                color: #ffffff; 
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#helpSubtitle { 
                font-size: 21px; 
                letter-spacing: 1px; 
                color: #60a5fa; 
                font-weight: 600; 
                margin-bottom: 10px;
            }
            QLabel#helpBody { 
                font-size: 23px;23px;23px;23px; 
                line-height: 1.8; 
                color: #d4d4d8;
                padding: 10px;
            }
            QPushButton#helpClose { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #3b82f6);
                border: none; 
                border-radius: 8px; 
                color: white; 
                padding: 10px 24px; 
                font-size: 21px;
                font-weight: 600; 
            }
            QPushButton#helpClose:hover { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #2563eb);
            }
            QPushButton#helpClose:pressed {
                background: #1e40af;
            }
            /* 分割线样式 */
            QFrame#hLine {
                background-color: #3f3f46;
                max-height: 1px;
                border: none;
            }
        """)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(15)

        # Title Section
        title_layout = QtWidgets.QHBoxLayout()
        icon_label = QtWidgets.QLabel("💡")
        icon_label.setStyleSheet("font-size: 28px; background: transparent;")
        title = QtWidgets.QLabel("FastGuard 功能指南")
        title.setObjectName("helpTitle")
        title_layout.addWidget(icon_label)
        title_layout.addSpacing(10)
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        subtitle = QtWidgets.QLabel("INTELLIGENT MONITORING SYSTEM GUIDE")
        subtitle.setObjectName("helpSubtitle")

        # Separator
        line = QtWidgets.QFrame()
        line.setObjectName("hLine")
        line.setFrameShape(QtWidgets.QFrame.HLine)

        # Body Content with HTML for better formatting
        body = QtWidgets.QLabel()
        body.setObjectName("helpBody")
        body.setTextFormat(QtCore.Qt.RichText)
        body.setText("""
            <style>
                ul { margin-left: -20px; }
                li { margin-bottom: 8px; }
                b { color: #60a5fa; }
            </style>
            <ul>
                <li><b>📹 三路视频：</b> 原始 / 预处理 / 推理结果，实时对比分析</li>
                <li><b>⚠️ 预警机制：</b> TTC 碰撞预警、侧向盲区警报、开门危险提示</li>
                <li><b>🎮 控制中心：</b> 支持开启摄像头/视频文件，回放进度拖拽与暂停</li>
                <li><b>📊 数据面板：</b> 实时显示帧率 (FPS)、活跃目标数及当前风险等级</li>
                <li><b>⚙️ 参数微调：</b> 自定义弱检测阈值与边缘增强强度，适应不同环境</li>
                <li><b>📜 系统日志：</b> 记录并查看所有历史警报与系统运行调试信息</li>
            </ul>
        """)
        body.setWordWrap(True)
        body.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Close Button Area
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        
        close_btn = QtWidgets.QPushButton("我已了解")
        close_btn.setObjectName("helpClose")
        close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        close_btn.clicked.connect(dialog.close)
        
        btn_layout.addWidget(close_btn)

        card_layout.addLayout(title_layout)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(line)
        card_layout.addSpacing(10)
        card_layout.addWidget(body)
        card_layout.addSpacing(20)
        card_layout.addLayout(btn_layout)

        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        # Add shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QtGui.QColor(0, 0, 0, 150))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)
        
        dialog_layout.addWidget(card)

        # Resize and Center
        dialog.setFixedWidth(550)
        dialog.adjustSize()
        
        # Center on parent
        if self.isVisible():
             geo = self.geometry()
             x = geo.x() + (geo.width() - dialog.width()) // 2
             y = geo.y() + (geo.height() - dialog.height()) // 2
             dialog.move(x, y)
        else:
             # Center on screen if parent not visible (fallback)
             screen = QtWidgets.QApplication.primaryScreen().geometry()
             dialog.move((screen.width() - dialog.width()) // 2, (screen.height() - dialog.height()) // 2)

        dialog.finished.connect(lambda _: setattr(self, "help_dialog", None))
        self.help_dialog = dialog
        dialog.show()

    def update_preprocess_from_dialog(self, weak_conf, edge_strength):
        # 从设置弹窗同步阈值
        self.weak_conf_threshold = float(weak_conf)
        self.edge_strength_threshold = float(edge_strength)
        self.apply_preprocess_params()

    def apply_preprocess_params(self, _value=None):
        if self.thread:
            self.thread.set_preprocess_thresholds(self.weak_conf_threshold, self.edge_strength_threshold)

    def reset_preprocess_defaults(self):
        self.weak_conf_threshold = self.default_weak_conf_threshold
        self.edge_strength_threshold = self.default_edge_strength_threshold
        # 同步到设置弹窗
        if hasattr(self, "settings_window"):
            self.settings_window.spin_weak_conf.setValue(self.default_weak_conf_threshold)
            self.settings_window.spin_edge_strength.setValue(self.default_edge_strength_threshold)
        self.apply_preprocess_params()



    def apply_modern_theme(self):
        # Global Application Theme
        # Note: Specific widget styles (like Sidebar buttons) are handled in setup_ui
        self.setStyleSheet("""
            QWidget {
                background-color: #09090b; /* Zinc-950 */
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                color: #e4e4e7; /* Zinc-200 */
            }
            
            /* Global Scrollbar Style */
            QScrollBar:vertical {
                border: none;
                background: #18181b;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3f3f46;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #18181b;
                height: 8px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #3f3f46;
                min-width: 20px;
                border-radius: 4px;
            }
            
            /* Global Menu Style */
            QMenu {
                background-color: #18181b;
                color: #e4e4e7;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #27272a;
                color: white;
            }
            
            /* Global Tooltip */
            QToolTip {
                background-color: #27272a;
                color: #ffffff;
                border: 1px solid #3f3f46;
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 12px;
            }
            
            /* Global Message Box */
            QMessageBox {
                background-color: #18181b;
            }
            QMessageBox QLabel {
                color: #e4e4e7;
            }
            QMessageBox QPushButton {
                background-color: #27272a;
                color: #e4e4e7;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 6px 16px;
            }
            QMessageBox QPushButton:hover {
                background-color: #3f3f46;
            }
        """)

    # --- Logic Methods (Adapted from old MainWindow) ---

    def open_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.start_thread(file_path)

    def open_camera(self):
        self.start_thread(0)

    def stop_camera(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        
        for lbl in [self.label_orig, self.label_pre, self.label_inf, self.label_bev]:
            lbl.clear()
            lbl.setText("无信号")
            lbl.setStyleSheet("color: #52525b; font-weight: 600; font-family: 'Microsoft YaHei', sans-serif; font-size: 24px; border: none;")
            
        self.card_risk.update_value("离线", "系统空闲")
        self.card_fps.update_value("0.0", "赫兹")
        self.card_objects.update_value("0", "个")
        self.append_system_log("设备已安全断开")


    def start_thread(self, source):
        if self.thread is not None:
            self.thread.stop()

        self.reset_playback_controls()
        self.total_duration = 0.0
        self.total_frames = 0
        self.log_window.log_list.clear()

        model_path = os.path.join(".", "assets", "weights", "yolo11n.pt")

        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.information(self, "下载", "正在下载 yolo11n.pt...")
        
        self.thread = VideoThread(
            source,
            model_path,
            self,
            weak_conf_threshold=self.weak_conf_threshold,
            edge_strength_threshold=self.edge_strength_threshold,
        )
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.status_signal.connect(self.append_system_log)
        self.thread.ttc_signal.connect(self.update_ttc)
        self.thread.side_warning_signal.connect(self.update_side_warning)
        self.thread.log_signal.connect(self.append_log)
        self.thread.latency_signal.connect(self.update_latency)
        self.thread.model_signal.connect(self.update_model_name)
        self.thread.perspective_signal.connect(self.update_perspective)
        self.thread.debug_signal.connect(self.update_debug_info)
        self.thread.position_signal.connect(self.update_position)
        self.thread.hud_signal.connect(self.update_hud)
        self.thread.start()
        
        self.card_risk.update_value("扫描中", "初始化...")
        self.reset_card_style(self.card_risk)

    def reset_playback_controls(self):
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏯")
        self.progress_slider.setEnabled(False)
        self.progress_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")

    def format_time(self, seconds):
        if seconds is None or seconds < 0: return "00:00"
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def update_position(self, frame_index, total_frames, current_seconds, total_seconds):
        self.total_frames = total_frames
        self.total_duration = total_seconds
        self.current_frame_idx = frame_index
        if total_frames > 0:
            self.progress_slider.setEnabled(True)
            self.progress_slider.setRange(0, max(total_frames - 1, 0))
            if not self.progress_slider.isSliderDown():
                self.progress_slider.setValue(frame_index)
        else:
            self.progress_slider.setEnabled(False)
        self.time_label.setText(f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}")

    def toggle_pause(self, checked):
        if not self.thread:
            self.pause_btn.setChecked(False)
            return
        if checked:
            self.pause_btn.setText("▶")
            self.thread.set_paused(True)
        else:
            # 如果当前已播放到末尾，重新点击播放则从头开始
            if hasattr(self, 'current_frame_idx') and self.total_frames > 0:
                if self.current_frame_idx >= self.total_frames - 1:
                    self.thread.set_frame(0)
            self.pause_btn.setText("⏸")
            self.thread.set_paused(False)

    def on_slider_pressed(self):
        if self.thread: self.thread.start_seek()

    def on_slider_moved(self, value):
        if self.thread: self.thread.set_frame(value)

    def on_slider_released(self):
        if self.thread:
            self.thread.set_frame(self.progress_slider.value())
            self.thread.finish_seek()
            if not self.pause_btn.isChecked():
                self.thread.set_paused(False)

    def reset_card_style(self, card):
        card.setStyleSheet("""
            QFrame#statCard {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 16px;
            }
            QFrame#statCard:hover {
                border: 1px solid #6366f1;
                background-color: #27272a;
            }
            QLabel { border: none; background: transparent; }
        """)

    def update_ttc(self, ttc, track_id):
        # Update risk card based on TTC
        if ttc < 1.5:
            self.card_risk.update_value("危险", f"即将碰撞 ({ttc:.1f}s)")
            self.card_risk.setStyleSheet("""
                QFrame#statCard { background: #450a0a; border: 1px solid #dc2626; border-radius: 16px; } 
                QLabel {background: transparent;}
            """)
        elif ttc < 3.0:
            self.card_risk.update_value("警告", f"正在接近 ({ttc:.1f}s)")
            self.card_risk.setStyleSheet("""
                QFrame#statCard { background: #431407; border: 1px solid #d97706; border-radius: 16px; } 
                QLabel {background: transparent;}
            """)
        else:
            self.card_risk.update_value("安全", "安全距离")
            self.reset_card_style(self.card_risk)

    def update_side_warning(self, level, message, object_id):
        self.side_warning_timer.start(3000)
        timestamp = time.strftime("%H:%M:%S")

        prefix = "⚠️ "
        if level == 'danger': prefix = "🚨 "

        log_msg = f"{prefix} [{timestamp}] {message}"
        self.log_window.append_log(log_msg)
        system_logger.info(log_msg)
        
        self.card_risk.update_value("侧向预警", message)
        self.card_risk.setStyleSheet("""
            QFrame#statCard { background: #431407; border: 1px solid #d97706; border-radius: 16px; } 
            QLabel {background: transparent;}
        """)

    def clear_side_warning(self):
        self.side_warning_timer.stop()
        self.card_risk.update_value("安全", "安全距离")
        self.reset_card_style(self.card_risk)

    def append_log(self, track_id, ttc):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"⚡ [{timestamp}] ID:{track_id} TTC:{ttc:.1f}s"
        self.log_window.append_log(log_msg)
        system_logger.info(log_msg)

    def append_system_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"ℹ️ [{timestamp}] {message}"
        self.log_window.append_log(log_msg)
        system_logger.info(log_msg)

    def update_model_name(self, name):

        self.model_name = name
        self.update_footer()

    def update_latency(self, ms):
        self.last_latency = ms
        self.update_footer()
        
    def update_footer(self):
        latency_str = f"{self.last_latency:.1f} ms" if self.last_latency else "-- ms"
        self.footer_label.setText(f"模型: {self.model_name} | 延迟: {latency_str}")

    def update_perspective(self, perspective_text):
        if not self.manual_perspective_set:
            self.append_system_log(f"当前视角: {perspective_text}")

    def set_manual_perspective(self, p_type):
        self.manual_perspective_set = True
        self.append_system_log(f"手动视角: {p_type}")
        # Need a way to inform thread if required, but logic was:
        # thread analyzes -> emits perspective -> UI updates
        # If manual, we just ignore thread emits.
        # But thread also switches detection logic based on its internal perspective.
        # So we really should tell the thread to lock perspective.
        # For now, we rely on the visual override. 
        # (Improvement: Add set_perspective to VideoThread)

    def reset_perspective_analysis(self):
        self.manual_perspective_set = False
        self.append_system_log("视角已重置")
        if self.thread:
            self.thread.perspective_locked = False
            self.thread.current_perspective = "分析中..."
            
    def toggle_debug_mode(self, checked):
        if self.thread:
            self.thread.perspective_debug = checked
        if checked:
            self.append_system_log("调试模式已开启")
        else:
            self.append_system_log("调试模式已关闭")

    def update_debug_info(self, debug_info):
        # Since we removed the large debug label, we can print to console or log occasionally
        # or maybe update a tooltip. For now, we'll just ignore or log if critical.
        pass

    def update_hud(self, payload):
        self.hud_info = payload
        fps = payload.get("fps", 0)
        tracked = payload.get("tracked", 0)
        
        self.card_fps.update_value(f"{fps:.1f}", "赫兹")
        self.card_objects.update_value(str(tracked), "个")

    def update_frame(self, img_orig, img_pre, img_inf, img_bev):
        self.last_images = (img_orig, img_pre, img_inf, img_bev)
        self.render_frames()

    def render_frames(self):
        if not hasattr(self, 'last_images') or not self.last_images: return
        
        imgs = self.last_images
        labels = [self.label_orig, self.label_pre, self.label_inf, self.label_bev]
        
        for img, lbl in zip(imgs, labels):
            if img.isNull() or lbl.width() <= 0 or lbl.height() <= 0: continue
            
            pixmap = QtGui.QPixmap.fromImage(img)
            # 使用 KeepAspectRatio 保持比例
            scaled = pixmap.scaled(lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            lbl.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_frames()


    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        super().closeEvent(event)




def main():
    print_versions()
    app = QtWidgets.QApplication(sys.argv)
    
    splash = SplashScreen()
    window = MainWindow()
    
    # When splash finishes, show main window
    splash.finished.connect(window.show)
    splash.show()
    
    sys.exit(app.exec_())


# ==================================================================================