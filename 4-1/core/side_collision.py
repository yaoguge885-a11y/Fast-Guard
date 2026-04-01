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

# === 3. 侧向碰撞检测 logic (侧向视角预警算法) ===
# ==================================================================================
class SideCollisionDetector:
    """侧向碰撞检测器，专门处理侧向视角的碰撞预警"""
    
    def __init__(self, frame_width, frame_height, fps):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # 定义侧向检测区域
        self.danger_zones = self._define_danger_zones()
        
        # 跟踪状态
        self.side_objects = {}
        self.last_warning_time = 0
        self.warning_cooldown = 1.0
        
        # 开门杀相关参数
        self.door_opening_zones = self._define_door_zones()
        self.door_warning_active = False
        
        # 盲区监测
        self.blind_zones = self._define_blind_zones()
        
        self.ego_speed = 0
        self.side_clearance_threshold = 50
        
    def _define_danger_zones(self):
        zones = {
            'immediate': {
                'left': (0, 0.1, 0.3, 0.7),
                'right': (0.9, 1.0, 0.3, 0.7)
            },
            'warning': {
                'left': (0.1, 0.25, 0.2, 0.8),
                'right': (0.75, 0.9, 0.2, 0.8)
            },
            'alert': {
                'left': (0.25, 0.4, 0.1, 0.9),
                'right': (0.6, 0.75, 0.1, 0.9)
            }
        }
        return zones
    
    def _define_door_zones(self):
        zones = {
            'front_door': {
                'left': (0, 0.15, 0.4, 0.6),
                'right': (0.85, 1.0, 0.4, 0.6)
            },
            'rear_door': {
                'left': (0, 0.15, 0.6, 0.8),
                'right': (0.85, 1.0, 0.6, 0.8)
            }
        }
        return zones
    
    def _define_blind_zones(self):
        zones = {
            'left_blind': (0, 0.2, 0.3, 0.7),
            'right_blind': (0.8, 1.0, 0.3, 0.7)
        }
        return zones
    
    def convert_ratio_to_pixel(self, zone_ratios):
        x1 = int(zone_ratios[0] * self.frame_width)
        x2 = int(zone_ratios[1] * self.frame_width)
        y1 = int(zone_ratios[2] * self.frame_height)
        y2 = int(zone_ratios[3] * self.frame_height)
        return (x1, y1, x2, y2)
    
    def update_object_tracking(self, object_id, bbox, class_name, confidence):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        current_time = time.time()
        
        if object_id not in self.side_objects:
            self.side_objects[object_id] = {
                'id': object_id,
                'class': class_name,
                'positions': [(cx, cy, current_time)],
                'velocities': [],
                'first_seen': current_time,
                'last_seen': current_time,
                'in_danger_zone': False,
                'in_door_zone': False,
                'in_blind_zone': False,
                'warning_level': 0,
                'lateral_speed': 0,
                'closing_rate': 0
            }
        else:
            obj = self.side_objects[object_id]
            obj['positions'].append((cx, cy, current_time))
            obj['last_seen'] = current_time
            
            if len(obj['positions']) >= 2:
                prev_cx, prev_cy, prev_time = obj['positions'][-2]
                time_diff = current_time - prev_time
                if time_diff > 0:
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    vx = dx / time_diff
                    vy = dy / time_diff
                    obj['velocities'].append((vx, vy))
                    if len(obj['velocities']) > 5:
                        obj['velocities'] = obj['velocities'][-5:]
                    if obj['velocities']:
                        avg_vx = np.mean([v[0] for v in obj['velocities']])
                        avg_vy = np.mean([v[1] for v in obj['velocities']])
                        obj['lateral_speed'] = avg_vx
                        obj['closing_rate'] = -avg_vy if avg_vy < 0 else 0
        
        self._check_danger_zones(object_id, cx, cy, width, height)
        self._check_door_zones(object_id, cx, cy)
        self._check_blind_zones(object_id, cx, cy)
        self._cleanup_old_objects(current_time)
    
    def _check_danger_zones(self, object_id, cx, cy, width, height):
        obj = self.side_objects[object_id]
        obj['in_danger_zone'] = False
        obj['warning_level'] = 0
        
        for side, ratios in self.danger_zones['immediate'].items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_danger_zone'] = True
                obj['warning_level'] = 3
                return
        
        for side, ratios in self.danger_zones['warning'].items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_danger_zone'] = True
                obj['warning_level'] = 2
                return
        
        for side, ratios in self.danger_zones['alert'].items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_danger_zone'] = True
                obj['warning_level'] = 1
                return
    
    def _check_door_zones(self, object_id, cx, cy):
        obj = self.side_objects[object_id]
        obj['in_door_zone'] = False
        
        for door_type, zones in self.door_opening_zones.items():
            for side, ratios in zones.items():
                x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    obj['in_door_zone'] = True
                    if abs(obj['lateral_speed']) > 20:
                        obj['warning_level'] = max(obj['warning_level'], 2)
                    else:
                        obj['warning_level'] = max(obj['warning_level'], 1)
                    return
    
    def _check_blind_zones(self, object_id, cx, cy):
        obj = self.side_objects[object_id]
        obj['in_blind_zone'] = False
        
        for zone_name, ratios in self.blind_zones.items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_blind_zone'] = True
                obj['warning_level'] = max(obj['warning_level'], 1)
                return
    
    def _cleanup_old_objects(self, current_time):
        to_remove = []
        for obj_id, obj in self.side_objects.items():
            if current_time - obj['last_seen'] > 2.0:
                to_remove.append(obj_id)
        for obj_id in to_remove:
            del self.side_objects[obj_id]
    
    def analyze_collision_risk(self):
        """分析碰撞风险并返回中文警告信息"""
        warnings = []
        current_time = time.time()
        
        if current_time - self.last_warning_time < self.warning_cooldown:
            return warnings
        
        for obj_id, obj in self.side_objects.items():
            if current_time - obj['first_seen'] < 0.5:
                continue
            
            warning = None
            
            if obj['warning_level'] >= 3:
                warning = {
                    'level': 'danger',
                    'message': f"危险！{obj['class']} 紧贴车辆",
                    'object_id': obj_id,
                    'object_class': obj['class'],
                    'position': obj['positions'][-1][:2] if obj['positions'] else (0, 0)
                }
            elif obj['warning_level'] >= 2:
                warning = {
                    'level': 'warning',
                    'message': f"警告！{obj['class']} 快速接近",
                    'object_id': obj_id,
                    'object_class': obj['class'],
                    'position': obj['positions'][-1][:2] if obj['positions'] else (0, 0)
                }
            elif obj['warning_level'] >= 1:
                warning = {
                    'level': 'alert',
                    'message': f"注意！{obj['class']} 在侧方",
                    'object_id': obj_id,
                    'object_class': obj['class'],
                    'position': obj['positions'][-1][:2] if obj['positions'] else (0, 0)
                }
            
            if obj['in_door_zone'] and obj['warning_level'] >= 2:
                warning = {
                    'level': 'door_danger',
                    'message': f"开门危险！{obj['class']} 接近车门",
                    'object_id': obj_id,
                    'object_class': obj['class'],
                    'position': obj['positions'][-1][:2] if obj['positions'] else (0, 0)
                }
                self.door_warning_active = True
            
            if warning:
                warnings.append(warning)
                self.last_warning_time = current_time
        
        return warnings
    
    def draw_zones(self, frame):
        """已完全移除区域绘制"""
        return frame


# ==================================================================================
