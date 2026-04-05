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
        
        # 定义侧向检测区域 - 缩小危险区域范围
        self.danger_zones = self._define_danger_zones()
        
        # 跟踪状态
        self.side_objects = {}
        self.last_warning_time = 0
        self.warning_cooldown = 1.5  # 增加冷却时间 1.0 -> 1.5秒
        
        # 开门杀相关参数
        self.door_opening_zones = self._define_door_zones()
        self.door_warning_active = False
        
        # 盲区监测
        self.blind_zones = self._define_blind_zones()
        
        self.ego_speed = 0
        self.side_clearance_threshold = 50
        
        # 新增：最小观察时间（秒）
        self.min_observation_time = 1.0  # 0.5 -> 1.0秒
        
        # 新增：距离验证阈值（米）
        self.min_distance_threshold = 0.5  # 至少0.5米才考虑
        self.max_distance_threshold = 8.0  # 超过8米忽略
        
        # 新增：连续确认帧数
        self.min_confirm_frames = 4  # 至少4帧确认
        
    def _define_danger_zones(self):
        """定义危险区域 - 缩小范围减少误报"""
        zones = {
            'immediate': {
                # 收窄immediate区域 0.1 -> 0.05, 0.3-0.7 -> 0.35-0.65
                'left': (0, 0.05, 0.35, 0.65),
                'right': (0.95, 1.0, 0.35, 0.65)
            },
            'warning': {
                # 收窄warning区域
                'left': (0.05, 0.20, 0.30, 0.70),
                'right': (0.80, 0.95, 0.30, 0.70)
            },
            'alert': {
                # 收窄alert区域
                'left': (0.20, 0.35, 0.25, 0.75),
                'right': (0.65, 0.80, 0.25, 0.75)
            }
        }
        return zones
    
    def _define_door_zones(self):
        """定义车门区域 - 缩小范围"""
        zones = {
            'front_door': {
                'left': (0, 0.12, 0.45, 0.58),  # 收窄
                'right': (0.88, 1.0, 0.45, 0.58)
            },
            'rear_door': {
                'left': (0, 0.12, 0.62, 0.78),
                'right': (0.88, 1.0, 0.62, 0.78)
            }
        }
        return zones
    
    def _define_blind_zones(self):
        """定义盲区 - 缩小范围"""
        zones = {
            'left_blind': (0, 0.15, 0.35, 0.65),  # 0.2 -> 0.15
            'right_blind': (0.85, 1.0, 0.35, 0.65)  # 0.8 -> 0.85
        }
        return zones
    
    def convert_ratio_to_pixel(self, zone_ratios):
        x1 = int(zone_ratios[0] * self.frame_width)
        x2 = int(zone_ratios[1] * self.frame_width)
        y1 = int(zone_ratios[2] * self.frame_height)
        y2 = int(zone_ratios[3] * self.frame_height)
        return (x1, y1, x2, y2)
    
    def update_object_tracking(self, object_id, bbox, class_name, confidence, world_pos=None):
        """
        更新目标跟踪
        
        Args:
            object_id: 目标ID
            bbox: (x1, y1, x2, y2)
            class_name: 类别名称
            confidence: 置信度
            world_pos: (x, y) 世界坐标位置（米），可选
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        current_time = time.time()
        
        # 距离验证 - 如果有世界坐标
        distance_valid = True
        if world_pos is not None:
            dist = abs(world_pos[0])  # 横向距离
            if dist < self.min_distance_threshold or dist > self.max_distance_threshold:
                distance_valid = False
        
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
                'closing_rate': 0,
                'frame_count': 1,  # 新增：帧计数
                'distance_valid': distance_valid,
                'world_pos': world_pos,
            }
        else:
            obj = self.side_objects[object_id]
            obj['positions'].append((cx, cy, current_time))
            obj['last_seen'] = current_time
            obj['frame_count'] += 1
            obj['distance_valid'] = distance_valid
            obj['world_pos'] = world_pos
            
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
        
        # 如果距离无效，不设置危险等级
        if not obj.get('distance_valid', True):
            obj['in_danger_zone'] = False
            obj['warning_level'] = 0
            return
        
        # 如果观察时间不足，降低风险等级
        current_time = time.time()
        observation_time = current_time - obj['first_seen']
        frame_count = obj.get('frame_count', 0)
        
        if observation_time < self.min_observation_time or frame_count < self.min_confirm_frames:
            # 观察期内的目标最多只给警告级别
            max_temp_level = 1
        else:
            max_temp_level = 3
        
        obj['in_danger_zone'] = False
        obj['warning_level'] = 0
        
        for side, ratios in self.danger_zones['immediate'].items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_danger_zone'] = True
                obj['warning_level'] = min(3, max_temp_level)
                return
        
        for side, ratios in self.danger_zones['warning'].items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_danger_zone'] = True
                obj['warning_level'] = min(2, max_temp_level)
                return
        
        for side, ratios in self.danger_zones['alert'].items():
            x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                obj['in_danger_zone'] = True
                obj['warning_level'] = min(1, max_temp_level)
                return
    
    def _check_door_zones(self, object_id, cx, cy):
        obj = self.side_objects[object_id]
        obj['in_door_zone'] = False
        
        # 距离验证
        if not obj.get('distance_valid', True):
            return
        
        for door_type, zones in self.door_opening_zones.items():
            for side, ratios in zones.items():
                x1, y1, x2, y2 = self.convert_ratio_to_pixel(ratios)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    obj['in_door_zone'] = True
                    # 提高速度阈值 20 -> 40 像素/秒
                    if abs(obj['lateral_speed']) > 40:
                        obj['warning_level'] = max(obj['warning_level'], 2)
                    else:
                        obj['warning_level'] = max(obj['warning_level'], 1)
                    return
    
    def _check_blind_zones(self, object_id, cx, cy):
        obj = self.side_objects[object_id]
        obj['in_blind_zone'] = False
        
        # 距离验证
        if not obj.get('distance_valid', True):
            return
        
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
            # 增加观察时间检查
            observation_time = current_time - obj['first_seen']
            frame_count = obj.get('frame_count', 0)
            
            if observation_time < self.min_observation_time:
                continue
            if frame_count < self.min_confirm_frames:
                continue
            if not obj.get('distance_valid', True):
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
