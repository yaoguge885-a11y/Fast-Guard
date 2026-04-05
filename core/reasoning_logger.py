"""
实时决策日志模块 (Reasoning Logger)

功能：管理系统实时决策日志，记录检测、报警等关键决策过程。
设计原则：纯文本逻辑，不依赖任何 UI 库，完全解耦。
"""

import time
from collections import deque
from typing import List


class ReasoningLogger:
    """
    实时决策日志管理器
    
    使用 deque(maxlen=15) 实现自动滚动的日志容器，
    最多保留最近 15 条日志记录。
    
    用法示例：
        logger = ReasoningLogger()
        logger.add_log("目标进入危险区域")
        logs = logger.get_logs()
    """
    
    def __init__(self, max_len: int = 15):
        """
        初始化日志管理器
        
        Args:
            max_len: 最大日志条目数，默认15，超出自动丢弃最旧记录
        """
        self._logs = deque(maxlen=max_len)
        self._max_len = max_len
    
    def add_log(self, message: str):
        """
        添加一条日志
        
        自动添加 [HH:MM:SS] 格式的时间戳前缀
        
        Args:
            message: 日志消息内容
        """
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self._logs.append(log_entry)
    
    def get_logs(self) -> List[str]:
        """
        获取当前所有日志
        
        Returns:
            日志字符串列表，按时间顺序排列（旧→新）
        """
        return list(self._logs)
    
    def get_latest(self, n: int = 5) -> List[str]:
        """
        获取最近的 N 条日志
        
        Args:
            n: 获取的条数
            
        Returns:
            最近 N 条日志列表
        """
        return list(self._logs)[-n:]
    
    def clear(self):
        """清空所有日志"""
        self._logs.clear()
    
    @property
    def count(self) -> int:
        """当前日志数量"""
        return len(self._logs)
    
    @property
    def max_length(self) -> int:
        """最大容量"""
        return self._max_len
    
    def __len__(self) -> int:
        return len(self._logs)
    
    def __repr__(self) -> str:
        return f"ReasoningLogger(count={len(self._logs)}, max={self._max_len})"
