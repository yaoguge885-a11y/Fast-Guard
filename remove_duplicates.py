#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动删除 main.py 中的所有重复代码
"""

import shutil
from pathlib import Path

def remove_all_duplicate_code():
    file_path = Path(r"F:\大学\大大的创\fast-guard-日志滚动显示\main.py")
    backup_path = file_path.parent / "main_backup_before_fix_v2.py"
    
    # 备份原文件
    shutil.copy2(file_path, backup_path)
    print(f"✅ 已备份原文件到: {backup_path}")
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📄 原文件行数: {len(lines)}")
    
    # 删除所有重复代码：
    # 1. 删除第2956-4565行（重复的UI代码）
    # 2. 删除第4567-7408行（重复的所有代码）
    
    # 保留第1-2955行（正确的程序入口）
    new_lines = lines[:2955]
    
    print(f"📄 删除后行数: {len(new_lines)}")
    print(f"🗑️  删除行数: {len(lines) - len(new_lines)}")
    
    # 写入新文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ 已删除所有重复代码，文件已更新！")
    print(f"📊 删除比例: {(len(lines) - len(new_lines)) / len(lines) * 100:.1f}%")

if __name__ == "__main__":
    remove_all_duplicate_code()