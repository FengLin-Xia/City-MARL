#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 vector_directions_latest.txt 提取坐标创建 north_slots.txt
"""

def create_north_slots():
    # 从 vector_directions_latest.txt 提取前两列坐标
    with open('vector_directions_latest.txt', 'r') as f:
        with open('north_slots.txt', 'w') as out:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    out.write(f'{parts[0]} {parts[1]}\n')
    print('已创建 north_slots.txt 文件')

if __name__ == '__main__':
    create_north_slots()




