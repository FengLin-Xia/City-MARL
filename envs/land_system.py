"""
地块功能系统模块
管理地块的属性和功能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import json

class LandType(Enum):
    """地块类型枚举"""
    EMPTY = 0
    RESIDENTIAL = 1      # 住宅
    COMMERCIAL = 2       # 商业
    INDUSTRIAL = 3       # 工业
    AGRICULTURAL = 4     # 农业
    RECREATIONAL = 5     # 娱乐
    INFRASTRUCTURE = 6   # 基础设施
    NATURAL = 7          # 自然保护

class LandFunction:
    """地块功能类"""
    
    def __init__(self, land_type: LandType, level: int = 1):
        """
        初始化地块功能
        
        Args:
            land_type: 地块类型
            level: 功能等级
        """
        self.land_type = land_type
        self.level = level
        self.efficiency = 1.0  # 效率系数
        self.condition = 1.0   # 状态系数
        self.age = 0           # 年龄
        
        # 功能属性
        self.properties = self._get_properties()
    
    def _get_properties(self) -> Dict:
        """获取地块功能属性"""
        base_properties = {
            LandType.EMPTY: {
                'name': '空地',
                'value': 0.0,
                'maintenance_cost': 0.0,
                'revenue': 0.0,
                'population_capacity': 0,
                'resource_production': 0.0,
                'happiness_effect': 0.0,
                'environmental_impact': 0.0
            },
            LandType.RESIDENTIAL: {
                'name': '住宅',
                'value': 100.0,
                'maintenance_cost': 5.0,
                'revenue': 10.0,
                'population_capacity': 100,
                'resource_production': 0.0,
                'happiness_effect': 0.1,
                'environmental_impact': -0.05
            },
            LandType.COMMERCIAL: {
                'name': '商业',
                'value': 200.0,
                'maintenance_cost': 10.0,
                'revenue': 25.0,
                'population_capacity': 0,
                'resource_production': 0.0,
                'happiness_effect': 0.2,
                'environmental_impact': -0.1
            },
            LandType.INDUSTRIAL: {
                'name': '工业',
                'value': 150.0,
                'maintenance_cost': 15.0,
                'revenue': 30.0,
                'population_capacity': 0,
                'resource_production': 5.0,
                'happiness_effect': -0.1,
                'environmental_impact': -0.3
            },
            LandType.AGRICULTURAL: {
                'name': '农业',
                'value': 80.0,
                'maintenance_cost': 3.0,
                'revenue': 15.0,
                'population_capacity': 0,
                'resource_production': 10.0,
                'happiness_effect': 0.05,
                'environmental_impact': 0.1
            },
            LandType.RECREATIONAL: {
                'name': '娱乐',
                'value': 120.0,
                'maintenance_cost': 8.0,
                'revenue': 5.0,
                'population_capacity': 0,
                'resource_production': 0.0,
                'happiness_effect': 0.3,
                'environmental_impact': 0.05
            },
            LandType.INFRASTRUCTURE: {
                'name': '基础设施',
                'value': 300.0,
                'maintenance_cost': 20.0,
                'revenue': 0.0,
                'population_capacity': 0,
                'resource_production': 0.0,
                'happiness_effect': 0.15,
                'environmental_impact': -0.05
            },
            LandType.NATURAL: {
                'name': '自然保护',
                'value': 50.0,
                'maintenance_cost': 2.0,
                'revenue': 0.0,
                'population_capacity': 0,
                'resource_production': 0.0,
                'happiness_effect': 0.2,
                'environmental_impact': 0.2
            }
        }
        
        props = base_properties[self.land_type].copy()
        # 根据等级调整属性
        level_multiplier = 1.0 + (self.level - 1) * 0.2
        for key in ['value', 'revenue', 'population_capacity', 'resource_production']:
            if key in props:
                props[key] *= level_multiplier
        
        return props
    
    def get_value(self) -> float:
        """获取地块价值"""
        base_value = self.properties['value']
        return base_value * self.efficiency * self.condition
    
    def get_revenue(self) -> float:
        """获取地块收入"""
        base_revenue = self.properties['revenue']
        return base_revenue * self.efficiency * self.condition
    
    def get_maintenance_cost(self) -> float:
        """获取维护成本"""
        base_cost = self.properties['maintenance_cost']
        return base_cost * (1.0 + self.age * 0.1)  # 年龄增加维护成本
    
    def get_net_revenue(self) -> float:
        """获取净收入"""
        return self.get_revenue() - self.get_maintenance_cost()
    
    def upgrade(self) -> bool:
        """升级地块功能"""
        if self.level < 5:  # 最大等级5
            self.level += 1
            self.properties = self._get_properties()
            return True
        return False
    
    def degrade(self) -> bool:
        """降级地块功能"""
        if self.level > 1:
            self.level -= 1
            self.properties = self._get_properties()
            return True
        return False
    
    def update(self, time_step: int = 1):
        """更新地块状态"""
        self.age += time_step
        
        # 效率随时间衰减
        self.efficiency = max(0.5, self.efficiency - 0.01 * time_step)
        
        # 状态随时间衰减
        self.condition = max(0.3, self.condition - 0.005 * time_step)
    
    def repair(self, cost: float) -> float:
        """修复地块"""
        repair_amount = min(cost / 10.0, 1.0 - self.condition)
        self.condition += repair_amount
        return repair_amount * 10.0  # 返回实际花费
    
    def improve_efficiency(self, cost: float) -> float:
        """提高效率"""
        improvement = min(cost / 20.0, 1.0 - self.efficiency)
        self.efficiency += improvement
        return improvement * 20.0  # 返回实际花费

class LandSystem:
    """地块系统类"""
    
    def __init__(self, width: int, height: int):
        """
        初始化地块系统
        
        Args:
            width: 地图宽度
            height: 地图高度
        """
        self.width = width
        self.height = height
        self.lands = np.full((height, width), None, dtype=object)
        
        # 初始化所有地块为空地
        for y in range(height):
            for x in range(width):
                self.lands[y, x] = LandFunction(LandType.EMPTY)
        
        # 地块统计
        self.stats = self._init_stats()
        
        # 地块连接关系
        self.connections = {}  # (x, y) -> set of connected positions
    
    def _init_stats(self) -> Dict:
        """初始化统计信息"""
        return {
            'total_lands': self.width * self.height,
            'developed_lands': 0,
            'land_types': {land_type: 0 for land_type in LandType},
            'total_value': 0.0,
            'total_revenue': 0.0,
            'total_maintenance': 0.0,
            'total_population_capacity': 0,
            'total_resource_production': 0.0,
            'average_happiness': 0.0,
            'environmental_impact': 0.0
        }
    
    def set_land_function(self, x: int, y: int, land_type: LandType, level: int = 1) -> bool:
        """
        设置地块功能
        
        Args:
            x, y: 地块坐标
            land_type: 地块类型
            level: 功能等级
            
        Returns:
            bool: 是否设置成功
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        # 更新统计信息
        old_land = self.lands[y, x]
        if old_land.land_type != LandType.EMPTY:
            self.stats['land_types'][old_land.land_type] -= 1
            self.stats['developed_lands'] -= 1
        
        # 设置新地块
        self.lands[y, x] = LandFunction(land_type, level)
        
        # 更新统计信息
        if land_type != LandType.EMPTY:
            self.stats['land_types'][land_type] += 1
            self.stats['developed_lands'] += 1
        
        self._update_stats()
        return True
    
    def get_land_function(self, x: int, y: int) -> Optional[LandFunction]:
        """获取地块功能"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.lands[y, x]
        return None
    
    def get_land_type(self, x: int, y: int) -> LandType:
        """获取地块类型"""
        land = self.get_land_function(x, y)
        return land.land_type if land else LandType.EMPTY
    
    def upgrade_land(self, x: int, y: int) -> bool:
        """升级地块"""
        land = self.get_land_function(x, y)
        if land and land.upgrade():
            self._update_stats()
            return True
        return False
    
    def degrade_land(self, x: int, y: int) -> bool:
        """降级地块"""
        land = self.get_land_function(x, y)
        if land and land.degrade():
            self._update_stats()
            return True
        return False
    
    def repair_land(self, x: int, y: int, cost: float) -> float:
        """修复地块"""
        land = self.get_land_function(x, y)
        if land:
            return land.repair(cost)
        return 0.0
    
    def improve_land_efficiency(self, x: int, y: int, cost: float) -> float:
        """提高地块效率"""
        land = self.get_land_function(x, y)
        if land:
            return land.improve_efficiency(cost)
        return 0.0
    
    def update_all_lands(self, time_step: int = 1):
        """更新所有地块状态"""
        for y in range(self.height):
            for x in range(self.width):
                self.lands[y, x].update(time_step)
        self._update_stats()
    
    def get_land_value(self, x: int, y: int) -> float:
        """获取地块价值"""
        land = self.get_land_function(x, y)
        return land.get_value() if land else 0.0
    
    def get_land_revenue(self, x: int, y: int) -> float:
        """获取地块收入"""
        land = self.get_land_function(x, y)
        return land.get_revenue() if land else 0.0
    
    def get_land_net_revenue(self, x: int, y: int) -> float:
        """获取地块净收入"""
        land = self.get_land_function(x, y)
        return land.get_net_revenue() if land else 0.0
    
    def get_connected_lands(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """获取连接的地块"""
        return self.connections.get((x, y), set())
    
    def add_connection(self, x1: int, y1: int, x2: int, y2: int):
        """添加地块连接"""
        pos1, pos2 = (x1, y1), (x2, y2)
        if pos1 not in self.connections:
            self.connections[pos1] = set()
        if pos2 not in self.connections:
            self.connections[pos2] = set()
        
        self.connections[pos1].add(pos2)
        self.connections[pos2].add(pos1)
    
    def remove_connection(self, x1: int, y1: int, x2: int, y2: int):
        """移除地块连接"""
        pos1, pos2 = (x1, y1), (x2, y2)
        if pos1 in self.connections:
            self.connections[pos1].discard(pos2)
        if pos2 in self.connections:
            self.connections[pos2].discard(pos1)
    
    def get_land_cluster(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """获取地块集群（连接的地块组）"""
        cluster = set()
        to_visit = [(x, y)]
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            cluster.add(current)
            
            # 添加连接的邻居
            for neighbor in self.get_connected_lands(*current):
                if neighbor not in visited:
                    to_visit.append(neighbor)
        
        return cluster
    
    def get_land_bonus(self, x: int, y: int) -> float:
        """获取地块加成（基于周围地块）"""
        land = self.get_land_function(x, y)
        if not land or land.land_type == LandType.EMPTY:
            return 0.0
        
        bonus = 0.0
        cluster = self.get_land_cluster(x, y)
        
        # 集群加成
        if len(cluster) >= 3:
            bonus += 0.1 * len(cluster)
        
        # 类型匹配加成
        for cx, cy in cluster:
            if (cx, cy) != (x, y):
                neighbor_land = self.get_land_function(cx, cy)
                if neighbor_land and neighbor_land.land_type == land.land_type:
                    bonus += 0.05
        
        return bonus
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats = self._init_stats()
        
        for y in range(self.height):
            for x in range(self.width):
                land = self.lands[y, x]
                if land.land_type != LandType.EMPTY:
                    self.stats['land_types'][land.land_type] += 1
                    self.stats['developed_lands'] += 1
                
                self.stats['total_value'] += land.get_value()
                self.stats['total_revenue'] += land.get_revenue()
                self.stats['total_maintenance'] += land.get_maintenance_cost()
                self.stats['total_population_capacity'] += land.properties['population_capacity']
                self.stats['total_resource_production'] += land.properties['resource_production']
                self.stats['environmental_impact'] += land.properties['environmental_impact']
        
        # 计算平均幸福度
        total_happiness = sum(land.properties['happiness_effect'] 
                            for y in range(self.height) 
                            for x in range(self.width) 
                            for land in [self.lands[y, x]])
        self.stats['average_happiness'] = total_happiness / (self.width * self.height)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def save_to_file(self, filepath: str) -> bool:
        """保存地块数据到文件"""
        try:
            data = {
                'width': self.width,
                'height': self.height,
                'lands': [],
                'connections': {f"{x},{y}": list(conns) for (x, y), conns in self.connections.items()}
            }
            
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    land = self.lands[y, x]
                    row.append({
                        'type': land.land_type.value,
                        'level': land.level,
                        'efficiency': land.efficiency,
                        'condition': land.condition,
                        'age': land.age
                    })
                data['lands'].append(row)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"保存地块数据失败: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """从文件加载地块数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.width = data['width']
            self.height = data['height']
            
            # 加载地块数据
            for y in range(self.height):
                for x in range(self.width):
                    land_data = data['lands'][y][x]
                    land_type = LandType(land_data['type'])
                    land = LandFunction(land_type, land_data['level'])
                    land.efficiency = land_data['efficiency']
                    land.condition = land_data['condition']
                    land.age = land_data['age']
                    land.properties = land._get_properties()
                    self.lands[y, x] = land
            
            # 加载连接数据
            self.connections = {}
            for key, conns in data['connections'].items():
                x, y = map(int, key.split(','))
                self.connections[(x, y)] = set(tuple(conn) for conn in conns)
            
            self._update_stats()
            return True
        except Exception as e:
            print(f"加载地块数据失败: {e}")
            return False
