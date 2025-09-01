import numpy as np
import random
from typing import List, Dict, Tuple

class PlacementLogic:
    def __init__(self):
        self.public_coverage_stats = {}  # 记录公共设施覆盖情况
        self.usage_stats = {}  # 记录使用情况
        
    def gov_add_public(self, stats: Dict, trunk: List[List[int]], radius: int = 20) -> List[Dict]:
        """政府补点：根据统计信息添加公共设施"""
        new_public = []
        
        # 检查学校覆盖
        if self._needs_school(stats):
            pos = self._find_public_position(trunk, radius, "school")
            if pos:
                new_public.append({
                    "id": f"school_{len(stats.get('public', [])) + 1}",
                    "type": "school",
                    "xy": pos
                })
        
        # 检查诊所覆盖
        if self._needs_clinic(stats):
            pos = self._find_public_position(trunk, radius, "clinic")
            if pos:
                new_public.append({
                    "id": f"clinic_{len(stats.get('public', [])) + 1}",
                    "type": "clinic",
                    "xy": pos
                })
        
        return new_public
    
    def _needs_school(self, stats: Dict) -> bool:
        """检查是否需要添加学校"""
        avg_time = stats.get('avg_school_time', 0)
        coverage = stats.get('school_coverage', 0)
        # 降低阈值，使其更容易触发
        return avg_time > 8 and coverage < 0.9
    
    def _needs_clinic(self, stats: Dict) -> bool:
        """检查是否需要添加诊所"""
        avg_time = stats.get('avg_clinic_time', 0)
        coverage = stats.get('clinic_coverage', 0)
        # 降低阈值，使其更容易触发
        return avg_time > 8 and coverage < 0.9
    
    def _find_public_position(self, trunk: List[List[int]], radius: int, poi_type: str) -> List[int]:
        """在主干线附近找到合适的公共设施位置"""
        trunk_start, trunk_end = trunk[0], trunk[1]
        
        # 在主干线附近随机选择位置
        for _ in range(10):
            # 在主干线中点附近随机
            mid_x = (trunk_start[0] + trunk_end[0]) / 2
            mid_y = (trunk_start[1] + trunk_end[1]) / 2
            
            x = mid_x + random.uniform(-radius, radius)
            y = mid_y + random.uniform(-radius, radius)
            
            # 确保在合理范围内
            if 20 <= x <= 236 and 20 <= y <= 236:
                return [int(x), int(y)]
        
        # 如果没找到合适位置，返回主干线中点
        mid_x = (trunk_start[0] + trunk_end[0]) / 2
        mid_y = (trunk_start[1] + trunk_end[1]) / 2
        return [int(mid_x), int(mid_y)]
    
    def firm_update(self, stats: Dict) -> Tuple[List[Dict], List[Dict]]:
        """企业更新：扩容和复制POI"""
        new_residential = []
        new_retail = []
        
        # 检查住宅扩容
        for res in stats.get('residential', []):
            if self._needs_expansion(res, 'residential'):
                new_pos = self._find_residential_position(res['xy'])
                if new_pos:
                    new_residential.append({
                        "id": f"res_{len(stats.get('residential', [])) + 1}",
                        "xy": new_pos,
                        "capacity": 200
                    })
        
        # 检查商业扩容
        for ret in stats.get('retail', []):
            if self._needs_expansion(ret, 'retail'):
                new_pos = self._find_retail_position(ret['xy'], stats.get('hubs', []))
                if new_pos:
                    new_retail.append({
                        "id": f"ret_{len(stats.get('retail', [])) + 1}",
                        "xy": new_pos,
                        "capacity": 800
                    })
        
        return new_residential, new_retail
    
    def _needs_expansion(self, poi: Dict, poi_type: str) -> bool:
        """检查POI是否需要扩容"""
        usage_ratio = poi.get('usage_ratio', 0)
        consecutive_days = poi.get('consecutive_high_usage', 0)
        # 降低阈值，使其更容易触发
        return usage_ratio > 0.6 and consecutive_days >= 2
    
    def _find_residential_position(self, base_pos: List[int]) -> List[int]:
        """找到住宅位置（距主干线10-60px）"""
        trunk_mid = [128, 128]  # 主干线中点
        
        for _ in range(10):
            # 在主干线附近10-60px范围内
            distance = random.uniform(10, 60)
            angle = random.uniform(0, 2 * np.pi)
            
            x = trunk_mid[0] + distance * np.cos(angle)
            y = trunk_mid[1] + distance * np.sin(angle)
            
            # 确保在合理范围内
            if 20 <= x <= 236 and 20 <= y <= 236:
                return [int(x), int(y)]
        
        # 如果没找到合适位置，返回主干线中点附近
        return [int(trunk_mid[0] + 30), int(trunk_mid[1] + 30)]
    
    def _find_retail_position(self, base_pos: List[int], hubs: List[Dict]) -> List[int]:
        """找到商业位置（靠近枢纽200-400px环带）"""
        if not hubs:
            return self._find_residential_position(base_pos)
        
        # 选择最近的枢纽
        hub = min(hubs, key=lambda h: np.linalg.norm(np.array(h['xy']) - np.array(base_pos)))
        
        for _ in range(10):
            # 在枢纽附近200-400px环带内
            distance = random.uniform(200, 400)
            angle = random.uniform(0, 2 * np.pi)
            
            x = hub['xy'][0] + distance * np.cos(angle)
            y = hub['xy'][1] + distance * np.sin(angle)
            
            # 确保在合理范围内
            if 20 <= x <= 236 and 20 <= y <= 236:
                return [int(x), int(y)]
        
        # 如果没找到合适位置，返回枢纽附近
        return [int(hub['xy'][0] + 300), int(hub['xy'][1] + 300)]

