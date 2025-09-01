import random
from typing import List, Dict

class ScheduleLogic:
    def __init__(self):
        pass
    
    def spawn_residents(self, residential_pois: List[Dict], daily_quota: int = 100) -> List[Dict]:
        """生成新居民"""
        residents = []
        
        # 计算总容量
        total_capacity = sum(poi.get('capacity', 200) for poi in residential_pois)
        current_population = sum(poi.get('current_population', 0) for poi in residential_pois)
        
        # 确定可以新增的居民数量
        available_slots = total_capacity - current_population
        new_residents = min(daily_quota, available_slots)
        
        for i in range(new_residents):
            # 随机选择一个住宅POI
            if residential_pois:
                home_poi = random.choice(residential_pois)
                home_id = home_poi['id']
                home_pos = home_poi['xy']
            else:
                # 如果没有住宅POI，使用默认位置
                home_id = "default_home"
                home_pos = [128, 128]
            
            resident = {
                "id": f"agent_{len(residents) + 1}",
                "pos": home_pos.copy(),
                "home": home_id,
                "plan": [],
                "target": home_pos.copy(),
                "current_activity": "home"
            }
            
            residents.append(resident)
            
            # 更新住宅POI的人口
            if residential_pois:
                home_poi['current_population'] = home_poi.get('current_population', 0) + 1
        
        return residents
    
    def assign_daily_plans(self, residents: List[Dict], retail_pois: List[Dict], public_pois: List[Dict]) -> None:
        """为居民分配每日计划"""
        for resident in residents:
            plan = []
            
            # 基本计划：home -> work -> shop -> home
            home_pos = resident['pos']
            
            # 工作：选择最近的商业POI或公共POI
            work_target = self._find_work_target(home_pos, retail_pois, public_pois)
            if work_target:
                plan.append(["work", work_target['id']])
            
            # 购物：选择商业POI
            shop_target = self._find_shop_target(home_pos, retail_pois)
            if shop_target:
                plan.append(["shop", shop_target['id']])
            
            # 回家
            plan.append(["home", resident['home']])
            
            resident['plan'] = plan
            resident['current_plan_index'] = 0
    
    def _find_work_target(self, home_pos: List[int], retail_pois: List[Dict], public_pois: List[Dict]) -> Dict:
        """找到工作目标"""
        all_targets = []
        
        # 添加商业POI
        for poi in retail_pois:
            all_targets.append({
                'id': poi['id'],
                'xy': poi['xy'],
                'type': 'retail'
            })
        
        # 添加公共POI（学校、诊所）
        for poi in public_pois:
            if poi['type'] in ['school', 'clinic']:
                all_targets.append({
                    'id': poi['id'],
                    'xy': poi['xy'],
                    'type': poi['type']
                })
        
        if not all_targets:
            return None
        
        # 选择最近的目标
        distances = [(target, self._calculate_distance(home_pos, target['xy'])) for target in all_targets]
        return min(distances, key=lambda x: x[1])[0]
    
    def _find_shop_target(self, home_pos: List[int], retail_pois: List[Dict]) -> Dict:
        """找到购物目标"""
        if not retail_pois:
            return None
        
        # 选择最近或随机的商业POI
        distances = [(poi, self._calculate_distance(home_pos, poi['xy'])) for poi in retail_pois]
        # 70%概率选择最近的，30%概率随机选择
        if random.random() < 0.7:
            return min(distances, key=lambda x: x[1])[0]
        else:
            return random.choice(retail_pois)
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def get_next_target(self, resident: Dict, all_pois: Dict) -> List[int]:
        """获取居民的下一个目标位置"""
        if resident['current_plan_index'] >= len(resident['plan']):
            return resident['pos']  # 计划完成，留在原地
        
        current_activity, target_id = resident['plan'][resident['current_plan_index']]
        
        # 在所有POI中查找目标
        for poi_type in ['public', 'residential', 'retail']:
            if poi_type in all_pois:
                for poi in all_pois[poi_type]:
                    if poi['id'] == target_id:
                        return poi['xy']
        
        # 如果找不到目标，返回当前位置
        return resident['pos']
    
    def advance_plan(self, resident: Dict) -> None:
        """推进居民的计划"""
        resident['current_plan_index'] += 1
        if resident['current_plan_index'] >= len(resident['plan']):
            resident['current_plan_index'] = 0  # 重新开始计划
