#!/usr/bin/env python3
"""
增强城市模拟系统 v3.2
基于PRD v3.2：政府骨架系统、统一决策器、特征化评分
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import math
import time
from dataclasses import dataclass
import cv2

# 导入现有模块
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.hysteresis_system import HysteresisSystem
from logic.public_facility_system import PublicFacilitySystem
from logic.enhanced_agents import GovernmentAgent, BusinessAgent, ResidentAgent
from logic.output_system import OutputSystem
from logic.trajectory_system import TrajectorySystem

@dataclass
class Slot:
    """建筑槽位"""
    pos: List[int]  # 位置 [x, y]
    used: bool = False  # 是否被占用
    dead: bool = False  # 是否为死槽
    allowed_types: List[str] = None  # 允许的建筑类型
    building_id: Optional[str] = None  # 占用的建筑ID
    features: Dict = None  # 特征值
    scores: Dict = None  # 评分
    
    def __post_init__(self):
        if self.allowed_types is None:
            self.allowed_types = ['commercial', 'residential', 'industrial']
        if self.features is None:
            self.features = {}
        if self.scores is None:
            self.scores = {}

@dataclass
class Layer:
    """建筑层"""
    layer_id: str  # 层标识
    status: str  # locked/active/complete
    activated_quarter: int  # 激活的季度
    slots: List[Slot]  # 槽位列表
    capacity: int  # 总容量
    dead_slots: int  # 死槽数量
    capacity_effective: int  # 有效容量
    placed: int  # 已放置数量
    density: float  # 密度 = placed / capacity_effective
    layer_type: str = 'road'  # 层类型: road/commercial_radial/industrial_radial/residential
    
    def update_stats(self):
        """更新统计信息"""
        self.dead_slots = sum(1 for slot in self.slots if slot.dead)
        self.capacity_effective = self.capacity - self.dead_slots
        self.placed = sum(1 for slot in self.slots if slot.used)
        self.density = self.placed / self.capacity_effective if self.capacity_effective > 0 else 0.0

class GovernmentBackboneSystem:
    """政府骨架系统 v3.2"""
    
    def __init__(self, config: Dict):
        self.config = config.get('government_backbone', {})
        
        # 走廊带配置
        road_config = self.config.get('road_corridor', {})
        self.sigma_perp_m = road_config.get('sigma_perp_m', 40)
        self.setback_m = road_config.get('setback_m', {'com': 8, 'res': 10, 'ind': 14})
        
        # 双顶点配置
        hubs_config = self.config.get('hubs', {})
        self.hub_commercial = hubs_config.get('commercial', {'sigma_perp_m': 30, 'sigma_parallel_m': 90})
        self.hub_industrial = hubs_config.get('industrial', {'sigma_perp_m': 35, 'sigma_parallel_m': 110})
        
        # 分区配置
        zoning_config = self.config.get('zoning', {})
        self.hub_com_radius_m = zoning_config.get('hub_com_radius_m', 350)
        self.hub_ind_radius_m = zoning_config.get('hub_ind_radius_m', 450)
        self.mid_corridor_residential = zoning_config.get('mid_corridor_residential', True)
        
        # 配额配置
        quotas_config = self.config.get('quotas_per_quarter', {})
        self.quotas = {
            'res': quotas_config.get('res', [10, 20]),
            'com': quotas_config.get('com', [5, 12]),
            'ind': quotas_config.get('ind', [4, 10])
        }
        
        # 纪律配置
        self.strict_layering = self.config.get('strict_layering', True)
        self.dead_slots_ratio_max = self.config.get('dead_slots_ratio_max', 0.05)
        
        # 骨架几何
        self.road_corridor = None  # 主干道中心线
        self.hub_commercial_pos = None  # 商业枢纽位置
        self.hub_industrial_pos = None  # 工业枢纽位置
        
        print(f"🏛️ 政府骨架系统 v3.2 初始化完成")
        print(f"   走廊带法向影响宽度: {self.sigma_perp_m}m")
        print(f"   商业枢纽半径: {self.hub_com_radius_m}m")
        print(f"   工业枢纽半径: {self.hub_ind_radius_m}m")
        print(f"   季度配额: 住宅{self.quotas['res']}, 商业{self.quotas['com']}, 工业{self.quotas['ind']}")
    
    def initialize_backbone(self, map_size: List[int], transport_hubs: List[List[int]]):
        """初始化政府骨架"""
        print("🏗️ 初始化政府骨架...")
        
        # 设置主干道中心线（水平走廊）
        center_y = map_size[1] // 2
        self.road_corridor = {
            'start': [0, center_y],
            'end': [map_size[0], center_y],
            'center_y': center_y
        }
        
        # 设置双顶点位置
        if len(transport_hubs) >= 2:
            self.hub_commercial_pos = transport_hubs[0]  # Hub1: 商业客运核
            self.hub_industrial_pos = transport_hubs[1]  # Hub2: 工业货运核
        else:
            # 默认位置 - 调整到离边缘更远的位置
            self.hub_commercial_pos = [map_size[0] // 3, center_y]  # 从1/4改为1/3
            self.hub_industrial_pos = [2 * map_size[0] // 3, center_y]  # 从3/4改为2/3
        
        print(f"✅ 政府骨架初始化完成")
        print(f"   主干道: {self.road_corridor['start']} -> {self.road_corridor['end']}")
        print(f"   商业枢纽: {self.hub_commercial_pos}")
        print(f"   工业枢纽: {self.hub_industrial_pos}")
    
    def get_road_corridor_layers(self, map_size: List[int]) -> List[Layer]:
        """生成走廊带Road-L0层"""
        layers = []
        
        # 计算道路退线
        center_y = self.road_corridor['center_y']
        
        # 商业层（靠近道路）
        com_offset = self.setback_m['com']
        com_layer = self._create_road_layer(
            'road_L0_com', 'commercial', 
            center_y - com_offset, center_y + com_offset,
            map_size, 'commercial'
        )
        if com_layer:
            layers.append(com_layer)
        
        # 住宅层（中等距离）
        res_offset = self.setback_m['res']
        res_layer = self._create_road_layer(
            'road_L0_res', 'residential',
            center_y - res_offset, center_y + res_offset,
            map_size, 'residential'
        )
        if res_layer:
            layers.append(res_layer)
        
        # 工业层（最远距离）
        ind_offset = self.setback_m['ind']
        ind_layer = self._create_road_layer(
            'road_L0_ind', 'industrial',
            center_y - ind_offset, center_y + ind_offset,
            map_size, 'industrial'
        )
        if ind_layer:
            layers.append(ind_layer)
        
        return layers
    
    def _create_road_layer(self, layer_id: str, building_type: str, 
                          y_min: int, y_max: int, map_size: List[int], 
                          layer_type: str) -> Optional[Layer]:
        """创建道路层槽位 - 沿线等弧长采样"""
        slots = []
        center_y = self.road_corridor['center_y']
        
        # 沿道路等距采样
        spacing = 25 if building_type == 'commercial' else 35 if building_type == 'residential' else 45
        
        # 只在两条退线 y = center_y ± offset 放槽位
        for side in (-1, +1):
            y_line = center_y + side * (y_max - center_y)  # 这里 y_max/y_min 其实就是 offset
            y_line = int(round(y_line))
            
            for x in range(0, map_size[0], spacing):
                pos = [x, y_line]
                if self._is_position_valid(pos, map_size):
                    slot = Slot(
                        pos=pos,
                        allowed_types=[building_type],
                        features={},
                        scores={}
                    )
                    slots.append(slot)
        
        if slots:
            layer = Layer(
                layer_id=layer_id,
                status="locked",
                activated_quarter=-1,
                slots=slots,
                capacity=len(slots),
                dead_slots=0,
                capacity_effective=len(slots),
                placed=0,
                density=0.0,
                layer_type='road'
            )
            return layer
        
        return None
    
    def _is_position_valid(self, pos: List[int], map_size: List[int]) -> bool:
        """检查位置是否有效"""
        return 0 <= pos[0] < map_size[0] and 0 <= pos[1] < map_size[1]
    
    def get_quarterly_quotas(self, quarter: int) -> Dict[str, int]:
        """获取季度配额"""
        return {
            'residential': random.randint(self.quotas['res'][0], self.quotas['res'][1]),
            'commercial': random.randint(self.quotas['com'][0], self.quotas['com'][1]),
            'industrial': random.randint(self.quotas['ind'][0], self.quotas['ind'][1])
        }
    
    def get_zoning_constraints(self, pos: List[int]) -> List[str]:
        """获取位置的分区约束"""
        allowed_types = {'commercial', 'residential', 'industrial'}
        
        # 像素到米的转换
        meters_per_pixel = 2.0
        
        # 计算到各枢纽的距离（米）
        dx_c = (pos[0] - self.hub_commercial_pos[0]) * meters_per_pixel
        dy_c = (pos[1] - self.hub_commercial_pos[1]) * meters_per_pixel
        dist_com_m = math.hypot(dx_c, dy_c)
        
        dx_i = (pos[0] - self.hub_industrial_pos[0]) * meters_per_pixel
        dy_i = (pos[1] - self.hub_industrial_pos[1]) * meters_per_pixel
        dist_ind_m = math.hypot(dx_i, dy_i)
        
        # 商业枢纽半径内不许工业
        if dist_com_m <= self.hub_com_radius_m:
            allowed_types.discard('industrial')
        
        # 工业枢纽半径内不许商业
        if dist_ind_m <= self.hub_ind_radius_m:
            allowed_types.discard('commercial')
        
        # 走廊中段优先住宅（不改变allowed_types，只在评分中体现）
        # if self.mid_corridor_residential:
        #     center_y = self.road_corridor['center_y']
        #     if abs(pos[1] - center_y) * meters_per_pixel <= 20:  # 走廊中段
        #         # 这里可以在评分系统中给住宅加分，而不是改变allowed_types
        #         pass
        
        return list(allowed_types)

class FeatureScoringSystem:
    """特征化评分系统 v3.2"""
    
    def __init__(self, config: Dict):
        self.config = config.get('scoring_weights', {})
        
        # 评分权重 - 统一使用长名
        self.weights = {
            'commercial': self.config.get('commercial', {
                'f_hub_com': 0.6, 'f_road': 0.2, 'f_heat': 0.15, 'f_access': 0.05,
                'crowding': -0.03, 'junction_penalty': -0.02
            }),
            'industrial': self.config.get('industrial', {
                'f_hub_ind': 0.55, 'f_road': 0.25, 'f_access': 0.05,
                'crowding': -0.10, 'junction_penalty': -0.05
            }),
            'residential': self.config.get('residential', {
                'f_road': 0.5, 'f_access': 0.15,
                'f_hub_com': -0.2, 'f_hub_ind': -0.15, 'crowding': -0.05
            })
        }
        
        # 特征计算参数
        self.meters_per_pixel = 2.0  # 像素到米的转换
        
        print(f"📊 特征化评分系统 v3.2 初始化完成")
        print(f"   商业权重: {self.weights['commercial']}")
        print(f"   工业权重: {self.weights['industrial']}")
        print(f"   住宅权重: {self.weights['residential']}")
    
    def compute_features(self, slot: Slot, backbone_system: 'GovernmentBackboneSystem', 
                        city_state: Dict) -> Dict[str, float]:
        """计算槽位特征"""
        pos = slot.pos
        features = {}
        
        # f_road: 到走廊线的法向核
        features['f_road'] = self._compute_road_feature(pos, backbone_system)
        
        # f_hub_com: 商业顶点核
        features['f_hub_com'] = self._compute_hub_feature(pos, backbone_system.hub_commercial_pos, 
                                                         backbone_system.hub_commercial)
        
        # f_hub_ind: 工业顶点核
        features['f_hub_ind'] = self._compute_hub_feature(pos, backbone_system.hub_industrial_pos, 
                                                         backbone_system.hub_industrial)
        
        # f_access: 公共设施可达性
        features['f_access'] = self._compute_access_feature(pos, city_state)
        
        # f_heat: 居民轨迹热力
        features['f_heat'] = self._compute_heat_feature(pos, city_state)
        
        # crowding: 拥挤惩罚
        features['crowding'] = self._compute_crowding_penalty(pos, city_state)
        
        # junction_penalty: 路口惩罚
        features['junction_penalty'] = self._compute_junction_penalty(pos, backbone_system)
        
        return features
    
    def _compute_road_feature(self, pos: List[int], backbone_system: 'GovernmentBackboneSystem') -> float:
        """计算道路特征：到走廊线的法向高斯核"""
        center_y = backbone_system.road_corridor['center_y']
        d_perp = abs(pos[1] - center_y) * self.meters_per_pixel
        sigma_perp = backbone_system.sigma_perp_m
        
        return math.exp(-(d_perp**2) / (2 * sigma_perp**2))
    
    def _compute_hub_feature(self, pos: List[int], hub_pos: List[int], hub_config: Dict) -> float:
        """计算枢纽特征：各向异性高斯核"""
        dx = (pos[0] - hub_pos[0]) * self.meters_per_pixel
        dy = (pos[1] - hub_pos[1]) * self.meters_per_pixel
        
        sigma_perp = hub_config.get('sigma_perp_m', 30)
        sigma_para = hub_config.get('sigma_parallel_m', 90)
        
        # 各向异性高斯（沿走廊方向更长）
        # 简化处理：假设走廊是水平的
        d_perp = abs(dy)
        d_para = abs(dx)
        
        return math.exp(-(d_perp**2 / (2 * sigma_perp**2) + d_para**2 / (2 * sigma_para**2)))
    
    def _compute_access_feature(self, pos: List[int], city_state: Dict) -> float:
        """计算公共设施可达性"""
        public_facilities = city_state.get('public', [])
        if not public_facilities:
            return 0.0
        
        max_access = 0.0
        for facility in public_facilities:
            facility_pos = facility.get('xy', [0, 0])
            distance = math.sqrt((pos[0] - facility_pos[0])**2 + (pos[1] - facility_pos[1])**2)
            distance_m = distance * self.meters_per_pixel
            
            # 服务半径内的可达性
            service_radius = facility.get('service_radius', 300)
            if distance_m <= service_radius:
                access = 1.0 - (distance_m / service_radius)
                max_access = max(max_access, access)
        
        return max_access
    
    def _compute_heat_feature(self, pos: List[int], city_state: Dict) -> float:
        """计算居民轨迹热力"""
        # 简化实现：基于附近居民密度
        residents = city_state.get('residents', [])
        if not residents:
            return 0.0
        
        heat = 0.0
        for resident in residents:
            resident_pos = resident.get('position', [0, 0])
            distance = math.sqrt((pos[0] - resident_pos[0])**2 + (pos[1] - resident_pos[1])**2)
            
            # 距离衰减
            if distance <= 50:  # 50像素范围内
                heat += math.exp(-distance / 20.0)
        
        return min(heat, 1.0)  # 归一化到[0,1]
    
    def _compute_crowding_penalty(self, pos: List[int], city_state: Dict) -> float:
        """计算拥挤惩罚"""
        # 计算附近建筑密度
        buildings = city_state.get('residential', []) + city_state.get('commercial', []) + city_state.get('industrial', [])
        
        nearby_buildings = 0
        for building in buildings:
            building_pos = building.get('xy', [0, 0])
            distance = math.sqrt((pos[0] - building_pos[0])**2 + (pos[1] - building_pos[1])**2)
            if distance <= 30:  # 30像素范围内
                nearby_buildings += 1
        
        # 拥挤惩罚：建筑密度过高
        return min(nearby_buildings / 10.0, 1.0)
    
    def _compute_junction_penalty(self, pos: List[int], backbone_system: 'GovernmentBackboneSystem') -> float:
        """计算路口惩罚"""
        # 简化实现：靠近枢纽的惩罚
        dist_com = math.sqrt((pos[0] - backbone_system.hub_commercial_pos[0])**2 + 
                           (pos[1] - backbone_system.hub_commercial_pos[1])**2)
        dist_ind = math.sqrt((pos[0] - backbone_system.hub_industrial_pos[0])**2 + 
                           (pos[1] - backbone_system.hub_industrial_pos[1])**2)
        
        min_dist = min(dist_com, dist_ind)
        
        # 距离枢纽太近的惩罚
        if min_dist <= 20:  # 20像素内
            return 1.0 - (min_dist / 20.0)
        
        return 0.0
    
    def compute_scores(self, features: Dict[str, float], building_type: str) -> float:
        """计算建筑类型评分"""
        weights = self.weights.get(building_type, {})
        
        score = 0.0
        for feature_name, weight in weights.items():
            if feature_name in features:
                score += weight * features[feature_name]
        
        return score
    
    def compute_all_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """计算所有建筑类型的评分"""
        scores = {}
        for building_type in ['commercial', 'industrial', 'residential']:
            scores[building_type] = self.compute_scores(features, building_type)
        
        return scores

class UnifiedDecisionMaker:
    """统一决策器 v3.2"""
    
    def __init__(self, backbone_system: 'GovernmentBackboneSystem', 
                 scoring_system: 'FeatureScoringSystem'):
        self.backbone_system = backbone_system
        self.scoring_system = scoring_system
        
        print(f"🎯 统一决策器 v3.2 初始化完成")
    
    def place_on_active_layer(self, active_layer: Layer, quotas: Dict[str, int], 
                            city_state: Dict) -> List[Dict]:
        """在激活层上放置建筑（统一投标机制）"""
        print(f"🏗️ 在层 {active_layer.layer_id} 上统一投标...")
        
        # 获取可用槽位
        free_slots = [slot for slot in active_layer.slots 
                     if not slot.used and not slot.dead]
        
        if not free_slots:
            print(f"⚠️ 层 {active_layer.layer_id} 没有可用槽位")
            return []
        
        # 为每个槽位计算特征和评分
        buckets = {"commercial": [], "residential": [], "industrial": []}
        
        for slot in free_slots:
            # 计算特征
            features = self.scoring_system.compute_features(slot, self.backbone_system, city_state)
            slot.features = features
            
            # 计算所有建筑类型的评分
            scores = self.scoring_system.compute_all_scores(features)
            slot.scores = scores
            
            # 应用分区约束
            allowed_types = self.backbone_system.get_zoning_constraints(slot.pos)
            
            # 找到最佳建筑类型
            best_type = None
            best_score = float('-inf')
            
            for building_type in ['commercial', 'residential', 'industrial']:
                if building_type in allowed_types and scores[building_type] > best_score:
                    best_score = scores[building_type]
                    best_type = building_type
            
            if best_type:
                buckets[best_type].append((slot, best_score))
        
        # 按评分排序
        for building_type in buckets:
            buckets[building_type].sort(key=lambda x: x[1], reverse=True)
        
        # 根据配额分配建筑
        placed = []
        for building_type, quota in quotas.items():
            if quota <= 0:
                continue
            
            # 取前N个最高评分的槽位
            candidates = buckets[building_type][:quota]
            
            for slot, score in candidates:
                building = self._create_building(slot, building_type, score, city_state)
                if building:
                    placed.append(building)
                    slot.used = True
                    slot.building_id = building['id']
                    
                    print(f"  ✅ {building_type}建筑 {building['id']} 中标，评分: {score:.3f}")
        
        # 更新层统计
        active_layer.update_stats()
        
        # 检查是否可以标记为完成
        if active_layer.density >= 0.95:  # 95%以上算满格
            active_layer.status = "complete"
            print(f"✅ 层 {active_layer.layer_id} 已完成 (密度: {active_layer.density:.1%})")
        
        return placed
    
    def _create_building(self, slot: Slot, building_type: str, score: float, 
                        city_state: Dict) -> Optional[Dict]:
        """创建建筑对象"""
        building_id = f"{building_type}_{len(city_state.get(building_type, [])) + 1}"
        
        # 建筑属性配置
        building_configs = {
            'commercial': {
                'type': 'commercial',
                'capacity': 800,
                'construction_cost': 1000,
                'revenue_per_person': 20
            },
            'residential': {
                'type': 'residential',
                'capacity': 200,
                'construction_cost': 500,
                'revenue_per_person': 10
            },
            'industrial': {
                'type': 'industrial',
                'capacity': 1200,
                'construction_cost': 1500,
                'revenue_per_person': 15
            }
        }
        
        config = building_configs.get(building_type, building_configs['residential'])
        
        building = {
            'id': building_id,
            'type': config['type'],
            'xy': slot.pos,
            'capacity': config['capacity'],
            'current_usage': 0,
            'construction_cost': config['construction_cost'],
            'revenue_per_person': config['revenue_per_person'],
            'revenue': 0,
            'land_price_value': 0.0,  # 将在后续更新
            'slot_id': f"{building_type}_{slot.pos[0]}_{slot.pos[1]}",
            'features': slot.features.copy(),
            'scores': slot.scores.copy(),
            'winning_score': score
        }
        
        return building
    
    def get_decision_log(self, active_layer: Layer) -> Dict:
        """获取决策日志"""
        log = {
            'layer_id': active_layer.layer_id,
            'total_slots': len(active_layer.slots),
            'free_slots': len([s for s in active_layer.slots if not s.used and not s.dead]),
            'used_slots': len([s for s in active_layer.slots if s.used]),
            'dead_slots': len([s for s in active_layer.slots if s.dead]),
            'density': active_layer.density,
            'slot_details': []
        }
        
        for slot in active_layer.slots:
            if slot.features and slot.scores:
                slot_detail = {
                    'pos': slot.pos,
                    'used': slot.used,
                    'dead': slot.dead,
                    'features': slot.features,
                    'scores': slot.scores,
                    'building_id': slot.building_id
                }
                log['slot_details'].append(slot_detail)
        
        return log

class ProgressiveGrowthSystemV3_2:
    """渐进式增长系统 v3.2"""
    
    def __init__(self, config: Dict):
        self.config = config.get('progressive_growth', {})
        self.strict_fill_required = self.config.get('strict_fill_required', True)
        self.allow_dead_slots_ratio = self.config.get('allow_dead_slots_ratio', 0.05)
        
        # 层管理
        self.layers = []  # 所有层
        self.active_layers = []  # 当前激活的层
        self.completed_layers = []  # 已完成的层
        
        # 生长阶段
        self.growth_phase = 'road_corridor'  # road_corridor -> radial_expansion
        self.road_layers_completed = False
        
        print(f"🏗️ 渐进式增长系统 v3.2 初始化完成")
        print(f"   严格满格要求: {self.strict_fill_required}")
        print(f"   死槽容忍率: {self.allow_dead_slots_ratio:.1%}")
    
    def initialize_layers(self, backbone_system: 'GovernmentBackboneSystem', map_size: List[int]):
        """初始化建筑层"""
        print("🔧 初始化建筑层...")
        
        # 生成走廊带层
        road_layers = backbone_system.get_road_corridor_layers(map_size)
        self.layers.extend(road_layers)
        
        print(f"✅ 建筑层初始化完成，共 {len(self.layers)} 个层")
        self._print_layer_status()
    
    def activate_road_layers(self, quarter: int):
        """激活走廊带层"""
        print("🎯 激活走廊带层...")
        
        road_layers = [layer for layer in self.layers if layer.layer_type == 'road']
        
        for layer in road_layers:
            layer.status = "active"
            layer.activated_quarter = quarter
            self.active_layers.append(layer)
            print(f"  ✅ 激活层 {layer.layer_id}")
        
        self.growth_phase = 'road_corridor'
    
    def check_road_layers_completion(self) -> bool:
        """检查走廊带层是否完成"""
        road_layers = [layer for layer in self.layers if layer.layer_type == 'road']
        
        for layer in road_layers:
            layer.update_stats()
            if layer.density < 0.95:  # 95%以上算满格
                return False
        
        # 所有走廊带层都完成
        for layer in road_layers:
            layer.status = "complete"
            if layer in self.active_layers:
                self.active_layers.remove(layer)
            self.completed_layers.append(layer)
        
        self.road_layers_completed = True
        self.growth_phase = 'radial_expansion'
        
        # 创建放射扩张层
        map_size = [110, 110]  # 从配置中获取
        self._create_radial_layers(quarter=1, phase=0, map_size=map_size)  # 走廊带完成后进入第1季度
        
        print("✅ 走廊带层全部完成，进入放射扩张阶段")
        return True
    
    def check_radial_layers_completion(self, quarter: int):
        """检查放射层是否完成，如果完成则创建新层"""
        if not self.road_layers_completed:
            return
        
        # 获取当前激活的放射层
        active_radial_layers = [layer for layer in self.active_layers if layer.layer_type == 'radial']
        
        if not active_radial_layers:
            return
        
        # 检查是否有层完成
        completed_radial_layers = []
        for layer in active_radial_layers:
            layer.update_stats()
            if layer.density >= 0.95:  # 95%以上算满格
                layer.status = "complete"
                completed_radial_layers.append(layer)
                if layer in self.active_layers:
                    self.active_layers.remove(layer)
                self.completed_layers.append(layer)
                print(f"✅ 放射层 {layer.layer_id} 已完成 (密度: {layer.density:.1%})")
        
        # 如果有层完成，创建新的放射层
        if completed_radial_layers:
            # 计算下一个阶段号
            max_phase = 0
            for layer in self.layers:
                if 'radial_P' in layer.layer_id:
                    try:
                        phase_num = int(layer.layer_id.split('P')[1])
                        max_phase = max(max_phase, phase_num)
                    except:
                        pass
            
            next_phase = max_phase + 1
            print(f"🎯 创建下一阶段放射层 P{next_phase}...")
            map_size = [110, 110]  # 从配置中获取
            self._create_radial_layers(quarter, next_phase, map_size)
    
    def _create_radial_layers(self, quarter: int = 1, phase: int = 0, map_size: List[int] = None):
        """创建放射扩张层"""
        print(f"🎯 创建放射扩张层 P{phase}...")
        
        # 获取地图尺寸
        if map_size is None:
            map_size = [110, 110]  # 默认尺寸
        
        # 创建商业放射层
        commercial_radial_layer = self._create_radial_layer(
            f"commercial_radial_P{phase}", "commercial", map_size, quarter, phase
        )
        if commercial_radial_layer:
            self.layers.append(commercial_radial_layer)
            self.active_layers.append(commercial_radial_layer)
            print(f"  ✅ 创建商业放射层: {commercial_radial_layer.layer_id}")
        
        # 创建工业放射层
        industrial_radial_layer = self._create_radial_layer(
            f"industrial_radial_P{phase}", "industrial", map_size, quarter, phase
        )
        if industrial_radial_layer:
            self.layers.append(industrial_radial_layer)
            self.active_layers.append(industrial_radial_layer)
            print(f"  ✅ 创建工业放射层: {industrial_radial_layer.layer_id}")
        
        # 创建住宅放射层（走廊中段）
        residential_radial_layer = self._create_radial_layer(
            f"residential_radial_P{phase}", "residential", map_size, quarter, phase
        )
        if residential_radial_layer:
            self.layers.append(residential_radial_layer)
            self.active_layers.append(residential_radial_layer)
            print(f"  ✅ 创建住宅放射层: {residential_radial_layer.layer_id}")
    
    def _create_radial_layer(self, layer_id: str, building_type: str, map_size: List[int], quarter: int, phase: int = 0) -> Optional[Layer]:
        """创建单个放射层"""
        slots = []
        
        # 根据建筑类型和阶段确定采样参数
        if building_type == "commercial":
            # 商业放射：围绕Hub1（商业枢纽）
            hub_pos = [37, 55]  # 110/3 ≈ 37
            radius_start = 30 + phase * 15  # 每阶段向外扩展15像素
            radius_end = radius_start + 15
            radius_spacing = 8  # 半径间隔8像素
            angle_spacing = 12   # 角度间隔12度
        elif building_type == "industrial":
            # 工业放射：围绕Hub2（工业枢纽）
            hub_pos = [73, 55]  # 2*110/3 ≈ 73
            radius_start = 30 + phase * 18
            radius_end = radius_start + 18
            radius_spacing = 9
            angle_spacing = 15
        else:  # residential
            # 住宅放射：走廊中段区域
            hub_pos = [55, 55]  # 走廊中心
            radius_start = 25 + phase * 20
            radius_end = radius_start + 20
            radius_spacing = 10
            angle_spacing = 18
        
        # 检查半径是否超出地图边界
        max_radius_x = min(map_size[0] - hub_pos[0], hub_pos[0])
        max_radius_y = min(map_size[1] - hub_pos[1], hub_pos[1])
        max_radius = min(max_radius_x, max_radius_y)
        
        if radius_start >= max_radius:
            print(f"⚠️ 层 {layer_id} 半径超出地图边界 (start={radius_start}, max={max_radius})，跳过创建")
            return None
        radius_end = min(radius_end, max_radius)
        
        print(f"🔍 层 {layer_id}: hub={hub_pos}, radius={radius_start}-{radius_end}, max_radius={max_radius}")
        
        # 在指定半径范围内创建槽位
        for radius in range(radius_start, radius_end, radius_spacing):
            for angle in range(0, 360, angle_spacing):  # 更密集的角度采样
                # 计算位置
                x = int(hub_pos[0] + radius * math.cos(math.radians(angle)))
                y = int(hub_pos[1] + radius * math.sin(math.radians(angle)))
                
                # 检查位置是否有效
                if 0 <= x < map_size[0] and 0 <= y < map_size[1]:
                    slot = Slot(
                        pos=[x, y],
                        allowed_types=[building_type],
                        features={},
                        scores={}
                    )
                    slots.append(slot)
        
        if slots:
            layer = Layer(
                layer_id=layer_id,
                status="active",  # 放射层直接激活
                activated_quarter=quarter,
                slots=slots,
                capacity=len(slots),
                dead_slots=0,
                capacity_effective=len(slots),
                placed=0,
                density=0.0,
                layer_type='radial'
            )
            return layer
        
        return None
    
    def get_active_layers(self) -> List[Layer]:
        """获取当前激活的层"""
        return self.active_layers
    
    def get_layer_status(self) -> Dict:
        """获取层状态信息"""
        status = {
            'growth_phase': self.growth_phase,
            'road_layers_completed': self.road_layers_completed,
            'total_layers': len(self.layers),
            'active_layers': len(self.active_layers),
            'completed_layers': len(self.completed_layers),
            'layers': []
        }
        
        for layer in self.layers:
            layer.update_stats()
            status['layers'].append({
                'layer_id': layer.layer_id,
                'status': layer.status,
                'layer_type': layer.layer_type,
                'activated_quarter': layer.activated_quarter,
                'capacity': layer.capacity,
                'dead_slots': layer.dead_slots,
                'capacity_effective': layer.capacity_effective,
                'placed': layer.placed,
                'density': layer.density
            })
        
        return status
    
    def _print_layer_status(self):
        """打印层状态"""
        print("\n📊 建筑层状态:")
        
        for layer in self.layers:
            status_icon = {
                'locked': '🔒',
                'active': '🟢',
                'complete': '✅'
            }.get(layer.status, '❓')
            
            print(f"  {status_icon} {layer.layer_id} ({layer.layer_type}): {layer.status}")
            print(f"     容量: {layer.placed}/{layer.capacity_effective} (死槽: {layer.dead_slots})")
            print(f"     密度: {layer.density:.1%}")

class BuildingStateTracker:
    """建筑状态追踪器 - 支持增量导出"""
    
    def __init__(self):
        self.current_buildings = {}  # {building_id: building_data}
        self.building_id_counter = 1
        self.state_cache = {}  # 缓存重建的状态
        self.cache_max_size = 5
    
    def get_new_buildings_this_month(self, city_state: Dict) -> List[Dict]:
        """获取这个月新增的建筑"""
        new_buildings = []
        
        for building_type in ['residential', 'commercial', 'industrial', 'public']:
            for building in city_state.get(building_type, []):
                building_id = building['id']
                if building_id not in self.current_buildings:
                    # 新建筑
                    new_buildings.append({
                        'id': building['id'],
                        'type': building['type'],
                        'position': building['xy'],
                        'land_price_value': building.get('land_price_value', 0.0),
                        'slot_id': building.get('slot_id', ''),
                        'features': building.get('features', {}),
                        'scores': building.get('scores', {}),
                        'winning_score': building.get('winning_score', 0.0)
                    })
                    # 更新当前状态
                    self.current_buildings[building_id] = building
        
        return new_buildings
    
    def get_full_state_at_month(self, target_month: int, output_dir: str = "enhanced_simulation_v3_2_output") -> Dict:
        """从增量数据重建到指定月份的状态"""
        # 检查缓存
        if target_month in self.state_cache:
            return self.state_cache[target_month]
        
        # 加载第1个月的完整状态
        full_state = {'buildings': []}
        month_01_file = f"{output_dir}/building_positions_month_01.json"
        
        if os.path.exists(month_01_file):
            with open(month_01_file, 'r', encoding='utf-8') as f:
                month_01_data = json.load(f)
                full_state['buildings'] = month_01_data.get('buildings', [])
        
        # 累加后续月份的新增建筑
        for month in range(2, target_month + 1):
            delta_file = f"{output_dir}/building_delta_month_{month:02d}.json"
            if os.path.exists(delta_file):
                with open(delta_file, 'r', encoding='utf-8') as f:
                    delta_data = json.load(f)
                full_state['buildings'].extend(delta_data.get('new_buildings', []))
        
        # 缓存结果
        if len(self.state_cache) >= self.cache_max_size:
            oldest_month = min(self.state_cache.keys())
            del self.state_cache[oldest_month]
        self.state_cache[target_month] = full_state
        
        return full_state

class EnhancedCitySimulationV3_2:
    """增强城市模拟系统 v3.2"""
    
    def __init__(self):
        """初始化模拟系统"""
        # 加载配置
        self.city_config = self._load_config('configs/city_config_v3_2.json')
        self.building_config = self._load_config('configs/building_config.json')
        self.agent_config = self._load_config('configs/agent_config.json')
        
        # 初始化新系统
        self.backbone_system = GovernmentBackboneSystem(self.city_config)
        self.scoring_system = FeatureScoringSystem(self.city_config)
        self.decision_maker = UnifiedDecisionMaker(self.backbone_system, self.scoring_system)
        self.progressive_growth_system = ProgressiveGrowthSystemV3_2(self.city_config)
        
        # 保留兼容性系统
        self.land_price_system = GaussianLandPriceSystem(self.city_config)
        self.isocontour_system = IsocontourBuildingSystem(self.city_config)
        self.hysteresis_system = HysteresisSystem(self.city_config)
        self.public_facility_system = PublicFacilitySystem(self.city_config)
        
        # 初始化智能体
        self.government_agent = GovernmentAgent(self.agent_config.get('government_agent', {}))
        self.business_agent = BusinessAgent(self.agent_config.get('business_agent', {}))
        self.resident_agent = ResidentAgent(self.agent_config.get('resident_agent', {}))
        
        # 初始化其他系统
        self.output_system = OutputSystem('enhanced_simulation_v3_2_output')
        self.trajectory_system = TrajectorySystem([256, 256], self.building_config)
        self.building_tracker = BuildingStateTracker()
        
        # 模拟状态
        self.current_month = 0
        self.current_quarter = 0
        self.current_year = 0
        self.city_state = {}
        
        print(f"🏙️ 增强城市模拟系统 v3.2 初始化完成")
        print(f"🎯 新特性：政府骨架系统、统一决策器、特征化评分、条带→放射生长")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告：配置文件 {config_path} 不存在，使用默认配置")
            return {}
    
    def initialize_simulation(self):
        """初始化模拟"""
        print("🔧 初始化模拟系统...")
        
        # 获取配置
        map_size = self.city_config.get('city', {}).get('map_size', [256, 256])
        transport_hubs = self.city_config.get('city', {}).get('transport_hubs', [[40, 128], [216, 128]])
        
        # 初始化政府骨架系统
        self.backbone_system.initialize_backbone(map_size, transport_hubs)
        
        # 初始化渐进式增长系统
        self.progressive_growth_system.initialize_layers(self.backbone_system, map_size)
        
        # 初始化兼容性系统
        self.land_price_system.initialize_system(transport_hubs, map_size)
        land_price_field = self.land_price_system.get_land_price_field()
        self.isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
        
        # 初始化城市状态
        self.city_state = {
            'core_point': [map_size[0]//2, map_size[1]//2],
            'transport_hubs': transport_hubs,
            'public': [],
            'residential': [],
            'commercial': [],
            'industrial': [],  # 新增工业建筑
            'residents': [],
            'land_price_field': land_price_field,
            'land_price_stats': self.land_price_system.get_land_price_stats(),
            'layers': self.progressive_growth_system.get_layer_status(),
            'backbone_info': {
                'road_corridor': self.backbone_system.road_corridor,
                'hub_commercial': self.backbone_system.hub_commercial_pos,
                'hub_industrial': self.backbone_system.hub_industrial_pos
            }
        }
        
        print(f"✅ 模拟系统初始化完成")
    
    def run_simulation(self):
        """运行模拟"""
        simulation_months = self.city_config.get('simulation', {}).get('total_months', 24)
        
        print(f"🚀 开始运行 {simulation_months} 个月模拟 (v3.2)...")
        
        for month in range(simulation_months):
            self.current_month = month
            self.current_quarter = month // 3
            self.current_year = month // 12
            
            # 每月更新
            self._monthly_update()
            
            # 季度更新
            if month % 3 == 0:
                self._quarterly_update()
            
            # 年度更新
            if month % 12 == 0:
                self._yearly_update()
            
            # 保存输出
            self._save_monthly_outputs(month)
        
        # 最终输出
        self._save_final_outputs(simulation_months)
        print("✅ v3.2模拟完成！")
    
    def _monthly_update(self):
        """每月更新"""
        # 居民增长
        self._spawn_new_residents()
        
        # 更新轨迹系统
        self.trajectory_system.update_trajectories(self.city_state['residents'], self.city_state)
    
    def _quarterly_update(self):
        """季度更新"""
        print(f"📅 第 {self.current_quarter} 季度更新...")
        
        # 第一个季度：激活走廊带层
        if self.current_quarter == 0:
            self._activate_road_layers()
        
        # 检查走廊带层是否完成
        if not self.progressive_growth_system.road_layers_completed:
            self.progressive_growth_system.check_road_layers_completion()
        
        # 检查放射层是否完成，如果完成则创建新层
        self.progressive_growth_system.check_radial_layers_completion(self.current_quarter)
        
        # 生成建筑（基于统一决策器）
        buildings_generated = self._generate_buildings_with_unified_decision()
        
        # 滞后替代评估
        self._evaluate_hysteresis_conversion()
        
        # 公共设施评估
        self._evaluate_public_facilities()
        
        # 更新层状态
        self.city_state['layers'] = self.progressive_growth_system.get_layer_status()
    
    def _yearly_update(self):
        """年度更新"""
        print(f"📅 第 {self.current_year} 年更新...")
        
        # 高斯核地价场演化
        self.land_price_system.update_land_price_field(self.current_month, self.city_state)
        
        # 更新城市状态中的地价场
        self.city_state['land_price_field'] = self.land_price_system.get_land_price_field()
        self.city_state['land_price_stats'] = self.land_price_system.get_land_price_stats()
        
        # 重新初始化等值线系统（地价场变化后）
        map_size = self.city_config.get('city', {}).get('map_size', [256, 256])
        self.isocontour_system.initialize_system(
            self.city_state['land_price_field'], 
            self.city_state['transport_hubs'], 
            map_size
        )
    
    def _activate_road_layers(self):
        """激活走廊带层"""
        print("🎯 激活走廊带层...")
        self.progressive_growth_system.activate_road_layers(self.current_quarter)
    
    def _generate_buildings_with_unified_decision(self) -> bool:
        """基于统一决策器生成建筑"""
        print(f"🏗️ 第 {self.current_quarter} 季度：基于统一决策器生成建筑...")
        
        # 获取季度配额
        quotas = self.backbone_system.get_quarterly_quotas(self.current_quarter)
        print(f"   季度配额: 住宅{quotas['residential']}, 商业{quotas['commercial']}, 工业{quotas['industrial']}")
        
        # 获取激活的层
        active_layers = self.progressive_growth_system.get_active_layers()
        
        if not active_layers:
            print("⚠️ 没有激活的层")
            return False
        
        buildings_generated = False
        
        # 在每个激活的层上放置建筑
        for layer in active_layers:
            if layer.status == "active":
                placed_buildings = self.decision_maker.place_on_active_layer(
                    layer, quotas, self.city_state
                )
                
                if placed_buildings:
                    # 添加到城市状态
                    for building in placed_buildings:
                        building_type = building['type']
                        if building_type in self.city_state:
                            self.city_state[building_type].append(building)
                    
                    buildings_generated = True
                    print(f"  ✅ 在层 {layer.layer_id} 上放置了 {len(placed_buildings)} 个建筑")
        
        return buildings_generated
    
    def _evaluate_hysteresis_conversion(self):
        """评估滞后替代"""
        # 更新滞后替代系统季度
        self.hysteresis_system.update_quarter(self.current_quarter)
        
        # 评估替代条件
        conversion_result = self.hysteresis_system.evaluate_conversion_conditions(
            self.city_state, self.land_price_system
        )
        
        if conversion_result['should_convert']:
            # 执行替代
            candidates = conversion_result['candidates']
            if candidates:
                # 选择评分差异最大的候选建筑
                best_candidate = candidates[0]
                conversion_result = self.hysteresis_system.convert_building(
                    best_candidate['building_id'], self.city_state
                )
                
                if conversion_result['success']:
                    print(f"🔄 第 {self.current_quarter} 季度：住宅 {best_candidate['building_id']} 转换为商业建筑")
    
    def _evaluate_public_facilities(self):
        """评估公共设施需求"""
        # 简化实现
        pass
    
    def _spawn_new_residents(self):
        """生成新居民"""
        # 简化实现
        pass
    
    def _save_monthly_outputs(self, month: int):
        """保存月度输出"""
        # 保存地价场帧
        self.land_price_system.save_land_price_frame(month, 'enhanced_simulation_v3_2_output')
        
        # 保存建筑位置
        self._save_building_positions(month)
        
        # 保存简化格式的建筑位置
        self._save_simplified_building_positions(month)
        
        # 保存层状态
        self._save_layer_state(month)
        
        # 保存决策日志
        self._save_decision_log(month)
        
        print(f"💾 第 {month} 个月输出已保存")
    
    def _save_building_positions(self, month: int):
        """保存建筑位置 - 增量式导出"""
        if month == 0:
            # 第0个月保存完整状态
            self._save_full_building_state(month)
        else:
            # 后续月份：只保存增量文件
            self._save_new_buildings_only(month)
    
    def _save_full_building_state(self, month: int):
        """保存第1个月的完整建筑状态"""
        building_data = {
            'timestamp': f'month_{month:02d}',
            'buildings': []
        }
        
        # 添加所有建筑
        for building_type in ['residential', 'commercial', 'industrial', 'public']:
            for building in self.city_state.get(building_type, []):
                building_data['buildings'].append({
                    'id': building['id'],
                    'type': building['type'],
                    'position': building['xy'],
                    'land_price_value': building.get('land_price_value', 0.0),
                    'slot_id': building.get('slot_id', ''),
                    'features': building.get('features', {}),
                    'scores': building.get('scores', {}),
                    'winning_score': building.get('winning_score', 0.0)
                })
        
        # 保存到文件
        output_file = f"enhanced_simulation_v3_2_output/building_positions_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(building_data, f, indent=2, ensure_ascii=False)
        
        print(f"📦 第 {month} 个月完整状态已保存：{len(building_data['buildings'])} 个建筑")
    
    def _save_new_buildings_only(self, month: int):
        """保存新增建筑（增量文件）"""
        # 获取这个月新增的建筑
        new_buildings = self.building_tracker.get_new_buildings_this_month(self.city_state)
        
        if new_buildings:  # 只有新增建筑时才保存文件
            # 计算总建筑数
            total_buildings = sum(len(self.city_state.get(building_type, [])) 
                                for building_type in ['residential', 'commercial', 'industrial', 'public'])
            
            delta_data = {
                'month': month,
                'timestamp': f'month_{month:02d}',
                'new_buildings': new_buildings,
                'metadata': {
                    'total_buildings': total_buildings,
                    'new_count': len(new_buildings)
                }
            }
            
            # 保存增量文件
            output_file = f"enhanced_simulation_v3_2_output/building_delta_month_{month:02d}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(delta_data, f, indent=2, ensure_ascii=False)
            
            print(f"📈 第 {month} 个月增量已保存：{len(new_buildings)} 个新建筑")
        else:
            print(f"📊 第 {month} 个月无新建筑，跳过增量文件")
    
    def _save_simplified_building_positions(self, month: int):
        """保存简化格式的建筑位置数据"""
        # 类型映射
        type_map = {'residential': 0, 'commercial': 1, 'industrial': 2, 'office': 3, 'public': 4}
        
        # 格式化建筑数据
        formatted = []
        for building_type in ['residential', 'commercial', 'industrial', 'public']:
            for building in self.city_state.get(building_type, []):
                t = str(building.get('type', 'unknown')).lower()
                mid = type_map.get(t, 5)
                pos = building.get('xy', [0.0, 0.0])
                x = float(pos[0]) if len(pos) > 0 else 0.0
                y = float(pos[1]) if len(pos) > 1 else 0.0
                z = 0.0  # 默认高度为0
                formatted.append(f"{mid}({x:.3f}, {y:.3f}, {z:.0f})")
        
        # 生成简化格式的字符串
        simplified_line = ", ".join(formatted)
        
        # 保存到JSON文件
        simplified_data = {
            'month': month,
            'timestamp': f'month_{month:02d}',
            'simplified_format': simplified_line,
            'building_count': len(formatted)
        }
        
        # 创建simplified子文件夹
        simplified_dir = "enhanced_simulation_v3_2_output/simplified"
        os.makedirs(simplified_dir, exist_ok=True)
        
        # 保存JSON文件（带顺序编号）
        json_file = f"{simplified_dir}/simplified_buildings_{month:02d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, indent=2, ensure_ascii=False)
        
        # 保存纯文本文件（带顺序编号）
        txt_file = f"{simplified_dir}/simplified_buildings_{month:02d}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(simplified_line)
        
        print(f"📝 第 {month} 个月简化格式已保存：{len(formatted)} 个建筑")
    
    def _save_layer_state(self, month: int):
        """保存层状态"""
        layer_data = {
            'month': month,
            'quarter': self.current_quarter,
            'layers': self.city_state['layers']
        }
        
        output_file = f"enhanced_simulation_v3_2_output/layer_state_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(layer_data, f, indent=2, ensure_ascii=False)
    
    def _save_decision_log(self, month: int):
        """保存决策日志"""
        decision_log = {
            'month': month,
            'quarter': self.current_quarter,
            'active_layers': []
        }
        
        # 获取所有激活层的决策日志
        active_layers = self.progressive_growth_system.get_active_layers()
        for layer in active_layers:
            if layer.status == "active":
                layer_log = self.decision_maker.get_decision_log(layer)
                decision_log['active_layers'].append(layer_log)
        
        output_file = f"enhanced_simulation_v3_2_output/decision_log_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(decision_log, f, indent=2, ensure_ascii=False)
    
    def _save_final_outputs(self, simulation_months: int):
        """保存最终输出"""
        # 保存最终总结
        final_summary = {
            'simulation_months': simulation_months,
            'final_layers': self.city_state['layers'],
            'final_buildings': {
                'public': len(self.city_state['public']),
                'residential': len(self.city_state['residential']),
                'commercial': len(self.city_state['commercial']),
                'industrial': len(self.city_state['industrial'])
            },
            'land_price_evolution': self.land_price_system.get_evolution_history(),
            'backbone_info': self.city_state['backbone_info']
        }
        
        output_file = "enhanced_simulation_v3_2_output/final_summary.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        print("📊 所有v3.2输出文件已保存到 enhanced_simulation_v3_2_output/ 目录")
    
    def get_full_state_at_month(self, target_month: int) -> Dict:
        """获取指定月份的完整建筑状态（从增量数据重建）"""
        return self.building_tracker.get_full_state_at_month(target_month)

def main():
    """主函数"""
    print("🏙️ 增强城市模拟系统 v3.2")
    print("=" * 60)
    print("🎯 新特性：")
    print("  • 政府骨架系统：走廊带 + 双顶点架构")
    print("  • 统一决策器：商业/住宅/工业统一评分投标")
    print("  • 特征化评分：f_road, f_hub_com, f_hub_ind等多特征融合")
    print('  • 条带→放射生长：先沿走廊"街墙"排满，再双顶点向外放射')
    print("  • 分区约束与配额管理：政府定骨架，智能体按规则投标")
    print("  • 增量式建筑位置导出（节省存储空间）")
    print("=" * 60)
    
    # 创建并运行模拟
    simulation = EnhancedCitySimulationV3_2()
    simulation.initialize_simulation()
    simulation.run_simulation()
    
    print("\n🎉 v3.2模拟完成！")
    print("📁 输出文件保存在 enhanced_simulation_v3_2_output/ 目录")

if __name__ == "__main__":
    import os
    main()
