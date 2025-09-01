#!/usr/bin/env python3
"""
增强城市模拟系统 v3.1
基于PRD v3.1：槽位化、冻结施工线、严格逐层满格机制
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
    
    def __post_init__(self):
        if self.allowed_types is None:
            self.allowed_types = ['commercial', 'residential']

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
    
    def update_stats(self):
        """更新统计信息"""
        self.dead_slots = sum(1 for slot in self.slots if slot.dead)
        self.capacity_effective = self.capacity - self.dead_slots
        self.placed = sum(1 for slot in self.slots if slot.used)
        self.density = self.placed / self.capacity_effective if self.capacity_effective > 0 else 0.0

class ProgressiveGrowthSystem:
    """渐进式增长系统 v3.1"""
    
    def __init__(self, config: Dict):
        self.config = config.get('progressive_growth', {})
        self.strict_fill_required = self.config.get('strict_fill_required', True)
        self.allow_dead_slots_ratio = self.config.get('allow_dead_slots_ratio', 0.05)
        self.carry_over_quota = self.config.get('carry_over_quota', True)
        self.freeze_contour_on_activation = self.config.get('freeze_contour_on_activation', True)
        self.min_segment_length_factor = self.config.get('min_segment_length_factor', 3.0)
        
        # 层状态管理
        self.layers = {
            'commercial': [],
            'residential': []
        }
        
        # 当前激活层
        self.active_layers = {
            'commercial': 0,
            'residential': 0
        }
        
        # 季度配额
        self.quarterly_quotas = {
            'commercial': {'residential': 0, 'commercial': 0},
            'residential': {'residential': 0, 'commercial': 0}
        }
        
        print(f"🏗️ 渐进式增长系统 v3.1 初始化完成")
        print(f"   严格满格要求: {self.strict_fill_required}")
        print(f"   死槽容忍率: {self.allow_dead_slots_ratio:.1%}")
        print(f"   配额结转: {self.carry_over_quota}")
        print(f"   冻结施工线: {self.freeze_contour_on_activation}")
    
    def initialize_layers(self, isocontour_system, land_price_field):
        """初始化建筑层"""
        print("🔧 初始化建筑层...")
        
        # 为商业和住宅建筑分别创建层
        for building_type in ['commercial', 'residential']:
            self._create_layers_for_type(building_type, isocontour_system, land_price_field)
        
        print(f"✅ 建筑层初始化完成")
        self._print_layer_status()
    
    def _create_layers_for_type(self, building_type: str, isocontour_system, land_price_field):
        """为指定建筑类型创建层"""
        # 获取等值线数据
        contour_data = isocontour_system.get_contour_data_for_visualization()
        
        if building_type == 'commercial':
            contours = contour_data.get('commercial_contours', [])
        else:  # residential
            contours = contour_data.get('residential_contours', [])
        
        layers = []
        
        for i, contour in enumerate(contours):
            # 移除长度过滤，让99%等值线的4个点也能被使用
            # if len(contour) < self.min_segment_length_factor * 20:  # 过滤太短的等值线
            #     continue
            
            # 创建槽位
            slots = self._create_slots_from_contour(contour, building_type)
            
            # 创建层
            layer = Layer(
                layer_id=f"{building_type}_P{i}",
                status="locked",  # 初始状态为锁定
                activated_quarter=-1,
                slots=slots,
                capacity=len(slots),
                dead_slots=0,
                capacity_effective=len(slots),
                placed=0,
                density=0.0
            )
            
            layers.append(layer)
        
        self.layers[building_type] = layers
        
        # 所有层初始状态都是locked，不激活任何层
        print(f"🔒 {building_type}建筑：创建了 {len(layers)} 个层，初始状态均为locked")
    
    def _create_slots_from_contour(self, contour: List[List[int]], building_type: str) -> List[Slot]:
        """从等值线创建槽位 - 基于固定长度间隔"""
        slots = []
        
        # 固定长度间隔（像素单位）
        if building_type == 'commercial':
            spacing_pixels = random.randint(10, 20)  # 10-15像素间隔
        else:  # residential
            spacing_pixels = random.randint(10, 20)  # 15-25像素间隔，减少住宅间隔
        
        # 计算总弧长
        total_length = self._calculate_contour_length(contour)
        
        # 基于固定间隔计算槽位位置
        current_distance = spacing_pixels  # 从第一个间隔开始
        
        while current_distance < total_length:
            # 计算当前位置在等值线上的比例
            t = current_distance / total_length
            
            # 插值得到位置
            pos = self._interpolate_contour_position(contour, t)
            
            # 检查位置是否有效（在地图范围内）
            if 0 <= pos[0] < 110 and 0 <= pos[1] < 110:
                # 检查是否与已有槽位距离太近
                too_close = False
                # 对于小等值线（如99%, 98%, 97%, 96%, 95%, 94%, 92%, 91%等值线），使用更小的最小距离
                if len(contour) <= 20:  # 小等值线（现在最多20个点：2个hub × 10个点）
                    min_distance = 1.5  # 更小的最小距离，允许更密集的分布
                else:
                    min_distance = 8  # 正常最小距离
                
                for existing_slot in slots:
                    distance = math.sqrt((pos[0] - existing_slot.pos[0])**2 + (pos[1] - existing_slot.pos[1])**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    slot = Slot(
                        pos=pos,
                        allowed_types=[building_type]
                    )
                    slots.append(slot)
            
            # 移动到下一个间隔
            current_distance += spacing_pixels
        
        # 确保至少有一个槽位
        if not slots and len(contour) > 0:
            # 在等值线中点创建一个槽位
            pos = self._interpolate_contour_position(contour, 0.5)
            slot = Slot(
                pos=pos,
                allowed_types=[building_type]
            )
            slots.append(slot)
        
        return slots
    
    def _calculate_contour_length(self, contour) -> float:
        """计算等值线弧长"""
        if len(contour) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(contour) - 1):
            # 处理不同的等值线格式
            if isinstance(contour[i], list) and len(contour[i]) == 2:
                # 点列表格式: [x, y]
                p1 = contour[i]
                p2 = contour[i + 1]
            elif isinstance(contour[i], list) and len(contour[i]) == 1:
                # OpenCV格式: [[[x, y]]]
                p1 = contour[i][0]
                p2 = contour[i + 1][0]
            else:
                # 其他格式，尝试直接访问
                p1 = contour[i]
                p2 = contour[i + 1]
            
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance
        
        return total_length
    
    def _interpolate_contour_position(self, contour, t: float) -> List[int]:
        """在等值线上插值位置"""
        if len(contour) == 1:
            # 处理不同的等值线格式
            if isinstance(contour[0], list) and len(contour[0]) == 2:
                return contour[0]  # 点列表格式
            elif isinstance(contour[0], list) and len(contour[0]) == 1:
                return contour[0][0]  # OpenCV格式
            else:
                return contour[0]  # 其他格式
        
        # 计算总弧长
        total_length = self._calculate_contour_length(contour)
        target_length = t * total_length
        
        # 找到目标位置
        current_length = 0.0
        for i in range(len(contour) - 1):
            # 处理不同的等值线格式
            if isinstance(contour[i], list) and len(contour[i]) == 2:
                # 点列表格式: [x, y]
                p1 = contour[i]
                p2 = contour[i + 1]
            elif isinstance(contour[i], list) and len(contour[i]) == 1:
                # OpenCV格式: [[[x, y]]]
                p1 = contour[i][0]
                p2 = contour[i + 1][0]
            else:
                # 其他格式，尝试直接访问
                p1 = contour[i]
                p2 = contour[i + 1]
            
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if current_length + segment_length >= target_length:
                # 在这个段内插值
                segment_t = (target_length - current_length) / segment_length
                x = int(p1[0] + segment_t * (p2[0] - p1[0]))
                y = int(p1[1] + segment_t * (p2[1] - p1[1]))
                return [x, y]
            
            current_length += segment_length
        
        # 如果到达末尾，返回最后一个点
        if isinstance(contour[-1], list) and len(contour[-1]) == 2:
            return contour[-1]  # 点列表格式
        elif isinstance(contour[-1], list) and len(contour[-1]) == 1:
            return contour[-1][0]  # OpenCV格式
        else:
            return contour[-1]  # 其他格式
    
    def _activate_layer(self, building_type: str, layer_index: int, quarter: int):
        """激活指定层"""
        if layer_index >= len(self.layers[building_type]):
            return False
        
        layer = self.layers[building_type][layer_index]
        layer.status = "active"
        layer.activated_quarter = quarter
        
        # 不再冻结槽位位置，允许动态调整
        print(f"🎯 {building_type}建筑：激活第{layer_index}层 (P{layer_index}) - 动态模式")
        return True
    
    def can_activate_next_layer(self, building_type: str) -> bool:
        """检查是否可以激活下一层"""
        current_layer_idx = self.active_layers[building_type]
        current_layer = self.layers[building_type][current_layer_idx]
        
        # 检查当前层是否已满格
        if self.strict_fill_required:
            return current_layer.density >= 0.95  # 95%以上算满格
        else:
            return current_layer.density >= 0.8  # 80%以上算满格
    
    def try_activate_next_layer(self, building_type: str, quarter: int) -> bool:
        """尝试激活下一层"""
        current_layer_idx = self.active_layers[building_type]
        
        if self.can_activate_next_layer(building_type):
            next_layer_idx = current_layer_idx + 1
            
            if next_layer_idx < len(self.layers[building_type]):
                # 检查死槽率
                next_layer = self.layers[building_type][next_layer_idx]
                dead_ratio = next_layer.dead_slots / next_layer.capacity
                
                if dead_ratio <= self.allow_dead_slots_ratio:
                    self._activate_layer(building_type, next_layer_idx, quarter)
                    self.active_layers[building_type] = next_layer_idx
                    return True
                else:
                    print(f"⚠️ {building_type}建筑：第{next_layer_idx}层死槽率过高 ({dead_ratio:.1%} > {self.allow_dead_slots_ratio:.1%})")
            else:
                print(f"✅ {building_type}建筑：所有层已完成")
        
        return False
    
    def get_available_slots(self, building_type: str, target_count: int) -> List[Slot]:
        """获取可用的槽位"""
        # 在所有激活的层中寻找可用槽位
        available_slots = []
        
        for layer_idx, layer in enumerate(self.layers[building_type]):
            if layer.status == "active":
                # 获取当前层的未使用槽位
                layer_slots = [
                    slot for slot in layer.slots 
                    if not slot.used and not slot.dead and building_type in slot.allowed_types
                ]
                available_slots.extend(layer_slots)
        
        # 限制数量
        return available_slots[:target_count]
    
    def place_building_in_slot(self, building_type: str, building_id: str, slot: Slot):
        """在槽位中放置建筑"""
        slot.used = True
        slot.building_id = building_id
        
        # 更新层统计
        current_layer_idx = self.active_layers[building_type]
        current_layer = self.layers[building_type][current_layer_idx]
        current_layer.update_stats()
        
        # 检查是否可以标记为完成
        if current_layer.density >= 0.95:
            current_layer.status = "complete"
            print(f"✅ {building_type}建筑：第{current_layer_idx}层已完成 (密度: {current_layer.density:.1%})")
    
    def mark_slot_as_dead(self, building_type: str, slot: Slot, reason: str = "unknown"):
        """标记槽位为死槽"""
        slot.dead = True
        
        # 更新层统计
        current_layer_idx = self.active_layers[building_type]
        current_layer = self.layers[building_type][current_layer_idx]
        current_layer.update_stats()
        
        print(f"💀 {building_type}建筑：第{current_layer_idx}层槽位 {slot.pos} 标记为死槽 ({reason})")
    
    def get_layer_status(self) -> Dict:
        """获取层状态信息"""
        status = {}
        
        for building_type in ['commercial', 'residential']:
            status[building_type] = []
            
            for i, layer in enumerate(self.layers[building_type]):
                layer.update_stats()  # 确保统计是最新的
                
                status[building_type].append({
                    'layer_id': layer.layer_id,
                    'status': layer.status,
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
        
        for building_type in ['commercial', 'residential']:
            print(f"\n{building_type.upper()} 建筑:")
            
            for i, layer in enumerate(self.layers[building_type]):
                status_icon = {
                    'locked': '🔒',
                    'active': '🟢',
                    'complete': '✅'
                }.get(layer.status, '❓')
                
                print(f"  {status_icon} 第{i}层 ({layer.layer_id}): {layer.status}")
                print(f"     容量: {layer.placed}/{layer.capacity_effective} (死槽: {layer.dead_slots})")
                print(f"     密度: {layer.density:.1%}")
                print(f"     激活季度: {layer.activated_quarter if layer.activated_quarter >= 0 else '未激活'}")

class EnhancedCitySimulationV3_1:
    """增强城市模拟系统 v3.1"""
    
    def __init__(self):
        """初始化模拟系统"""
        # 加载配置
        self.city_config = self._load_config('configs/city_config_v3_1.json')
        self.building_config = self._load_config('configs/building_config.json')
        self.agent_config = self._load_config('configs/agent_config.json')
        
        # 初始化系统
        self.land_price_system = GaussianLandPriceSystem(self.city_config)
        self.isocontour_system = IsocontourBuildingSystem(self.city_config)
        self.hysteresis_system = HysteresisSystem(self.city_config)
        self.public_facility_system = PublicFacilitySystem(self.city_config)
        self.progressive_growth_system = ProgressiveGrowthSystem(self.city_config)
        
        # 初始化智能体
        self.government_agent = GovernmentAgent(self.agent_config.get('government_agent', {}))
        self.business_agent = BusinessAgent(self.agent_config.get('business_agent', {}))
        self.resident_agent = ResidentAgent(self.agent_config.get('resident_agent', {}))
        
        # 初始化其他系统
        self.output_system = OutputSystem('enhanced_simulation_v3_1_output')
        self.trajectory_system = TrajectorySystem([256, 256], self.building_config)
        
        # 模拟状态
        self.current_month = 0
        self.current_quarter = 0
        self.current_year = 0
        self.city_state = {}
        
        print(f"🏙️ 增强城市模拟系统 v3.1 初始化完成")
        print(f"🎯 新特性：槽位化、冻结施工线、严格逐层满格机制")
    
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
        
        # 初始化高斯核地价场系统
        self.land_price_system.initialize_system(transport_hubs, map_size)
        
        # 初始化等值线系统
        land_price_field = self.land_price_system.get_land_price_field()
        self.isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
        
        # 初始化渐进式增长系统
        self.progressive_growth_system.initialize_layers(self.isocontour_system, land_price_field)
        
        # 初始化城市状态
        self.city_state = {
            'core_point': [128, 128],
            'transport_hubs': transport_hubs,
            'public': [],
            'residential': [],
            'commercial': [],
            'residents': [],
            'land_price_field': land_price_field,
            'land_price_stats': self.land_price_system.get_land_price_stats(),
            'layers': self.progressive_growth_system.get_layer_status()
        }
        
        print(f"✅ 模拟系统初始化完成")
    
    def run_simulation(self):
        """运行模拟"""
        simulation_months = self.city_config.get('simulation', {}).get('total_months', 24)
        
        print(f"🚀 开始运行 {simulation_months} 个月模拟 (v3.1)...")
        
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
        print("✅ v3.1模拟完成！")
    
    def _monthly_update(self):
        """每月更新"""
        # 居民增长
        self._spawn_new_residents()
        
        # 更新轨迹系统
        self.trajectory_system.update_trajectories(self.city_state['residents'], self.city_state)
    
    def _quarterly_update(self):
        """季度更新"""
        print(f"📅 第 {self.current_quarter} 季度更新...")
        
        # 第一个季度：手动激活第一层
        if self.current_quarter == 0:
            self._activate_first_layers()
        
        # 生成建筑（基于槽位系统）
        buildings_generated = self._generate_buildings_with_slots()
        
        # 滞后替代评估
        self._evaluate_hysteresis_conversion()
        
        # 公共设施评估
        self._evaluate_public_facilities()
        
        # 尝试激活下一层
        self._try_activate_next_layers()
        
        # 检查是否需要创建新的等值层（当没有新建筑生成时）
        if not buildings_generated:
            self._create_new_isocontour_layers_when_no_growth()
        
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
        self.isocontour_system.initialize_system(
            self.city_state['land_price_field'], 
            self.city_state['transport_hubs'], 
            [110, 110]  # 修正地图尺寸
        )
        
        # 动态调整槽位系统
        self._update_slots_for_land_price_changes()
        
        # 尝试激活新的层
        self._try_activate_new_layers_after_update()
    
    def _update_slots_for_land_price_changes(self):
        """根据地价场变化动态调整槽位"""
        print("🔄 动态调整槽位系统...")
        
        # 获取新的等值线数据
        contour_data = self.isocontour_system.get_contour_data_for_visualization()
        
        # 执行原位替换和新增槽位
        self._perform_in_place_replacement(contour_data)
        
        print("✅ 槽位系统动态调整完成")
    
    def _perform_in_place_replacement(self, contour_data: Dict):
        """执行原位替换和新增槽位"""
        print("🔄 执行原位替换和新增槽位...")
        
        # 1. 评估现有建筑是否需要替换
        self._evaluate_building_replacements()
        
        # 2. 为新的等值线添加额外槽位
        self._add_slots_for_new_contours(contour_data)
        
        print("✅ 原位替换和新增槽位完成")
    
    def _evaluate_building_replacements(self):
        """评估建筑是否需要替换"""
        print("🔄 评估建筑替换需求...")
        
        # 获取当前地价场
        current_land_price_field = self.city_state['land_price_field']
        
        # 评估住宅建筑
        for building in self.city_state['residential']:
            self._evaluate_single_building_replacement(building, 'residential', current_land_price_field)
        
        # 评估商业建筑
        for building in self.city_state['commercial']:
            self._evaluate_single_building_replacement(building, 'commercial', current_land_price_field)
    
    def _evaluate_single_building_replacement(self, building: Dict, building_type: str, land_price_field):
        """评估单个建筑是否需要替换"""
        position = building['xy']
        current_land_price = land_price_field[position[1], position[0]]
        
        # 根据建筑类型确定合适的地价范围
        if building_type == 'commercial':
            # 商业建筑需要较高地价
            replacement_threshold = 0.6  # 低于60%地价时考虑替换
        else:  # residential
            # 住宅建筑适合中等偏低地价
            replacement_threshold = 0.8  # 高于80%地价时考虑替换
        
        # 检查是否需要替换
        if building_type == 'commercial' and current_land_price < replacement_threshold:
            # 商业建筑地价过低，考虑替换为住宅
            self._replace_building_type(building, 'residential')
            print(f"  🔄 商业建筑 {building['id']} 因地价过低 ({current_land_price:.3f}) 替换为住宅")
            
        elif building_type == 'residential' and current_land_price > replacement_threshold:
            # 住宅建筑地价过高，考虑替换为商业
            self._replace_building_type(building, 'commercial')
            print(f"  🔄 住宅建筑 {building['id']} 因地价过高 ({current_land_price:.3f}) 替换为商业")
    
    def _replace_building_type(self, building: Dict, new_type: str):
        """原位替换建筑类型"""
        old_type = building['type']
        
        # 更新建筑属性
        building['type'] = new_type
        if new_type == 'commercial':
            building['capacity'] = 800
            building['construction_cost'] = 1000
            building['revenue_per_person'] = 20
        else:  # residential
            building['capacity'] = 200
            building['construction_cost'] = 500
            building['revenue_per_person'] = 10
        
        # 更新地价值
        position = building['xy']
        building['land_price_value'] = float(self.city_state['land_price_field'][position[1], position[0]])
        
        # 更新槽位ID
        building['slot_id'] = f"{new_type}_{position[0]}_{position[1]}"
        
        # 记录替换历史
        if 'replacement_history' not in building:
            building['replacement_history'] = []
        
        building['replacement_history'].append({
            'quarter': self.current_quarter,
            'from_type': old_type,
            'to_type': new_type,
            'reason': 'land_price_change'
        })
    
    def _add_slots_for_new_contours(self, contour_data: Dict):
        """为新的等值线添加额外槽位"""
        print("🔄 为新的等值线添加槽位...")
        
        # 检查商业等值线
        commercial_contours = contour_data.get('commercial_contours', [])
        self._add_slots_for_contours('commercial', commercial_contours)
        
        # 检查住宅等值线
        residential_contours = contour_data.get('residential_contours', [])
        self._add_slots_for_contours('residential', residential_contours)
    
    def _add_slots_for_contours(self, building_type: str, new_contours: List):
        """为指定建筑类型的新等值线添加槽位"""
        existing_layers = self.progressive_growth_system.layers[building_type]
        
        for i, contour in enumerate(new_contours):
            if len(contour) < 20:  # 过滤太短的等值线
                continue
            
            # 检查这个等值线是否已经有对应的层
            contour_has_layer = False
            for layer in existing_layers:
                for slot in layer.slots:
                    if self._is_slot_on_contour(slot, contour):
                        contour_has_layer = True
                        break
                if contour_has_layer:
                    break
            
            if not contour_has_layer:
                # 创建新的层
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
                
                new_layer = Layer(
                    layer_id=f"{building_type}_P{len(existing_layers)}",
                    status="locked",  # 新层初始状态为锁定
                    activated_quarter=-1,
                    slots=new_slots,
                    capacity=len(new_slots),
                    dead_slots=0,
                    capacity_effective=len(new_slots),
                    placed=0,
                    density=0.0
                )
                
                existing_layers.append(new_layer)
                print(f"  🆕 {building_type}建筑：为等值线 {i+1} 创建了新层 {new_layer.layer_id}")
    
    def _try_activate_new_layers_after_update(self):
        """年度更新后尝试激活新的层"""
        print("🔄 尝试激活年度更新后的新层...")
        
        for building_type in ['commercial', 'residential']:
            layers = self.progressive_growth_system.layers[building_type]
            
            # 检查是否有新的锁定层可以激活
            for i, layer in enumerate(layers):
                if layer.status == "locked":
                    # 激活新的层
                    self.progressive_growth_system._activate_layer(building_type, i, self.current_quarter)
                    print(f"  🎯 {building_type}建筑：年度更新后激活新层 {layer.layer_id}")
                    break  # 每次只激活一个层
    
    def _create_new_isocontour_layers_when_no_growth(self):
        """当没有新建筑生成时，创建新的等值层"""
        print("🆕 检测到无增长状态，创建新的等值层...")
        
        # 获取当前地价场
        current_land_price_field = self.city_state['land_price_field']
        
        # 为商业建筑创建新的等值层
        self._create_new_isocontour_layers_for_type('commercial', current_land_price_field)
        
        # 为住宅建筑创建新的等值层
        self._create_new_isocontour_layers_for_type('residential', current_land_price_field)
        
        print("✅ 新等值层创建完成")
    
    def _create_new_isocontour_layers_for_type(self, building_type: str, land_price_field):
        """为指定建筑类型创建新的等值层"""
        print(f"  🏗️ 为 {building_type} 建筑创建新等值层...")
        
        # 获取当前配置
        config = self.city_config.get('isocontour_layout', {}).get(building_type, {})
        percentiles = config.get('percentiles', [95, 90, 85])
        
        # 计算新的等值线阈值（使用更低的百分位数，创建更多层）
        new_percentiles = []
        for i, p in enumerate(percentiles):
            new_p = max(5, p - 20 - i * 8)  # 更激进地降低百分位数
            new_percentiles.append(new_p)
        
        print(f"    📊 新百分位数: {new_percentiles}")
        
        # 为每个新的百分位数创建等值线
        for i, percentile in enumerate(new_percentiles):
            # 计算阈值
            threshold = np.percentile(land_price_field, percentile)
            
            # 创建等值线掩码
            mask = (land_price_field >= threshold).astype(np.uint8)
            
            # 查找等值线
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤太短的等值线
            valid_contours = [c for c in contours if len(c) >= 20]
            
            if valid_contours:
                # 选择最长的等值线
                longest_contour = max(valid_contours, key=len)
                
                # 将OpenCV格式转换为点列表格式
                contour_points = []
                for point in longest_contour:
                    x, y = point[0][0], point[0][1]
                    contour_points.append([x, y])
                
                # 创建槽位
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour_points, building_type)
                
                if new_slots:
                    # 创建新层
                    existing_layers = self.progressive_growth_system.layers[building_type]
                    new_layer = Layer(
                        layer_id=f"{building_type}_P{len(existing_layers)}_new",
                        status="active",  # 直接激活新层
                        activated_quarter=self.current_quarter,
                        slots=new_slots,
                        capacity=len(new_slots),
                        dead_slots=0,
                        capacity_effective=len(new_slots),
                        placed=0,
                        density=0.0
                    )
                    
                    existing_layers.append(new_layer)
                    print(f"    🆕 创建新层 {new_layer.layer_id}，阈值 {threshold:.3f}，槽位 {len(new_slots)}")
                    break  # 每次只创建一个新层
    
    def _reinitialize_slots_for_land_price_changes(self, contour_data: Dict):
        """根据地价场变化重新初始化槽位系统"""
        print("🔄 重新初始化槽位系统...")
        
        # 保存当前建筑信息
        current_buildings = {
            'residential': self.city_state['residential'].copy(),
            'commercial': self.city_state['commercial'].copy()
        }
        
        # 重新创建所有层
        for building_type in ['commercial', 'residential']:
            contours = contour_data.get(f'{building_type}_contours', [])
            self._recreate_layers_for_type(building_type, contours)
        
        # 重新分配建筑到新的槽位
        self._redistribute_buildings_to_new_slots(current_buildings)
    
    def _recreate_layers_for_type(self, building_type: str, contours: List):
        """重新创建指定建筑类型的层"""
        existing_layers = self.progressive_growth_system.layers[building_type]
        
        # 清空现有层
        existing_layers.clear()
        
        # 重新创建层
        for i, contour in enumerate(contours):
            if len(contour) < 20:  # 过滤太短的等值线
                continue
            
            # 创建槽位
            slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
            
            # 创建层
            layer = Layer(
                layer_id=f"{building_type}_P{i}",
                status="locked",  # 初始状态为锁定
                activated_quarter=-1,
                slots=slots,
                capacity=len(slots),
                dead_slots=0,
                capacity_effective=len(slots),
                placed=0,
                density=0.0
            )
            
            existing_layers.append(layer)
        
        print(f"  🔄 {building_type}建筑：重新创建了 {len(existing_layers)} 个层")
    
    def _redistribute_buildings_to_new_slots(self, current_buildings: Dict):
        """重新分配建筑到新的槽位"""
        print("🔄 重新分配建筑到新槽位...")
        
        # 清空当前建筑列表
        self.city_state['residential'].clear()
        self.city_state['commercial'].clear()
        
        # 重新分配住宅建筑
        for building in current_buildings['residential']:
            self._redistribute_building(building, 'residential')
        
        # 重新分配商业建筑
        for building in current_buildings['commercial']:
            self._redistribute_building(building, 'commercial')
        
        print(f"  ✅ 重新分配完成：{len(current_buildings['residential'])} 个住宅，{len(current_buildings['commercial'])} 个商业建筑")
    
    def _redistribute_building(self, building: Dict, building_type: str):
        """重新分配单个建筑"""
        # 找到最近的可用槽位
        best_slot = self._find_best_slot_for_building(building, building_type)
        
        if best_slot:
            # 更新建筑位置
            building['xy'] = best_slot.pos
            building['land_price_value'] = float(self.city_state['land_price_field'][best_slot.pos[1], best_slot.pos[0]])
            building['slot_id'] = f"{building_type}_{best_slot.pos[0]}_{best_slot.pos[1]}"
            
            # 标记槽位为已使用
            best_slot.used = True
            best_slot.building_id = building['id']
            
            # 添加到城市状态
            self.city_state[building_type].append(building)
        else:
            # 如果找不到合适的槽位，标记为死槽
            print(f"  ⚠️ 建筑 {building['id']} 无法找到合适的槽位")
    
    def _find_best_slot_for_building(self, building: Dict, building_type: str) -> Optional[object]:
        """为建筑找到最佳槽位"""
        building_pos = building['xy']
        layers = self.progressive_growth_system.layers[building_type]
        
        best_slot = None
        min_distance = float('inf')
        
        # 在所有层中寻找最近的可用槽位
        for layer in layers:
            for slot in layer.slots:
                if not slot.used and not slot.dead and building_type in slot.allowed_types:
                    distance = ((slot.pos[0] - building_pos[0])**2 + (slot.pos[1] - building_pos[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_slot = slot
        
        return best_slot
    
    def _coordinate_isocontours_with_slots(self, contour_data: Dict):
        """协调等值线与槽位系统"""
        print("🔄 协调等值线与槽位系统...")
        
        # 检查是否有新的等值线出现
        new_commercial_contours = contour_data.get('commercial_contours', [])
        new_residential_contours = contour_data.get('residential_contours', [])
        
        # 为新的等值线创建额外的层
        self._create_additional_layers_for_new_contours('commercial', new_commercial_contours)
        self._create_additional_layers_for_new_contours('residential', new_residential_contours)
        
        print("✅ 等值线与槽位系统协调完成")
    
    def _create_additional_layers_for_new_contours(self, building_type: str, new_contours: List):
        """为新的等值线创建额外的层"""
        existing_layers = self.progressive_growth_system.layers[building_type]
        
        # 检查是否有新的等值线需要创建层
        for i, contour in enumerate(new_contours):
            if len(contour) < 20:  # 过滤太短的等值线
                continue
            
            # 检查这个等值线是否已经有对应的层
            contour_has_layer = False
            for layer in existing_layers:
                for slot in layer.slots:
                    if self._is_slot_on_contour(slot, contour):
                        contour_has_layer = True
                        break
                if contour_has_layer:
                    break
            
            if not contour_has_layer:
                # 创建新的层
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
                
                new_layer = self.progressive_growth_system.Layer(
                    layer_id=f"{building_type}_P{len(existing_layers)}",
                    status="locked",  # 新层初始状态为锁定
                    activated_quarter=-1,
                    slots=new_slots,
                    capacity=len(new_slots),
                    dead_slots=0,
                    capacity_effective=len(new_slots),
                    placed=0,
                    density=0.0
                )
                
                existing_layers.append(new_layer)
                print(f"  🆕 {building_type}建筑：为等值线 {i+1} 创建了新层 {new_layer.layer_id}")
    
    def _update_slots_for_type(self, building_type: str, new_contours: List):
        """为指定建筑类型更新槽位"""
        if not new_contours:
            return
        
        # 获取当前激活层
        current_layer_idx = self.progressive_growth_system.active_layers[building_type]
        current_layer = self.progressive_growth_system.layers[building_type][current_layer_idx]
        
        # 如果当前层已完成，不需要调整
        if current_layer.status == "complete":
            return
        
        # 检查是否有新的等值线可以添加槽位
        for i, contour in enumerate(new_contours):
            if len(contour) < 20:  # 过滤太短的等值线
                continue
            
            # 检查这个等值线是否已经有对应的槽位
            existing_slots = [slot for slot in current_layer.slots if self._is_slot_on_contour(slot, contour)]
            
            if not existing_slots:
                # 创建新的槽位
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
                
                # 添加到当前层
                current_layer.slots.extend(new_slots)
                current_layer.capacity += len(new_slots)
                current_layer.capacity_effective += len(new_slots)
                
                print(f"  📍 {building_type}建筑：为等值线 {i+1} 添加了 {len(new_slots)} 个新槽位")
    
    def _is_slot_on_contour(self, slot, contour: List) -> bool:
        """检查槽位是否在指定等值线上"""
        slot_pos = slot.pos
        tolerance = 5  # 容差范围
        
        for contour_point in contour:
            distance = ((slot_pos[0] - contour_point[0])**2 + (slot_pos[1] - contour_point[1])**2)**0.5
            if distance <= tolerance:
                return True
        
        return False
    
    def _generate_buildings_with_slots(self):
        """基于槽位系统生成建筑"""
        print(f"🏗️ 第 {self.current_quarter} 季度：基于槽位系统生成建筑...")
        
        # 获取季度建筑增长目标（确保有足够的建筑生成）
        available_residential_slots = len(self.progressive_growth_system.get_available_slots('residential', 100))
        available_commercial_slots = len(self.progressive_growth_system.get_available_slots('commercial', 100))
        
        # 根据可用槽位确定目标（增加基础生成量）
        residential_target = min(random.randint(12, 20), available_residential_slots)
        commercial_target = min(random.randint(5, 12), available_commercial_slots)
        
        # 如果是年度更新后的第一个季度，增加生成目标
        if self.current_month % 12 == 0:
            residential_target = min(residential_target + 8, available_residential_slots)
            commercial_target = min(commercial_target + 5, available_commercial_slots)
            print(f"  📈 年度更新后增加生成目标：住宅 +8，商业 +5")
        
        # 生成住宅建筑
        new_residential = self._generate_residential_with_slots(residential_target)
        
        # 生成商业建筑
        new_commercial = self._generate_commercial_with_slots(commercial_target)
        
        # 添加到城市状态
        self.city_state['residential'].extend(new_residential)
        self.city_state['commercial'].extend(new_commercial)
        
        buildings_generated = len(new_residential) > 0 or len(new_commercial) > 0
        
        if buildings_generated:
            print(f"✅ 生成完成：{len(new_residential)} 个住宅，{len(new_commercial)} 个商业建筑")
            print(f"   可用槽位：住宅 {available_residential_slots}，商业 {available_commercial_slots}")
        else:
            print(f"⚠️ 没有生成新建筑 - 所有层已完成")
        
        return buildings_generated
    
    def _generate_residential_with_slots(self, target_count: int) -> List[Dict]:
        """基于槽位生成住宅建筑"""
        available_slots = self.progressive_growth_system.get_available_slots('residential', target_count)
        
        new_buildings = []
        for i, slot in enumerate(available_slots):
            building = {
                'id': f'res_{len(self.city_state["residential"]) + i + 1}',
                'type': 'residential',
                'xy': slot.pos,
                'capacity': 200,
                'current_usage': 0,
                'construction_cost': 500,
                'revenue_per_person': 10,
                'revenue': 0,
                'land_price_value': float(self.city_state['land_price_field'][slot.pos[1], slot.pos[0]]),
                'slot_id': f"residential_{slot.pos[0]}_{slot.pos[1]}"
            }
            
            new_buildings.append(building)
            
            # 标记槽位为已使用
            self.progressive_growth_system.place_building_in_slot('residential', building['id'], slot)
        
        return new_buildings
    
    def _generate_commercial_with_slots(self, target_count: int) -> List[Dict]:
        """基于槽位生成商业建筑"""
        available_slots = self.progressive_growth_system.get_available_slots('commercial', target_count)
        
        new_buildings = []
        for i, slot in enumerate(available_slots):
            building = {
                'id': f'com_{len(self.city_state["commercial"]) + i + 1}',
                'type': 'commercial',
                'xy': slot.pos,
                'capacity': 800,
                'current_usage': 0,
                'construction_cost': 1000,
                'revenue_per_person': 20,
                'revenue': 0,
                'land_price_value': float(self.city_state['land_price_field'][slot.pos[1], slot.pos[0]]),
                'slot_id': f"commercial_{slot.pos[0]}_{slot.pos[1]}"
            }
            
            new_buildings.append(building)
            
            # 标记槽位为已使用
            self.progressive_growth_system.place_building_in_slot('commercial', building['id'], slot)
        
        return new_buildings
    
    def _activate_first_layers(self):
        """激活前几层"""
        print("🎯 激活前几层...")
        
        # 商业建筑：只激活第一层（99%等值线），实现逐层生长
        commercial_layers = self.progressive_growth_system.layers['commercial']
        if commercial_layers:
            # 只激活第0层，其他层保持locked状态
            self.progressive_growth_system._activate_layer('commercial', 0, 0)
            print(f"✅ 商业建筑：激活第0层（99%等值线），实现逐层生长")
        
        # 住宅建筑：激活第一层（沿道路）
        residential_layers = self.progressive_growth_system.layers['residential']
        if residential_layers:
            self.progressive_growth_system._activate_layer('residential', 0, 0)
            print(f"✅ 住宅建筑：激活第0层（沿道路）")
    
    def _try_activate_next_layers(self):
        """尝试激活下一层"""
        for building_type in ['commercial', 'residential']:
            if self.progressive_growth_system.try_activate_next_layer(building_type, self.current_quarter):
                print(f"🎯 {building_type}建筑：成功激活下一层")
    
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
                    
                    # 更新槽位系统
                    self._update_slot_after_conversion(best_candidate['building_id'])
    
    def _update_slot_after_conversion(self, building_id: str):
        """转换后更新槽位系统"""
        # 找到被转换的建筑
        converted_building = None
        for building in self.city_state['residential']:
            if building['id'] == building_id:
                converted_building = building
                break
        
        if converted_building:
            # 从住宅列表中移除
            self.city_state['residential'].remove(converted_building)
            
            # 添加到商业列表
            converted_building['type'] = 'commercial'
            converted_building['capacity'] = 800  # 更新容量
            self.city_state['commercial'].append(converted_building)
            
            # 更新槽位状态
            slot_id = converted_building.get('slot_id', '')
            if slot_id:
                # 找到对应的槽位对象并标记为死槽
                self._mark_slot_as_dead_by_id('residential', slot_id, 'converted_to_commercial')
    
    def _mark_slot_as_dead_by_id(self, building_type: str, slot_id: str, reason: str):
        """根据槽位ID标记槽位为死槽"""
        current_layer_idx = self.progressive_growth_system.active_layers[building_type]
        current_layer = self.progressive_growth_system.layers[building_type][current_layer_idx]
        
        # 解析槽位ID获取位置信息
        # slot_id格式: "residential_x_y" 或 "commercial_x_y"
        try:
            parts = slot_id.split('_')
            if len(parts) >= 3:
                x, y = int(parts[-2]), int(parts[-1])
                
                # 找到对应的槽位
                for slot in current_layer.slots:
                    if slot.pos[0] == x and slot.pos[1] == y:
                        self.progressive_growth_system.mark_slot_as_dead(building_type, slot, reason)
                        break
        except (ValueError, IndexError):
            print(f"⚠️ 无法解析槽位ID: {slot_id}")
    
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
        self.land_price_system.save_land_price_frame(month, 'enhanced_simulation_v3_1_output')
        
        # 保存建筑位置
        self._save_building_positions(month)
        
        # 保存简化格式的建筑位置
        self._save_simplified_building_positions(month)
        
        # 保存层状态
        self._save_layer_state(month)
        
        print(f"💾 第 {month} 个月输出已保存")
    
    def _save_building_positions(self, month: int):
        """保存建筑位置"""
        building_data = {
            'timestamp': f'month_{month:02d}',
            'buildings': []
        }
        
        # 添加所有建筑
        for building_type in ['residential', 'commercial', 'public']:
            for building in self.city_state.get(building_type, []):
                building_data['buildings'].append({
                    'id': building['id'],
                    'type': building['type'],
                    'position': building['xy'],
                    'land_price_value': building.get('land_price_value', 0.0),
                    'slot_id': building.get('slot_id', '')
                })
        
        # 保存到文件
        output_file = f"enhanced_simulation_v3_1_output/building_positions_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(building_data, f, indent=2, ensure_ascii=False)
    
    def _save_simplified_building_positions(self, month: int):
        """保存简化格式的建筑位置数据"""
        # 类型映射
        type_map = {'residential': 0, 'commercial': 1, 'office': 2, 'public': 3}
        
        # 格式化建筑数据
        formatted = []
        for building_type in ['residential', 'commercial', 'public']:
            for building in self.city_state.get(building_type, []):
                t = str(building.get('type', 'unknown')).lower()
                mid = type_map.get(t, 4)
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
        simplified_dir = "enhanced_simulation_v3_1_output/simplified"
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
        
        output_file = f"enhanced_simulation_v3_1_output/layer_state_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(layer_data, f, indent=2, ensure_ascii=False)
    
    def _save_final_outputs(self, simulation_months: int):
        """保存最终输出"""
        # 保存最终总结
        final_summary = {
            'simulation_months': simulation_months,
            'final_layers': self.city_state['layers'],
            'final_buildings': {
                'public': len(self.city_state['public']),
                'residential': len(self.city_state['residential']),
                'commercial': len(self.city_state['commercial'])
            },
            'land_price_evolution': self.land_price_system.get_evolution_history()
        }
        
        output_file = "enhanced_simulation_v3_1_output/final_summary.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        print("📊 所有v3.1输出文件已保存到 enhanced_simulation_v3_1_output/ 目录")

def main():
    """主函数"""
    print("🏙️ 增强城市模拟系统 v3.1")
    print("=" * 60)
    print("🎯 新特性：")
    print("  • 槽位化与冻结施工线")
    print("  • 严格逐层满格机制")
    print("  • 死槽机制与容忍率")
    print("  • 高斯核地价潜力场")
    print("  • 逐层涟漪式生长感")
    print("=" * 60)
    
    # 创建并运行模拟
    simulation = EnhancedCitySimulationV3_1()
    simulation.initialize_simulation()
    simulation.run_simulation()
    
    print("\n🎉 v3.1模拟完成！")
    print("📁 输出文件保存在 enhanced_simulation_v3_1_output/ 目录")

if __name__ == "__main__":
    import os
    main()
