#!/usr/bin/env python3
"""
增强城市模拟系统 v3.5
基于PRD v3.5：阶段门控演化、槽位系统同步、Hub2工业后处理、增量导出
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

# Force-safe print for non-UTF8 consoles (strip non-ASCII on encode errors)
import builtins as _builtins
def print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode('ascii', 'ignore').decode('ascii'))
            else:
                safe_args.append(a)
        _builtins.print(*safe_args, **kwargs)

# 复用现有逻辑模块
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.hysteresis_system import HysteresisSystem
from logic.public_facility_system import PublicFacilitySystem
from logic.enhanced_agents import GovernmentAgent, BusinessAgent, ResidentAgent
from logic.output_system import OutputSystem
from logic.trajectory_system import TrajectorySystem


@dataclass
class Slot:
    pos: List[int]
    used: bool = False
    dead: bool = False
    allowed_types: List[str] = None
    building_id: Optional[str] = None

    def __post_init__(self):
        if self.allowed_types is None:
            self.allowed_types = ['commercial', 'residential']


@dataclass
class Layer:
    layer_id: str
    status: str
    activated_quarter: int
    slots: List[Slot]
    capacity: int
    dead_slots: int
    capacity_effective: int
    placed: int
    density: float

    def update_stats(self):
        self.dead_slots = sum(1 for slot in self.slots if slot.dead)
        self.capacity_effective = self.capacity - self.dead_slots
        self.placed = sum(1 for slot in self.slots if slot.used)
        self.density = self.placed / self.capacity_effective if self.capacity_effective > 0 else 0.0


class ProgressiveGrowthSystem:
    """渐进式增长系统 v3.5（沿用v3.1实现，修正空层判定与动态重建接口）"""

    def __init__(self, config: Dict):
        self.config = config.get('progressive_growth', {})
        self.strict_fill_required = self.config.get('strict_fill_required', True)
        self.allow_dead_slots_ratio = self.config.get('allow_dead_slots_ratio', 0.05)
        self.carry_over_quota = self.config.get('carry_over_quota', True)
        self.freeze_contour_on_activation = self.config.get('freeze_contour_on_activation', True)
        self.min_segment_length_factor = self.config.get('min_segment_length_factor', 3.0)

        self.layers = {'commercial': [], 'residential': []}
        self.active_layers = {'commercial': 0, 'residential': 0}
        self.quarterly_quotas = {
            'commercial': {'residential': 0, 'commercial': 0},
            'residential': {'residential': 0, 'commercial': 0}
        }
        print("[Progressive] system v3.5 initialized")

    def initialize_layers(self, isocontour_system, land_price_field):
        print("[Progressive] init layers...")
        for building_type in ['commercial', 'residential']:
            self._create_layers_for_type(building_type, isocontour_system, land_price_field)
        print("[Progressive] layers initialized")
        self._print_layer_status()

    def _create_layers_for_type(self, building_type: str, isocontour_system, land_price_field):
        contour_data = isocontour_system.get_contour_data_for_visualization()
        contours = contour_data.get(f'{building_type}_contours', [])
        layers = []
        for i, contour in enumerate(contours):
            slots = self._create_slots_from_contour(contour, building_type)
            layer = Layer(
                layer_id=f"{building_type}_P{i}",
                status="locked",
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
        print(f"[Progressive] {building_type}: created {len(layers)} layers, all locked")

    def _create_slots_from_contour(self, contour: List[List[int]], building_type: str) -> List[Slot]:
        slots = []
        # 从配置获取按米定义的弧长采样范围，并换算为像素
        meters_per_pixel = self.config.get('gaussian_land_price_system', {}).get('meters_per_pixel', 2.0)
        iso_cfg = self.config.get('isocontour_layout', {}).get(building_type, {})
        spacing_range_m = iso_cfg.get('arc_spacing_m', [10, 20])
        spacing_min_px = max(3, int(spacing_range_m[0] / meters_per_pixel))
        spacing_max_px = max(spacing_min_px, int(spacing_range_m[1] / meters_per_pixel))
        spacing_pixels = random.randint(spacing_min_px, spacing_max_px)
        total_length = self._calculate_contour_length(contour)
        current_distance = spacing_pixels
        while current_distance < total_length:
            t = current_distance / total_length
            pos = self._interpolate_contour_position(contour, t)
            if 0 <= pos[0] < 110 and 0 <= pos[1] < 110:
                too_close = False
                pg_cfg = self.config.get('progressive_growth', {})
                min_distance_large = float(pg_cfg.get('min_slot_distance_px', 8))
                min_distance_small = float(pg_cfg.get('small_contour_min_slot_distance_px', 1.5))
                min_distance = min_distance_small if len(contour) <= 20 else min_distance_large
                for existing_slot in slots:
                    distance = math.sqrt((pos[0] - existing_slot.pos[0])**2 + (pos[1] - existing_slot.pos[1])**2)
                    if distance < min_distance:
                        too_close = True
                        break
                if not too_close:
                    slots.append(Slot(pos=pos, allowed_types=[building_type]))
            current_distance += spacing_pixels
        if not slots and len(contour) > 0:
            pos = self._interpolate_contour_position(contour, 0.5)
            slots.append(Slot(pos=pos, allowed_types=[building_type]))
        return slots

    def _calculate_contour_length(self, contour) -> float:
        if len(contour) < 2:
            return 0.0
        total_length = 0.0
        for i in range(len(contour) - 1):
            p1 = contour[i][0] if isinstance(contour[i], list) and len(contour[i]) == 1 else contour[i]
            p2 = contour[i + 1][0] if isinstance(contour[i + 1], list) and len(contour[i + 1]) == 1 else contour[i + 1]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance
        return total_length

    def _interpolate_contour_position(self, contour, t: float) -> List[int]:
        if len(contour) == 1:
            return contour[0][0] if isinstance(contour[0], list) and len(contour[0]) == 1 else contour[0]
        total_length = self._calculate_contour_length(contour)
        target_length = t * total_length
        current_length = 0.0
        for i in range(len(contour) - 1):
            p1 = contour[i][0] if isinstance(contour[i], list) and len(contour[i]) == 1 else contour[i]
            p2 = contour[i + 1][0] if isinstance(contour[i + 1], list) and len(contour[i + 1]) == 1 else contour[i + 1]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if current_length + segment_length >= target_length:
                segment_t = (target_length - current_length) / segment_length
                x = int(p1[0] + segment_t * (p2[0] - p1[0]))
                y = int(p1[1] + segment_t * (p2[1] - p1[1]))
                return [x, y]
            current_length += segment_length
        last = contour[-1]
        return last[0] if isinstance(last, list) and len(last) == 1 else last

    def _activate_layer(self, building_type: str, layer_index: int, quarter: int):
        if layer_index >= len(self.layers[building_type]):
            return False
        layer = self.layers[building_type][layer_index]
        layer.status = "active"
        layer.activated_quarter = quarter
        print(f"[Progressive] {building_type}: activate layer {layer_index} (P{layer_index})")
        return True

    def can_activate_next_layer(self, building_type: str) -> bool:
        current_layer_idx = self.active_layers[building_type]
        current_layer = self.layers[building_type][current_layer_idx]
        if self.strict_fill_required:
            return current_layer.density >= 0.95
        else:
            return current_layer.density >= 0.8

    def try_activate_next_layer(self, building_type: str, quarter: int) -> bool:
        current_layer_idx = self.active_layers[building_type]
        if self.can_activate_next_layer(building_type):
            next_layer_idx = current_layer_idx + 1
            if next_layer_idx < len(self.layers[building_type]):
                next_layer = self.layers[building_type][next_layer_idx]
                dead_ratio = next_layer.dead_slots / next_layer.capacity if next_layer.capacity > 0 else 0.0
                if dead_ratio <= self.allow_dead_slots_ratio:
                    self._activate_layer(building_type, next_layer_idx, quarter)
                    self.active_layers[building_type] = next_layer_idx
                    return True
                else:
                    print(f"[Progressive] {building_type}: next layer {next_layer_idx} dead ratio high ({dead_ratio:.1%} > {self.allow_dead_slots_ratio:.1%})")
            else:
                print(f"[Progressive] {building_type}: all layers complete")
        return False

    def get_available_slots(self, building_type: str, target_count: int) -> List[Slot]:
        available_slots = []
        for layer in self.layers[building_type]:
            if layer.status == "active":
                layer_slots = [slot for slot in layer.slots if (not slot.used and not slot.dead and building_type in slot.allowed_types)]
                available_slots.extend(layer_slots)
        return available_slots[:target_count]

    def place_building_in_slot(self, building_type: str, building_id: str, slot: Slot):
        slot.used = True
        slot.building_id = building_id
        current_layer_idx = self.active_layers[building_type]
        current_layer = self.layers[building_type][current_layer_idx]
        current_layer.update_stats()
        if current_layer.density >= 0.95:
            current_layer.status = "complete"
            print(f"[Progressive] {building_type}: layer {current_layer_idx} complete (density {current_layer.density:.1%})")

    def mark_slot_as_dead(self, building_type: str, slot: Slot, reason: str = "unknown"):
        slot.dead = True
        current_layer_idx = self.active_layers[building_type]
        current_layer = self.layers[building_type][current_layer_idx]
        current_layer.update_stats()
        print(f"[Progressive] {building_type}: mark slot {slot.pos} dead ({reason}) on layer {current_layer_idx}")

    def get_layer_status(self) -> Dict:
        status = {}
        for building_type in ['commercial', 'residential']:
            status[building_type] = []
            for i, layer in enumerate(self.layers[building_type]):
                layer.update_stats()
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
        print("\n[Progressive] layer status:")
        for building_type in ['commercial', 'residential']:
            print(f"\n{building_type.upper()} 建筑:")
            for i, layer in enumerate(self.layers[building_type]):
                print(f"  layer {i} ({layer.layer_id}): {layer.status}")
                print(f"     容量: {layer.placed}/{layer.capacity_effective} (死槽: {layer.dead_slots})")
                print(f"     密度: {layer.density:.1%}")
                print(f"     激活季度: {layer.activated_quarter if layer.activated_quarter >= 0 else '未激活'}")


class BuildingStateTracker:
    def __init__(self):
        self.current_buildings = {}
        self.building_id_counter = 1
        self.state_cache = {}
        self.cache_max_size = 5

    def get_new_buildings_this_month(self, city_state: Dict) -> List[Dict]:
        new_buildings = []
        for building_type in ['residential', 'commercial', 'public', 'industrial']:
            for building in city_state.get(building_type, []):
                building_id = building['id']
                if building_id not in self.current_buildings:
                    new_buildings.append({
                        'id': building['id'],
                        'type': building['type'],
                        'position': building['xy'],
                        'land_price_value': building.get('land_price_value', 0.0),
                        'slot_id': building.get('slot_id', '')
                    })
                    self.current_buildings[building_id] = building
        return new_buildings

    def get_full_state_at_month(self, target_month: int, output_dir: str = "enhanced_simulation_v3_1_output") -> Dict:
        if target_month in self.state_cache:
            return self.state_cache[target_month]
        full_state = {'buildings': []}
        month_01_file = f"{output_dir}/building_positions_month_01.json"
        if os.path.exists(month_01_file):
            with open(month_01_file, 'r', encoding='utf-8') as f:
                month_01_data = json.load(f)
                full_state['buildings'] = month_01_data.get('buildings', [])
        for month in range(2, target_month + 1):
            delta_file = f"{output_dir}/building_delta_month_{month:02d}.json"
            if os.path.exists(delta_file):
                with open(delta_file, 'r', encoding='utf-8') as f:
                    delta_data = json.load(f)
                full_state['buildings'].extend(delta_data.get('new_buildings', []))
        if len(self.state_cache) >= self.cache_max_size:
            oldest_month = min(self.state_cache.keys())
            del self.state_cache[oldest_month]
        self.state_cache[target_month] = full_state
        return full_state


class EnhancedCitySimulationV3_5:
    def __init__(self):
        self.city_config = self._load_config('configs/city_config_v3_5.json')
        self.building_config = self._load_config('configs/building_config.json')
        self.agent_config = self._load_config('configs/agent_config.json')

        self.land_price_system = GaussianLandPriceSystem(self.city_config)
        self.isocontour_system = IsocontourBuildingSystem(self.city_config)
        self.hysteresis_system = HysteresisSystem(self.city_config)
        self.public_facility_system = PublicFacilitySystem(self.city_config)
        self.progressive_growth_system = ProgressiveGrowthSystem(self.city_config)

        self.government_agent = GovernmentAgent(self.agent_config.get('government_agent', {}))
        self.business_agent = BusinessAgent(self.agent_config.get('business_agent', {}))
        self.resident_agent = ResidentAgent(self.agent_config.get('resident_agent', {}))

        # v3.5 独立输出目录
        self.output_dir = 'enhanced_simulation_v3_5_output'
        self.output_system = OutputSystem(self.output_dir)
        self.trajectory_system = TrajectorySystem([256, 256], self.building_config)
        self.building_tracker = BuildingStateTracker()

        self.current_month = 0
        self.current_quarter = 0
        self.current_year = 0
        self.city_state = {}

        print("[Simulation] v3.5 initialized")

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告：配置文件 {config_path} 不存在，使用默认配置")
            return {}

    def initialize_simulation(self):
        print("[Simulation] initializing...")
        map_size = self.city_config.get('city', {}).get('map_size', [256, 256])
        transport_hubs = self.city_config.get('city', {}).get('transport_hubs', [[40, 128], [216, 128]])
        self.land_price_system.initialize_system(transport_hubs, map_size)
        land_price_field = self.land_price_system.get_land_price_field()
        self.isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, 0, self.land_price_system)
        print("[Simulation] slot system will initialize in first quarter")
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
        print(f"[Simulation] init done")

    def run_simulation(self):
        simulation_months = self.city_config.get('simulation', {}).get('total_months', 36)
        print(f"🚀 开始运行 {simulation_months} 个月模拟 (v3.5)...")
        for month in range(simulation_months):
            self.current_month = month
            self.current_quarter = month // 3
            self.current_year = month // 12
            self._monthly_update()
            if month % 3 == 0:
                self._quarterly_update()
            if month % 12 == 0 and month > 0:
                self._yearly_update()
            self._save_monthly_outputs(month)
        self._save_final_outputs(simulation_months)
        print("[Simulation] v3.5 finished")

    def _monthly_update(self):
        self._spawn_new_residents()
        self.trajectory_system.update_trajectories(self.city_state['residents'], self.city_state)

    def _quarterly_update(self):
        print(f"[Quarter] update q={self.current_quarter}...")
        if self.current_quarter == 0 and self.current_month == 0:
            self._initialize_slots_for_current_land_price()
            self._activate_first_layers()
        print(f"[Quarter] debug: q={self.current_quarter}, month={self.current_month}")
        buildings_generated = self._generate_buildings_with_slots()
        print(f"[Quarter] debug: buildings_generated={buildings_generated}")
        self._evaluate_hysteresis_conversion()
        self._evaluate_public_facilities()
        self._try_activate_next_layers()
        if not buildings_generated:
            self._create_new_isocontour_layers_when_no_growth()
        if hasattr(self.progressive_growth_system, 'layers') and len(self.progressive_growth_system.layers['commercial']) > 0:
            self.city_state['layers'] = self.progressive_growth_system.get_layer_status()
        else:
            self.city_state['layers'] = {'commercial': [], 'residential': []}

    def _yearly_update(self):
        print(f"[Year] update y={self.current_year}...")
        self.land_price_system.update_land_price_field(self.current_month, self.city_state)
        evolution_stage = self.land_price_system._get_evolution_stage(self.current_month)
        print(f"[Year] evolution: {evolution_stage['description']} ({evolution_stage['name']})")
        component_strengths = evolution_stage.get('component_strengths', {})
        print(f"[Year] strengths: road={component_strengths.get('road', 0):.1f}, hub1={component_strengths.get('hub1', 0):.1f}, hub2={component_strengths.get('hub2', 0):.1f}, hub3={component_strengths.get('hub3', 0):.1f}")
        self.city_state['land_price_field'] = self.land_price_system.get_land_price_field()
        self.city_state['land_price_stats'] = self.land_price_system.get_land_price_stats()
        self.isocontour_system.initialize_system(
            self.city_state['land_price_field'],
            self.city_state['transport_hubs'],
            [110, 110],
            self.current_month,
            self.land_price_system
        )
        self._rebuild_slots_for_land_price_changes()
        self._try_activate_new_layers_after_update()

    def _initialize_slots_for_current_land_price(self):
        print("[Slots] delayed init...")
        current_land_price_field = self.city_state['land_price_field']
        self.progressive_growth_system.initialize_layers(self.isocontour_system, current_land_price_field)
        print("[Slots] delayed init done")

    def _rebuild_slots_for_land_price_changes(self):
        print("[Slots] rebuild...")
        contour_data = self.isocontour_system.get_contour_data_for_visualization()
        current_buildings = {
            'residential': self.city_state['residential'].copy(),
            'commercial': self.city_state['commercial'].copy()
        }
        for building_type in ['commercial', 'residential']:
            contours = contour_data.get(f'{building_type}_contours', [])
            self._recreate_layers_for_type(building_type, contours)
        self._redistribute_buildings_to_new_slots(current_buildings)
        print("[Slots] rebuild done")

    def _recreate_layers_for_type(self, building_type: str, contours: List):
        """重新创建指定建筑类型的层（与v3.1一致的实现）"""
        existing_layers = self.progressive_growth_system.layers[building_type]
        existing_layers.clear()
        for i, contour in enumerate(contours):
            if len(contour) < 20:  # 过滤太短的等值线
                continue
            slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
            layer = Layer(
                layer_id=f"{building_type}_P{i}",
                status="locked",
                activated_quarter=-1,
                slots=slots,
                capacity=len(slots),
                dead_slots=0,
                capacity_effective=len(slots),
                placed=0,
                density=0.0
            )
            existing_layers.append(layer)
        print(f"  [Slots] {building_type}: recreated {len(existing_layers)} layers")

    def _perform_in_place_replacement(self, contour_data: Dict):
        print("[Slots] in-place replacement and additions...")
        self._evaluate_building_replacements()
        self._add_slots_for_new_contours(contour_data)
        print("[Slots] in-place done")

    def _evaluate_building_replacements(self):
        print("[Slots] evaluate replacements...")
        current_land_price_field = self.city_state['land_price_field']
        for building in self.city_state['residential']:
            self._evaluate_single_building_replacement(building, 'residential', current_land_price_field)
        for building in self.city_state['commercial']:
            self._evaluate_single_building_replacement(building, 'commercial', current_land_price_field)

    def _evaluate_single_building_replacement(self, building: Dict, building_type: str, land_price_field):
        position = building['xy']
        current_land_price = land_price_field[position[1], position[0]]
        if building_type == 'commercial':
            replacement_threshold = 0.6
        else:
            replacement_threshold = 0.8
        if building_type == 'commercial' and current_land_price < replacement_threshold:
            self._replace_building_type(building, 'residential')
            print(f"  [Slots] commercial {building['id']} -> residential (low price {current_land_price:.3f})")
        elif building_type == 'residential' and current_land_price > replacement_threshold:
            self._replace_building_type(building, 'commercial')
            print(f"  [Slots] residential {building['id']} -> commercial (high price {current_land_price:.3f})")

    def _replace_building_type(self, building: Dict, new_type: str):
        old_type = building['type']
        building['type'] = new_type
        if new_type == 'commercial':
            building['capacity'] = 800
            building['construction_cost'] = 1000
            building['revenue_per_person'] = 20
        else:
            building['capacity'] = 200
            building['construction_cost'] = 500
            building['revenue_per_person'] = 10
        position = building['xy']
        building['land_price_value'] = float(self.city_state['land_price_field'][position[1], position[0]])
        building['slot_id'] = f"{new_type}_{position[0]}_{position[1]}"
        if 'replacement_history' not in building:
            building['replacement_history'] = []
        building['replacement_history'].append({
            'quarter': self.current_quarter,
            'from_type': old_type,
            'to_type': new_type,
            'reason': 'land_price_change'
        })

    def _add_slots_for_new_contours(self, contour_data: Dict):
        print("[Slots] add for new contours...")
        self._add_slots_for_contours('commercial', contour_data.get('commercial_contours', []))
        self._add_slots_for_contours('residential', contour_data.get('residential_contours', []))

    def _add_slots_for_contours(self, building_type: str, new_contours: List):
        existing_layers = self.progressive_growth_system.layers[building_type]
        for i, contour in enumerate(new_contours):
            if len(contour) < 20:
                continue
            contour_has_layer = False
            for layer in existing_layers:
                for slot in layer.slots:
                    if self._is_slot_on_contour(slot, contour):
                        contour_has_layer = True
                        break
                if contour_has_layer:
                    break
            if not contour_has_layer:
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
                new_layer = Layer(
                    layer_id=f"{building_type}_P{len(existing_layers)}",
                    status="locked",
                    activated_quarter=-1,
                    slots=new_slots,
                    capacity=len(new_slots),
                    dead_slots=0,
                    capacity_effective=len(new_slots),
                    placed=0,
                    density=0.0
                )
                existing_layers.append(new_layer)
                print(f"  [Slots] {building_type}: new layer for contour {i+1} => {new_layer.layer_id}")

    def _try_activate_new_layers_after_update(self):
        print("[Slots] try activate new layers after yearly update...")
        if not hasattr(self.progressive_growth_system, 'layers') or len(self.progressive_growth_system.layers['commercial']) == 0:
            print("[Slots] not initialized, skip new-layer activation")
            return
        for building_type in ['commercial', 'residential']:
            layers = self.progressive_growth_system.layers[building_type]
            for i, layer in enumerate(layers):
                if layer.status == "locked":
                    self.progressive_growth_system._activate_layer(building_type, i, self.current_quarter)
                    print(f"  [Slots] {building_type}: yearly activate {layer.layer_id}")
                    break

    def _create_new_isocontour_layers_when_no_growth(self):
        print("[Slots] no growth, create new layers...")
        if not hasattr(self.progressive_growth_system, 'layers') or len(self.progressive_growth_system.layers['commercial']) == 0:
            print("[Slots] not initialized, skip new layers creation")
            return
        current_land_price_field = self.city_state['land_price_field']
        self._create_new_isocontour_layers_for_type('commercial', current_land_price_field)
        self._create_new_isocontour_layers_for_type('residential', current_land_price_field)
        print("[Slots] new layers created")

    def _create_new_isocontour_layers_for_type(self, building_type: str, land_price_field):
        print(f"  [Slots] create new layers for {building_type}...")
        config = self.city_config.get('isocontour_layout', {}).get(building_type, {})
        percentiles = config.get('percentiles', [95, 90, 85])
        new_percentiles = []
        for i, p in enumerate(percentiles):
            new_p = max(5, p - 20 - i * 8)
            new_percentiles.append(new_p)
        print(f"    [Slots] new percentiles: {new_percentiles}")
        for i, percentile in enumerate(new_percentiles):
            threshold = np.percentile(land_price_field, percentile)
            mask = (land_price_field >= threshold).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if len(c) >= 20]
            if valid_contours:
                longest_contour = max(valid_contours, key=len)
                contour_points = [[point[0][0], point[0][1]] for point in longest_contour]
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour_points, building_type)
                if new_slots:
                    existing_layers = self.progressive_growth_system.layers[building_type]
                    new_layer = Layer(
                        layer_id=f"{building_type}_P{len(existing_layers)}_new",
                        status="active",
                        activated_quarter=self.current_quarter,
                        slots=new_slots,
                        capacity=len(new_slots),
                        dead_slots=0,
                        capacity_effective=len(new_slots),
                        placed=0,
                        density=0.0
                    )
                    existing_layers.append(new_layer)
                    print(f"    [Slots] new {new_layer.layer_id}, thr {threshold:.3f}, slots {len(new_slots)}")
                    break

    def _redistribute_buildings_to_new_slots(self, current_buildings: Dict):
        print("[Slots] redistribute buildings...")
        self.city_state['residential'].clear()
        self.city_state['commercial'].clear()
        for building in current_buildings['residential']:
            self._redistribute_building(building, 'residential')
        for building in current_buildings['commercial']:
            self._redistribute_building(building, 'commercial')
        print(f"  [Slots] redistribution done: res {len(current_buildings['residential'])}, com {len(current_buildings['commercial'])}")

    def _redistribute_building(self, building: Dict, building_type: str):
        best_slot = self._find_best_slot_for_building(building, building_type)
        if best_slot:
            building['xy'] = best_slot.pos
            building['land_price_value'] = float(self.city_state['land_price_field'][best_slot.pos[1], best_slot.pos[0]])
            building['slot_id'] = f"{building_type}_{best_slot.pos[0]}_{best_slot.pos[1]}"
            best_slot.used = True
            best_slot.building_id = building['id']
            self.city_state[building_type].append(building)
        else:
            print(f"  [Slots] building {building['id']} no suitable slot")

    def _find_best_slot_for_building(self, building: Dict, building_type: str) -> Optional[object]:
        building_pos = building['xy']
        layers = self.progressive_growth_system.layers[building_type]
        best_slot = None
        min_distance = float('inf')
        for layer in layers:
            for slot in layer.slots:
                if not slot.used and not slot.dead and building_type in slot.allowed_types:
                    distance = ((slot.pos[0] - building_pos[0])**2 + (slot.pos[1] - building_pos[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_slot = slot
        return best_slot

    def _coordinate_isocontours_with_slots(self, contour_data: Dict):
        print("[Slots] coordinate contours and slots...")
        new_commercial_contours = contour_data.get('commercial_contours', [])
        new_residential_contours = contour_data.get('residential_contours', [])
        self._create_additional_layers_for_new_contours('commercial', new_commercial_contours)
        self._create_additional_layers_for_new_contours('residential', new_residential_contours)
        print("[Slots] coordination done")

    def _create_additional_layers_for_new_contours(self, building_type: str, new_contours: List):
        existing_layers = self.progressive_growth_system.layers[building_type]
        for i, contour in enumerate(new_contours):
            if len(contour) < 20:
                continue
            contour_has_layer = False
            for layer in existing_layers:
                for slot in layer.slots:
                    if self._is_slot_on_contour(slot, contour):
                        contour_has_layer = True
                        break
                if contour_has_layer:
                    break
            if not contour_has_layer:
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
                new_layer = Layer(
                    layer_id=f"{building_type}_P{len(existing_layers)}",
                    status="locked",
                    activated_quarter=-1,
                    slots=new_slots,
                    capacity=len(new_slots),
                    dead_slots=0,
                    capacity_effective=len(new_slots),
                    placed=0,
                    density=0.0
                )
                existing_layers.append(new_layer)
                print(f"  [Slots] {building_type}: add new layer for contour {i+1} => {new_layer.layer_id}")

    def _update_slots_for_type(self, building_type: str, new_contours: List):
        if not new_contours:
            return
        current_layer_idx = self.progressive_growth_system.active_layers[building_type]
        current_layer = self.progressive_growth_system.layers[building_type][current_layer_idx]
        if current_layer.status == "complete":
            return
        for i, contour in enumerate(new_contours):
            if len(contour) < 20:
                continue
            existing_slots = [slot for slot in current_layer.slots if self._is_slot_on_contour(slot, contour)]
            if not existing_slots:
                new_slots = self.progressive_growth_system._create_slots_from_contour(contour, building_type)
                current_layer.slots.extend(new_slots)
                current_layer.capacity += len(new_slots)
                current_layer.capacity_effective += len(new_slots)
                print(f"  📍 {building_type}建筑：为等值线 {i+1} 添加了 {len(new_slots)} 个新槽位")

    def _is_slot_on_contour(self, slot, contour: List) -> bool:
        slot_pos = slot.pos
        tolerance = float(self.config.get('progressive_growth', {}).get('slot_on_contour_tolerance_px', 5))
        for contour_point in contour:
            cp = contour_point[0] if isinstance(contour_point, list) and len(contour_point) == 1 else contour_point
            distance = ((slot_pos[0] - cp[0])**2 + (slot_pos[1] - cp[1])**2)**0.5
            if distance <= tolerance:
                return True
        return False

    def _generate_buildings_with_slots(self):
        print(f"[Generate] quarter {self.current_quarter} using slot system...")
        if not hasattr(self.progressive_growth_system, 'layers') or len(self.progressive_growth_system.layers['commercial']) == 0:
            print("[Generate] slots not initialized, skip")
            return False
        available_residential_slots = len(self.progressive_growth_system.get_available_slots('residential', 100))
        available_commercial_slots = len(self.progressive_growth_system.get_available_slots('commercial', 100))
        print(f"[Generate] available slots - res: {available_residential_slots}, com: {available_commercial_slots}")
        residential_target = min(random.randint(12, 20), available_residential_slots)
        commercial_target = min(random.randint(5, 12), available_commercial_slots)
        # 去掉硬编码年度加量；改为与可用槽位比例挂钩（温和提升）
        utilization_boost = float(self.city_config.get('progressive_growth', {}).get('annual_boost_factor', 0.0))
        if self.current_month % 12 == 0 and utilization_boost > 0:
            residential_target = min(int(residential_target * (1.0 + utilization_boost)), available_residential_slots)
            commercial_target = min(int(commercial_target * (1.0 + utilization_boost)), available_commercial_slots)
            print(f"  [Generate] yearly utilization boost x{1.0 + utilization_boost:.2f}")
        new_residential = self._generate_residential_with_slots(residential_target)
        new_commercial = self._generate_commercial_with_slots(commercial_target)
        self.city_state['residential'].extend(new_residential)
        self.city_state['commercial'].extend(new_commercial)
        buildings_generated = len(new_residential) > 0 or len(new_commercial) > 0
        if buildings_generated:
            print(f"[Generate] done: res {len(new_residential)}, com {len(new_commercial)}")
            print(f"[Generate] available slots: res {available_residential_slots}, com {available_commercial_slots}")
        else:
            print(f"[Generate] no new buildings - all layers complete")
        return buildings_generated

    def _generate_residential_with_slots(self, target_count: int) -> List[Dict]:
        available_slots = self.progressive_growth_system.get_available_slots('residential', target_count)
        print(f"[Generate] residential - target {target_count}, available {len(available_slots)}")
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
            self.progressive_growth_system.place_building_in_slot('residential', building['id'], slot)
        return new_buildings

    def _generate_commercial_with_slots(self, target_count: int) -> List[Dict]:
        available_slots = self.progressive_growth_system.get_available_slots('commercial', target_count)
        print(f"[Generate] commercial - target {target_count}, available {len(available_slots)}")
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
            self.progressive_growth_system.place_building_in_slot('commercial', building['id'], slot)
        return new_buildings

    def _activate_first_layers(self):
        print("[Progressive] activate initial layers...")
        if not hasattr(self.progressive_growth_system, 'layers') or len(self.progressive_growth_system.layers['commercial']) == 0:
            print("[Progressive] slots not initialized, skip activation")
            return
        # 阶段门控：道路阶段优先激活“靠近道路或活跃Hub”的层
        def pick_first_layer(building_type: str) -> int:
            layers = self.progressive_growth_system.layers[building_type]
            if not layers:
                return -1
            # Month 0：优先激活包含活跃Hub（此时只有Hub3）的层；否则退化为0
            active_hubs = self.city_state.get('transport_hubs', [])
            # 简化：选容量最大的层作为更稳妥的起点
            best_idx = max(range(len(layers)), key=lambda i: layers[i].capacity) if layers else -1
            return 0 if best_idx < 0 else best_idx

        ci = pick_first_layer('commercial')
        if ci >= 0:
            self.progressive_growth_system._activate_layer('commercial', ci, 0)
            print(f"[Progressive] commercial: activated layer {ci}")
        ri = pick_first_layer('residential')
        if ri >= 0:
            self.progressive_growth_system._activate_layer('residential', ri, 0)
            print(f"[Progressive] residential: activated layer {ri}")

    def _try_activate_next_layers(self):
        if not hasattr(self.progressive_growth_system, 'layers') or len(self.progressive_growth_system.layers['commercial']) == 0:
            return
        for building_type in ['commercial', 'residential']:
            if self.progressive_growth_system.try_activate_next_layer(building_type, self.current_quarter):
                print(f"[Progressive] {building_type}: activated next layer")

    def _evaluate_hysteresis_conversion(self):
        self.hysteresis_system.update_quarter(self.current_quarter)
        conversion_result = self.hysteresis_system.evaluate_conversion_conditions(
            self.city_state, self.land_price_system
        )
        if conversion_result['should_convert'] and conversion_result['candidates']:
            best_candidate = conversion_result['candidates'][0]
            conversion_result = self.hysteresis_system.convert_building(
                best_candidate['building_id'], self.city_state
            )
            if conversion_result['success']:
                print(f"[Hysteresis] quarter {self.current_quarter}: residential {best_candidate['building_id']} -> commercial")
                self._update_slot_after_conversion(best_candidate['building_id'])

    def _update_slot_after_conversion(self, building_id: str):
        converted_building = None
        for building in self.city_state['residential']:
            if building['id'] == building_id:
                converted_building = building
                break
        if converted_building:
            self.city_state['residential'].remove(converted_building)
            converted_building['type'] = 'commercial'
            converted_building['capacity'] = 800
            self.city_state['commercial'].append(converted_building)
            slot_id = converted_building.get('slot_id', '')
            if slot_id:
                self._mark_slot_as_dead_by_id('residential', slot_id, 'converted_to_commercial')

    def _mark_slot_as_dead_by_id(self, building_type: str, slot_id: str, reason: str):
        current_layer_idx = self.progressive_growth_system.active_layers[building_type]
        current_layer = self.progressive_growth_system.layers[building_type][current_layer_idx]
        try:
            parts = slot_id.split('_')
            if len(parts) >= 3:
                x, y = int(parts[-2]), int(parts[-1])
                for slot in current_layer.slots:
                    if slot.pos[0] == x and slot.pos[1] == y:
                        self.progressive_growth_system.mark_slot_as_dead(building_type, slot, reason)
                        break
        except (ValueError, IndexError):
            print(f"[Hysteresis] cannot parse slot id: {slot_id}")

    def _evaluate_public_facilities(self):
        print(f"[Facility] quarter {self.current_quarter}: evaluate needs...")
        facility_needs = self.public_facility_system.evaluate_facility_needs(self.city_state)
        new_facilities = self.public_facility_system.generate_facilities(self.city_state, facility_needs)
        if new_facilities:
            self.city_state['public'].extend(new_facilities)
            print(f"[Facility] created {len(new_facilities)} facilities")
        else:
            print(f"[Facility] no new facilities needed")
        self._print_facility_needs(facility_needs)

    def _print_facility_needs(self, facility_needs: Dict):
        print(f"[Facility] need summary:")
        for facility_type, need_info in facility_needs.items():
            status = "NEED" if need_info['needed'] else "NO_NEED"
            reason = need_info.get('reason', 'unknown')
            print(f"  - {facility_type}: {status} ({reason})")

    def _spawn_new_residents(self):
        pass

    def _save_monthly_outputs(self, month: int):
        self.land_price_system.save_land_price_frame(month, self.output_dir)
        new_buildings = self.building_tracker.get_new_buildings_this_month(self.city_state)
        self._save_building_positions(month, new_buildings)
        self._save_simplified_building_positions(month, new_buildings)
        self._save_layer_state(month)
        print(f"[Output] month {month} saved")

    def _save_building_positions(self, month: int, new_buildings: List[Dict]):
        if month == 1:
            self._save_full_building_state(month)
        else:
            self._save_new_buildings_only(month, new_buildings)

    def _save_full_building_state(self, month: int):
        building_data = {'timestamp': f'month_{month:02d}', 'buildings': []}
        for building_type in ['residential', 'commercial', 'public', 'industrial']:
            for building in self.city_state.get(building_type, []):
                building_data['buildings'].append({
                    'id': building['id'],
                    'type': building['type'],
                    'position': building['xy'],
                    'land_price_value': building.get('land_price_value', 0.0),
                    'slot_id': building.get('slot_id', '')
                })
        building_data['buildings'] = self._post_process_building_types(building_data['buildings'], month)
        output_file = f"{self.output_dir}/building_positions_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(building_data, f, indent=2, ensure_ascii=False)
        print(f"[Output] month {month} full saved: {len(building_data['buildings'])} buildings")

    def _post_process_building_types(self, buildings: List[Dict], month: int) -> List[Dict]:
        hub2_position = [90, 55]
        hub2_radius = 30
        processed_buildings = []
        for building in buildings:
            processed_building = building.copy()
            if building['type'] == 'commercial':
                x, y = building['position']
                distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
                if distance <= hub2_radius:
                    processed_building['type'] = 'industrial'
                    processed_building['original_type'] = 'commercial'
                    processed_building['hub_influence'] = 'hub2_industrial_zone'
                    processed_building['conversion_reason'] = f'Hub2工业中心影响 (距离: {distance:.1f})'
            processed_buildings.append(processed_building)
        return processed_buildings

    def _save_new_buildings_only(self, month: int, new_buildings: List[Dict]):
        if new_buildings:
            processed_new_buildings = self._post_process_building_types(new_buildings, month)
            total_buildings = sum(len(self.city_state.get(bt, [])) for bt in ['residential', 'commercial', 'public', 'industrial'])
            delta_data = {
                'month': month,
                'timestamp': f'month_{month:02d}',
                'new_buildings': processed_new_buildings,
                'metadata': { 'total_buildings': total_buildings, 'new_count': len(processed_new_buildings) }
            }
            output_file = f"{self.output_dir}/building_delta_month_{month:02d}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(delta_data, f, indent=2, ensure_ascii=False)
            print(f"[Output] month {month} delta saved: {len(processed_new_buildings)} new buildings")
        else:
            print(f"[Output] month {month} no new buildings, skip delta")

    def _save_simplified_building_positions(self, month: int, new_buildings: List[Dict]):
        type_map = {'residential': 0, 'commercial': 1, 'industrial': 2, 'public': 3}
        all_buildings = []
        for building_type in ['residential', 'commercial', 'public', 'industrial']:
            for building in self.city_state.get(building_type, []):
                all_buildings.append({
                    'id': building['id'],
                    'type': building['type'],
                    'position': building['xy'],
                    'land_price_value': building.get('land_price_value', 0.0),
                    'slot_id': building.get('slot_id', '')
                })
        processed_buildings = self._post_process_building_types(all_buildings, month)
        formatted = []
        for building in processed_buildings:
            t = str(building.get('type', 'unknown')).lower()
            mid = type_map.get(t, 4)
            pos = building.get('position', [0.0, 0.0])
            x = float(pos[0]) if len(pos) > 0 else 0.0
            y = float(pos[1]) if len(pos) > 1 else 0.0
            z = 0.0
            formatted.append(f"{mid}({x:.3f}, {y:.3f}, {z:.0f})")
        simplified_line = ", ".join(formatted)
        simplified_dir = f"{self.output_dir}/simplified"
        os.makedirs(simplified_dir, exist_ok=True)
        json_file = f"{simplified_dir}/simplified_buildings_{month:02d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({'month': month, 'timestamp': f'month_{month:02d}', 'simplified_format': simplified_line, 'building_count': len(formatted)}, f, indent=2, ensure_ascii=False)
        txt_file = f"{simplified_dir}/simplified_buildings_{month:02d}.txt"
        if month == 0:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(simplified_line)
        else:
            if new_buildings:
                new_formatted = []
                for building in self._post_process_building_types(new_buildings, month):
                    t = str(building.get('type', 'unknown')).lower()
                    mid = type_map.get(t, 4)
                    pos = building.get('position', [0.0, 0.0])
                    x = float(pos[0]) if len(pos) > 0 else 0.0
                    y = float(pos[1]) if len(pos) > 1 else 0.0
                    z = 0.0
                    new_formatted.append(f"{mid}({x:.3f}, {y:.3f}, {z:.0f})")
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(", ".join(new_formatted))
                print(f"[Output] month {month} TXT saved: {len(new_formatted)} new")
            else:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("")
                print(f"[Output] month {month} TXT empty")
        print(f"[Output] month {month} simplified saved: JSON {len(formatted)} buildings, TXT delta style")

    def _save_layer_state(self, month: int):
        layer_data = {'month': month, 'quarter': self.current_quarter, 'layers': self.city_state['layers']}
        output_file = f"{self.output_dir}/layer_state_month_{month:02d}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(layer_data, f, indent=2, ensure_ascii=False)

    def _save_final_outputs(self, simulation_months: int):
        final_summary = {
            'simulation_months': simulation_months,
            'final_layers': self.city_state['layers'],
            'final_buildings': {
                'public': len(self.city_state.get('public', [])),
                'residential': len(self.city_state.get('residential', [])),
                'commercial': len(self.city_state.get('commercial', []))
            },
            'land_price_evolution': self.land_price_system.get_evolution_history()
        }
        output_file = f"{self.output_dir}/final_summary.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        print(f"[Output] all v3.5 files saved to {self.output_dir}/")

    def get_full_state_at_month(self, target_month: int) -> Dict:
        return self.building_tracker.get_full_state_at_month(target_month)


def main():
    print("[Simulation] v3.5 start")
    print("=" * 60)
    print("[Simulation] features:")
    print("  • 阶段门控地价演化 + Hub3 恒定")
    print("  • 槽位系统同步（初始化修正、年度重建、阶段激活）")
    print("  • Hub2 工业中心（数据后处理 + 简化TXT映射）")
    print("  • 增量式建筑位置导出")
    print("=" * 60)
    simulation = EnhancedCitySimulationV3_5()
    simulation.initialize_simulation()
    simulation.run_simulation()
    print("\n🎉 v3.5模拟完成！")
    print(f"📁 输出文件保存在 {simulation.output_dir}/ 目录")


if __name__ == "__main__":
    main()


