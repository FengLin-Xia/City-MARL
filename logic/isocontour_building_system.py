#!/usr/bin/env python3
"""
等值线建筑生成系统 v2.3 - 重构版
实现基于SDF等值线的精确建筑选址和分带逻辑
按照PRD要求：建筑放置在等值线上，等值线间距等距
"""

import numpy as np
import math
from typing import List, Dict, Tuple
import random
from scipy import ndimage
from scipy.spatial import distance
import cv2

class IsocontourBuildingSystem:
    """等值线建筑生成系统 - 重构版"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.isocontour_config = config.get('isocontour_layout', {})
        
        # 商业建筑配置
        self.commercial_config = self.isocontour_config.get('commercial', {
            'percentiles': [80, 70, 60],  # 基于分位数的等值线
            'arc_spacing_m': [25, 35],
            'normal_offset_m': 4,
            'jitter_m': 1.5
        })
        
        # 住宅建筑配置
        self.residential_config = self.isocontour_config.get('residential', {
            'percentiles': [50, 40, 30, 20],  # 基于分位数的等值线
            'arc_spacing_m': [35, 55],
            'normal_offset_m': 4,
            'jitter_m': 1.5
        })
        
        # 如果配置中没有percentiles，尝试从fallback_percentiles获取
        if 'percentiles' not in self.commercial_config and 'fallback_percentiles' in self.isocontour_config:
            self.commercial_config['percentiles'] = self.isocontour_config['fallback_percentiles'].get('commercial', [80, 70, 60])
        
        if 'percentiles' not in self.residential_config and 'fallback_percentiles' in self.isocontour_config:
            self.residential_config['percentiles'] = self.isocontour_config['fallback_percentiles'].get('residential', [50, 40, 30, 20])
        
        # 通用配置
        self.normal_offset_m = self.isocontour_config.get('normal_offset_m', 4)
        self.jitter_m = self.isocontour_config.get('jitter_m', 1.5)
        
        # 分带配置
        self.front_zone_distance = 120  # 前排区域距离（米）
        self.residential_zone_start = 120  # 住宅带起始距离
        self.residential_zone_end = 260   # 住宅带结束距离
        
        # 系统状态
        self.sdf_field = None
        self.transport_hubs = []
        self.map_size = [256, 256]
        
    def initialize_system(self, land_price_field: np.ndarray, transport_hubs: List[List[int]], map_size: List[int]):
        """初始化系统"""
        self.sdf_field = land_price_field  # 保持兼容性，但实际是地价场
        self.transport_hubs = transport_hubs
        self.map_size = map_size
        
        print(f"🏗️ 等值线建筑系统初始化完成")
        print(f"地价场值范围: [{np.min(land_price_field):.3f}, {np.max(land_price_field):.3f}]")
        
    def generate_commercial_buildings(self, city_state: Dict, target_count: int, target_layer: int = None) -> List[Dict]:
        """生成商业建筑（基于等值线）"""
        if self.sdf_field is None:
            return []
        
        # 如果指定了目标层，使用对应的等值线
        if target_layer is not None and target_layer < len(self.commercial_config['percentiles']):
            percentiles = [self.commercial_config['percentiles'][target_layer]]
            print(f"🎯 商业建筑：目标第{target_layer}层，使用分位数{percentiles}")
        else:
            percentiles = self.commercial_config['percentiles']
            print(f"🎯 商业建筑：使用所有分位数{percentiles}")
        
        # 获取商业等值线（基于分位数）
        commercial_contours = self._extract_equidistant_contours(
            percentiles, 
            'commercial'
        )
        
        if not commercial_contours:
            print(f"⚠️ 未找到商业等值线")
            return []
        
        # 在等值线上生成建筑位置
        building_positions = self._place_buildings_on_contours(
            commercial_contours, target_count, 'commercial'
        )
        
        # 创建商业建筑
        new_buildings = []
        for i, position in enumerate(building_positions):
            building = {
                'id': f'com_{len(city_state.get("commercial", [])) + i + 1}',
                'type': 'commercial',
                'xy': position,
                'capacity': 800,
                'current_usage': 0,
                'construction_cost': 1000,
                'revenue_per_person': 20,
                'revenue': 0,
                'land_price_value': float(self.sdf_field[position[1], position[0]]),
                'contour_generated': True
            }
            new_buildings.append(building)
        
        print(f"🏢 生成 {len(new_buildings)} 个商业建筑，等值线数量: {len(commercial_contours)}")
        return new_buildings
    
    def generate_residential_buildings(self, city_state: Dict, target_count: int, target_layer: int = None) -> List[Dict]:
        """生成住宅建筑（基于等值线和分带）"""
        if self.sdf_field is None:
            return []
        
        # 检查分带限制
        if not self._check_residential_zone_availability(city_state):
            print(f"❌ 住宅分带检查失败")
            return []
        
        # 如果指定了目标层，使用对应的等值线
        if target_layer is not None and target_layer < len(self.residential_config['percentiles']):
            percentiles = [self.residential_config['percentiles'][target_layer]]
            print(f"🎯 住宅建筑：目标第{target_layer}层，使用分位数{percentiles}")
        else:
            percentiles = self.residential_config['percentiles']
            print(f"🎯 住宅建筑：使用所有分位数{percentiles}")
        
        # 获取住宅等值线（基于分位数）
        residential_contours = self._extract_equidistant_contours(
            percentiles, 
            'residential'
        )
        
        if not residential_contours:
            print(f"⚠️ 未找到住宅等值线")
            return []
        
        # 在等值线上生成建筑位置
        building_positions = self._place_buildings_on_contours(
            residential_contours, target_count, 'residential'
        )
        
        print(f"🔍 住宅建筑位置生成: {len(building_positions)} 个位置")
        
        # 创建住宅建筑
        new_buildings = []
        for i, position in enumerate(building_positions):
            building = {
                'id': f'res_{len(city_state.get("residential", [])) + i + 1}',
                'type': 'residential',
                'xy': position,
                'capacity': 200,
                'current_usage': 0,
                'construction_cost': 500,
                'revenue_per_person': 10,
                'revenue': 0,
                'land_price_value': float(self.sdf_field[position[1], position[0]]),
                'contour_generated': True
            }
            new_buildings.append(building)
        
        print(f"🏠 生成 {len(new_buildings)} 个住宅建筑，等值线数量: {len(residential_contours)}")
        return new_buildings
    
    def _extract_equidistant_contours(self, percentiles: List[int], building_type: str) -> List[List[Tuple[int, int]]]:
        """基于分位数提取等距等值线"""
        if self.sdf_field is None:
            return []
        
        # 计算分位数对应的SDF值
        sdf_flat = self.sdf_field.flatten()
        sdf_percentiles = np.percentile(sdf_flat, percentiles)
        
        print(f"📊 {building_type} 等值线分位数: {percentiles}")
        print(f"📊 {building_type} SDF阈值: {[f'{p:.3f}' for p in sdf_percentiles]}")
        
        contours = []
        
        for i, threshold in enumerate(sdf_percentiles):
            # 提取等值线
            contour = self._extract_contour_at_level_cv2(threshold)
            
            if len(contour) > 20:  # 足够长的等值线
                contours.append(contour)
                print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}")
            else:
                # 等值线太小，在hub周围等分4个点
                small_contour = self._create_small_contour_around_hubs(threshold, building_type)
                if small_contour:
                    contours.append(small_contour)
                    print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)} (使用hub周围4点)")
                else:
                    print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)} (跳过)")
        
        return contours
    
    def _extract_contour_at_level_cv2(self, level: float) -> List[Tuple[int, int]]:
        """使用OpenCV在指定SDF值水平提取等值线"""
        if self.sdf_field is None:
            return []
        
        # 创建二值图像
        binary = (self.sdf_field >= level).astype(np.uint8) * 255
        
        # 使用OpenCV的findContours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 转换为点列表
        contour_points = []
        for point in largest_contour:
            x, y = point[0][0], point[0][1]
            contour_points.append((x, y))
        
        return contour_points
    
    def _create_small_contour_around_hubs(self, threshold: float, building_type: str) -> List[Tuple[int, int]]:
        """当等值线太小时，在hub周围生成更多点"""
        if not self.transport_hubs:
            return []
        
        # 计算到hub的距离，基于阈值
        # 阈值越高，距离越近
        max_distance = 20  # 最大距离（像素）
        min_distance = 3   # 最小距离（像素）
        
        # 根据阈值调整距离（阈值越高，距离越近）
        threshold_ratio = threshold / np.max(self.sdf_field)
        distance = min_distance + (max_distance - min_distance) * (1 - threshold_ratio)
        
        contour_points = []
        
        for hub in self.transport_hubs:
            hub_x, hub_y = hub[0], hub[1]
            
            # 根据阈值决定点的数量
            if threshold_ratio > 0.98:  # 99%等值线
                num_points = 8  # 8个点
            elif threshold_ratio > 0.95:  # 98%, 97%, 96%, 95%等值线
                num_points = 6  # 6个点
            elif threshold_ratio > 0.90:  # 94%, 92%, 91%, 90%等值线
                num_points = 5  # 5个点
            else:  # 88%, 85%, 80%等值线
                num_points = 4  # 4个点
            
            # 在hub周围等分点
            for i in range(num_points):
                angle = i * (2 * math.pi / num_points)  # 均匀分布
                x = int(hub_x + distance * math.cos(angle))
                y = int(hub_y + distance * math.sin(angle))
                
                # 确保坐标在地图范围内
                x = max(0, min(x, self.map_size[0] - 1))
                y = max(0, min(y, self.map_size[1] - 1))
                
                contour_points.append((x, y))
        
        return contour_points
    
    def _place_buildings_on_contours(self, contours: List[List[Tuple[int, int]]], 
                                   target_count: int, building_type: str) -> List[List[int]]:
        """在等值线上放置建筑"""
        if not contours:
            return []
        
        positions = []
        config = self.commercial_config if building_type == 'commercial' else self.residential_config
        
        # 计算建筑间距
        min_spacing, max_spacing = config['arc_spacing_m']
        
        for contour in contours:
            if len(contour) < 10:
                continue
            
            # 在等值线上均匀采样建筑位置
            contour_positions = self._sample_contour_uniformly(
                contour, min_spacing, max_spacing, target_count - len(positions)
            )
            
            for pos in contour_positions:
                # 应用法向偏移
                offset_pos = self._apply_normal_offset(pos, contour)
                
                # 应用切向抖动
                final_pos = self._apply_tangential_jitter(offset_pos)
                
                # 检查位置是否合适
                if self._is_valid_building_position(final_pos, building_type):
                    positions.append(final_pos)
                
                if len(positions) >= target_count:
                    break
            
            if len(positions) >= target_count:
                break
        
        return positions
    
    def _sample_contour_uniformly(self, contour: List[Tuple[int, int]], 
                                min_spacing: int, max_spacing: int, 
                                max_buildings: int) -> List[List[int]]:
        """在等值线上均匀采样建筑位置"""
        if len(contour) < 10:
            return []
        
        positions = []
        contour_length = len(contour)
        
        # 对于小等值线（如99%等值线的4个点），直接使用所有点
        if contour_length <= 8:  # 99%等值线通常是8个点（2个hub × 4个点）
            for point in contour:
                positions.append([point[0], point[1]])
            return positions[:max_buildings]  # 限制最大数量
        
        # 计算采样间距
        spacing = random.randint(min_spacing, max_spacing)
        
        # 计算可以放置的建筑数量
        num_buildings = min(max_buildings, contour_length // spacing)
        
        for i in range(num_buildings):
            # 在等值线上均匀分布
            idx = (i * spacing) % contour_length
            base_pos = contour[idx]
            positions.append([base_pos[0], base_pos[1]])
        
        return positions
    
    def _apply_normal_offset(self, position: List[int], contour: List[Tuple[int, int]]) -> List[int]:
        """应用法向偏移（垂直于等值线方向）"""
        x, y = position[0], position[1]
        
        # 找到最近的点在轮廓上的索引
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (cx, cy) in enumerate(contour):
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 计算法向量
        if len(contour) > 2:
            # 使用前后点计算切线
            prev_idx = (closest_idx - 1) % len(contour)
            next_idx = (closest_idx + 1) % len(contour)
            
            prev_pos = contour[prev_idx]
            next_pos = contour[next_idx]
            
            # 切线向量
            tangent_x = next_pos[0] - prev_pos[0]
            tangent_y = next_pos[1] - prev_pos[1]
            
            # 法向量（垂直于切线）
            normal_x = -tangent_y
            normal_y = tangent_x
            
            # 归一化
            length = math.sqrt(normal_x**2 + normal_y**2)
            if length > 0:
                normal_x /= length
                normal_y /= length
                
                # 应用偏移
                offset_distance = random.uniform(-self.normal_offset_m, self.normal_offset_m)
                new_x = int(x + normal_x * offset_distance)
                new_y = int(y + normal_y * offset_distance)
                
                # 确保在边界内
                new_x = max(0, min(self.map_size[0] - 1, new_x))
                new_y = max(0, min(self.map_size[1] - 1, new_y))
                
                return [new_x, new_y]
        
        return position
    
    def _apply_tangential_jitter(self, position: List[int]) -> List[int]:
        """应用切向抖动（沿等值线方向）"""
        jitter_x = random.uniform(-self.jitter_m, self.jitter_m)
        jitter_y = random.uniform(-self.jitter_m, self.jitter_m)
        
        new_x = int(position[0] + jitter_x)
        new_y = int(position[1] + jitter_y)
        
        # 确保在边界内
        new_x = max(0, min(self.map_size[0] - 1, new_x))
        new_y = max(0, min(self.map_size[1] - 1, new_y))
        
        return [new_x, new_y]
    
    def _is_valid_building_position(self, position: List[int], building_type: str) -> bool:
        """检查建筑位置是否有效"""
        x, y = position[0], position[1]
        
        # 检查边界
        if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
            return False
        
        # 检查SDF值
        if self.sdf_field is not None:
            sdf_value = self.sdf_field[y, x]
            if building_type == 'commercial' and sdf_value < np.percentile(self.sdf_field.flatten(), 60):
                return False
            elif building_type == 'residential' and sdf_value < np.percentile(self.sdf_field.flatten(), 20):
                return False
        
        # 检查分带限制
        if building_type == 'residential':
            # 检查是否在住宅带内
            min_distance_to_hub = float('inf')
            for hub in self.transport_hubs:
                distance = math.sqrt((x - hub[0])**2 + (y - hub[1])**2)
                min_distance_to_hub = min(min_distance_to_hub, distance)
            
            # 放宽住宅带限制：60-300像素，避免过于严格
            if min_distance_to_hub < 60 or min_distance_to_hub > 300:
                return False
        
        return True
    
    def _check_residential_zone_availability(self, city_state: Dict) -> bool:
        """检查住宅带是否有可用空间"""
        # 检查前排区域是否被占用
        front_zone_buildings = 0
        for building in city_state.get('commercial', []):
            building_pos = building['xy']
            min_distance_to_hub = float('inf')
            for hub in self.transport_hubs:
                distance = math.sqrt((building_pos[0] - hub[0])**2 + (building_pos[1] - hub[1])**2)
                min_distance_to_hub = min(min_distance_to_hub, distance)
            
            if min_distance_to_hub < self.front_zone_distance:
                front_zone_buildings += 1
        
        # 放宽限制：如果前排区域建筑过多，限制住宅建设
        if front_zone_buildings > 20:  # 从10增加到20
            print(f"⚠️ 前排区域建筑过多 ({front_zone_buildings})，限制住宅建设")
            return False
        
        return True
    
    def get_zone_statistics(self, city_state: Dict) -> Dict:
        """获取分带统计信息"""
        stats = {
            'front_zone_buildings': 0,
            'residential_zone_buildings': 0,
            'commercial_zone_buildings': 0
        }
        
        for building in city_state.get('commercial', []):
            building_pos = building['xy']
            min_distance_to_hub = float('inf')
            for hub in self.transport_hubs:
                distance = math.sqrt((building_pos[0] - hub[0])**2 + (building_pos[1] - hub[1])**2)
                min_distance_to_hub = min(min_distance_to_hub, distance)
            
            if min_distance_to_hub < self.front_zone_distance:
                stats['front_zone_buildings'] += 1
            else:
                stats['commercial_zone_buildings'] += 1
        
        for building in city_state.get('residential', []):
            building_pos = building['xy']
            min_distance_to_hub = float('inf')
            for hub in self.transport_hubs:
                distance = math.sqrt((building_pos[0] - hub[0])**2 + (building_pos[1] - hub[1])**2)
                min_distance_to_hub = min(min_distance_to_hub, distance)
            
            if self.residential_zone_start <= min_distance_to_hub <= self.residential_zone_end:
                stats['residential_zone_buildings'] += 1
        
        return stats
    
    def get_contour_data_for_visualization(self) -> Dict:
        """获取等值线数据用于可视化"""
        if self.sdf_field is None:
            return {}
        
        # 获取分位数配置，优先使用fallback_percentiles
        commercial_percentiles = self.commercial_config.get('percentiles', 
            self.isocontour_config.get('fallback_percentiles', {}).get('commercial', [95, 90, 85]))
        residential_percentiles = self.residential_config.get('percentiles',
            self.isocontour_config.get('fallback_percentiles', {}).get('residential', [80, 70, 60, 50]))
        
        contour_data = {
            'commercial_contours': [],
            'residential_contours': [],
            'commercial_percentiles': commercial_percentiles,
            'residential_percentiles': residential_percentiles
        }
        
        # 获取商业等值线
        commercial_contours = self._extract_equidistant_contours(
            commercial_percentiles, 'commercial'
        )
        contour_data['commercial_contours'] = commercial_contours
        
        # 获取住宅等值线
        residential_contours = self._extract_equidistant_contours(
            residential_percentiles, 'residential'
        )
        contour_data['residential_contours'] = residential_contours
        
        return contour_data
    
    def get_fallback_statistics(self) -> Dict:
        """获取分位数回退统计信息"""
        # 简化实现：返回基本的回退统计
        stats = {
            'total_events': 0,
            'commercial_fallbacks': 0,
            'residential_fallbacks': 0,
            'fallback_reasons': []
        }
        
        # 这里可以添加更详细的回退统计逻辑
        # 目前返回默认值以保持兼容性
        
        return stats
