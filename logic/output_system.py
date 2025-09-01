#!/usr/bin/env python3
"""
输出系统模块
生成各种JSON输出文件
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import math

class OutputSystem:
    """输出系统：生成各种JSON输出文件"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'animations').mkdir(exist_ok=True)
    
    def save_city_state_output(self, city_state: Dict, day: int):
        """保存城市状态输出"""
        output_data = {
            "simulation_info": {
                "day": day,
                "total_residents": len(city_state.get('residents', [])),
                "total_buildings": (
                    len(city_state.get('public', [])) +
                    len(city_state.get('residential', [])) +
                    len(city_state.get('commercial', []))
                ),
                "average_land_price": city_state.get('land_price_stats', {}).get('avg_price', 100)
            },
            "land_prices": city_state.get('land_price_stats', {}),
            "buildings": {
                "public": city_state.get('public', []),
                "residential": city_state.get('residential', []),
                "commercial": city_state.get('commercial', [])
            },
            "residents": city_state.get('residents', []),
            "statistics": city_state.get('statistics', {})
        }
        
        filepath = self.output_dir / 'city_state_output.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def save_visualization_config(self, building_config: Dict):
        """保存可视化配置"""
        output_data = {
            "color_scheme": {
                "background": building_config.get('visualization', {}).get('background', '#FFFFFF'),
                "trunk_road": building_config.get('visualization', {}).get('trunk_road', '#9AA4B2'),
                "core_point": building_config.get('visualization', {}).get('core_point', '#0B5ED7'),
                "heat_map": building_config.get('visualization', {}).get('heat_map', '#FF00FF'),
                "buildings": {
                    building_type: config.get('color', '#000000')
                    for building_type, config in building_config.get('building_types', {}).items()
                },
                "residents": building_config.get('visualization', {}).get('residents', '#FFFFFF')
            },
            "symbols": {
                building_type: config.get('symbol', '🏢')
                for building_type, config in building_config.get('building_types', {}).items()
            },
            "rendering_settings": {
                "image_size": [800, 600],
                "dpi": 300,
                "frame_rate": 5,
                "heat_map_alpha": 0.6
            }
        }
        
        filepath = self.output_dir / 'visualization_config.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def save_coordinates_output(self, city_state: Dict, day: int):
        """保存坐标信息输出"""
        output_data = {
            "day": day,
            "core_point": {
                "xy": city_state.get('core_point', [128, 128]),
                "type": "government_core",
                "description": "城市核心点"
            },
            "trunk_road": {
                "points": city_state.get('trunk_road', [[40, 128], [216, 128]]),
                "type": "main_road",
                "description": "主干道"
            },
            "buildings": {
                "public": [
                    {
                        "id": building['id'],
                        "xy": building['xy'],
                        "type": "public",
                        "description": f"公共建筑{building['id']}"
                    }
                    for building in city_state.get('public', [])
                ],
                "residential": [
                    {
                        "id": building['id'],
                        "xy": building['xy'],
                        "type": "residential",
                        "description": f"住宅建筑{building['id']}"
                    }
                    for building in city_state.get('residential', [])
                ],
                "commercial": [
                    {
                        "id": building['id'],
                        "xy": building['xy'],
                        "type": "commercial",
                        "description": f"商业建筑{building['id']}"
                    }
                    for building in city_state.get('commercial', [])
                ]
            },
            "residents": [
                {
                    "id": resident['id'],
                    "xy": resident['pos'],
                    "type": "resident",
                    "description": f"居民{resident['id']}"
                }
                for resident in city_state.get('residents', [])
            ],
            "land_price_zones": self._generate_land_price_zones(city_state)
        }
        
        filepath = self.output_dir / 'coordinates_output.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def _generate_land_price_zones(self, city_state: Dict) -> List[Dict]:
        """生成地价区域信息"""
        land_price_stats = city_state.get('land_price_stats', {})
        avg_price = land_price_stats.get('avg_price', 100)
        max_price = land_price_stats.get('max_price', 150)
        min_price = land_price_stats.get('min_price', 50)
        
        zones = []
        
        # 高价值区域
        if max_price > avg_price * 1.2:
            zones.append({
                "zone_id": "high_value",
                "boundary": [[120, 120], [136, 136]],
                "average_price": max_price,
                "description": "高价值区域"
            })
        
        # 中价值区域
        zones.append({
            "zone_id": "medium_value",
            "boundary": [[80, 80], [176, 176]],
            "average_price": avg_price,
            "description": "中价值区域"
        })
        
        # 低价值区域
        if min_price < avg_price * 0.8:
            zones.append({
                "zone_id": "low_value",
                "boundary": [[20, 20], [236, 236]],
                "average_price": min_price,
                "description": "低价值区域"
            })
        
        return zones
    
    def save_daily_stats(self, daily_stats: List[Dict]):
        """保存每日统计数据"""
        filepath = self.output_dir / 'daily_stats.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(daily_stats, f, indent=2, ensure_ascii=False)
    
    def save_land_price_evolution(self, land_price_history: List[Dict]):
        """保存地价演化数据"""
        filepath = self.output_dir / 'land_price_evolution.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(land_price_history, f, indent=2, ensure_ascii=False)
    
    def save_building_distribution(self, building_distribution: Dict):
        """保存建筑分布数据"""
        filepath = self.output_dir / 'building_distribution.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(building_distribution, f, indent=2, ensure_ascii=False)
    
    def save_final_summary(self, city_state: Dict, simulation_days: int):
        """保存最终总结报告"""
        summary = {
            "simulation_summary": {
                "total_days": simulation_days,
                "final_population": len(city_state.get('residents', [])),
                "total_buildings": {
                    "public": len(city_state.get('public', [])),
                    "residential": len(city_state.get('residential', [])),
                    "commercial": len(city_state.get('commercial', []))
                },
                "land_price_summary": city_state.get('land_price_stats', {}),
                "economic_summary": self._calculate_economic_summary(city_state)
            },
            "development_patterns": self._analyze_development_patterns(city_state),
            "recommendations": self._generate_recommendations(city_state)
        }
        
        filepath = self.output_dir / 'final_summary.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _calculate_economic_summary(self, city_state: Dict) -> Dict:
        """计算经济总结"""
        buildings = []
        buildings.extend(city_state.get('public', []))
        buildings.extend(city_state.get('residential', []))
        buildings.extend(city_state.get('commercial', []))
        
        total_construction_cost = sum(building.get('construction_cost', 0) for building in buildings)
        total_revenue = sum(building.get('revenue', 0) for building in buildings)
        
        return {
            "total_construction_cost": total_construction_cost,
            "total_revenue": total_revenue,
            "profit_margin": (total_revenue - total_construction_cost) / total_construction_cost if total_construction_cost > 0 else 0,
            "average_land_price": city_state.get('land_price_stats', {}).get('avg_price', 100)
        }
    
    def _analyze_development_patterns(self, city_state: Dict) -> Dict:
        """分析发展模式"""
        public_buildings = city_state.get('public', [])
        residential_buildings = city_state.get('residential', [])
        commercial_buildings = city_state.get('commercial', [])
        
        # 分析建筑分布
        public_positions = [building['xy'] for building in public_buildings]
        residential_positions = [building['xy'] for building in residential_buildings]
        commercial_positions = [building['xy'] for building in commercial_buildings]
        
        # 计算到核心点的平均距离
        core_point = city_state.get('core_point', [128, 128])
        
        def avg_distance(positions):
            if not positions:
                return 0
            distances = [math.sqrt((pos[0]-core_point[0])**2 + (pos[1]-core_point[1])**2) for pos in positions]
            return sum(distances) / len(distances)
        
        return {
            "spatial_distribution": {
                "public_avg_distance_to_core": avg_distance(public_positions),
                "residential_avg_distance_to_core": avg_distance(residential_positions),
                "commercial_avg_distance_to_core": avg_distance(commercial_positions)
            },
            "building_density": {
                "public_density": len(public_buildings) / 256**2 * 10000,  # 每平方公里
                "residential_density": len(residential_buildings) / 256**2 * 10000,
                "commercial_density": len(commercial_buildings) / 256**2 * 10000
            },
            "development_stage": self._determine_development_stage(city_state)
        }
    
    def _determine_development_stage(self, city_state: Dict) -> str:
        """确定发展阶段"""
        total_buildings = (
            len(city_state.get('public', [])) +
            len(city_state.get('residential', [])) +
            len(city_state.get('commercial', []))
        )
        
        if total_buildings < 10:
            return "初期发展阶段"
        elif total_buildings < 30:
            return "快速发展阶段"
        elif total_buildings < 50:
            return "成熟发展阶段"
        else:
            return "高度发展阶段"
    
    def _generate_recommendations(self, city_state: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于人口密度
        population = len(city_state.get('residents', []))
        residential_buildings = city_state.get('residential', [])
        
        if population > 0 and residential_buildings:
            avg_population_per_building = population / len(residential_buildings)
            if avg_population_per_building > 150:
                recommendations.append("建议增加住宅建筑以满足居住需求")
            elif avg_population_per_building < 50:
                recommendations.append("住宅建筑可能过剩，建议控制住宅建设")
        
        # 基于公共设施覆盖率
        public_buildings = city_state.get('public', [])
        if population > 0 and public_buildings:
            coverage_ratio = len(public_buildings) / (population / 100)  # 每100人一个公共设施
            if coverage_ratio < 0.5:
                recommendations.append("建议增加公共设施以提高服务质量")
        
        # 基于地价分布
        land_price_stats = city_state.get('land_price_stats', {})
        if land_price_stats:
            price_variance = land_price_stats.get('max_price', 0) - land_price_stats.get('min_price', 0)
            if price_variance > 100:
                recommendations.append("地价差异较大，建议优化土地利用规划")
        
        return recommendations
