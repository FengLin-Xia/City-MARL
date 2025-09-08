import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional
import json

class CityVisualizer:
    def __init__(self, grid_size: tuple = (256, 256)):
        self.grid_size = grid_size
        self.colors = self._load_colors()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.new_pois = []  # 记录新增的POI用于高亮显示
        
    def _load_colors(self) -> Dict:
        """加载颜色配置"""
        try:
            with open('configs/colors.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # 默认颜色配置
            return {
                "gov": { 
                    "hub": "#0B5ED7", 
                    "school": "#22A6B3", 
                    "clinic": "#3CA6FF", 
                    "park": "#2ECC71" 
                },
                "firm": { 
                    "residential": "#F6C344", 
                    "retail": "#FD7E14" 
                },
                "layers": { 
                    "trunk": "#9AA4B2", 
                    "heat": "#FF00FF", 
                    "agent": "#FFFFFF" 
                }
            }
    
    def render_layers(self, hubs: List[Dict], trunk: List[List[int]], 
                     public_pois: List[Dict], residential_pois: List[Dict], 
                     retail_pois: List[Dict], heat_map: Optional[np.ndarray] = None, 
                     agents: Optional[List[Dict]] = None, show_agents: bool = False,
                     contour_data: Optional[Dict] = None) -> None:
        """分层渲染城市"""
        self.ax.clear()
        
        # 设置背景
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 1. 渲染热力图
        if heat_map is not None:
            self._render_heatmap(heat_map)
        
        # 2. 渲染主干线
        self._render_trunk(trunk)
        
        # 3. 渲染枢纽
        self._render_hubs(hubs)
        
        # 4. 渲染公共POI
        self._render_public_pois(public_pois)
        
        # 5. 渲染住宅POI
        self._render_residential_pois(residential_pois)
        
        # 6. 渲染商业POI
        self._render_retail_pois(retail_pois)
        
        # 7. 渲染居民（可选）
        if show_agents and agents:
            self._render_agents(agents)
        
        # 8. 渲染等值线（如果提供）
        if contour_data:
            self._render_contours(contour_data)
        
        # 9. 高亮新增POI
        self._highlight_new_pois()
        
        # 设置标题和标签
        self.ax.set_title('City Simulation', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        
    def _render_heatmap(self, heat_map: np.ndarray) -> None:
        """渲染热力图"""
        if heat_map is not None and heat_map.size > 0:
            # 归一化热力图
            if heat_map.max() > 0:
                normalized_heat = heat_map / heat_map.max()
            else:
                normalized_heat = heat_map
            
            # 使用半透明的热力图，只有当热力值大于阈值时才显示
            if normalized_heat.max() > 0.01:  # 只有当热力值足够大时才显示
                # 热力图矩阵形状为 (height, width) = (y, x)
                # imshow 需要正确的方向和范围
                # 不使用转置，直接使用原始矩阵，但需要调整 origin 和 extent
                self.ax.imshow(normalized_heat, origin='lower', 
                              extent=[0, self.grid_size[0], 0, self.grid_size[1]],
                              alpha=0.6, cmap='hot', vmin=0, vmax=1)
    
    def _render_trunk(self, trunk: List[List[int]]) -> None:
        """渲染主干线"""
        if len(trunk) >= 2:
            trunk_color = self.colors['layers']['trunk']
            self.ax.plot([trunk[0][0], trunk[1][0]], [trunk[0][1], trunk[1][1]], 
                        color=trunk_color, linewidth=8, alpha=0.8, label='Trunk Road')
    
    def _render_hubs(self, hubs: List[Dict]) -> None:
        """渲染交通枢纽"""
        hub_color = self.colors['gov']['hub']
        for hub in hubs:
            x, y = hub['xy']
            circle = patches.Circle((x, y), radius=8, color=hub_color, 
                                  alpha=0.8, label=f"Hub {hub['id']}")
            self.ax.add_patch(circle)
            self.ax.text(x, y, hub['id'], ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
    
    def _render_public_pois(self, public_pois: List[Dict]) -> None:
        """渲染公共POI"""
        for poi in public_pois:
            x, y = poi['xy']
            poi_type = poi['type']
            
            if poi_type in self.colors['gov']:
                color = self.colors['gov'][poi_type]
            else:
                color = self.colors['gov']['clinic']  # 默认颜色
            
            # 根据类型选择形状
            if poi_type == 'school':
                shape = patches.Rectangle((x-4, y-4), 8, 8, color=color, alpha=0.8)
            elif poi_type == 'clinic':
                shape = patches.Circle((x, y), radius=4, color=color, alpha=0.8)
            elif poi_type == 'park':
                shape = patches.RegularPolygon((x, y), numVertices=6, radius=4, 
                                             color=color, alpha=0.8)
            else:
                shape = patches.Circle((x, y), radius=3, color=color, alpha=0.8)
            
            self.ax.add_patch(shape)
            self.ax.text(x, y+8, poi_type, ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
    
    def _render_residential_pois(self, residential_pois: List[Dict]) -> None:
        """渲染住宅POI"""
        color = self.colors['firm']['residential']
        for poi in residential_pois:
            x, y = poi['xy']
            # 住宅用方形表示
            rect = patches.Rectangle((x-3, y-3), 6, 6, color=color, alpha=0.8)
            self.ax.add_patch(rect)
            self.ax.text(x, y+6, 'Res', ha='center', va='bottom', 
                        fontsize=7, fontweight='bold')
    
    def _render_retail_pois(self, retail_pois: List[Dict]) -> None:
        """渲染商业POI"""
        color = self.colors['firm']['retail']
        for poi in retail_pois:
            x, y = poi['xy']
            # 商业用圆形表示
            circle = patches.Circle((x, y), radius=4, color=color, alpha=0.8)
            self.ax.add_patch(circle)
            self.ax.text(x, y+6, 'Ret', ha='center', va='bottom', 
                        fontsize=7, fontweight='bold')
    
    def _render_agents(self, agents: List[Dict]) -> None:
        """渲染居民"""
        agent_color = self.colors['layers']['agent']
        for agent in agents:
            x, y = agent['pos']
            # 居民用小点表示
            self.ax.plot(x, y, 'o', color=agent_color, markersize=2, alpha=0.7)
    
    def _render_contours(self, contour_data: Dict) -> None:
        """渲染等值线"""
        # 渲染商业等值线
        if 'commercial_contours' in contour_data:
            for i, contour in enumerate(contour_data['commercial_contours']):
                if contour:
                    x_coords = [point[0] for point in contour]
                    y_coords = [point[1] for point in contour]
                    self.ax.plot(x_coords, y_coords, 'r--', linewidth=1, alpha=0.6, 
                               label=f'Commercial Contour {i+1}' if i == 0 else "")
        
        # 渲染住宅等值线
        if 'residential_contours' in contour_data:
            for i, contour in enumerate(contour_data['residential_contours']):
                if contour:
                    x_coords = [point[0] for point in contour]
                    y_coords = [point[1] for point in contour]
                    self.ax.plot(x_coords, y_coords, 'b--', linewidth=1, alpha=0.6,
                               label=f'Residential Contour {i+1}' if i == 0 else "")
    
    def _highlight_new_pois(self) -> None:
        """高亮新增的POI"""
        for poi in self.new_pois:
            x, y = poi['xy']
            # 添加闪烁效果（红色边框）
            circle = patches.Circle((x, y), radius=6, fill=False, 
                                  color='red', linewidth=2, alpha=0.8)
            self.ax.add_patch(circle)
    
    def add_new_poi(self, poi: Dict) -> None:
        """添加新的POI到高亮列表"""
        self.new_pois.append(poi)
    
    def clear_new_pois(self) -> None:
        """清除新增POI列表"""
        self.new_pois = []
    
    def save_frame(self, filename: str) -> None:
        """保存当前帧"""
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    def show(self) -> None:
        """显示图像"""
        plt.tight_layout()
        plt.show()
    
    def close(self) -> None:
        """关闭图像"""
        plt.close(self.fig)
