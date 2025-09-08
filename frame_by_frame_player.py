#!/usr/bin/env python3
"""
Enhanced City Simulation v3.1.1 逐帧播放器
支持手动控制播放过程，观察每一帧的演化细节
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, List, Tuple, Optional
import glob

class FrameByFramePlayer:
    """逐帧播放器"""
    
    def __init__(self, data_dir: str = "enhanced_simulation_v3_1_output"):
        self.data_dir = data_dir
        self.config = self._load_config()
        self.months = self._get_available_months()
        self.current_frame = 0
        
        # 颜色方案
        self.colors = {
            'hubs': {
                'commercial': '#FF6B6B',  # 红色 - Hub1
                'industrial': '#4ECDC4',  # 青色 - Hub2 (工业中心)
                'mixed': '#45B7D1'        # 蓝色 - Hub3
            },
            'buildings': {
                'residential': '#F6C344',  # 黄色
                'commercial': '#FD7E14',   # 橙色
                'industrial': '#22A6B3'    # 青色 (工业建筑)
            },
            'land_price': 'viridis',
            'contours': '#FFD700'  # 金色
        }
        
        # Hub 配置
        self.transport_hubs = [
            {'position': [20, 55], 'type': 'commercial', 'label': 'Hub1 (商业)'},
            {'position': [90, 55], 'type': 'industrial', 'label': 'Hub2 (工业中心)'},
            {'position': [67, 94], 'type': 'mixed', 'label': 'Hub3 (混合)'}
        ]
        
        # Hub2 工业中心影响半径
        self.hub2_industrial_radius = 30
        
        # 创建图形
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.tight_layout(pad=3.0)
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # 显示控制说明
        self._show_controls()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_path = os.path.join(self.data_dir, "configs", "city_config_v3_1.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _get_available_months(self) -> List[int]:
        """获取可用的月份数据"""
        months = []
        for file in glob.glob(os.path.join(self.data_dir, "building_positions_month_*.json")):
            filename = os.path.basename(file)
            month_str = filename.replace("building_positions_month_", "").replace(".json", "")
            try:
                month = int(month_str)
                months.append(month)
            except ValueError:
                continue
        return sorted(months)
    
    def _load_month_data(self, month: int) -> Dict:
        """加载指定月份的数据"""
        data = {}
        
        # 加载建筑位置
        building_file = os.path.join(self.data_dir, f"building_positions_month_{month:02d}.json")
        if os.path.exists(building_file):
            with open(building_file, 'r', encoding='utf-8') as f:
                building_data = json.load(f)
                # 提取 buildings 数组
                if isinstance(building_data, dict) and 'buildings' in building_data:
                    data['buildings'] = building_data['buildings']
                else:
                    data['buildings'] = building_data
        
        # 加载地价场
        land_price_file = os.path.join(self.data_dir, f"land_price_field_month_{month:02d}.json")
        if os.path.exists(land_price_file):
            with open(land_price_file, 'r', encoding='utf-8') as f:
                data['land_price'] = json.load(f)
        
        # 加载等值线
        contour_file = os.path.join(self.data_dir, f"isocontour_building_slots_month_{month:02d}.json")
        if os.path.exists(contour_file):
            with open(contour_file, 'r', encoding='utf-8') as f:
                data['contours'] = json.load(f)
        
        return data
    
    def _is_near_hub2(self, x: float, y: float) -> bool:
        """检查是否在 Hub2 工业中心附近"""
        hub2_x, hub2_y = 90, 55
        distance = np.sqrt((x - hub2_x)**2 + (y - hub2_y)**2)
        return distance <= self.hub2_industrial_radius
    
    def _get_building_color(self, building_type: str, x: float, y: float) -> Tuple[str, str]:
        """获取建筑颜色和标签（支持 Hub2 工业中心伪装）"""
        if building_type == 'commercial' and self._is_near_hub2(x, y):
            # Hub2 附近的商业建筑显示为工业建筑颜色
            return self.colors['buildings']['industrial'], 'industrial'
        else:
            return self.colors['buildings'][building_type], building_type
    
    def _plot_transport_hubs(self, ax):
        """绘制交通枢纽（支持不同类型）"""
        for hub in self.transport_hubs:
            x, y = hub['position']
            hub_type = hub['type']
            label = hub['label']
            color = self.colors['hubs'][hub_type]
            
            # 绘制 Hub
            ax.plot(x, y, 'o', color=color, markersize=12, 
                   label=label, markeredgecolor='black', markeredgewidth=2)
            
            # 为 Hub2 工业中心添加特殊标记
            if hub_type == 'industrial':
                # 添加工业中心标识
                ax.plot(x, y, 's', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=1)
    
    def _plot_buildings(self, ax, buildings: List[Dict], month: int):
        """绘制建筑（支持 Hub2 工业中心伪装）"""
        if not buildings:
            return
        
        # 按类型分组建筑
        building_groups = {}
        for building in buildings:
            x, y = building['position']
            building_type = building['type']
            color, display_type = self._get_building_color(building_type, x, y)
            
            if display_type not in building_groups:
                building_groups[display_type] = {'x': [], 'y': [], 'color': color}
            
            building_groups[display_type]['x'].append(x)
            building_groups[display_type]['y'].append(y)
        
        # 绘制不同类型的建筑
        for display_type, group in building_groups.items():
            ax.scatter(group['x'], group['y'], c=group['color'], s=20, alpha=0.7, 
                      label=f'{display_type}建筑', edgecolors='black', linewidth=0.5)
    
    def _plot_land_price_field(self, ax, land_price_data: Dict, month: int):
        """绘制地价场"""
        if not land_price_data:
            return
        
        # 提取地价场数据
        field = np.array(land_price_data.get('field', []))
        if field.size == 0:
            return
        
        # 绘制地价场热力图
        im = ax.imshow(field, cmap=self.colors['land_price'], alpha=0.6, 
                      extent=[0, field.shape[1], 0, field.shape[0]], origin='lower')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('地价水平', fontsize=10)
        
        # 添加标题
        ax.set_title(f'地价场分布 (第{month}月)', fontsize=12, fontweight='bold')
    
    def _plot_contours(self, ax, contour_data: Dict, month: int):
        """绘制等值线"""
        if not contour_data:
            return
        
        # 绘制等值线
        for level, contours in contour_data.items():
            for contour in contours:
                if len(contour) > 2:
                    contour_array = np.array(contour)
                    ax.plot(contour_array[:, 0], contour_array[:, 1], 
                           color=self.colors['contours'], linewidth=2, alpha=0.8)
        
        # 添加标题
        ax.set_title(f'等值线建筑槽位 (第{month}月)', fontsize=12, fontweight='bold')
    
    def _create_frame(self, month: int):
        """创建单帧图像"""
        data = self._load_month_data(month)
        
        # 清除之前的图像
        for ax in self.axes.flat:
            ax.clear()
        
        # 设置图像标题
        self.fig.suptitle(f'Enhanced City Simulation v3.1.1 - 第{month}月演化 (帧 {self.current_frame + 1}/{len(self.months)})', 
                         fontsize=16, fontweight='bold')
        
        # 绘制地价场
        self._plot_land_price_field(self.axes[0, 0], data.get('land_price', {}), month)
        self._plot_transport_hubs(self.axes[0, 0])
        
        # 绘制等值线
        self._plot_contours(self.axes[0, 1], data.get('contours', {}), month)
        self._plot_transport_hubs(self.axes[0, 1])
        
        # 绘制建筑分布
        self._plot_buildings(self.axes[1, 0], data.get('buildings', []), month)
        self._plot_transport_hubs(self.axes[1, 0])
        
        # 绘制综合视图
        self._plot_land_price_field(self.axes[1, 1], data.get('land_price', {}), month)
        self._plot_contours(self.axes[1, 1], data.get('contours', {}), month)
        self._plot_buildings(self.axes[1, 1], data.get('buildings', []), month)
        self._plot_transport_hubs(self.axes[1, 1])
        
        # 设置坐标轴
        for ax in self.axes.flat:
            ax.set_xlim(0, 110)
            ax.set_ylim(0, 110)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        # 设置子图标题
        self.axes[0, 0].set_title('地价场分布', fontsize=12, fontweight='bold')
        self.axes[0, 1].set_title('等值线建筑槽位', fontsize=12, fontweight='bold')
        self.axes[1, 0].set_title('建筑分布', fontsize=12, fontweight='bold')
        self.axes[1, 1].set_title('综合视图', fontsize=12, fontweight='bold')
        
        # 添加 Hub2 工业中心说明
        self.axes[1, 1].text(0.02, 0.98, 'Hub2 附近商业建筑显示为工业建筑颜色', 
                            transform=self.axes[1, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', 
                            facecolor='lightblue', alpha=0.8))
        
        # 添加帧信息
        self.axes[0, 0].text(0.02, 0.02, f'帧: {self.current_frame + 1}/{len(self.months)}', 
                            transform=self.axes[0, 0].transAxes, fontsize=12,
                            verticalalignment='bottom', bbox=dict(boxstyle='round', 
                            facecolor='white', alpha=0.8))
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 'right' or event.key == ' ':
            # 下一帧
            self.next_frame()
        elif event.key == 'left':
            # 上一帧
            self.prev_frame()
        elif event.key == 'home':
            # 第一帧
            self.first_frame()
        elif event.key == 'end':
            # 最后一帧
            self.last_frame()
        elif event.key == 'r':
            # 重新开始
            self.restart()
        elif event.key == 's':
            # 保存当前帧
            self.save_current_frame()
        elif event.key == 'q':
            # 退出
            plt.close()
        elif event.key == 'h':
            # 显示帮助
            self._show_controls()
    
    def next_frame(self):
        """下一帧"""
        if self.current_frame < len(self.months) - 1:
            self.current_frame += 1
            month = self.months[self.current_frame]
            self._create_frame(month)
            print(f"播放到第 {month} 月 (帧 {self.current_frame + 1}/{len(self.months)})")
    
    def prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.current_frame -= 1
            month = self.months[self.current_frame]
            self._create_frame(month)
            print(f"播放到第 {month} 月 (帧 {self.current_frame + 1}/{len(self.months)})")
    
    def first_frame(self):
        """第一帧"""
        self.current_frame = 0
        month = self.months[self.current_frame]
        self._create_frame(month)
        print(f"播放到第 {month} 月 (帧 {self.current_frame + 1}/{len(self.months)})")
    
    def last_frame(self):
        """最后一帧"""
        self.current_frame = len(self.months) - 1
        month = self.months[self.current_frame]
        self._create_frame(month)
        print(f"播放到第 {month} 月 (帧 {self.current_frame + 1}/{len(self.months)})")
    
    def restart(self):
        """重新开始"""
        self.first_frame()
        print("重新开始播放")
    
    def save_current_frame(self):
        """保存当前帧"""
        month = self.months[self.current_frame]
        filename = f"frame_{self.current_frame:02d}_month_{month:02d}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"当前帧已保存: {filename}")
    
    def _show_controls(self):
        """显示控制说明"""
        print("\n" + "="*60)
        print("Enhanced City Simulation v3.1.1 逐帧播放器")
        print("="*60)
        print("键盘控制:")
        print("  → 或 空格键: 下一帧")
        print("  ←: 上一帧")
        print("  Home: 第一帧")
        print("  End: 最后一帧")
        print("  R: 重新开始")
        print("  S: 保存当前帧")
        print("  H: 显示帮助")
        print("  Q: 退出")
        print("="*60)
        print(f"总帧数: {len(self.months)}")
        print(f"月份范围: {min(self.months)} - {max(self.months)}")
        print("="*60)
    
    def play(self):
        """开始播放"""
        if not self.months:
            print("错误: 没有找到可用的数据文件")
            return
        
        # 显示第一帧
        self.first_frame()
        
        # 显示窗口
        plt.show()

def main():
    """主函数"""
    print("Enhanced City Simulation v3.1.1 逐帧播放器")
    print("支持手动控制播放过程，观察每一帧的演化细节")
    print()
    
    # 创建播放器
    player = FrameByFramePlayer()
    
    if not player.months:
        print("错误: 没有找到可用的数据文件")
        return
    
    # 开始播放
    player.play()

if __name__ == "__main__":
    main()