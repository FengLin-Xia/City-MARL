#!/usr/bin/env python3
"""
增强城市模拟系统 v3.1 可视化播放器
逐帧显示地价场变化、建筑分布、层状态等
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
from typing import Dict, List, Tuple
import glob

class V3_1EvolutionPlayback:
    """v3.1系统演化可视化播放器"""
    
    def __init__(self, output_dir: str = "enhanced_simulation_v3_1_output"):
        self.output_dir = output_dir
        self.land_price_frames = []
        self.building_frames = []
        self.layer_frames = []
        self.months = []
        self._load_frames()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"🎬 v3.1可视化播放器初始化完成")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 加载了 {len(self.land_price_frames)} 个地价场帧")
        print(f"🏗️ 加载了 {len(self.building_frames)} 个建筑帧")
        print(f"📋 加载了 {len(self.layer_frames)} 个层状态帧")
    
    def _load_frames(self):
        """加载所有帧数据"""
        # 加载地价场帧
        land_price_files = sorted(glob.glob(f"{self.output_dir}/land_price_frame_month_*.json"))
        for file_path in land_price_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    month = frame_data.get('month', 0)
                    self.land_price_frames.append(frame_data)
                    if month not in self.months:
                        self.months.append(month)
            except Exception as e:
                print(f"⚠️ 加载地价场帧失败: {file_path}, 错误: {e}")
        
        # 加载建筑位置帧（支持增量数据重建）
        self._load_building_frames()
        
        # 加载层状态帧
        layer_files = sorted(glob.glob(f"{self.output_dir}/layer_state_month_*.json"))
        for file_path in layer_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    month = frame_data.get('month', 0)
                    self.layer_frames.append(frame_data)
                    if month not in self.months:
                        self.months.append(month)
            except Exception as e:
                print(f"⚠️ 加载层状态帧失败: {file_path}, 错误: {e}")
        
        # 排序月份
        self.months.sort()
        
        print(f"📅 模拟月份范围: {min(self.months)} - {max(self.months)}")
        print(f"📊 成功加载 {len(self.land_price_frames)} 个地价场帧")
        print(f"🏗️ 成功加载 {len(self.building_frames)} 个建筑帧")
        print(f"📋 成功加载 {len(self.layer_frames)} 个层状态帧")
    
    def _load_building_frames(self):
        """加载建筑位置帧（支持增量数据重建）"""
        # 首先加载所有完整的建筑位置文件
        building_files = sorted(glob.glob(f"{self.output_dir}/building_positions_month_*.json"))
        for file_path in building_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    # 建筑文件没有month字段，从文件名提取
                    filename = os.path.basename(file_path)
                    if 'month_' in filename:
                        month_str = filename.split('month_')[1].split('.')[0]
                        month = int(month_str)
                        frame_data['month'] = month  # 添加month字段
                    else:
                        month = frame_data.get('month', 0)
                    
                    self.building_frames.append(frame_data)
                    if month not in self.months:
                        self.months.append(month)
            except Exception as e:
                print(f"⚠️ 加载建筑帧失败: {file_path}, 错误: {e}")
        
        # 然后处理增量数据，重建缺失的月份
        self._rebuild_missing_building_frames()
    
    def _rebuild_missing_building_frames(self):
        """重建缺失的建筑帧（从增量数据）"""
        # 找到所有可用的增量文件
        delta_files = sorted(glob.glob(f"{self.output_dir}/building_delta_month_*.json"))
        
        if not delta_files:
            print("📊 没有找到增量建筑数据文件")
            return
        
        # 找到最大的月份
        max_month = 0
        for file_path in delta_files:
            filename = os.path.basename(file_path)
            if 'month_' in filename:
                month_str = filename.split('month_')[1].split('.')[0]
                month = int(month_str)
                max_month = max(max_month, month)
        
        print(f"🔄 开始重建建筑帧，最大月份: {max_month}")
        
        # 重建每个月的建筑状态
        for month in range(max_month + 1):
            if month in [frame['month'] for frame in self.building_frames]:
                continue  # 已经存在，跳过
            
            # 重建这个月的建筑状态
            rebuilt_data = self._rebuild_building_state_for_month(month)
            if rebuilt_data:
                self.building_frames.append(rebuilt_data)
                if month not in self.months:
                    self.months.append(month)
                print(f"✅ 重建第 {month} 个月建筑状态: {len(rebuilt_data['buildings'])} 个建筑")
    
    def _rebuild_building_state_for_month(self, target_month: int) -> Dict:
        """重建指定月份的完整建筑状态"""
        # 加载第0个月的完整状态作为基础
        month_0_file = f"{self.output_dir}/building_positions_month_00.json"
        if not os.path.exists(month_0_file):
            print(f"⚠️ 第0个月完整状态文件不存在: {month_0_file}")
            return None
        
        try:
            with open(month_0_file, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
                buildings = base_data.get('buildings', []).copy()
        except Exception as e:
            print(f"⚠️ 加载第0个月状态失败: {e}")
            return None
        
        # 累加后续月份的新增建筑
        for month in range(1, target_month + 1):
            delta_file = f"{self.output_dir}/building_delta_month_{month:02d}.json"
            if os.path.exists(delta_file):
                try:
                    with open(delta_file, 'r', encoding='utf-8') as f:
                        delta_data = json.load(f)
                        new_buildings = delta_data.get('new_buildings', [])
                        buildings.extend(new_buildings)
                except Exception as e:
                    print(f"⚠️ 加载第{month}个月增量数据失败: {e}")
        
        # 创建重建的数据结构
        rebuilt_data = {
            'month': target_month,
            'timestamp': f'month_{target_month:02d}',
            'buildings': buildings
        }
        
        return rebuilt_data
    
    def _plot_transport_hubs(self, ax):
        """绘制交通枢纽"""
        # 从配置或数据中获取交通枢纽位置
        transport_hubs = [[20, 55], [90, 55], [67, 94]]  # Hub1, Hub2, Hub3
        
        for i, hub in enumerate(transport_hubs):
            ax.plot(hub[0], hub[1], 'ro', markersize=10, label=f'Hub {i+1}')
        
        ax.legend()
    
    def _get_frame_data(self, month: int) -> Tuple[Dict, Dict, Dict]:
        """获取指定月份的所有帧数据"""
        land_price_data = None
        building_data = None
        layer_data = None
        
        # 查找地价场数据
        for frame in self.land_price_frames:
            if frame.get('month') == month:
                land_price_data = frame
                break
        
        # 查找建筑数据
        for frame in self.building_frames:
            if frame.get('month') == month:
                building_data = frame
                break
        
        # 查找层状态数据
        for frame in self.layer_frames:
            if frame.get('month') == month:
                layer_data = frame
                break
        
        return land_price_data, building_data, layer_data
    
    def _create_frame(self, frame_idx):
        """创建单个帧的可视化"""
        if frame_idx >= len(self.months):
            return None
        
        month = self.months[frame_idx]
        land_price_data, building_data, layer_data = self._get_frame_data(month)
        
        # 创建2x3的子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🏙️ 增强城市模拟系统 v3.1 - 第 {month} 个月', fontsize=16, fontweight='bold')
        
        # 1. 地价场热力图
        if land_price_data and 'land_price_field' in land_price_data:
            land_price_field = np.array(land_price_data['land_price_field'])
            im1 = axes[0, 0].imshow(land_price_field, cmap='viridis', aspect='equal')
            axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('X (像素)')
            axes[0, 0].set_ylabel('Y (像素)')
            plt.colorbar(im1, ax=axes[0, 0], label='地价值')
            
            # 添加交通枢纽标记
            self._plot_transport_hubs(axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, '无地价场数据', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
        
        # 2. 建筑分布图
        if building_data and 'buildings' in building_data:
            buildings = building_data['buildings']
            
            # 分类建筑
            residential = [b for b in buildings if b['type'] == 'residential']
            commercial = [b for b in buildings if b['type'] == 'commercial']
            public = [b for b in buildings if b['type'] == 'public']
            
            # 绘制建筑
            if residential:
                res_x = [b['position'][0] for b in residential]
                res_y = [b['position'][1] for b in residential]
                axes[0, 1].scatter(res_x, res_y, c='#F6C344', s=50, alpha=0.8, label=f'住宅 ({len(residential)})')
            
            if commercial:
                com_x = [b['position'][0] for b in commercial]
                com_y = [b['position'][1] for b in commercial]
                axes[0, 1].scatter(com_x, com_y, c='#FD7E14', s=50, alpha=0.8, label=f'商业 ({len(commercial)})')
            
            if public:
                pub_x = [b['position'][0] for b in public]
                pub_y = [b['position'][1] for b in public]
                axes[0, 1].scatter(pub_x, pub_y, c='#22A6B3', s=50, alpha=0.8, label=f'公共 ({len(public)})')
            
            # 添加交通枢纽
            self._plot_transport_hubs(axes[0, 1])
            
            axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('X (像素)')
            axes[0, 1].set_ylabel('Y (像素)')
            axes[0, 1].legend()
            axes[0, 1].set_xlim(0, 110)
            axes[0, 1].set_ylim(0, 110)
        else:
            axes[0, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
        
        # 3. 层状态可视化
        if layer_data and 'layers' in layer_data:
            layers = layer_data['layers']
            
            # 商业建筑层状态
            if 'commercial' in layers:
                commercial_layers = layers['commercial']
                for i, layer in enumerate(commercial_layers):
                    status = layer['status']
                    density = layer['density']
                    placed = layer['placed']
                    capacity = layer['capacity_effective']
                    
                    # 状态颜色
                    if status == 'locked':
                        color = 'gray'
                        status_text = '🔒'
                    elif status == 'active':
                        color = 'green'
                        status_text = '🟢'
                    else:  # complete
                        color = 'blue'
                        status_text = '✅'
                    
                    # 绘制层进度条
                    y_pos = 0.8 - i * 0.15
                    axes[0, 2].barh(y_pos, density, height=0.1, color=color, alpha=0.7)
                    axes[0, 2].text(0.5, y_pos, f'{status_text} P{i}: {placed}/{capacity}', 
                                   ha='center', va='center', fontweight='bold')
                
                axes[0, 2].set_xlim(0, 1)
                axes[0, 2].set_ylim(0, 1)
                axes[0, 2].set_title('🏢 商业建筑层状态', fontsize=12, fontweight='bold')
                axes[0, 2].set_xlabel('填充密度')
        
        # 4. 地价场统计
        if land_price_data and 'land_price_stats' in land_price_data:
            stats = land_price_data['land_price_stats']
            
            # 创建统计图表
            labels = ['最小值', '平均值', '最大值']
            values = [stats.get('min_price', 0), stats.get('avg_price', 0), stats.get('max_price', 0)]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = axes[1, 0].bar(labels, values, color=colors, alpha=0.7)
            axes[1, 0].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('地价值')
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, '无地价统计', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
        
        # 5. 建筑类型统计
        if building_data and 'buildings' in building_data:
            buildings = building_data['buildings']
            
            # 统计建筑类型
            building_types = {}
            for building in buildings:
                btype = building['type']
                building_types[btype] = building_types.get(btype, 0) + 1
            
            if building_types:
                labels = list(building_types.keys())
                values = list(building_types.values())
                colors = ['#F6C344', '#FD7E14', '#22A6B3']
                
                # 中文标签映射
                label_map = {'residential': '住宅', 'commercial': '商业', 'public': '公共'}
                chinese_labels = [label_map.get(label, label) for label in labels]
                
                wedges, texts, autotexts = axes[1, 1].pie(values, labels=chinese_labels, colors=colors, 
                                                          autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
                
                # 在饼图上添加数量标签
                for i, (wedge, value) in enumerate(zip(wedges, values)):
                    angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
                    x = 0.8 * np.cos(np.radians(angle))
                    y = 0.8 * np.sin(np.radians(angle))
                    axes[1, 1].text(x, y, f'{value}个', ha='center', va='center', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
        
        # 6. 演化阶段信息
        if land_price_data and 'evolution_stage' in land_price_data:
            stage = land_price_data['evolution_stage']
            
            # 显示演化阶段信息
            axes[1, 2].text(0.1, 0.8, f"演化阶段: {stage.get('name', '未知')}", fontsize=14, fontweight='bold')
            axes[1, 2].text(0.1, 0.6, f"Hub σ: {stage.get('hub_sigma', 0):.1f}", fontsize=12)
            axes[1, 2].text(0.1, 0.4, f"Road σ: {stage.get('road_sigma', 0):.1f}", fontsize=12)
            axes[1, 2].text(0.1, 0.2, f"当前月份: {month}", fontsize=12)
            
            axes[1, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, '无演化数据', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        return fig
    
    def play_animation(self, interval: int = 1000, save_gif: bool = False):
        """播放动画"""
        if not self.months:
            print("❌ 没有可播放的帧数据")
            return
        
        print(f"🎬 开始播放动画，共 {len(self.months)} 帧，间隔 {interval}ms")
        
        # 创建初始图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        def animate(frame_idx):
            if frame_idx >= len(self.months):
                return []
            
            month = self.months[frame_idx]
            land_price_data, building_data, layer_data = self._get_frame_data(month)
            
            # 清除所有子图
            for ax in axes.flat:
                ax.clear()
            
            # 设置总标题
            fig.suptitle(f'🏙️ 增强城市模拟系统 v3.1 - 第 {month} 个月', fontsize=16, fontweight='bold')
            
            # 1. 地价场热力图
            if land_price_data and 'land_price_field' in land_price_data:
                land_price_field = np.array(land_price_data['land_price_field'])
                im1 = axes[0, 0].imshow(land_price_field, cmap='viridis', aspect='equal')
                axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('X (像素)')
                axes[0, 0].set_ylabel('Y (像素)')
                plt.colorbar(im1, ax=axes[0, 0], label='地价值')
                
                # 添加交通枢纽标记
                self._plot_transport_hubs(axes[0, 0])
            else:
                axes[0, 0].text(0.5, 0.5, '无地价场数据', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
            
            # 2. 建筑分布图
            if building_data and 'buildings' in building_data:
                buildings = building_data['buildings']
                
                # 分类建筑
                residential = [b for b in buildings if b['type'] == 'residential']
                commercial = [b for b in buildings if b['type'] == 'commercial']
                public = [b for b in buildings if b['type'] == 'public']
                
                # 绘制建筑
                if residential:
                    res_x = [b['position'][0] for b in residential]
                    res_y = [b['position'][1] for b in residential]
                    axes[0, 1].scatter(res_x, res_y, c='#F6C344', s=50, alpha=0.8, label=f'住宅 ({len(residential)})')
                
                if commercial:
                    com_x = [b['position'][0] for b in commercial]
                    com_y = [b['position'][1] for b in commercial]
                    axes[0, 1].scatter(com_x, com_y, c='#FD7E14', s=50, alpha=0.8, label=f'商业 ({len(commercial)})')
                
                if public:
                    pub_x = [b['position'][0] for b in public]
                    pub_y = [b['position'][1] for b in public]
                    axes[0, 1].scatter(pub_x, pub_y, c='#22A6B3', s=50, alpha=0.8, label=f'公共 ({len(public)})')
                
                # 添加交通枢纽
                self._plot_transport_hubs(axes[0, 1])
                
                axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('X (像素)')
                axes[0, 1].set_ylabel('Y (像素)')
                axes[0, 1].legend()
                axes[0, 1].set_xlim(0, 110)
                axes[0, 1].set_ylim(0, 110)
            else:
                axes[0, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
            
            # 3. 层状态可视化
            if layer_data and 'layers' in layer_data:
                layers = layer_data['layers']
                
                # 商业建筑层状态
                if 'commercial' in layers:
                    commercial_layers = layers['commercial']
                    for i, layer in enumerate(commercial_layers):
                        status = layer['status']
                        density = layer['density']
                        placed = layer['placed']
                        capacity = layer['capacity_effective']
                        
                        # 状态颜色
                        if status == 'locked':
                            color = 'gray'
                            status_text = '🔒'
                        elif status == 'active':
                            color = 'green'
                            status_text = '🟢'
                        else:  # complete
                            color = 'blue'
                            status_text = '✅'
                        
                        # 绘制层进度条
                        y_pos = 0.8 - i * 0.15
                        axes[0, 2].barh(y_pos, density, height=0.1, color=color, alpha=0.7)
                        axes[0, 2].text(0.5, y_pos, f'{status_text} P{i}: {placed}/{capacity}', 
                                       ha='center', va='center', fontweight='bold')
                    
                    axes[0, 2].set_xlim(0, 1)
                    axes[0, 2].set_ylim(0, 1)
                    axes[0, 2].set_title('🏢 商业建筑层状态', fontsize=12, fontweight='bold')
                    axes[0, 2].set_xlabel('填充密度')
            
            # 4. 地价场统计
            if land_price_data and 'land_price_stats' in land_price_data:
                stats = land_price_data['land_price_stats']
                
                # 创建统计图表
                labels = ['最小值', '平均值', '最大值']
                values = [stats.get('min_price', 0), stats.get('avg_price', 0), stats.get('max_price', 0)]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                bars = axes[1, 0].bar(labels, values, color=colors, alpha=0.7)
                axes[1, 0].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
                axes[1, 0].set_ylabel('地价值')
                
                # 在柱状图上添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value:.2f}', ha='center', va='bottom')
            else:
                axes[1, 0].text(0.5, 0.5, '无地价统计', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
            
            # 5. 建筑类型统计
            if building_data and 'buildings' in building_data:
                buildings = building_data['buildings']
                
                # 统计建筑类型
                building_types = {}
                for building in buildings:
                    btype = building['type']
                    building_types[btype] = building_types.get(btype, 0) + 1
                
                if building_types:
                    labels = list(building_types.keys())
                    values = list(building_types.values())
                    colors = ['#F6C344', '#FD7E14', '#22A6B3']
                    
                    # 中文标签映射
                    label_map = {'residential': '住宅', 'commercial': '商业', 'public': '公共'}
                    chinese_labels = [label_map.get(label, label) for label in labels]
                    
                    wedges, texts, autotexts = axes[1, 1].pie(values, labels=chinese_labels, colors=colors, 
                                                              autopct='%1.1f%%', startangle=90)
                    axes[1, 1].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
                    
                    # 在饼图上添加数量标签
                    for i, (wedge, value) in enumerate(zip(wedges, values)):
                        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
                        x = 0.8 * np.cos(np.radians(angle))
                        y = 0.8 * np.sin(np.radians(angle))
                        axes[1, 1].text(x, y, f'{value}个', ha='center', va='center', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
            
            # 6. 演化阶段信息
            if land_price_data and 'evolution_stage' in land_price_data:
                stage = land_price_data['evolution_stage']
                
                # 显示演化阶段信息
                axes[1, 2].text(0.1, 0.8, f"演化阶段: {stage.get('name', '未知')}", fontsize=14, fontweight='bold')
                axes[1, 2].text(0.1, 0.6, f"Hub σ: {stage.get('hub_sigma', 0):.1f}", fontsize=12)
                axes[1, 2].text(0.1, 0.4, f"Road σ: {stage.get('road_sigma', 0):.1f}", fontsize=12)
                axes[1, 2].text(0.1, 0.2, f"当前月份: {month}", fontsize=12)
                
                axes[1, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].axis('off')
            else:
                axes[1, 2].text(0.5, 0.5, '无演化数据', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            return []
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, 
            animate, 
            frames=len(self.months),
            interval=interval, 
            repeat=True,
            blit=False
        )
        
        if save_gif:
            print("💾 保存GIF动画...")
            anim.save(f'{self.output_dir}/v3_1_evolution.gif', writer='pillow', fps=1)
            print(f"✅ GIF已保存到 {self.output_dir}/v3_1_evolution.gif")
        
        plt.show()
        return anim
    
    def show_frame(self, month: int):
        """显示指定月份的单帧"""
        if month not in self.months:
            print(f"❌ 月份 {month} 不存在")
            return
        
        fig = self._create_frame(self.months.index(month))
        if fig:
            plt.show()
    
    def show_all_frames(self):
        """显示所有帧（静态）"""
        if not self.months:
            print("❌ 没有可显示的帧数据")
            return
        
        print(f"📺 显示所有 {len(self.months)} 帧（按任意键继续）")
        
        for i, month in enumerate(self.months):
            print(f"\n📅 显示第 {month} 个月 (第 {i+1}/{len(self.months)} 帧)")
            fig = self._create_frame(i)
            if fig:
                plt.show()
                input("按回车键继续下一帧...")

def main():
    """主函数"""
    print("🎬 增强城市模拟系统 v3.1 可视化播放器")
    print("=" * 60)
    
    # 创建播放器
    player = V3_1EvolutionPlayback()
    
    if not player.months:
        print("❌ 没有找到可播放的数据，请先运行模拟")
        return
    
    print("\n🎮 播放选项:")
    print("1. 播放动画 (GIF)")
    print("2. 播放动画 (实时)")
    print("3. 显示单帧")
    print("4. 逐帧浏览")
    print("5. 退出")
    print("\n💡 提示: 在任何时候按 Ctrl+C 可以强制结束程序")
    
    while True:
        try:
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == '1':
                print("🎬 开始生成GIF动画... (按 Ctrl+C 可强制结束)")
                try:
                    player.play_animation(interval=1000, save_gif=True)
                except KeyboardInterrupt:
                    print("\n⚠️ 动画生成被中断")
                    plt.close('all')  # 关闭所有图形
            elif choice == '2':
                print("🎬 开始播放动画... (按 Ctrl+C 可强制结束)")
                try:
                    player.play_animation(interval=1000, save_gif=False)
                except KeyboardInterrupt:
                    print("\n⚠️ 动画播放被中断")
                    plt.close('all')  # 关闭所有图形
            elif choice == '3':
                try:
                    month = int(input("请输入月份: "))
                    player.show_frame(month)
                except ValueError:
                    print("❌ 请输入有效的月份数字")
            elif choice == '4':
                print("📺 开始逐帧浏览... (按 Ctrl+C 可强制结束)")
                try:
                    player.show_all_frames()
                except KeyboardInterrupt:
                    print("\n⚠️ 逐帧浏览被中断")
                    plt.close('all')  # 关闭所有图形
            elif choice == '5':
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入1-5")
        
        except KeyboardInterrupt:
            print("\n👋 再见！")
            plt.close('all')  # 确保关闭所有图形
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            plt.close('all')  # 确保关闭所有图形

if __name__ == "__main__":
    main()
