#!/usr/bin/env python3
"""
增强城市模拟系统 v3.2 可视化播放器
逐帧显示地价场变化、建筑分布、层状态、决策日志等
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
from typing import Dict, List, Tuple
import glob

class V3_2EvolutionPlayback:
    """v3.2系统演化可视化播放器"""
    
    def __init__(self, output_dir: str = "enhanced_simulation_v3_2_output"):
        self.output_dir = output_dir
        self.land_price_frames = []
        self.building_frames = []
        self.layer_frames = []
        self.decision_frames = []
        self.months = []
        self._load_frames()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"🎬 v3.2可视化播放器初始化完成")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 加载了 {len(self.land_price_frames)} 个地价场帧")
        print(f"🏗️ 加载了 {len(self.building_frames)} 个建筑帧")
        print(f"📋 加载了 {len(self.layer_frames)} 个层状态帧")
        print(f"🎯 加载了 {len(self.decision_frames)} 个决策日志帧")
    
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
        
        # 加载建筑位置帧 - 从增量文件重建完整状态
        self._load_building_frames_from_deltas()
        
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
        
        # 加载决策日志帧
        decision_files = sorted(glob.glob(f"{self.output_dir}/decision_log_month_*.json"))
        for file_path in decision_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    month = frame_data.get('month', 0)
                    self.decision_frames.append(frame_data)
                    if month not in self.months:
                        self.months.append(month)
            except Exception as e:
                print(f"⚠️ 加载决策日志帧失败: {file_path}, 错误: {e}")
        
        # 排序月份
        self.months.sort()
        
        print(f"📅 模拟月份范围: {min(self.months)} - {max(self.months)}")
        print(f"📊 成功加载 {len(self.land_price_frames)} 个地价场帧")
        print(f"🏗️ 成功加载 {len(self.building_frames)} 个建筑帧")
        print(f"📋 成功加载 {len(self.layer_frames)} 个层状态帧")
        print(f"🎯 成功加载 {len(self.decision_frames)} 个决策日志帧")
    
    def _load_building_frames_from_deltas(self):
        """从增量文件重建建筑帧数据"""
        # 首先尝试加载第0个月的完整状态
        month_0_file = f"{self.output_dir}/building_positions_month_0.json"
        if os.path.exists(month_0_file):
            try:
                with open(month_0_file, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    frame_data['month'] = 0
                    self.building_frames.append(frame_data)
                    if 0 not in self.months:
                        self.months.append(0)
                print(f"✅ 加载第0个月完整建筑状态")
            except Exception as e:
                print(f"⚠️ 加载第0个月建筑状态失败: {e}")
        
        # 然后从增量文件重建后续月份
        delta_files = sorted(glob.glob(f"{self.output_dir}/building_delta_month_*.json"))
        current_buildings = {}
        
        # 如果有第0个月数据，初始化当前建筑状态
        if self.building_frames:
            current_buildings = self.building_frames[0].get('buildings', {}).copy()
        
        for delta_file in delta_files:
            try:
                # 从文件名提取月份
                filename = os.path.basename(delta_file)
                month_str = filename.split('month_')[1].split('.')[0]
                month = int(month_str)
                
                with open(delta_file, 'r', encoding='utf-8') as f:
                    delta_data = json.load(f)
                
                # 应用增量：添加新建筑
                new_buildings = delta_data.get('new_buildings', [])
                for building in new_buildings:
                    building_id = building.get('id', f"building_{len(current_buildings)}")
                    current_buildings[building_id] = building
                
                # 创建当前月份的完整建筑状态
                frame_data = {
                    'month': month,
                    'buildings': current_buildings.copy(),
                    'total_buildings': len(current_buildings),
                    'new_buildings_this_month': len(new_buildings)
                }
                
                self.building_frames.append(frame_data)
                if month not in self.months:
                    self.months.append(month)
                
                print(f"✅ 从增量重建第{month}个月建筑状态: {len(new_buildings)}个新建筑")
                
            except Exception as e:
                print(f"⚠️ 处理增量文件失败: {delta_file}, 错误: {e}")
        
        # 如果没有找到任何建筑数据，尝试加载完整的建筑位置文件作为备选
        if not self.building_frames:
            building_files = sorted(glob.glob(f"{self.output_dir}/building_positions_month_*.json"))
            for file_path in building_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                        filename = os.path.basename(file_path)
                        if 'month_' in filename:
                            month_str = filename.split('month_')[1].split('.')[0]
                            month = int(month_str)
                            frame_data['month'] = month
                        else:
                            month = frame_data.get('month', 0)
                        
                        self.building_frames.append(frame_data)
                        if month not in self.months:
                            self.months.append(month)
                except Exception as e:
                    print(f"⚠️ 备选加载建筑帧失败: {file_path}, 错误: {e}")
    
    def _get_frame_data(self, month: int) -> Tuple[Dict, Dict, Dict, Dict]:
        """获取指定月份的所有帧数据"""
        land_price_data = None
        building_data = None
        layer_data = None
        decision_data = None
        
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
        
        # 查找决策日志数据
        for frame in self.decision_frames:
            if frame.get('month') == month:
                decision_data = frame
                break
        
        return land_price_data, building_data, layer_data, decision_data
    
    def _create_frame(self, frame_idx):
        """创建单个帧的可视化"""
        if frame_idx >= len(self.months):
            return None
        
        month = self.months[frame_idx]
        land_price_data, building_data, layer_data, decision_data = self._get_frame_data(month)
        
        # 创建3x3的子图布局
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'🏙️ 增强城市模拟系统 v3.2 - 第 {month} 个月', fontsize=18, fontweight='bold')
        
        # 1. 地价场热力图
        if land_price_data and 'land_price_field' in land_price_data:
            land_price_field = np.array(land_price_data['land_price_field'])
            im1 = axes[0, 0].imshow(land_price_field, cmap='viridis', aspect='equal')
            axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('X (像素)')
            axes[0, 0].set_ylabel('Y (像素)')
            plt.colorbar(im1, ax=axes[0, 0], label='地价值')
            
            # 添加交通枢纽标记
            axes[0, 0].plot(20, 55, 'ro', markersize=10, label='Hub 1 (商业)')
            axes[0, 0].plot(90, 55, 'bo', markersize=10, label='Hub 2 (工业)')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, '无地价场数据', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
        
        # 2. 建筑分布图
        if building_data and 'buildings' in building_data:
            buildings = building_data['buildings']
            
            # 分类建筑
            residential = [b for b in buildings if b['type'] == 'residential']
            commercial = [b for b in buildings if b['type'] == 'commercial']
            industrial = [b for b in buildings if b['type'] == 'industrial']
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
            
            if industrial:
                ind_x = [b['position'][0] for b in industrial]
                ind_y = [b['position'][1] for b in industrial]
                axes[0, 1].scatter(ind_x, ind_y, c='#8E44AD', s=50, alpha=0.8, label=f'工业 ({len(industrial)})')
            
            if public:
                pub_x = [b['position'][0] for b in public]
                pub_y = [b['position'][1] for b in public]
                axes[0, 1].scatter(pub_x, pub_y, c='#22A6B3', s=50, alpha=0.8, label=f'公共 ({len(public)})')
            
            # 添加交通枢纽
            axes[0, 1].plot(20, 55, 'ro', markersize=10, label='Hub 1 (商业)')
            axes[0, 1].plot(90, 55, 'bo', markersize=10, label='Hub 2 (工业)')
            
            axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('X (像素)')
            axes[0, 1].set_ylabel('Y (像素)')
            axes[0, 1].legend()
            axes[0, 1].set_xlim(0, 110)
            axes[0, 1].set_ylim(0, 110)
        else:
            axes[0, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
        
        # 3. 政府骨架系统
        axes[0, 2].text(0.1, 0.9, "🏛️ 政府骨架系统", fontsize=14, fontweight='bold')
        axes[0, 2].text(0.1, 0.8, "走廊带: 主干道中心线", fontsize=12)
        axes[0, 2].text(0.1, 0.7, "Hub1: 商业客运核", fontsize=12)
        axes[0, 2].text(0.1, 0.6, "Hub2: 工业货运核", fontsize=12)
        axes[0, 2].text(0.1, 0.5, "分区约束: 政府规划", fontsize=12)
        axes[0, 2].text(0.1, 0.4, "配额管理: 季度动态", fontsize=12)
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].axis('off')
        
        # 4. 层状态可视化
        if layer_data and 'layers' in layer_data:
            layers_info = layer_data['layers']
            
            # 显示生长阶段
            growth_phase = layers_info.get('growth_phase', 'unknown')
            road_completed = layers_info.get('road_layers_completed', False)
            
            # 生长阶段图标
            phase_icon = {'road_corridor': '🛣️', 'radial_expansion': '🎯', 'unknown': '❓'}.get(growth_phase, '❓')
            axes[1, 0].text(0.1, 0.9, f"生长阶段: {phase_icon} {growth_phase}", fontsize=12, fontweight='bold')
            axes[1, 0].text(0.1, 0.8, f"走廊层完成: {'是' if road_completed else '否'}", fontsize=12)
            axes[1, 0].text(0.1, 0.7, f"总层数: {layers_info.get('total_layers', 0)}", fontsize=12)
            axes[1, 0].text(0.1, 0.6, f"激活层数: {layers_info.get('active_layers', 0)}", fontsize=12)
            axes[1, 0].text(0.1, 0.5, f"完成层数: {layers_info.get('completed_layers', 0)}", fontsize=12)
            
            # 显示层详情
            layers = layers_info.get('layers', [])
            y_pos = 0.4
            for layer in layers[:5]:  # 只显示前5个层
                layer_id = layer.get('layer_id', 'unknown')
                status = layer.get('status', 'unknown')
                density = layer.get('density', 0)
                layer_type = layer.get('layer_type', 'unknown')
                
                status_icon = {'locked': '🔒', 'active': '🟢', 'complete': '✅'}.get(status, '❓')
                layer_type_icon = {'road': '🛣️', 'radial': '🎯', 'unknown': '❓'}.get(layer_type, '❓')
                axes[1, 0].text(0.1, y_pos, f"{status_icon}{layer_type_icon} {layer_id}: {density:.1%}", fontsize=10)
                y_pos -= 0.05
            
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, '无层状态数据', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        axes[1, 0].set_title('📋 层状态信息', fontsize=12, fontweight='bold')
        
        # 5. 决策日志可视化
        if decision_data and 'active_layers' in decision_data:
            active_layers = decision_data['active_layers']
            
            if active_layers:
                layer = active_layers[0]  # 显示第一个激活层
                layer_id = layer.get('layer_id', 'unknown')
                total_slots = layer.get('total_slots', 0)
                used_slots = layer.get('used_slots', 0)
                free_slots = layer.get('free_slots', 0)
                dead_slots = layer.get('dead_slots', 0)
                density = layer.get('density', 0)
                
                axes[1, 1].text(0.1, 0.9, f"层ID: {layer_id}", fontsize=12, fontweight='bold')
                axes[1, 1].text(0.1, 0.8, f"总槽位: {total_slots}", fontsize=12)
                axes[1, 1].text(0.1, 0.7, f"已用: {used_slots}", fontsize=12)
                axes[1, 1].text(0.1, 0.6, f"空闲: {free_slots}", fontsize=12)
                axes[1, 1].text(0.1, 0.5, f"死槽: {dead_slots}", fontsize=12)
                axes[1, 1].text(0.1, 0.4, f"密度: {density:.1%}", fontsize=12)
                
                # 绘制密度进度条
                axes[1, 1].barh(0.2, density, height=0.1, color='green', alpha=0.7)
                axes[1, 1].text(0.5, 0.2, f'{density:.1%}', ha='center', va='center', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, '无激活层', ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, '无决策数据', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('🎯 决策日志', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        # 6. 建筑类型统计
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
                colors = ['#F6C344', '#FD7E14', '#8E44AD', '#22A6B3']
                
                # 中文标签映射
                label_map = {'residential': '住宅', 'commercial': '商业', 'industrial': '工业', 'public': '公共'}
                chinese_labels = [label_map.get(label, label) for label in labels]
                
                wedges, texts, autotexts = axes[1, 2].pie(values, labels=chinese_labels, colors=colors, 
                                                          autopct='%1.1f%%', startangle=90)
                axes[1, 2].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
                
                # 在饼图上添加数量标签
                for i, (wedge, value) in enumerate(zip(wedges, values)):
                    angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
                    x = 0.8 * np.cos(np.radians(angle))
                    y = 0.8 * np.sin(np.radians(angle))
                    axes[1, 2].text(x, y, f'{value}个', ha='center', va='center', fontweight='bold')
        else:
            axes[1, 2].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
        
        # 7. 特征评分可视化
        if building_data and 'buildings' in building_data:
            buildings = building_data['buildings']
            
            # 收集所有建筑的评分信息
            scores_data = {'com': [], 'res': [], 'ind': []}
            for building in buildings:
                if 'scores' in building:
                    scores = building['scores']
                    for score_type in ['com', 'res', 'ind']:
                        if score_type in scores:
                            scores_data[score_type].append(scores[score_type])
            
            # 绘制评分分布
            if any(scores_data.values()):
                score_labels = ['商业', '住宅', '工业']
                score_colors = ['#FD7E14', '#F6C344', '#8E44AD']
                
                for i, (score_type, scores) in enumerate(scores_data.items()):
                    if scores:
                        avg_score = np.mean(scores)
                        axes[2, 0].bar(score_labels[i], avg_score, color=score_colors[i], alpha=0.7)
                        axes[2, 0].text(i, avg_score + 0.01, f'{avg_score:.3f}', ha='center', va='bottom')
                
                axes[2, 0].set_title('📊 平均评分分布', fontsize=12, fontweight='bold')
                axes[2, 0].set_ylabel('平均评分')
            else:
                axes[2, 0].text(0.5, 0.5, '无评分数据', ha='center', va='center', transform=axes[2, 0].transAxes)
                axes[2, 0].set_title('📊 平均评分分布', fontsize=12, fontweight='bold')
        else:
            axes[2, 0].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('📊 平均评分分布', fontsize=12, fontweight='bold')
        
        # 8. 地价场统计
        if land_price_data and 'land_price_stats' in land_price_data:
            stats = land_price_data['land_price_stats']
            
            # 创建统计图表
            labels = ['最小值', '平均值', '最大值']
            values = [stats.get('min_price', 0), stats.get('avg_price', 0), stats.get('max_price', 0)]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = axes[2, 1].bar(labels, values, color=colors, alpha=0.7)
            axes[2, 1].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
            axes[2, 1].set_ylabel('地价值')
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom')
        else:
            axes[2, 1].text(0.5, 0.5, '无地价统计', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
        
        # 9. 演化阶段信息
        if land_price_data and 'evolution_stage' in land_price_data:
            stage = land_price_data['evolution_stage']
            
            # 显示演化阶段信息
            axes[2, 2].text(0.1, 0.8, f"演化阶段: {stage.get('name', '未知')}", fontsize=14, fontweight='bold')
            axes[2, 2].text(0.1, 0.6, f"Hub σ: {stage.get('hub_sigma', 0):.1f}", fontsize=12)
            axes[2, 2].text(0.1, 0.4, f"Road σ: {stage.get('road_sigma', 0):.1f}", fontsize=12)
            axes[2, 2].text(0.1, 0.2, f"当前月份: {month}", fontsize=12)
            
            axes[2, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
            axes[2, 2].set_xlim(0, 1)
            axes[2, 2].set_ylim(0, 1)
            axes[2, 2].axis('off')
        else:
            axes[2, 2].text(0.5, 0.5, '无演化数据', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
        
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
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        def animate(frame_idx):
            if frame_idx >= len(self.months):
                return []
            
            month = self.months[frame_idx]
            land_price_data, building_data, layer_data, decision_data = self._get_frame_data(month)
            
            # 清除所有子图
            for ax in axes.flat:
                ax.clear()
            
            # 设置总标题
            fig.suptitle(f'🏙️ 增强城市模拟系统 v3.2 - 第 {month} 个月', fontsize=18, fontweight='bold')
            
            # 重新绘制所有内容（简化版本，避免重复代码）
            # 这里可以调用_create_frame的逻辑，但为了简化，我们只显示基本信息
            
            # 1. 地价场
            if land_price_data and 'land_price_field' in land_price_data:
                land_price_field = np.array(land_price_data['land_price_field'])
                im1 = axes[0, 0].imshow(land_price_field, cmap='viridis', aspect='equal')
                axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
                axes[0, 0].plot(20, 55, 'ro', markersize=8, label='Hub 1')
                axes[0, 0].plot(90, 55, 'bo', markersize=8, label='Hub 2')
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, '无地价场数据', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
            
            # 2. 建筑分布
            if building_data and 'buildings' in building_data:
                buildings = building_data['buildings']
                residential = [b for b in buildings if b['type'] == 'residential']
                commercial = [b for b in buildings if b['type'] == 'commercial']
                industrial = [b for b in buildings if b['type'] == 'industrial']
                public = [b for b in buildings if b['type'] == 'public']
                
                if residential:
                    res_x = [b['position'][0] for b in residential]
                    res_y = [b['position'][1] for b in residential]
                    axes[0, 1].scatter(res_x, res_y, c='#F6C344', s=30, alpha=0.8, label=f'住宅 ({len(residential)})')
                
                if commercial:
                    com_x = [b['position'][0] for b in commercial]
                    com_y = [b['position'][1] for b in commercial]
                    axes[0, 1].scatter(com_x, com_y, c='#FD7E14', s=30, alpha=0.8, label=f'商业 ({len(commercial)})')
                
                if industrial:
                    ind_x = [b['position'][0] for b in industrial]
                    ind_y = [b['position'][1] for b in industrial]
                    axes[0, 1].scatter(ind_x, ind_y, c='#8E44AD', s=30, alpha=0.8, label=f'工业 ({len(industrial)})')
                
                if public:
                    pub_x = [b['position'][0] for b in public]
                    pub_y = [b['position'][1] for b in public]
                    axes[0, 1].scatter(pub_x, pub_y, c='#22A6B3', s=30, alpha=0.8, label=f'公共 ({len(public)})')
                
                axes[0, 1].plot(20, 55, 'ro', markersize=8, label='Hub 1')
                axes[0, 1].plot(90, 55, 'bo', markersize=8, label='Hub 2')
                axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
                axes[0, 1].legend()
                axes[0, 1].set_xlim(0, 110)
                axes[0, 1].set_ylim(0, 110)
            else:
                axes[0, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
            
            # 3. 政府骨架系统
            axes[0, 2].text(0.1, 0.9, "🏛️ 政府骨架系统", fontsize=14, fontweight='bold')
            axes[0, 2].text(0.1, 0.8, "走廊带: 主干道中心线", fontsize=12)
            axes[0, 2].text(0.1, 0.7, "Hub1: 商业客运核", fontsize=12)
            axes[0, 2].text(0.1, 0.6, "Hub2: 工业货运核", fontsize=12)
            axes[0, 2].text(0.1, 0.5, "分区约束: 政府规划", fontsize=12)
            axes[0, 2].text(0.1, 0.4, "配额管理: 季度动态", fontsize=12)
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].axis('off')
            
            # 其他子图显示基本信息
            if layer_data and 'layers' in layer_data:
                growth_phase = layer_data['layers'].get('growth_phase', 'unknown')
                phase_icon = {'road_corridor': '🛣️', 'radial_expansion': '🎯', 'unknown': '❓'}.get(growth_phase, '❓')
                axes[1, 0].text(0.5, 0.5, f'层状态信息\n月份: {month}\n阶段: {phase_icon} {growth_phase}', ha='center', va='center', transform=axes[1, 0].transAxes)
            else:
                axes[1, 0].text(0.5, 0.5, f'层状态信息\n月份: {month}', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('📋 层状态信息', fontsize=12, fontweight='bold')
            
            axes[1, 1].text(0.5, 0.5, f'决策日志\n月份: {month}', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('🎯 决策日志', fontsize=12, fontweight='bold')
            
            axes[1, 2].text(0.5, 0.5, f'建筑类型分布\n月份: {month}', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
            
            axes[2, 0].text(0.5, 0.5, f'平均评分分布\n月份: {month}', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('📊 平均评分分布', fontsize=12, fontweight='bold')
            
            axes[2, 1].text(0.5, 0.5, f'地价场统计\n月份: {month}', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('📊 地价场统计', fontsize=12, fontweight='bold')
            
            axes[2, 2].text(0.5, 0.5, f'地价场演化\n月份: {month}', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('🔄 地价场演化', fontsize=12, fontweight='bold')
            
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
            anim.save(f'{self.output_dir}/v3_2_evolution.gif', writer='pillow', fps=1)
            print(f"✅ GIF已保存到 {self.output_dir}/v3_2_evolution.gif")
        
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
    print("🎬 增强城市模拟系统 v3.2 可视化播放器")
    print("=" * 60)
    
    # 创建播放器
    player = V3_2EvolutionPlayback()
    
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
