#!/usr/bin/env python3
"""
测试v3.1系统可视化
直接加载并显示一帧数据
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def test_v3_1_visualization():
    """测试v3.1可视化"""
    print("🧪 测试v3.1系统可视化...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 加载数据
    output_dir = "enhanced_simulation_v3_1_output"
    
    # 测试加载地价场数据
    try:
        land_price_file = f"{output_dir}/land_price_frame_month_23.json"
        with open(land_price_file, 'r', encoding='utf-8') as f:
            land_price_data = json.load(f)
        print(f"✅ 地价场数据加载成功: {land_price_file}")
        print(f"   月份: {land_price_data.get('month')}")
        print(f"   地价场形状: {len(land_price_data.get('land_price_field', []))} x {len(land_price_data.get('land_price_field', [[]])[0]) if land_price_data.get('land_price_field') else 0}")
    except Exception as e:
        print(f"❌ 地价场数据加载失败: {e}")
        return
    
    # 测试加载建筑数据
    try:
        building_file = f"{output_dir}/building_positions_month_23.json"
        with open(building_file, 'r', encoding='utf-8') as f:
            building_data = json.load(f)
        print(f"✅ 建筑数据加载成功: {building_file}")
        print(f"   建筑数量: {len(building_data.get('buildings', []))}")
        
        # 统计建筑类型
        buildings = building_data.get('buildings', [])
        building_types = {}
        for building in buildings:
            btype = building['type']
            building_types[btype] = building_types.get(btype, 0) + 1
        
        print(f"   建筑类型分布: {building_types}")
        
    except Exception as e:
        print(f"❌ 建筑数据加载失败: {e}")
        return
    
    # 测试加载层状态数据
    try:
        layer_file = f"{output_dir}/layer_state_month_23.json"
        with open(layer_file, 'r', encoding='utf-8') as f:
            layer_data = json.load(f)
        print(f"✅ 层状态数据加载成功: {layer_file}")
        print(f"   层数据: {layer_data.keys()}")
        
        if 'layers' in layer_data:
            layers = layer_data['layers']
            for building_type, type_layers in layers.items():
                print(f"   {building_type}: {len(type_layers)} 层")
                for i, layer in enumerate(type_layers):
                    print(f"     第{i}层: {layer['status']}, 密度: {layer['density']:.1%}")
        
    except Exception as e:
        print(f"❌ 层状态数据加载失败: {e}")
        return
    
    # 创建可视化
    print("\n🎨 创建可视化...")
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏙️ 增强城市模拟系统 v3.1 - 第 23 个月测试', fontsize=16, fontweight='bold')
    
    # 1. 地价场热力图
    if 'land_price_field' in land_price_data:
        land_price_field = np.array(land_price_data['land_price_field'])
        print(f"   地价场数据形状: {land_price_field.shape}")
        print(f"   地价场值范围: {land_price_field.min():.4f} - {land_price_field.max():.4f}")
        
        im1 = axes[0, 0].imshow(land_price_field, cmap='viridis', aspect='equal')
        axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X (像素)')
        axes[0, 0].set_ylabel('Y (像素)')
        plt.colorbar(im1, ax=axes[0, 0], label='地价值')
        
        # 添加交通枢纽标记
        axes[0, 0].plot(40, 128, 'ro', markersize=10, label='Hub 1')
        axes[0, 0].plot(216, 128, 'ro', markersize=10, label='Hub 2')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, '无地价场数据', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('🏔️ 高斯核地价场', fontsize=12, fontweight='bold')
    
    # 2. 建筑分布图
    if 'buildings' in building_data:
        buildings = building_data['buildings']
        
        # 分类建筑
        residential = [b for b in buildings if b['type'] == 'residential']
        commercial = [b for b in buildings if b['type'] == 'commercial']
        public = [b for b in buildings if b['type'] == 'public']
        
        print(f"   住宅建筑: {len(residential)} 个")
        print(f"   商业建筑: {len(commercial)} 个")
        print(f"   公共建筑: {len(public)} 个")
        
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
        axes[0, 1].plot(40, 128, 'ro', markersize=10, label='Hub 1')
        axes[0, 1].plot(216, 128, 'ro', markersize=10, label='Hub 2')
        
        axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('X (像素)')
        axes[0, 1].set_ylabel('Y (像素)')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 256)
        axes[0, 1].set_ylim(0, 256)
    else:
        axes[0, 1].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('🏗️ 建筑分布', fontsize=12, fontweight='bold')
    
    # 3. 建筑类型统计饼图
    if 'buildings' in building_data:
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
            
            wedges, texts, autotexts = axes[1, 0].pie(values, labels=chinese_labels, colors=colors, 
                                                      autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
            
            # 在饼图上添加数量标签
            for i, (wedge, value) in enumerate(zip(wedges, values)):
                angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
                x = 0.8 * np.cos(np.radians(angle))
                y = 0.8 * np.sin(np.radians(angle))
                axes[1, 0].text(x, y, f'{value}个', ha='center', va='center', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, '无建筑数据', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('🏘️ 建筑类型分布', fontsize=12, fontweight='bold')
    
    # 4. 层状态可视化
    if 'layers' in layer_data:
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
                axes[1, 1].barh(y_pos, density, height=0.1, color=color, alpha=0.7)
                axes[1, 1].text(0.5, y_pos, f'{status_text} P{i}: {placed}/{capacity}', 
                               ha='center', va='center', fontweight='bold')
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('🏢 商业建筑层状态', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('填充密度')
        else:
            axes[1, 1].text(0.5, 0.5, '无商业层数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('🏢 商业建筑层状态', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, '无层状态数据', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('🏢 商业建筑层状态', fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    print("✅ 可视化创建完成！")
    plt.show()

if __name__ == "__main__":
    test_v3_1_visualization()


