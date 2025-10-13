#!/usr/bin/env python3
"""
测试长期训练结果分析工具
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TestLongTermAnalyzer:
    def __init__(self, output_dir='test_long_term_output'):
        self.output_dir = Path(output_dir)
        self.daily_stats = []
        self.poi_evolution = []
        self.final_heatmap = None
        
    def load_data(self):
        """加载分析数据"""
        try:
            # 加载每日统计
            stats_file = self.output_dir / 'daily_stats.json'
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.daily_stats = json.load(f)
                print(f"✅ 加载每日统计: {len(self.daily_stats)}天")
            
            # 加载POI演化
            evolution_file = self.output_dir / 'poi_evolution.json'
            if evolution_file.exists():
                with open(evolution_file, 'r', encoding='utf-8') as f:
                    self.poi_evolution = json.load(f)
                print(f"✅ 加载POI演化: {len(self.poi_evolution)}天")
            
            # 加载最终热力图
            heat_file = self.output_dir / 'final_heatmap.npy'
            if heat_file.exists():
                self.final_heatmap = np.load(heat_file)
                print(f"✅ 加载最终热力图: {self.final_heatmap.shape}")
            
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def analyze_population_growth(self):
        """分析人口增长趋势"""
        if not self.daily_stats:
            return
        
        days = [stat['day'] for stat in self.daily_stats]
        populations = [stat['total_residents'] for stat in self.daily_stats]
        
        plt.figure(figsize=(15, 10))
        
        # 人口增长曲线
        plt.subplot(2, 3, 1)
        plt.plot(days, populations, 'b-', linewidth=2, marker='o')
        plt.title('人口增长趋势', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('居民数量')
        plt.grid(True, alpha=0.3)
        
        # 人口增长率
        plt.subplot(2, 3, 2)
        growth_rates = []
        for i in range(1, len(populations)):
            if populations[i-1] > 0:
                rate = (populations[i] - populations[i-1]) / populations[i-1] * 100
                growth_rates.append(rate)
            else:
                growth_rates.append(0)
        
        plt.plot(days[1:], growth_rates, 'g-', linewidth=2, marker='s')
        plt.title('人口增长率 (%)', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('增长率 (%)')
        plt.grid(True, alpha=0.3)
        
        # 人口密度分析
        plt.subplot(2, 3, 3)
        final_population = populations[-1]
        area = 256 * 256  # 网格面积
        density = final_population / area
        plt.bar(['人口密度'], [density], color='orange', alpha=0.7)
        plt.title(f'最终人口密度\n{final_population}人/{area}像素²', fontsize=12)
        plt.ylabel('人/像素²')
        
        # 增长阶段分析
        plt.subplot(2, 3, 4)
        early_pop = populations[5] if len(populations) > 5 else populations[0]
        mid_pop = populations[len(populations)//2]
        late_pop = populations[-1]
        
        stages = ['早期', '中期', '后期']
        stage_pops = [early_pop, mid_pop, late_pop]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        plt.bar(stages, stage_pops, color=colors, alpha=0.7)
        plt.title('不同阶段人口对比', fontsize=14, fontweight='bold')
        plt.ylabel('居民数量')
        
        # 每日新增人口
        plt.subplot(2, 3, 5)
        daily_new = []
        for i in range(1, len(populations)):
            daily_new.append(populations[i] - populations[i-1])
        
        plt.plot(days[1:], daily_new, 'r-', linewidth=2, marker='^')
        plt.title('每日新增人口', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('新增数量')
        plt.grid(True, alpha=0.3)
        
        # 累积人口
        plt.subplot(2, 3, 6)
        plt.plot(days, populations, 'purple', linewidth=3, marker='o')
        plt.fill_between(days, populations, alpha=0.3, color='purple')
        plt.title('累积人口增长', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('累积居民数')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'population_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_poi_evolution(self):
        """分析POI演化趋势"""
        if not self.poi_evolution:
            return
        
        days = [poi['day'] for poi in self.poi_evolution]
        public = [poi['public'] for poi in self.poi_evolution]
        residential = [poi['residential'] for poi in self.poi_evolution]
        retail = [poi['retail'] for poi in self.poi_evolution]
        
        plt.figure(figsize=(15, 10))
        
        # POI数量变化
        plt.subplot(2, 3, 1)
        plt.plot(days, public, 'b-', label='公共设施', linewidth=2, marker='o')
        plt.plot(days, residential, 'g-', label='住宅设施', linewidth=2, marker='s')
        plt.plot(days, retail, 'r-', label='零售设施', linewidth=2, marker='^')
        plt.title('POI数量演化', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('设施数量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 设施增长率
        plt.subplot(2, 3, 2)
        public_growth = [(public[i] - public[i-1]) for i in range(1, len(public))]
        residential_growth = [(residential[i] - residential[i-1]) for i in range(1, len(residential))]
        retail_growth = [(retail[i] - retail[i-1]) for i in range(1, len(retail))]
        
        plt.plot(days[1:], public_growth, 'b-', label='公共设施', linewidth=2, marker='o')
        plt.plot(days[1:], residential_growth, 'g-', label='住宅设施', linewidth=2, marker='s')
        plt.plot(days[1:], retail_growth, 'r-', label='零售设施', linewidth=2, marker='^')
        plt.title('设施增长率', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('新增数量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 设施比例变化
        plt.subplot(2, 3, 3)
        total_pois = [p + r + ret for p, r, ret in zip(public, residential, retail)]
        public_ratio = [p/t if t > 0 else 0 for p, t in zip(public, total_pois)]
        residential_ratio = [r/t if t > 0 else 0 for r, t in zip(residential, total_pois)]
        retail_ratio = [ret/t if t > 0 else 0 for ret, t in zip(retail, total_pois)]
        
        plt.plot(days, public_ratio, 'b-', label='公共设施', linewidth=2, marker='o')
        plt.plot(days, residential_ratio, 'g-', label='住宅设施', linewidth=2, marker='s')
        plt.plot(days, retail_ratio, 'r-', label='零售设施', linewidth=2, marker='^')
        plt.title('设施比例变化', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('比例')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最终设施分布
        plt.subplot(2, 3, 4)
        final_public = public[-1]
        final_residential = residential[-1]
        final_retail = retail[-1]
        
        labels = ['公共设施', '住宅设施', '零售设施']
        sizes = [final_public, final_residential, final_retail]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('最终设施分布', fontsize=14, fontweight='bold')
        
        # 设施密度分析
        plt.subplot(2, 3, 5)
        area = 256 * 256
        public_density = final_public / area
        residential_density = final_residential / area
        retail_density = final_retail / area
        
        densities = [public_density, residential_density, retail_density]
        plt.bar(labels, densities, color=colors, alpha=0.7)
        plt.title('设施密度 (个/像素²)', fontsize=14, fontweight='bold')
        plt.ylabel('密度')
        plt.xticks(rotation=45)
        
        # 发展效率分析
        plt.subplot(2, 3, 6)
        if len(self.daily_stats) > 0:
            final_population = self.daily_stats[-1]['total_residents']
            public_per_capita = final_public / final_population if final_population > 0 else 0
            residential_per_capita = final_residential / final_population if final_population > 0 else 0
            retail_per_capita = final_retail / final_population if final_population > 0 else 0
            
            per_capita = [public_per_capita, residential_per_capita, retail_per_capita]
            plt.bar(labels, per_capita, color=colors, alpha=0.7)
            plt.title('人均设施数量', fontsize=14, fontweight='bold')
            plt.ylabel('设施/人')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'poi_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_heatmap_evolution(self):
        """分析热力图演化"""
        if not self.daily_stats or self.final_heatmap is None:
            return
        
        days = [stat['day'] for stat in self.daily_stats]
        heat_sums = [stat['heat_sum'] for stat in self.daily_stats]
        heat_maxs = [stat['heat_max'] for stat in self.daily_stats]
        heat_means = [stat['heat_mean'] for stat in self.daily_stats]
        
        plt.figure(figsize=(15, 10))
        
        # 热力图统计变化
        plt.subplot(2, 3, 1)
        plt.plot(days, heat_sums, 'r-', linewidth=2, marker='o')
        plt.title('热力图总和变化', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('热力值')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(days, heat_maxs, 'orange', linewidth=2, marker='s')
        plt.title('热力图最大值变化', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('最大热力值')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(days, heat_means, 'purple', linewidth=2, marker='^')
        plt.title('热力图平均值变化', fontsize=14, fontweight='bold')
        plt.xlabel('天数')
        plt.ylabel('平均热力值')
        plt.grid(True, alpha=0.3)
        
        # 最终热力图可视化
        plt.subplot(2, 3, 4)
        plt.imshow(self.final_heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='热力值')
        plt.title('最终热力图', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 热力图分布
        plt.subplot(2, 3, 5)
        heat_flat = self.final_heatmap.flatten()
        heat_flat = heat_flat[heat_flat > 0]  # 只显示有热力的区域
        if len(heat_flat) > 0:
            plt.hist(heat_flat, bins=50, color='red', alpha=0.7, edgecolor='black')
            plt.title('热力值分布', fontsize=14, fontweight='bold')
            plt.xlabel('热力值')
            plt.ylabel('频次')
        
        # 热力集中度分析
        plt.subplot(2, 3, 6)
        if len(heat_flat) > 0:
            total_heat = np.sum(self.final_heatmap)
            top_10_percent = np.percentile(heat_flat, 90)
            high_heat_areas = self.final_heatmap[self.final_heatmap >= top_10_percent]
            concentration = np.sum(high_heat_areas) / total_heat if total_heat > 0 else 0
            
            plt.pie([concentration, 1-concentration], 
                   labels=['高热力区域', '其他区域'], 
                   colors=['red', 'lightgray'], 
                   autopct='%1.1f%%')
            plt.title('热力集中度', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成总结报告"""
        if not self.daily_stats:
            return
        
        final_stats = self.daily_stats[-1]
        
        print("\n" + "="*80)
        print("🏙️ 测试长期城市仿真分析报告")
        print("="*80)
        
        # 基础统计
        print(f"\n📊 基础统计:")
        print(f"   仿真天数: {final_stats['day']}")
        print(f"   最终居民数: {final_stats['total_residents']}")
        print(f"   公共设施: {final_stats['public_pois']}")
        print(f"   住宅设施: {final_stats['residential_pois']}")
        print(f"   零售设施: {final_stats['retail_pois']}")
        
        # 发展趋势
        if len(self.daily_stats) > 5:
            early_stats = self.daily_stats[5]
            late_stats = self.daily_stats[-1]
            
            print(f"\n📈 发展趋势 (第5天 → 第{final_stats['day']}天):")
            print(f"   居民增长: {early_stats['total_residents']} → {late_stats['total_residents']} (+{late_stats['total_residents'] - early_stats['total_residents']})")
            print(f"   公共设施: {early_stats['public_pois']} → {late_stats['public_pois']} (+{late_stats['public_pois'] - early_stats['public_pois']})")
            print(f"   住宅设施: {early_stats['residential_pois']} → {late_stats['residential_pois']} (+{late_stats['residential_pois'] - early_stats['residential_pois']})")
            print(f"   零售设施: {early_stats['retail_pois']} → {late_stats['retail_pois']} (+{late_stats['retail_pois'] - early_stats['retail_pois']})")
        
        # 热力图分析
        print(f"\n🔥 热力图分析:")
        print(f"   热力图总和: {final_stats['heat_sum']:.2f}")
        print(f"   热力图最大值: {final_stats['heat_max']:.2f}")
        print(f"   热力图平均值: {final_stats['heat_mean']:.2f}")
        
        # 城市发展评估
        print(f"\n🏆 城市发展评估:")
        
        # 人口密度评估
        area = 256 * 256
        density = final_stats['total_residents'] / area
        if density > 0.005:
            print("   🟢 人口密度: 适中")
        elif density > 0.002:
            print("   🟡 人口密度: 较低")
        else:
            print("   🔴 人口密度: 过低")
        
        # 设施配套评估
        total_pois = final_stats['public_pois'] + final_stats['residential_pois'] + final_stats['retail_pois']
        if total_pois > 10:
            print("   🟢 设施配套: 完善")
        elif total_pois > 5:
            print("   🟡 设施配套: 一般")
        else:
            print("   🔴 设施配套: 不足")
        
        # 热力分布评估
        if final_stats['heat_max'] > 1000:
            print("   🟢 热力分布: 高度集中")
        elif final_stats['heat_max'] > 100:
            print("   🟡 热力分布: 适中")
        else:
            print("   🔴 热力分布: 分散")
        
        print(f"\n📁 分析结果保存在: {self.output_dir}")
        print("="*80)
    
    def run_analysis(self):
        """运行完整分析"""
        print("🔍 开始测试长期训练结果分析...")
        
        if not self.load_data():
            print("❌ 数据加载失败，无法进行分析")
            return
        
        # 生成分析图表
        print("📊 生成人口增长分析...")
        self.analyze_population_growth()
        
        print("📊 生成POI演化分析...")
        self.analyze_poi_evolution()
        
        print("📊 生成热力图分析...")
        self.analyze_heatmap_evolution()
        
        # 生成总结报告
        print("📋 生成总结报告...")
        self.generate_summary_report()
        
        print("✅ 分析完成！")

if __name__ == "__main__":
    analyzer = TestLongTermAnalyzer()
    analyzer.run_analysis()



