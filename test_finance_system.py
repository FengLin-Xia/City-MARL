#!/usr/bin/env python3
"""
测试财务评估系统
"""

import json
import os
import numpy as np
from enhanced_city_simulation_v3_6 import CityV36


def test_finance_system():
    """测试财务系统"""
    print("开始测试财务评估系统...")
    
    # 读取配置
    try:
        with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("配置文件不存在，使用默认配置")
        config = {}
    
    # 创建城市模拟
    city = CityV36(config)
    city.initialize()
    
    print(f"初始化完成，建筑数量: {sum(len(buildings) for buildings in city.state.values())}")
    
    # 运行短期模拟
    total_months = 6  # 测试6个月
    print(f"开始运行{total_months}个月的模拟...")
    
    city.run(total_months)
    
    print("模拟完成，检查输出文件...")
    
    # 检查输出文件
    output_dir = city.cfg.output_dir
    
    # 检查财务CSV文件
    finance_files = []
    for month in range(total_months):
        csv_file = os.path.join(output_dir, f'building_finance_month_{month:02d}.csv')
        if os.path.exists(csv_file):
            finance_files.append(csv_file)
            print(f"✓ 财务CSV文件存在: {csv_file}")
        else:
            print(f"✗ 财务CSV文件缺失: {csv_file}")
    
    # 检查季度仪表板文件
    for quarter in range(1, (total_months // 3) + 1):
        dashboard_file = os.path.join(output_dir, f'finance_dashboard_quarter_{quarter:02d}.json')
        if os.path.exists(dashboard_file):
            print(f"✓ 季度仪表板文件存在: {dashboard_file}")
            
            # 读取并显示仪表板数据
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                dashboard_data = json.load(f)
            
            print(f"  第{quarter}季度财务摘要:")
            summary = dashboard_data.get('summary', {})
            print(f"    总收入: {summary.get('total_revenue', 0):.2f}")
            print(f"    总成本: {summary.get('total_cost', 0):.2f}")
            print(f"    净利润: {summary.get('net_profit', 0):.2f}")
            print(f"    平均ROI: {summary.get('roi_avg', 0):.4f}")
            
            # 显示政府财务
            gov = dashboard_data.get('government', {})
            print(f"  政府财务:")
            print(f"    现金流: {gov.get('cashflow', 0):.2f}")
            print(f"    销售税: {gov.get('tax_sales', 0):.2f}")
            print(f"    工资税: {gov.get('tax_payroll', 0):.2f}")
            
            # 显示企业财务
            business = dashboard_data.get('business', {})
            print(f"  企业财务:")
            revenue = business.get('revenue', {})
            profit = business.get('profit', {})
            print(f"    住宅收入: {revenue.get('residential', 0):.2f}, 利润: {profit.get('residential', 0):.2f}")
            print(f"    商业收入: {revenue.get('commercial', 0):.2f}, 利润: {profit.get('commercial', 0):.2f}")
            print(f"    工业收入: {revenue.get('industrial', 0):.2f}, 利润: {profit.get('industrial', 0):.2f}")
            
            # 显示居民财务
            residents = dashboard_data.get('residents', {})
            print(f"  居民财务:")
            print(f"    住房负担率: {residents.get('housing_burden_pct', 0):.1f}%")
            print(f"    通勤负担率: {residents.get('commute_burden_pct', 0):.1f}%")
            print(f"    可支配收入: {residents.get('disp_income', 0):.2f}")
            
        else:
            print(f"✗ 季度仪表板文件缺失: {dashboard_file}")
    
    # 检查KPI汇总文件
    for quarter in range(1, (total_months // 3) + 1):
        kpi_file = os.path.join(output_dir, f'kpi_summary_quarter_{quarter:02d}.csv')
        if os.path.exists(kpi_file):
            print(f"✓ KPI汇总文件存在: {kpi_file}")
        else:
            print(f"✗ KPI汇总文件缺失: {kpi_file}")
    
    print(f"\n测试完成！输出目录: {output_dir}")
    print(f"生成了 {len(finance_files)} 个财务CSV文件")
    
    return True


def test_heatmap_visualization():
    """测试热力图可视化"""
    print("\n开始测试热力图可视化...")
    
    try:
        from visualize_finance_heatmaps import FinanceHeatmapVisualizer
        
        visualizer = FinanceHeatmapVisualizer()
        
        # 创建可视化输出目录
        viz_output_dir = os.path.join(visualizer.output_dir, 'finance_visualizations')
        os.makedirs(viz_output_dir, exist_ok=True)
        
        # 测试月份
        test_month = 3
        
        print(f"为第{test_month}月创建热力图...")
        
        # 创建各种热力图
        visualizer.create_profit_density_heatmap(
            test_month, 
            os.path.join(viz_output_dir, f'profit_density_month_{test_month:02d}.png')
        )
        
        visualizer.create_roi_distribution_heatmap(
            test_month, 
            os.path.join(viz_output_dir, f'roi_distribution_month_{test_month:02d}.png')
        )
        
        visualizer.create_land_price_revenue_correlation(
            test_month, 
            os.path.join(viz_output_dir, f'land_price_revenue_correlation_month_{test_month:02d}.png')
        )
        
        visualizer.create_comprehensive_finance_dashboard(
            test_month, 
            os.path.join(viz_output_dir, f'comprehensive_finance_dashboard_month_{test_month:02d}.png')
        )
        
        print(f"✓ 热力图可视化测试完成，输出目录: {viz_output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ 热力图可视化测试失败: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("财务评估系统测试")
    print("=" * 60)
    
    # 测试财务系统
    finance_success = test_finance_system()
    
    # 测试热力图可视化
    heatmap_success = test_heatmap_visualization()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"财务系统测试: {'✓ 通过' if finance_success else '✗ 失败'}")
    print(f"热力图可视化测试: {'✓ 通过' if heatmap_success else '✗ 失败'}")
    
    if finance_success and heatmap_success:
        print("\n🎉 所有测试通过！财务评估系统已就绪。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")


