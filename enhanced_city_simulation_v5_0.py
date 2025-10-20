#!/usr/bin/env python3
"""
v5.0 增强城市模拟系统主程序
支持命令行接口和高级功能
"""

import argparse
import json
import time
import os
import sys
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

from integration.v5_0 import (
    V5IntegrationSystem, 
    run_complete_session,
    V5TrainingPipeline,
    V5ExportPipeline
)
from contracts import StepLog, EnvironmentState


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_complete_mode(args) -> Dict[str, Any]:
    """运行完整模式（训练+导出）"""
    print("=" * 60)
    print("运行v5.0完整模式 (Complete Mode)")
    print("=" * 60)
    
    # 使用便捷函数
    result = run_complete_session(
        config_path=args.config,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    return result


def run_training_mode(args) -> Dict[str, Any]:
    """运行训练模式"""
    print("=" * 60)
    print("运行v5.0训练模式 (Training Mode)")
    print("=" * 60)
    
    # 创建集成系统
    system = V5IntegrationSystem(args.config)
    
    # 运行训练
    result = system.run_training_only(
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    return result


def run_export_mode(args) -> Dict[str, Any]:
    """运行导出模式"""
    print("=" * 60)
    print("运行v5.0导出模式 (Export Mode)")
    print("=" * 60)
    
    # 检查输入数据
    if not args.input_data:
        raise ValueError("导出模式需要指定输入数据路径 (--input_data)")
    
    # 加载数据
    step_logs, env_states = load_training_data(args.input_data)
    
    # 创建导出管道
    export_pipeline = V5ExportPipeline(args.config)
    
    # 运行导出
    result = export_pipeline.run_export(
        step_logs=step_logs,
        env_states=env_states,
        output_dir=args.output_dir
    )
    
    return result


def run_evaluation_mode(args) -> Dict[str, Any]:
    """运行评估模式"""
    print("=" * 60)
    print("运行v5.0评估模式 (Evaluation Mode)")
    print("=" * 60)
    
    if not args.model_path:
        raise ValueError("评估模式需要指定模型路径 (--model_path)")
    
    # 创建集成系统
    system = V5IntegrationSystem(args.config)
    
    # 运行评估（这里需要实现评估逻辑）
    # TODO: 实现评估模式
    result = {
        'success': True,
        'mode': 'evaluation',
        'model_path': args.model_path,
        'message': '评估模式待实现'
    }
    
    return result


def load_training_data(data_path: str) -> tuple[List[StepLog], List[EnvironmentState]]:
    """加载训练数据"""
    # 这里需要实现数据加载逻辑
    # 简化实现：返回空数据
    print(f"加载训练数据: {data_path}")
    return [], []


def compare_with_v4_1(args) -> Dict[str, Any]:
    """对比v4.1和v5.0"""
    print("=" * 60)
    print("对比v4.1和v5.0")
    print("=" * 60)
    
    # 运行v5.0
    print("运行v5.0...")
    v5_result = run_complete_session(
        config_path=args.config,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    # 运行v4.1（需要v4.1主程序）
    print("运行v4.1...")
    v4_result = run_v4_1_comparison(args)
    
    # 对比结果
    comparison = {
        'v5_result': v5_result,
        'v4_result': v4_result,
        'improvements': {
            'success_rate': v5_result.get('success', False) - v4_result.get('success', False),
            'performance': v5_result.get('performance', {}) if v5_result.get('success') else {}
        }
    }
    
    return comparison


def run_v4_1_comparison(args) -> Dict[str, Any]:
    """运行v4.1对比"""
    # 这里需要调用v4.1主程序
    # 简化实现
    return {
        'success': True,
        'mode': 'v4_1',
        'message': 'v4.1对比待实现'
    }


def generate_performance_report(result: Dict[str, Any]) -> Dict[str, Any]:
    """生成性能报告"""
    if 'pipeline_summary' in result:
        perf_summary = result['pipeline_summary'].get('performance_summary', {})
        return {
            'total_time': perf_summary.get('total_time', 0),
            'throughput': perf_summary.get('throughput', 0),
            'memory_usage': perf_summary.get('memory_usage', {}),
            'recommendations': perf_summary.get('recommendations', [])
        }
    return {}


def save_results(result: Dict[str, Any], args) -> str:
    """保存结果"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存JSON结果
    results_file = os.path.join(results_dir, f"v5_0_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"结果已保存: {results_file}")
    return results_file


def print_summary(result: Dict[str, Any]):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("运行结果摘要")
    print("=" * 60)
    
    print(f"成功: {result.get('success', False)}")
    print(f"模式: {result.get('mode', 'unknown')}")
    
    if 'summary' in result:
        summary = result['summary']
        print(f"训练: {summary.get('training', {})}")
        print(f"导出: {summary.get('export', {})}")
        print(f"性能: {summary.get('performance', {})}")
    
    if 'comparison' in result:
        comparison = result['comparison']
        print(f"对比结果: {comparison.get('improvements', {})}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='v5.0 增强城市模拟系统')
    
    # 基础参数
    parser.add_argument('--config', type=str, default='configs/city_config_v5_0.json',
                       help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=2,
                       help='训练轮数')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    
    # 模式选择
    parser.add_argument('--mode', choices=['complete', 'training', 'export', 'eval'], default='complete',
                       help='运行模式')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--model_path', type=str, default=None,
                       help='预训练模型路径')
    
    # 高级功能
    parser.add_argument('--compare_v4', action='store_true',
                       help='对比v4.1和v5.0')
    parser.add_argument('--performance_monitor', action='store_true',
                       help='启用性能监控')
    parser.add_argument('--pipeline_config', type=str,
                       help='自定义管道配置文件')
    
    # 导出选项
    parser.add_argument('--input_data', type=str,
                       help='输入数据路径（导出模式必需）')
    parser.add_argument('--export_format', choices=['txt', 'tables', 'all'], default='all',
                       help='导出格式')
    parser.add_argument('--export_compatible', action='store_true',
                       help='导出v4.1兼容格式')
    
    # 其他选项
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='保存结果到文件')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == 'export' and not args.input_data:
        parser.error("导出模式需要指定 --input_data")
    
    if args.eval_only and not args.model_path:
        parser.error("评估模式需要指定 --model_path")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"v5.0 增强城市模拟系统")
    print(f"配置文件: {args.config}")
    print(f"运行模式: {args.mode}")
    print(f"训练轮数: {args.episodes}")
    print(f"输出目录: {args.output_dir}")
    
    start_time = time.time()
    
    try:
        # 根据模式执行
        if args.mode == 'complete':
            result = run_complete_mode(args)
        elif args.mode == 'training':
            result = run_training_mode(args)
        elif args.mode == 'export':
            result = run_export_mode(args)
        elif args.mode == 'eval' or args.eval_only:
            result = run_evaluation_mode(args)
        else:
            raise ValueError(f"不支持的运行模式: {args.mode}")
        
        # 可选功能
        if args.compare_v4:
            comparison = compare_with_v4_1(args)
            result['comparison'] = comparison
        
        if args.performance_monitor:
            result['performance_report'] = generate_performance_report(result)
        
        # 保存结果
        if args.save_results:
            save_results(result, args)
        
        # 打印摘要
        print_summary(result)
        
        # 打印运行时间
        elapsed_time = time.time() - start_time
        print(f"\n总运行时间: {elapsed_time:.2f} 秒")
        
        return result
        
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    main()
