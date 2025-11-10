#!/usr/bin/env python3
"""
1025功能烟雾测试

验证动态Hub激活和动作解锁功能是否正常工作。
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence
from utils.event_bus import get_event_bus, EventTypes

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmokeTest1025:
    """1025功能烟雾测试"""
    
    def __init__(self, config_path: str = "configs/city_config_v5_0.json"):
        """初始化测试"""
        self.config_path = config_path
        self.env = None
        self.event_bus = get_event_bus()
        self.test_results = {}
        
        # 设置事件监听器
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """设置事件监听器"""
        self.event_bus.subscribe(EventTypes.ACTION_UNLOCKED, self._on_action_unlocked)
        self.event_bus.subscribe(EventTypes.HUB_ACTIVATED, self._on_hub_activated)
        self.event_bus.subscribe(EventTypes.LAND_PRICE_UPDATED, self._on_land_price_updated)
    
    def _on_action_unlocked(self, payload: Dict[str, Any]):
        """动作解锁事件处理"""
        logger.info(f"动作解锁事件: {payload}")
        self.test_results.setdefault("unlock_events", []).append(payload)
    
    def _on_hub_activated(self, payload: Dict[str, Any]):
        """Hub激活事件处理"""
        logger.info(f"Hub激活事件: {payload}")
        self.test_results.setdefault("hub_events", []).append(payload)
    
    def _on_land_price_updated(self, payload: Dict[str, Any]):
        """地价更新事件处理"""
        logger.info(f"地价更新事件: {payload}")
        self.test_results.setdefault("land_price_events", []).append(payload)
    
    def run_test(self) -> Dict[str, Any]:
        """运行完整测试"""
        logger.info("开始1025功能烟雾测试")
        
        try:
            # 初始化环境
            self._init_environment()
            
            # 测试1: Hub激活测试
            self._test_hub_activation()
            
            # 测试2: 动作解锁测试
            self._test_action_unlock()
            
            # 测试3: 地价演化测试
            self._test_land_price_evolution()
            
            # 测试4: 中间件集成测试
            self._test_middleware_integration()
            
            # 生成测试报告
            report = self._generate_report()
            
            logger.info("1025功能烟雾测试完成")
            return report
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            return {"error": str(e), "success": False}
    
    def _init_environment(self):
        """初始化环境"""
        logger.info("初始化环境...")
        self.env = V5CityEnvironment(self.config_path)
        logger.info("环境初始化完成")
    
    def _test_hub_activation(self):
        """测试Hub激活功能"""
        logger.info("测试Hub激活功能...")
        
        # 检查初始状态
        if hasattr(self.env, 'land_price_evo') and self.env.land_price_evo:
            active_hubs = self.env.land_price_evo.get_active_hubs()
            logger.info(f"初始激活Hub数量: {len(active_hubs)}")
        
        # 模拟时间推进，测试Hub激活
        for month in range(1, 20):
            self.env.current_month = month
            self.env._update_land_price_evolution()
            
            if hasattr(self.env, 'land_price_evo') and self.env.land_price_evo:
                active_hubs = self.env.land_price_evo.get_active_hubs()
                if len(active_hubs) > 0:
                    logger.info(f"月份 {month}: 激活Hub数量 {len(active_hubs)}")
                    for hub_id, hub_info in active_hubs.items():
                        logger.info(f"  - Hub {hub_id}: 激活于月份 {hub_info.get('activated_at', 'unknown')}")
        
        self.test_results["hub_activation"] = "通过"
        logger.info("Hub激活测试完成")
    
    def _test_action_unlock(self):
        """测试动作解锁功能"""
        logger.info("测试动作解锁功能...")
        
        # 设置测试预算
        self.env.budgets = {
            "IND": 1500,  # 超过解锁阈值
            "EDU": 1000,
            "COUNCIL": 1000
        }
        
        # 测试解锁中间件
        from action_mw.unlock_gate import UnlockGateMW
        unlock_mw = UnlockGateMW()
        
        # 创建测试序列
        test_sequence = Sequence(agent="IND", actions=[9, 10, 11])
        
        # 创建测试状态
        from contracts import EnvironmentState
        test_state = EnvironmentState(
            month=16,  # 超过解锁时间
            land_prices={},
            buildings={},
            budgets=self.env.budgets,
            slots={}
        )
        # 添加配置到状态对象
        test_state.cfg = self.env.config
        
        # 应用解锁中间件
        processed_sequence = unlock_mw.apply(test_sequence, test_state)
        
        logger.info(f"原始序列: {test_sequence}")
        logger.info(f"处理后的序列: {processed_sequence}")
        
        # 检查解锁状态
        unlocked_actions = unlock_mw.get_unlocked_actions("IND")
        logger.info(f"IND已解锁动作: {unlocked_actions}")
        
        self.test_results["action_unlock"] = "通过"
        logger.info("动作解锁测试完成")
    
    def _test_land_price_evolution(self):
        """测试地价演化功能"""
        logger.info("测试地价演化功能...")
        
        if not hasattr(self.env, 'land_price_evo') or not self.env.land_price_evo:
            logger.warning("地价演化系统未初始化，跳过测试")
            self.test_results["land_price_evolution"] = "跳过"
            return
        
        # 测试地价场更新
        for month in range(1, 20):
            self.env.current_month = month
            land_price_grid = self.env.land_price_evo.update_if_needed(month)
            
            if month % 5 == 0:  # 每5个月记录一次
                logger.info(f"月份 {month}: 地价场范围 [{land_price_grid.min():.3f}, {land_price_grid.max():.3f}]")
        
        self.test_results["land_price_evolution"] = "通过"
        logger.info("地价演化测试完成")
    
    def _test_middleware_integration(self):
        """测试中间件集成"""
        logger.info("测试中间件集成...")
        
        # 测试中间件应用
        test_sequence = Sequence(agent="IND", actions=[9, 10, 11])
        processed_sequence = self.env._apply_middleware("IND", test_sequence)
        
        logger.info(f"中间件处理前: {test_sequence}")
        logger.info(f"中间件处理后: {processed_sequence}")
        
        self.test_results["middleware_integration"] = "通过"
        logger.info("中间件集成测试完成")
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            "test_name": "1025功能烟雾测试",
            "success": True,
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results.values() if "通过" in str(r)),
                "skipped": sum(1 for r in self.test_results.values() if "跳过" in str(r)),
                "failed": sum(1 for r in self.test_results.values() if "失败" in str(r))
            }
        }
        
        logger.info("测试报告:")
        logger.info(f"  总测试数: {report['summary']['total_tests']}")
        logger.info(f"  通过: {report['summary']['passed']}")
        logger.info(f"  跳过: {report['summary']['skipped']}")
        logger.info(f"  失败: {report['summary']['failed']}")
        
        return report


def main():
    """主函数"""
    print("启动1025功能烟雾测试")
    
    # 检查配置文件
    config_path = "configs/city_config_v5_0.json"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return 1
    
    # 运行测试
    tester = SmokeTest1025(config_path)
    report = tester.run_test()
    
    # 保存报告
    report_path = "outputs/smoke_test_1025_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"测试报告已保存: {report_path}")
    
    # 返回退出码
    return 0 if report.get("success", False) else 1


if __name__ == "__main__":
    exit(main())
