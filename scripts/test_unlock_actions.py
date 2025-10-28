#!/usr/bin/env python3
"""
测试动作解锁功能

验证15月后IND智能体是否能解锁新动作9,10,11。
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


class UnlockActionTest:
    """动作解锁测试"""
    
    def __init__(self, config_path: str = "configs/city_config_v5_0.json"):
        """初始化测试"""
        self.config_path = config_path
        self.env = None
        self.event_bus = get_event_bus()
        self.unlock_events = []
        
        # 设置事件监听器
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """设置事件监听器"""
        self.event_bus.subscribe(EventTypes.ACTION_UNLOCKED, self._on_action_unlocked)
    
    def _on_action_unlocked(self, payload: Dict[str, Any]):
        """动作解锁事件处理"""
        logger.info(f"动作解锁事件: {payload}")
        self.unlock_events.append(payload)
    
    def run_test(self) -> Dict[str, Any]:
        """运行完整测试"""
        logger.info("开始动作解锁测试")
        
        try:
            # 初始化环境
            self._init_environment()
            
            # 模拟运行到15月
            self._simulate_to_month_15()
            
            # 测试动作解锁
            self._test_action_unlock()
            
            # 生成测试报告
            report = self._generate_report()
            
            logger.info("动作解锁测试完成")
            return report
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            return {"error": str(e), "success": False}
    
    def _init_environment(self):
        """初始化环境"""
        logger.info("初始化环境...")
        self.env = V5CityEnvironment(self.config_path)
        
        # 设置初始预算，确保满足解锁条件
        self.env.budgets = {
            "IND": 1500,  # 超过解锁阈值1000
            "EDU": 1000,
            "COUNCIL": 1000
        }
        
        logger.info("环境初始化完成")
    
    def _simulate_to_month_15(self):
        """模拟运行到15月"""
        logger.info("模拟运行到15月...")
        
        for month in range(1, 16):
            self.env.current_month = month
            
            # 更新地价演化系统
            if hasattr(self.env, '_update_land_price_evolution'):
                self.env._update_land_price_evolution()
            
            # 记录Hub激活状态
            if hasattr(self.env, 'land_price_evo') and self.env.land_price_evo:
                active_hubs = self.env.land_price_evo.get_active_hubs()
                if active_hubs:
                    logger.info(f"月份 {month}: 激活Hub数量 {len(active_hubs)}")
        
        logger.info("已运行到15月")
    
    def _test_action_unlock(self):
        """测试动作解锁"""
        logger.info("测试动作解锁...")
        
        # 直接测试解锁中间件
        from action_mw.unlock_gate import UnlockGateMW
        unlock_mw = UnlockGateMW()
        
        # 创建测试状态
        from contracts import EnvironmentState
        test_state = EnvironmentState(
            month=15,  # 超过解锁时间
            land_prices={},
            buildings={},
            budgets=self.env.budgets,
            slots={}
        )
        # 添加配置到状态对象
        test_state.cfg = self.env.config
        
        # 创建测试序列（包含新动作）
        test_sequence = Sequence(agent="IND", actions=[9, 10, 11])
        
        # 应用解锁中间件
        processed_sequence = unlock_mw.apply(test_sequence, test_state)
        
        logger.info(f"原始序列: {test_sequence}")
        logger.info(f"解锁中间件处理后: {processed_sequence}")
        
        # 检查解锁状态
        unlocked_actions = unlock_mw.get_unlocked_actions("IND")
        logger.info(f"IND已解锁动作: {unlocked_actions}")
        
        # 测试获取候选动作
        self._test_get_candidates()
    
    def _test_get_candidates(self):
        """测试获取候选动作"""
        logger.info("测试获取候选动作...")
        
        try:
            # 获取IND的候选动作
            candidates = self.env.get_action_candidates("IND")
            logger.info(f"IND候选动作数量: {len(candidates)}")
            
            # 检查是否包含新动作
            new_action_ids = set()
            for candidate in candidates:
                if hasattr(candidate, 'action_id'):
                    new_action_ids.add(candidate.action_id)
                elif hasattr(candidate, 'atype'):
                    new_action_ids.add(candidate.atype)
            
            logger.info(f"候选动作ID: {new_action_ids}")
            
            # 检查是否包含动作9,10,11
            target_actions = {9, 10, 11}
            found_actions = new_action_ids.intersection(target_actions)
            
            if found_actions:
                logger.info(f"找到新动作: {found_actions}")
                return True
            else:
                logger.warning(f"未找到新动作，目标动作: {target_actions}")
                return False
                
        except Exception as e:
            logger.error(f"获取候选动作失败: {e}")
            return False
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            "test_name": "动作解锁测试",
            "success": True,
            "current_month": self.env.current_month,
            "budgets": self.env.budgets,
            "unlock_events": self.unlock_events,
            "summary": {
                "total_events": len(self.unlock_events),
                "month_15_reached": self.env.current_month >= 15,
                "budget_sufficient": self.env.budgets.get("IND", 0) >= 1000
            }
        }
        
        logger.info("测试报告:")
        logger.info(f"  当前月份: {report['current_month']}")
        logger.info(f"  IND预算: {report['budgets']['IND']}")
        logger.info(f"  解锁事件数量: {report['summary']['total_events']}")
        logger.info(f"  达到15月: {report['summary']['month_15_reached']}")
        logger.info(f"  预算充足: {report['summary']['budget_sufficient']}")
        
        return report


def main():
    """主函数"""
    print("开始动作解锁测试")
    
    # 检查配置文件
    config_path = "configs/city_config_v5_0.json"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return 1
    
    # 运行测试
    tester = UnlockActionTest(config_path)
    report = tester.run_test()
    
    # 保存报告
    report_path = "outputs/unlock_action_test_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"测试报告已保存: {report_path}")
    
    # 返回退出码
    return 0 if report.get("success", False) else 1


if __name__ == "__main__":
    exit(main())
