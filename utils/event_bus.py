# utils/event_bus.py
"""
事件总线系统

提供事件发布-订阅机制，用于系统组件间的解耦通信。
"""

from typing import Dict, List, Callable, Any
import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventBus:
    """事件总线"""
    
    def __init__(self):
        """初始化事件总线"""
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        logger.info("事件总线初始化")
    
    def subscribe(self, event_name: str, callback: Callable) -> None:
        """
        订阅事件
        
        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        with self._lock:
            self._listeners[event_name].append(callback)
            logger.debug(f"订阅事件 {event_name}, 当前监听器数量: {len(self._listeners[event_name])}")
    
    def unsubscribe(self, event_name: str, callback: Callable) -> bool:
        """
        取消订阅事件
        
        Args:
            event_name: 事件名称
            callback: 回调函数
            
        Returns:
            是否成功取消订阅
        """
        with self._lock:
            if callback in self._listeners[event_name]:
                self._listeners[event_name].remove(callback)
                logger.debug(f"取消订阅事件 {event_name}, 剩余监听器数量: {len(self._listeners[event_name])}")
                return True
            return False
    
    def emit(self, event_name: str, payload: Any = None) -> None:
        """
        发布事件
        
        Args:
            event_name: 事件名称
            payload: 事件载荷
        """
        with self._lock:
            listeners = self._listeners[event_name].copy()
        
        if not listeners:
            logger.debug(f"事件 {event_name} 无监听器")
            return
        
        logger.debug(f"发布事件 {event_name}, 监听器数量: {len(listeners)}")
        
        # 异步调用所有监听器
        for callback in listeners:
            try:
                callback(payload)
            except Exception as e:
                logger.error(f"事件 {event_name} 监听器执行失败: {e}")
    
    def get_listener_count(self, event_name: str) -> int:
        """
        获取指定事件的监听器数量
        
        Args:
            event_name: 事件名称
            
        Returns:
            监听器数量
        """
        with self._lock:
            return len(self._listeners[event_name])
    
    def get_all_events(self) -> List[str]:
        """
        获取所有事件名称
        
        Returns:
            事件名称列表
        """
        with self._lock:
            return list(self._listeners.keys())
    
    def clear(self) -> None:
        """清空所有监听器"""
        with self._lock:
            self._listeners.clear()
        logger.info("事件总线已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取事件总线统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                "total_events": len(self._listeners),
                "total_listeners": sum(len(listeners) for listeners in self._listeners.values()),
                "events": {
                    event: len(listeners) 
                    for event, listeners in self._listeners.items()
                }
            }


# 全局事件总线实例
_global_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """
    获取全局事件总线实例
    
    Returns:
        全局事件总线实例
    """
    return _global_event_bus


def subscribe(event_name: str, callback: Callable) -> None:
    """
    订阅全局事件
    
    Args:
        event_name: 事件名称
        callback: 回调函数
    """
    _global_event_bus.subscribe(event_name, callback)


def emit(event_name: str, payload: Any = None) -> None:
    """
    发布全局事件
    
    Args:
        event_name: 事件名称
        payload: 事件载荷
    """
    _global_event_bus.emit(event_name, payload)


# 常用事件类型
class EventTypes:
    """事件类型常量"""
    
    # 动作解锁事件
    ACTION_UNLOCKED = "ActionUnlocked"
    
    # Hub激活事件
    HUB_ACTIVATED = "HubActivated"
    
    # 地价场更新事件
    LAND_PRICE_UPDATED = "LandPriceUpdated"
    
    # 智能体动作执行事件
    AGENT_ACTION_EXECUTED = "AgentActionExecuted"
    
    # 环境状态变化事件
    ENVIRONMENT_STATE_CHANGED = "EnvironmentStateChanged"
    
    # 预算变化事件
    BUDGET_CHANGED = "BudgetChanged"
    
    # 错误事件
    ERROR_OCCURRED = "ErrorOccurred"
    
    # 调试事件
    DEBUG_INFO = "DebugInfo"



