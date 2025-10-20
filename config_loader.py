"""
v5.0 配置加载器

支持路径引用解析和配置验证。
"""

import json
import os
from typing import Dict, Any, Optional
import re


class ConfigLoader:
    """v5.0配置加载器"""
    
    def __init__(self):
        self.resolved_config = {}
    
    def load_v5_config(self, path: str) -> Dict[str, Any]:
        """
        加载v5.0配置并解析路径引用
        
        Args:
            path: 配置文件路径
            
        Returns:
            解析后的配置字典
        """
        # 加载原始配置
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 解析路径引用
        self.resolved_config = self.resolve_paths(config)
        
        # 验证配置
        self.validate_config(self.resolved_config)
        
        return self.resolved_config
    
    def resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析路径引用，如 ${paths.slots_txt}
        
        Args:
            config: 原始配置
            
        Returns:
            解析后的配置
        """
        # 先处理所有非引用值，建立完整的配置树
        resolved = {}
        
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # 暂时保留引用，稍后处理
                resolved[key] = value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                resolved[key] = self.resolve_paths(value)
            elif isinstance(value, list):
                # 处理列表中的路径引用
                resolved[key] = [self._resolve_list_item(item, config) for item in value]
            else:
                resolved[key] = value
        
        # 现在处理所有路径引用
        resolved = self._resolve_all_references(resolved)
        
        return resolved
    
    def _resolve_reference(self, config: Dict[str, Any], ref_path: str) -> str:
        """
        解析单个路径引用
        
        Args:
            config: 配置字典
            ref_path: 引用路径，如 "paths.slots_txt"
            
        Returns:
            解析后的路径
        """
        parts = ref_path.split(".")
        current = config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Invalid reference path: {ref_path}")
        
        if not isinstance(current, str):
            raise ValueError(f"Referenced value must be string: {ref_path}")
        
        return current
    
    def _resolve_all_references(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析所有路径引用
        
        Args:
            config: 包含引用的配置
            
        Returns:
            解析后的配置
        """
        resolved = {}
        
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # 解析路径引用
                ref_path = value[2:-1]  # 去掉 ${ 和 }
                try:
                    resolved_value = self._resolve_reference(config, ref_path)
                    resolved[key] = resolved_value
                except ValueError:
                    # 如果引用失败，保留原始值
                    resolved[key] = value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                resolved[key] = self._resolve_all_references(value)
            elif isinstance(value, list):
                # 处理列表中的路径引用
                resolved[key] = [self._resolve_list_item_recursive(item, config) for item in value]
            else:
                resolved[key] = value
        
        return resolved
    
    def _resolve_list_item_recursive(self, item: Any, config: Dict[str, Any]) -> Any:
        """递归解析列表项中的路径引用"""
        if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
            ref_path = item[2:-1]
            try:
                return self._resolve_reference(config, ref_path)
            except ValueError:
                return item
        elif isinstance(item, dict):
            return self._resolve_all_references(item)
        else:
            return item
    
    def _resolve_list_item(self, item: Any, config: Dict[str, Any]) -> Any:
        """解析列表项中的路径引用"""
        if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
            ref_path = item[2:-1]
            return self._resolve_reference(config, ref_path)
        elif isinstance(item, dict):
            return self.resolve_paths(item)
        else:
            return item
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置完整性
        
        Args:
            config: 配置字典
            
        Returns:
            验证是否通过
        """
        # 验证必需字段
        required_fields = ["schema_version", "agents", "action_params"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
        
        # 验证智能体配置
        self._validate_agents_config(config["agents"])
        
        # 验证动作参数
        self._validate_action_params(config["action_params"])
        
        # 验证调度器配置
        if "scheduler" in config:
            self._validate_scheduler_config(config["scheduler"])
        
        return True
    
    def _validate_agents_config(self, agents_config: Dict[str, Any]):
        """验证智能体配置"""
        assert "order" in agents_config, "Missing agents.order"
        assert "defs" in agents_config, "Missing agents.defs"
        
        order = agents_config["order"]
        defs = agents_config["defs"]
        
        # 验证所有智能体都有定义
        for agent in order:
            assert agent in defs, f"Missing definition for agent: {agent}"
            agent_def = defs[agent]
            assert "action_ids" in agent_def, f"Missing action_ids for agent: {agent}"
    
    def _validate_action_params(self, action_params: Dict[str, Any]):
        """验证动作参数配置"""
        for action_id, params in action_params.items():
            assert "desc" in params, f"Missing desc for action {action_id}"
            assert "cost" in params, f"Missing cost for action {action_id}"
            assert "reward" in params, f"Missing reward for action {action_id}"
            assert "prestige" in params, f"Missing prestige for action {action_id}"
    
    def _validate_scheduler_config(self, scheduler_config: Dict[str, Any]):
        """验证调度器配置"""
        assert "name" in scheduler_config, "Missing scheduler.name"
        assert "params" in scheduler_config, "Missing scheduler.params"
        
        params = scheduler_config["params"]
        assert "phases" in params, "Missing scheduler.params.phases"
        
        phases = params["phases"]
        assert len(phases) > 0, "At least one phase must be defined"
        
        for phase in phases:
            assert "agents" in phase, "Missing agents in phase"
            assert "mode" in phase, "Missing mode in phase"
            assert phase["mode"] in ["concurrent", "sequential"], f"Invalid mode: {phase['mode']}"
    
    def get_action_params(self, action_id: int) -> Dict[str, Any]:
        """
        获取动作参数
        
        Args:
            action_id: 动作ID
            
        Returns:
            动作参数字典
        """
        action_params = self.resolved_config.get("action_params", {})
        return action_params.get(str(action_id), {})
    
    def get_agent_config(self, agent: str) -> Dict[str, Any]:
        """
        获取智能体配置
        
        Args:
            agent: 智能体名称
            
        Returns:
            智能体配置字典
        """
        agents_config = self.resolved_config.get("agents", {})
        defs = agents_config.get("defs", {})
        return defs.get(agent, {})
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取调度器配置"""
        return self.resolved_config.get("scheduler", {})
    
    def get_paths_config(self) -> Dict[str, str]:
        """获取路径配置"""
        return self.resolved_config.get("paths", {})
