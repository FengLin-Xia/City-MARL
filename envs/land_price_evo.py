# envs/land_price_evo.py
"""
地价演化系统

实现动态Hub激活和地价场变化，支持fade-in效果。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LandPriceEvo:
    """地价演化系统"""
    
    def __init__(self, cfg: Dict[str, Any], grid_shape: Tuple[int, int]):
        """
        初始化地价演化系统（整合原有高斯地价系统）
        
        Args:
            cfg: 配置字典
            grid_shape: 地价网格形状 (height, width)
        """
        self.cfg = cfg.get("land_price", {}).get("evolution", {})
        self.grid_shape = grid_shape
        self.active_hubs: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[int, np.ndarray] = {}
        
        # 初始化原有高斯地价系统
        self._init_legacy_system(cfg)
        
        logger.info(f"地价演化系统初始化: grid_shape={grid_shape}")
        logger.info(f"Hub日程配置: {self.cfg.get('hubs_schedule', [])}")
    
    def _init_legacy_system(self, cfg: Dict[str, Any]):
        """初始化原有高斯地价系统"""
        try:
            from logic.enhanced_sdf_system import GaussianLandPriceSystem
            
            # 创建原有地价系统
            self.legacy_system = GaussianLandPriceSystem(cfg)
            
            # 获取交通枢纽位置
            transport_hubs = cfg.get("city", {}).get("transport_hubs", [])
            map_size = cfg.get("city", {}).get("map_size", [200, 200])
            
            # 初始化系统
            self.legacy_system.initialize_system(transport_hubs, map_size)
            
            logger.info("原有高斯地价系统初始化成功")
            
        except Exception as e:
            logger.warning(f"原有高斯地价系统初始化失败: {e}")
            self.legacy_system = None
    
    def update_if_needed(self, t: int) -> np.ndarray:
        """
        按需更新地价场（整合原有系统）
        
        Args:
            t: 当前时间步（月份）
            
        Returns:
            更新后的地价场网格
        """
        evo = self.cfg
        if not evo.get("enabled", False):
            # 如果演化未启用，使用原有系统
            if self.legacy_system:
                return self.legacy_system.get_land_price_field()
            return self.cache.setdefault(t, self.cache.get(t-1, np.zeros(self.grid_shape, dtype=float)))
        
        # 检查并激活新的Hub
        self._activate_hubs(t)
        
        # 使用原有系统作为基础
        if self.legacy_system:
            legacy_lp = self.legacy_system.get_land_price_field()
            # 调整形状以匹配新系统的形状
            if legacy_lp.shape != self.grid_shape:
                # 如果形状不匹配，创建新的零矩阵
                base_lp = np.zeros(self.grid_shape, dtype=float)
                # 可以选择将原有数据复制到新形状中，或者直接使用零矩阵
                logger.warning(f"形状不匹配: legacy={legacy_lp.shape}, target={self.grid_shape}")
            else:
                base_lp = legacy_lp
        else:
            base_lp = np.zeros(self.grid_shape, dtype=float)
        
        # 应用动态Hub影响
        lp = base_lp.copy()
        
        # 应用所有激活的Hub影响
        for hub_id, hub_info in self.active_hubs.items():
            strength = hub_info.get("peak", 1.0)
            
            # 处理fade-in效果
            fade = hub_info.get("fade_in_months", 0)
            if fade and t - hub_info["activated_at"] < fade:
                k = (t - hub_info["activated_at"] + 1) / float(fade)
                strength *= max(0.0, min(1.0, k))
                logger.debug(f"Hub {hub_id} fade-in: k={k:.3f}, strength={strength:.3f}")
            
            # 添加Hub影响
            hub_effect = self._gaussian_blob(
                hub_info["x"], 
                hub_info["y"], 
                hub_info.get("sigma_m", 32), 
                strength
            )
            lp += hub_effect
            
            logger.debug(f"Hub {hub_id} 影响: peak={strength:.3f}, sigma={hub_info.get('sigma_m', 32)}")
        
        # 缓存结果
        self.cache[t] = lp
        self.current_land_price = lp  # 保存当前地价场
        
        if t % 5 == 0:  # 每5个月记录一次
            logger.info(f"Month {t}: 激活Hub数量={len(self.active_hubs)}, 地价场范围=[{lp.min():.3f}, {lp.max():.3f}]")
        
        return lp
    
    def _activate_hubs(self, t: int) -> None:
        """激活到期的Hub"""
        for hub in self.cfg.get("hubs_schedule", []):
            hub_id = hub["id"]
            activation_month = hub["activation_month"]
            
            if activation_month <= t and hub_id not in self.active_hubs:
                self.active_hubs[hub_id] = dict(hub) | {"activated_at": t}
                logger.info(f"Hub {hub_id} 在月份 {t} 激活 (计划月份: {activation_month})")
    
    def _gaussian_blob(self, cx: int, cy: int, sigma_m: float, peak: float) -> np.ndarray:
        """
        生成高斯分布的地价影响
        
        Args:
            cx: Hub中心x坐标
            cy: Hub中心y坐标
            sigma_m: 高斯分布标准差（米）
            peak: 峰值强度
            
        Returns:
            高斯分布的地价影响矩阵
        """
        H, W = self.grid_shape
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        
        # 计算距离平方
        dist2 = (x - cx) ** 2 + (y - cy) ** 2
        
        # 确保sigma为正数
        sigma2 = max(1.0, sigma_m) ** 2
        
        # 计算高斯分布
        gaussian = peak * np.exp(-0.5 * dist2 / sigma2)
        
        return gaussian
    
    def get_active_hubs(self) -> Dict[str, Dict[str, Any]]:
        """获取当前激活的Hub信息"""
        return self.active_hubs.copy()
    
    def is_hub_active(self, hub_id: str, current_month: int) -> bool:
        """
        检查指定Hub是否在当前月份激活
        
        Args:
            hub_id: Hub标识
            current_month: 当前月份
            
        Returns:
            是否激活
        """
        if hub_id not in self.active_hubs:
            return False
        
        hub_info = self.active_hubs[hub_id]
        activation_month = hub_info.get("activated_at", 0)
        
        return current_month >= activation_month
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("地价演化缓存已清空")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "cache_size": len(self.cache),
            "active_hubs": len(self.active_hubs),
            "cached_months": list(self.cache.keys())
        }
    
    def get_land_price(self, position: List[float]) -> float:
        """
        获取指定位置的地价（兼容原有接口）
        
        Args:
            position: 位置 [x, y]
            
        Returns:
            地价值
        """
        if not hasattr(self, 'current_land_price') or self.current_land_price is None:
            return 0.0
        
        x, y = position
        if 0 <= x < self.grid_shape[1] and 0 <= y < self.grid_shape[0]:
            return float(self.current_land_price[int(y), int(x)])
        return 0.0
    
    def get_land_price_field(self) -> np.ndarray:
        """
        获取地价场（兼容原有接口）
        
        Returns:
            地价场矩阵
        """
        if hasattr(self, 'current_land_price') and self.current_land_price is not None:
            return self.current_land_price
        return np.zeros(self.grid_shape, dtype=float)
