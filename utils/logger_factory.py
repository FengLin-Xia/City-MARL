#!/usr/bin/env python3
"""
集中式日志工厂（配置驱动）

功能：
- 读取 config.logging，初始化全局日志
- 按模块设置日志级别
- 主题开关（topics.*）与采样（every_n_steps / sample_agents / sample_months）
- 提供 get_logger 与辅助判定函数
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


_CONFIG: Dict[str, Any] = {}
_INITIALIZED = False


_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _resolve_log_path(path: str, config_paths: Optional[Dict[str, Any]]) -> str:
    if not path:
        return "./logs/v5_0.log"
    # 支持 ${paths.logs_dir}
    if config_paths and "${paths.logs_dir}" in path:
        logs_dir = config_paths.get("logs_dir", "./logs")
        return path.replace("${paths.logs_dir}", logs_dir)
    return path


def init_logging(config: Dict[str, Any]) -> None:
    global _CONFIG, _INITIALIZED
    _CONFIG = config.get("logging", {}) or {}
    if not _CONFIG.get("enabled", True):
        _INITIALIZED = True
        return

    level = _LEVELS.get(str(_CONFIG.get("level", "INFO")).upper(), logging.INFO)
    fmt_plain = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt_plain)

    root = logging.getLogger()
    root.setLevel(level)

    # 清理旧 handler，避免重复
    while root.handlers:
        root.handlers.pop()

    sinks = _CONFIG.get("sinks", {})
    # console
    if sinks.get("console", True):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # file with rotation
    file_path = sinks.get("file", None)
    if file_path:
        # 尝试解析 ${paths.logs_dir}
        paths_cfg = config.get("paths", {})
        file_path = _resolve_log_path(file_path, paths_cfg)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        rotate_cfg = sinks.get("rotate", {"enabled": True, "max_bytes": 10 * 1024 * 1024, "backup_count": 3})
        if rotate_cfg.get("enabled", True):
            fh = RotatingFileHandler(file_path, maxBytes=int(rotate_cfg.get("max_bytes", 10 * 1024 * 1024)),
                                     backupCount=int(rotate_cfg.get("backup_count", 3)), encoding="utf-8")
        else:
            fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # 每个模块单独级别
    modules = _CONFIG.get("modules", {})
    for module_name, lvl in modules.items():
        logging.getLogger(module_name).setLevel(_LEVELS.get(str(lvl).upper(), level))

    _INITIALIZED = True


def get_logger(module_name: str) -> logging.Logger:
    if not _INITIALIZED:
        # 兜底初始化，避免空调用
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger(module_name)


def topic_enabled(topic: str) -> bool:
    topics = _CONFIG.get("topics", {})
    return bool(topics.get(topic, False))


def sampling_allows(agent: Optional[str], month: Optional[int], step: Optional[int]) -> bool:
    sampling = _CONFIG.get("sampling", {})
    every_n = int(sampling.get("every_n_steps", 1) or 1)
    if step is not None and every_n > 1 and step % every_n != 0:
        return False
    sample_agents = sampling.get("sample_agents", [])
    if sample_agents and agent is not None and agent not in sample_agents:
        return False
    sample_months = sampling.get("sample_months", [])
    if sample_months and month is not None and month not in sample_months:
        return False
    return True


def export_strict_mode() -> bool:
    return bool(_CONFIG.get("export", {}).get("strict_slot_positions", False))


def export_error_policy() -> str:
    return str(_CONFIG.get("export", {}).get("error_policy", "WARN")).upper()



