#!/usr/bin/env python3
"""
增强的训练记录系统 - 记录详细的PPO训练指标
"""

import json
import os
import datetime
from typing import Dict, List, Any
import numpy as np

class EnhancedTrainingLogger:
    """增强的训练记录器，记录详细的PPO训练指标"""
    
    def __init__(self, save_dir: str, experiment_name: str = "ppo_training"):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化记录
        self.training_log = {
            "experiment_info": {
                "name": experiment_name,
                "timestamp": self.timestamp,
                "start_time": datetime.datetime.now().isoformat()
            },
            "episodes": [],
            "training_updates": [],
            "final_summary": {}
        }
        
        # 当前episode记录
        self.current_episode = None
        self.current_update = None
        
    def start_episode(self, episode_id: int):
        """开始记录新的episode"""
        self.current_episode = {
            "episode_id": episode_id,
            "start_time": datetime.datetime.now().isoformat(),
            "steps": [],
            "episode_metrics": {
                "total_return": 0.0,
                "edu_return": 0.0,
                "ind_return": 0.0,
                "episode_length": 0,
                "final_kl_divergence": 0.0,
                "final_clip_fraction": 0.0,
                "final_policy_loss": 0.0,
                "final_value_loss": 0.0
            }
        }
    
    def record_step(self, agent: str, month: int, reward: float, 
                   selected_slots: List[str], action_score: float,
                   available_actions: int, candidate_slots: int):
        """记录episode中的一个step"""
        if self.current_episode is None:
            return
            
        step_record = {
            "agent": agent,
            "month": month,
            "reward": reward,
            "selected_slots": selected_slots,
            "action_score": action_score,
            "available_actions": available_actions,
            "candidate_slots": candidate_slots,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.current_episode["steps"].append(step_record)
        
        # 更新episode指标
        self.current_episode["episode_metrics"]["total_return"] += reward
        if agent == "EDU":
            self.current_episode["episode_metrics"]["edu_return"] += reward
        elif agent == "IND":
            self.current_episode["episode_metrics"]["ind_return"] += reward
        
        self.current_episode["episode_metrics"]["episode_length"] += 1
    
    def finish_episode(self, episode_metrics: Dict[str, Any] = None):
        """完成当前episode的记录"""
        if self.current_episode is None:
            return
            
        self.current_episode["end_time"] = datetime.datetime.now().isoformat()
        
        # 添加额外的episode指标
        if episode_metrics:
            self.current_episode["episode_metrics"].update(episode_metrics)
        
        self.training_log["episodes"].append(self.current_episode)
        self.current_episode = None
    
    def start_training_update(self, update_id: int, total_experiences: int):
        """开始记录新的训练更新"""
        self.current_update = {
            "update_id": update_id,
            "total_experiences": total_experiences,
            "start_time": datetime.datetime.now().isoformat(),
            "epochs": [],
            "update_summary": {
                "avg_policy_loss": 0.0,
                "avg_value_loss": 0.0,
                "avg_kl_divergence": 0.0,
                "avg_clip_fraction": 0.0,
                "avg_entropy": 0.0,
                "final_advantages_mean": 0.0,
                "final_advantages_std": 0.0
            }
        }
    
    def record_epoch(self, epoch_id: int, epoch_metrics: Dict[str, Any]):
        """记录训练更新中的一个epoch"""
        if self.current_update is None:
            return
            
        epoch_record = {
            "epoch_id": epoch_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": epoch_metrics.copy()
        }
        
        self.current_update["epochs"].append(epoch_record)
    
    def finish_training_update(self, update_summary: Dict[str, Any] = None):
        """完成当前训练更新的记录"""
        if self.current_update is None:
            return
            
        self.current_update["end_time"] = datetime.datetime.now().isoformat()
        
        # 计算epoch平均值
        if self.current_update["epochs"]:
            epoch_metrics = [epoch["metrics"] for epoch in self.current_update["epochs"]]
            
            # 计算平均值
            self.current_update["update_summary"]["avg_policy_loss"] = np.mean([m.get("policy_loss", 0) for m in epoch_metrics])
            self.current_update["update_summary"]["avg_value_loss"] = np.mean([m.get("value_loss", 0) for m in epoch_metrics])
            self.current_update["update_summary"]["avg_kl_divergence"] = np.mean([m.get("kl_divergence", 0) for m in epoch_metrics])
            self.current_update["update_summary"]["avg_clip_fraction"] = np.mean([m.get("clip_fraction", 0) for m in epoch_metrics])
            self.current_update["update_summary"]["avg_entropy"] = np.mean([m.get("entropy", 0) for m in epoch_metrics])
            
            # 使用最后一个epoch的advantages统计
            last_epoch = epoch_metrics[-1]
            self.current_update["update_summary"]["final_advantages_mean"] = last_epoch.get("advantages_mean", 0)
            self.current_update["update_summary"]["final_advantages_std"] = last_epoch.get("advantages_std", 0)
        
        # 添加额外的更新摘要
        if update_summary:
            self.current_update["update_summary"].update(update_summary)
        
        self.training_log["training_updates"].append(self.current_update)
        self.current_update = None
    
    def set_final_summary(self, summary: Dict[str, Any]):
        """设置最终训练摘要"""
        self.training_log["final_summary"] = summary
        self.training_log["experiment_info"]["end_time"] = datetime.datetime.now().isoformat()
    
    def save_log(self):
        """保存训练日志到文件"""
        log_path = os.path.join(self.save_dir, f"{self.experiment_name}_detailed_log_{self.timestamp}.json")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
        
        print(f"详细训练日志已保存: {log_path}")
        return log_path
    
    def save_summary_csv(self):
        """保存训练摘要为CSV格式"""
        import pandas as pd
        
        # Episode摘要
        episode_summary = []
        for episode in self.training_log["episodes"]:
            episode_summary.append({
                "episode_id": episode["episode_id"],
                "total_return": episode["episode_metrics"]["total_return"],
                "edu_return": episode["episode_metrics"]["edu_return"],
                "ind_return": episode["episode_metrics"]["ind_return"],
                "episode_length": episode["episode_metrics"]["episode_length"],
                "final_kl_divergence": episode["episode_metrics"].get("final_kl_divergence", 0),
                "final_clip_fraction": episode["episode_metrics"].get("final_clip_fraction", 0)
            })
        
        # 训练更新摘要
        update_summary = []
        for update in self.training_log["training_updates"]:
            update_summary.append({
                "update_id": update["update_id"],
                "avg_policy_loss": update["update_summary"]["avg_policy_loss"],
                "avg_value_loss": update["update_summary"]["avg_value_loss"],
                "avg_kl_divergence": update["update_summary"]["avg_kl_divergence"],
                "avg_clip_fraction": update["update_summary"]["avg_clip_fraction"],
                "avg_entropy": update["update_summary"]["avg_entropy"],
                "final_advantages_mean": update["update_summary"]["final_advantages_mean"],
                "final_advantages_std": update["update_summary"]["final_advantages_std"]
            })
        
        # 保存CSV文件
        csv_dir = os.path.join(self.save_dir, "csv_summaries")
        os.makedirs(csv_dir, exist_ok=True)
        
        if episode_summary:
            episode_df = pd.DataFrame(episode_summary)
            episode_csv_path = os.path.join(csv_dir, f"{self.experiment_name}_episodes_{self.timestamp}.csv")
            episode_df.to_csv(episode_csv_path, index=False)
            print(f"Episode摘要CSV已保存: {episode_csv_path}")
        
        if update_summary:
            update_df = pd.DataFrame(update_summary)
            update_csv_path = os.path.join(csv_dir, f"{self.experiment_name}_updates_{self.timestamp}.csv")
            update_df.to_csv(update_csv_path, index=False)
            print(f"训练更新摘要CSV已保存: {update_csv_path}")
    
    def print_training_summary(self):
        """打印训练摘要"""
        print("\n" + "="*60)
        print("训练摘要")
        print("="*60)
        
        if self.training_log["episodes"]:
            episode_returns = [ep["episode_metrics"]["total_return"] for ep in self.training_log["episodes"]]
            print(f"Episode数量: {len(episode_returns)}")
            print(f"Episode回报范围: [{min(episode_returns):.3f}, {max(episode_returns):.3f}]")
            print(f"平均Episode回报: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
        
        if self.training_log["training_updates"]:
            kl_divergences = [up["update_summary"]["avg_kl_divergence"] for up in self.training_log["training_updates"]]
            clip_fractions = [up["update_summary"]["avg_clip_fraction"] for up in self.training_log["training_updates"]]
            
            print(f"训练更新数量: {len(kl_divergences)}")
            print(f"最终KL散度: {kl_divergences[-1]:.6f}")
            print(f"最终Clip Fraction: {clip_fractions[-1]:.3f}")
        
        print("="*60)

# 全局训练记录器实例
_training_logger = None

def get_training_logger() -> EnhancedTrainingLogger:
    """获取全局训练记录器实例"""
    return _training_logger

def initialize_training_logger(save_dir: str, experiment_name: str = "ppo_training") -> EnhancedTrainingLogger:
    """初始化全局训练记录器"""
    global _training_logger
    _training_logger = EnhancedTrainingLogger(save_dir, experiment_name)
    return _training_logger

