"""
v5.0 PPO训练器

基于契约对象和配置的训练系统。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import datetime
import json
from collections import deque

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts import ActionCandidate, Sequence, StepLog, EnvironmentState
from config_loader import ConfigLoader
from envs.v5_0.city_env import V5CityEnvironment
from solvers.v5_0.rl_selector import V5RLSelector
from utils.logger_factory import get_logger, topic_enabled


class V5PPOTrainer:
    """v5.0 PPO训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化PPO训练器
        
        Args:
            config_path: v5.0配置文件路径
        """
        # 加载配置
        self.loader = ConfigLoader()
        self.config = self.loader.load_v5_config(config_path)
        
        # 初始化日志
        self.logger = get_logger("trainer")
        
        # 获取RL配置
        self.rl_config = self.config.get("mappo", {})
        self.ppo_config = self.rl_config.get("ppo", {})
        
        # PPO超参数（从 mappo.ppo 读取）
        self.gamma = self.ppo_config.get("gamma", 0.99)
        self.gae_lambda = self.ppo_config.get("gae_lambda", 0.95)
        self.clip_eps = self.ppo_config.get("clip_eps", 0.2)
        self.lr = self.ppo_config.get("lr", 3e-4)
        self.value_loss_coef = self.ppo_config.get("value_coef", 0.5)
        self.entropy_coef = self.ppo_config.get("entropy_coef", 0.01)
        self.max_grad_norm = self.ppo_config.get("max_grad_norm", 0.5)
        
        # Reward归一化配置（从 constraints.reward_scaling 读取）
        reward_scaling = self.config.get("constraints", {}).get("reward_scaling", {})
        self.reward_scaling_enabled = reward_scaling.get("enabled", False)
        self.reward_scale = reward_scaling.get("reward_scale", 3000.0)
        self.reward_clip = reward_scaling.get("reward_clip", 1.0)
        self.reward_target_range = reward_scaling.get("target_range", [-1.0, 1.0])
        
        # 训练参数
        rollout_cfg = self.rl_config.get("rollout", {})
        self.rollout_horizon = rollout_cfg.get("horizon", 20)
        self.minibatch_size = rollout_cfg.get("minibatch_size", 32)
        self.updates_per_iter = rollout_cfg.get("updates_per_iter", 8)
        self.max_updates = rollout_cfg.get("max_updates", 10)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 更新计数器
        self.current_update = 0
        
        # 温度退火参数（从 mappo.exploration 读取）
        exploration_cfg = self.rl_config.get("exploration", {})
        self.initial_temperature = exploration_cfg.get("temperature", 3.0)
        self.final_temperature = exploration_cfg.get("anneal_to", 1.0)
        self.anneal_steps = exploration_cfg.get("anneal_steps", 300000)
        
        # 初始化环境
        self.env = V5CityEnvironment(config_path)
        
        # 初始化RL选择器
        self.selector = V5RLSelector(self.config)
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # 当前温度（用于退火）
        self.current_temperature = self.initial_temperature
        
        # 历史记录
        self.training_history = []
        self.episode_rewards = {agent: [] for agent in self.config.get("agents", {}).get("order", [])}
        
        # 优化器
        self._setup_optimizers()
    
    def _compute_current_temperature(self) -> float:
        """
        计算当前温度（线性退火）
        
        Returns:
            当前温度值
        """
        progress = min(1.0, self.total_steps / self.anneal_steps)
        current_temp = self.initial_temperature * (1 - progress) + self.final_temperature * progress
        return current_temp
    
    def _update_temperature(self):
        """更新当前温度并同步到配置"""
        self.current_temperature = self._compute_current_temperature()
        # 同步到 multi_action 和 exploration 配置
        if "multi_action" in self.config:
            self.config["multi_action"]["temperature"] = self.current_temperature
        self.config["mappo"]["exploration"]["temperature"] = self.current_temperature
    
    def _setup_optimizers(self):
        """设置优化器"""
        self.optimizers = {}
        agents = self.config.get("agents", {}).get("order", [])
        multi_enabled = self.config.get("multi_action", {}).get("enabled", False)
        
        for agent in agents:
            # 根据模式选择actor网络
            if multi_enabled:
                actor_params = self.selector.actor_networks_multi[agent].parameters()
            else:
                actor_params = self.selector.actor_networks[agent].parameters()
            
            self.optimizers[agent] = {
                'actor': optim.Adam(actor_params, lr=self.lr),
                'critic': optim.Adam(self.selector.critic_networks[agent].parameters(), lr=self.lr)
            }
    
    def collect_experience(self, num_steps: int) -> List[Dict]:
        """
        收集经验数据
        
        Args:
            num_steps: 收集步数
            
        Returns:
            经验列表
        """
        all_experiences = []
        steps_collected = 0
        episode_counter = 0  # 追踪episode编号
        
        while steps_collected < num_steps:
            # 重置环境
            state = self.env.reset()
            episode_counter += 1  # 每个episode递增
            
            episode_experiences = []
            done = False
            
            step_count = 0
            max_steps = 1000  # 防止无限循环
            
            while not done and steps_collected < num_steps and step_count < max_steps:
                # 获取当前phase的所有智能体
                phase_agents = self.env.get_phase_agents()
                execution_mode = self.env.get_phase_execution_mode()
                
                if topic_enabled("experience_collection"):
                    self.logger.info(f"[EXP_COLLECT] 步骤 {step_count}: phase_agents={phase_agents}, mode={execution_mode}")
                
                # 为每个智能体获取动作候选并用策略选择序列（动态更新候选集）
                phase_sequences = {}
                phase_candidates = {}
                sel_dict = {}  # 保存sel用于后续构建经验
                
                for agent in phase_agents:
                    # 获取动作候选（考虑已占用槽位）
                    # 检查是否启用多动作模式
                    multi_enabled = self.config.get("multi_action", {}).get("enabled", False)
                    if multi_enabled:
                        candidates, cand_idx = self.env.get_action_candidates_with_index(agent)
                        phase_candidates[agent] = (candidates, cand_idx)
                        
                        if candidates and cand_idx:
                            max_actions = self.config.get("multi_action", {}).get("max_actions_per_step", 3)
                            sel = self.selector.select_action_multi(agent, candidates, cand_idx, state, max_k=max_actions, greedy=False)
                            if sel is not None:
                                # 🔍 追踪日志：记录保存到sel_dict时的logprob（前50步）
                                if step_count < 50:
                                    trace_id_sel = sel.get('trace_id', None)
                                    logprob_sel = sel.get('logprob', 0.0)
                                    self.logger.warning(
                                        f"[SEL_SAVE_TO_DICT] agent={agent} episode={episode_counter} step={step_count} trace_id={trace_id_sel} "
                                        f"sel['logprob']={logprob_sel:.6f} "
                                        f"保存到sel_dict[{agent}]"
                                    )
                                phase_sequences[agent] = sel['sequence']
                                sel_dict[agent] = sel  # 保存sel用于后续构建经验
                            else:
                                phase_sequences[agent] = None
                                sel_dict[agent] = None
                        else:
                            phase_sequences[agent] = None
                            sel_dict[agent] = None
                    else:
                        candidates = self.env.get_action_candidates(agent)
                        phase_candidates[agent] = candidates
                        
                        if candidates:
                            sel = self.selector.select_action(agent, candidates, state, greedy=False)
                            if sel is not None:
                                phase_sequences[agent] = sel['sequence']
                                sel_dict[agent] = sel  # 保存sel用于后续构建经验
                            else:
                                phase_sequences[agent] = None
                                sel_dict[agent] = None
                        else:
                            phase_sequences[agent] = None
                            sel_dict[agent] = None
                
                # 执行phase
                next_state, phase_rewards, done, info = self.env.step_phase(phase_agents, phase_sequences)
                
                # 🔍 追踪日志：记录step_phase后的sel logprob（前50步）
                if step_count < 50:
                    for agent in phase_agents:
                        sel_after_step = sel_dict.get(agent)
                        if sel_after_step:
                            trace_id_after = sel_after_step.get('trace_id', None)
                            logprob_after = sel_after_step.get('logprob', 0.0)
                            self.logger.warning(
                                f"[SEL_AFTER_STEP] agent={agent} episode={episode_counter} step={step_count} trace_id={trace_id_after} "
                                f"sel['logprob']={logprob_after:.6f} "
                                f"step_phase后检查"
                            )
                
                # 创建经验记录（包含 logprob/value/动作ID 等，并附带 StepLog 与 next_state 用于导出）
                for agent in phase_agents:
                    # 直接使用保存的sel，而不是重新调用
                    sel = sel_dict.get(agent)
                    
                    # 🔍 追踪日志：记录存储经验前的sel logprob（前50步）
                    if step_count < 50:
                        if sel:
                            trace_id_before_store = sel.get('trace_id', None)
                            logprob_before_store = sel.get('logprob', 0.0)
                            self.logger.warning(
                                f"[SEL_BEFORE_STORE] agent={agent} episode={episode_counter} step={step_count} trace_id={trace_id_before_store} "
                                f"sel['logprob']={logprob_before_store:.6f} "
                                f"准备存储到experience"
                            )
                    obs_vec = self.env.get_observation(agent)
                    next_obs_vec = self.env.get_observation(agent)
                    # 匹配该agent的step_log（来自phase_logs）
                    agent_log = None
                    if info.get('phase_logs'):
                        for lg in info['phase_logs']:
                            if getattr(lg, 'agent', None) == agent:
                                agent_log = lg
                                break
                    # 检查是否是多动作模式
                    multi_enabled = self.config.get("multi_action", {}).get("enabled", False)
                    
                    if multi_enabled and isinstance(phase_candidates.get(agent), tuple):
                        # 多动作模式：存储详细信息
                        # 只处理有效的sequence，如果sel为None或sequence为空，跳过该经验
                        if sel and 'sequence' in sel and sel['sequence'] and sel['sequence'].actions:
                            actions_detail = []
                            for atomic_action in sel['sequence'].actions:
                                actions_detail.append({
                                    'point_idx': atomic_action.point,
                                    'type_idx': atomic_action.atype,  # 类型索引（ratio=0修复需要）
                                    'action_id': atomic_action.meta.get('action_id', -1),
                                    'available_types': atomic_action.meta.get('available_types', []),  # 该点的类型列表（ratio=0修复需要）
                                })
                            
                            # 🔍 调试日志：确认logprob从sel中读取（前50步）
                            old_logprob_from_sel = sel['logprob'] if sel else 0.0
                            # 获取sequence中的actions数量
                            seq_actions_count = len(sel['sequence'].actions) if sel and 'sequence' in sel and sel['sequence'] else 0
                            if step_count < 50:
                                # 验证logprob是否合理
                                num_actions = len(actions_detail)
                                expected_min = num_actions * -5.5 if num_actions > 0 else -20.0
                                expected_max = num_actions * -3.5 if num_actions > 0 else -5.0
                                is_abnormal = (num_actions > 0 and 
                                             (old_logprob_from_sel < expected_min or old_logprob_from_sel > expected_max))
                                
                                # 检查num_actions是否匹配
                                actions_count_mismatch = (seq_actions_count > 0 and num_actions != seq_actions_count)
                                
                                # 获取trace_id用于匹配
                                trace_id_from_sel = sel.get('trace_id', None) if sel else None
                                
                                log_level = "error" if is_abnormal or actions_count_mismatch else "warning"
                                log_msg = (
                                    f"[LOGPROB_LOAD] agent={agent} episode={episode_counter} step={step_count} "
                                    f"trace_id={trace_id_from_sel} "
                                    f"sel['logprob']={old_logprob_from_sel:.6f} "
                                    f"seq_actions_count={seq_actions_count} "
                                    f"actions_detail_count={num_actions} "
                                    f"stored_to_experience['logprob']={old_logprob_from_sel:.6f} "
                                    f"expected_range=[{expected_min:.1f}, {expected_max:.1f}]"
                                )
                                
                                if actions_count_mismatch:
                                    log_msg += f" ⚠️ ACTIONS_COUNT_MISMATCH: seq有{seq_actions_count}个动作，但actions_detail只有{num_actions}个！"
                                
                                if is_abnormal:
                                    log_msg += f" ⚠️ LOGPROB_ABNORMAL: logprob异常！可能是累积不完整"
                                
                                if log_level == "error":
                                    self.logger.error(log_msg)
                                else:
                                    self.logger.warning(log_msg)
                            
                            # 修复：将mask tensor转换为numpy数组以节省内存
                            initial_point_mask = None
                            point_masks_per_action = []
                            if sel.get('initial_point_mask') is not None:
                                initial_mask = sel['initial_point_mask']
                                if isinstance(initial_mask, torch.Tensor):
                                    initial_point_mask = initial_mask.numpy().astype(bool)
                                else:
                                    initial_point_mask = initial_mask
                            
                            if sel.get('point_masks_per_action'):
                                for mask in sel['point_masks_per_action']:
                                    if isinstance(mask, torch.Tensor):
                                        point_masks_per_action.append(mask.numpy().astype(bool))
                                    else:
                                        point_masks_per_action.append(mask)
                            
                            experience = {
                                'agent': agent,
                                'obs': obs_vec,
                                'actions_detail': actions_detail,  # 详细动作列表
                                'num_actions': len(actions_detail),
                                'logprob': sel['logprob'],  # 保留：用于GAE
                                'value': sel['value'] if sel else 0.0,
                                'reward': phase_rewards.get(agent, 0.0),
                                'next_obs': next_obs_vec,
                                'done': done,
                                'step_log': agent_log,
                                'next_state': next_state,
                                # 追踪字段：来自收集阶段
                                'trace_id': sel.get('trace_id', None),
                                'month': sel.get('month', None),
                                # 修复：存储mask信息（numpy数组格式，节省内存）
                                'initial_point_mask': initial_point_mask,
                                'point_masks_per_action': point_masks_per_action,
                                'cand_idx_points_count': sel.get('cand_idx_points_count', None),
                            }
                            episode_experiences.append(experience)
                            all_experiences.append(experience)
                        # 如果sel为None或sequence为空，跳过该经验（不创建经验，不降级）
                    else:
                        # 单动作模式：保持原有结构
                        action_id = -1
                        if sel:
                            if 'action_id' in sel:
                                action_id = sel['action_id']  # 单动作模式
                            elif 'sequence' in sel and sel['sequence'] and sel['sequence'].actions:
                                # 兼容处理：从AtomicAction的meta中获取action_id
                                first_action = sel['sequence'].actions[0]
                                action_id = first_action.meta.get('action_id', -1)
                        
                        experience = {
                            'agent': agent,
                            'obs': obs_vec,
                            'action_id': action_id,
                            'logprob': sel['logprob'] if sel else 0.0,
                            'value': sel['value'] if sel else 0.0,
                            'reward': phase_rewards.get(agent, 0.0),
                            'next_obs': next_obs_vec,
                            'done': done,
                            'step_log': agent_log,
                            'next_state': next_state,
                        }
                        episode_experiences.append(experience)
                        all_experiences.append(experience)
                
                # 更新状态
                state = next_state
                steps_collected += 1
                step_count += 1
        
        print(f"收集了 {len(all_experiences)} 步经验")
        return all_experiences
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        计算GAE优势估计
        
        Args:
            rewards: 奖励列表
            values: 价值估计列表
            next_value: 下一个状态的价值
            dones: 是否结束标志
            
        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []
        
        # 计算GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def check_network_initial_output(self, agent: str):
        """检查网络初始输出是否正常"""
        network = self.selector.actor_networks_multi[agent]
        network.eval()
        
        # 创建虚拟输入（正常范围的特征）
        dummy_feat = torch.randn(1, self.selector.hidden_size).to(self.device) * 0.1  # 小范围随机值
        
        with torch.no_grad():
            # 检查point输出
            point_logits = network.forward_point(dummy_feat)
            point_logits_min = point_logits.min().item()
            point_logits_max = point_logits.max().item()
            point_logits_std = point_logits.std().item()
            
            # 检查type输出
            type_logits = network.forward_type(dummy_feat, torch.tensor([0], device=self.device))
            type_logits_min = type_logits.min().item()
            type_logits_max = type_logits.max().item()
            type_logits_std = type_logits.std().item()
            
            self.logger.warning(
                f"[NETWORK_INIT_CHECK] agent={agent} "
                f"point_logits: min={point_logits_min:.2f} max={point_logits_max:.2f} std={point_logits_std:.2f} "
                f"type_logits: min={type_logits_min:.2f} max={type_logits_max:.2f} std={type_logits_std:.2f}"
            )
            
            # 检查是否异常
            if abs(point_logits_max) > 100 or abs(point_logits_min) > 100 or point_logits_std > 50:
                self.logger.error(
                    f"[NETWORK_INIT_ABNORMAL] agent={agent} 网络初始输出异常！可能需要重新初始化"
                )
        
        network.train()
    
    def train_step(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        执行一步训练
        
        Args:
            experiences: 经验数据
            
        Returns:
            训练统计信息
        """
        # 按智能体分组经验
        agent_experiences = {}
        for exp in experiences:
            agent = exp['agent']
            if agent not in agent_experiences:
                agent_experiences[agent] = []
            agent_experiences[agent].append(exp)
        
        # 为每个智能体训练（标准 PPO 近似）
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        # 检测是否是多动作模式
        multi_enabled = self.config.get("multi_action", {}).get("enabled", False)
        
        for agent, agent_exps in agent_experiences.items():
            if not agent_exps:
                if topic_enabled("training_step"):
                    self.logger.info(f"[TRAIN_STEP] 智能体 {agent} 没有经验数据")
                continue
            
            if topic_enabled("training_step"):
                self.logger.info(f"[TRAIN_STEP] 训练智能体 {agent}: {len(agent_exps)} 个经验")
            
            # 按模式提取数据并训练
            # 多动作模式：直接使用多动作训练，不降级到单动作模式
            if multi_enabled:
                # 多动作模式：使用新的训练逻辑
                stats = self._train_step_multi(agent, agent_exps)
                total_loss += stats.get('total_loss', 0.0)
                total_actor_loss += stats.get('actor_loss', 0.0)
                total_critic_loss += stats.get('critic_loss', 0.0)
                total_entropy_loss += stats.get('entropy_loss', 0.0)
            else:
                # 单动作模式：使用原有逻辑
                stats = self._train_step_single(agent, agent_exps)
                total_loss += stats.get('total_loss', 0.0)
                total_actor_loss += stats.get('actor_loss', 0.0)
                total_critic_loss += stats.get('critic_loss', 0.0)
                total_entropy_loss += stats.get('entropy_loss', 0.0)
        
        # 🔍 检查5：在第一次训练步骤时检查网络初始输出和优化器
        if self.training_step == 0:
            # 检查网络初始输出
            for agent in self.selector.actor_networks_multi.keys():
                self.check_network_initial_output(agent)
            
            # 检查优化器学习率
            for agent in self.optimizers.keys():
                actor_lr = self.optimizers[agent]['actor'].param_groups[0]['lr']
                critic_lr = self.optimizers[agent]['critic'].param_groups[0]['lr']
                
                self.logger.warning(
                    f"[OPTIMIZER_CHECK] agent={agent} update={self.current_update} "
                    f"actor_lr={actor_lr} critic_lr={critic_lr}"
                )
                
                # 检查学习率是否异常
                if actor_lr > 1.0 or critic_lr > 1.0:
                    self.logger.error(
                        f"[OPTIMIZER_ABNORMAL] agent={agent} 学习率异常大！可能导致训练不稳定"
                    )
        
        # 更新训练步数
        self.training_step += 1
        
        # 计算平均指标
        num_agents = len(agent_experiences) if agent_experiences else 1
        
        return {
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            'actor_loss': total_actor_loss / num_agents,
            'critic_loss': total_critic_loss / num_agents,
            'entropy_loss': total_entropy_loss / num_agents,
            'training_step': self.training_step,
            'temperature': self.current_temperature,
            'total_steps': self.total_steps,
            'current_update': self.current_update,
            'num_agents': num_agents,
            'total_experiences': len(experiences)
        }
    
    def _train_step_single(self, agent: str, agent_exps: List[Dict]) -> Dict[str, float]:
        """单动作模式的训练逻辑（原有逻辑）"""
        # 提取张量数据
        obs = torch.FloatTensor([exp['obs'] for exp in agent_exps])
        actions = torch.LongTensor([max(0, exp['action_id']) for exp in agent_exps])
        rewards_raw = torch.FloatTensor([exp['reward'] for exp in agent_exps])
        dones = torch.FloatTensor([1.0 if exp['done'] else 0.0 for exp in agent_exps])
        values_old = torch.FloatTensor([exp['value'] for exp in agent_exps])
        logprobs_old = torch.FloatTensor([exp['logprob'] for exp in agent_exps])
        
        # 修复：对reward进行归一化（仅用于训练，不影响StepLog.reward_terms）
        if self.reward_scaling_enabled:
            # 使用配置的reward_scale归一化
            rewards = rewards_raw / self.reward_scale
            # Clip到target_range
            rewards = torch.clamp(rewards, self.reward_target_range[0], self.reward_target_range[1])
        else:
            # 如果没有启用，使用经验统计归一化（防止reward过大）
            reward_mean = rewards_raw.mean().item()
            reward_std = rewards_raw.std().item() + 1e-8
            if abs(reward_mean) > 1000 or reward_std > 1000:
                # reward值过大，使用clip到合理范围
                rewards = torch.clamp(rewards_raw / 3000.0, -1.0, 1.0)
                if self.current_update < 2:
                    self.logger.warning(
                        f"[REWARD_NORM] agent={agent} reward_raw: mean={reward_mean:.2f} std={reward_std:.2f}, "
                        f"使用clip归一化到[-1, 1]"
                    )
            else:
                rewards = rewards_raw
        
        # 引导值：用最后一个样本的 next_obs 估一个 next_value（简化）
        with torch.no_grad():
            last_next = torch.FloatTensor(agent_exps[-1]['next_obs']).unsqueeze(0)
            next_value = self.selector.critic_networks[agent](last_next).squeeze().item()
        
        # 计算GAE（使用归一化后的rewards）
        advantages, returns = self.compute_gae(rewards.tolist(), values_old.tolist(), next_value, dones.tolist())
        
        # 转换为张量
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练网络
        agent_kl_divergences = []
        agent_clip_fractions = []
        agent_entropy_values = []
        agent_ratio_values = []
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        
        for epoch in range(self.updates_per_iter):
            # 检查是否达到最大更新次数
            if self.current_update >= self.max_updates:
                print(f"达到最大更新次数: {self.max_updates}")
                break
            
            # 随机采样批次
            idx = torch.randperm(len(agent_exps))[:min(self.minibatch_size, len(agent_exps))]
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_adv = advantages[idx]
            batch_ret = returns[idx]
            batch_old_logp = logprobs_old[idx]
            
            # 新分布与 value
            batch_obs_t = batch_obs
            logits = self.selector.actor_networks[agent](batch_obs_t)
            # 掩码（按agent允许动作）
            allow = torch.zeros((logits.size(0), logits.size(1)))
            allowed_ids = self.selector._agent_allowed_actions(agent)
            if allowed_ids:
                allow[:, allowed_ids] = 1.0
            masked_logits = logits + (allow + 1e-45).log()  # 局部近似mask
            logp_all = torch.log_softmax(masked_logits, dim=-1)
            new_logp = logp_all.gather(1, batch_actions.view(-1,1)).squeeze(1)
            
            values_pred = self.selector.critic_networks[agent](batch_obs_t).squeeze(1)
            
            # PPO目标
            ratio = (new_logp - batch_old_logp).exp()
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算critic loss
            critic_loss = F.mse_loss(values_pred, batch_ret)
            
            # 🔍 检查returns和value_pred值范围（用于调试）
            if epoch == 0 and self.current_update < 2:
                ret_min = batch_ret.min().item()
                ret_max = batch_ret.max().item()
                ret_mean = batch_ret.mean().item()
                ret_std = batch_ret.std().item()
                val_min = values_pred.min().item()
                val_max = values_pred.max().item()
                val_mean = values_pred.mean().item()
                val_std = values_pred.std().item()
                self.logger.warning(
                    f"[RETURNS_CHECK] agent={agent} update={self.current_update} epoch={epoch} "
                    f"returns: min={ret_min:.2f} max={ret_max:.2f} mean={ret_mean:.2f} std={ret_std:.2f}, "
                    f"values_pred: min={val_min:.2f} max={val_max:.2f} mean={val_mean:.2f} std={val_std:.2f}"
                )
                if abs(ret_max) > 100 or abs(ret_min) > 100 or ret_std > 50:
                    self.logger.error(
                        f"[RETURNS_ABNORMAL] agent={agent} returns异常！可能是reward归一化未生效"
                    )
            
            entropy = -(torch.softmax(masked_logits, dim=-1) * logp_all).sum(dim=-1).mean()
            entropy_loss = -entropy
            
            total_loss_batch = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            total_loss = total_loss_batch
            
            # 反向传播
            self.optimizers[agent]['actor'].zero_grad()
            self.optimizers[agent]['critic'].zero_grad()
            
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.selector.actor_networks[agent].parameters(), 
                self.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.selector.critic_networks[agent].parameters(), 
                self.max_grad_norm
            )
            
            self.optimizers[agent]['actor'].step()
            self.optimizers[agent]['critic'].step()
            
            # 增加更新计数器
            self.current_update += 1
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy.item()
            
            # 记录详细的训练指标
            with torch.no_grad():
                # KL散度计算
                kl_div = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                agent_kl_divergences.append(kl_div.item())
                
                # 裁剪比例
                clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                agent_clip_fractions.append(clip_fraction.item())
                
                # 熵值
                agent_entropy_values.append(entropy.item())
                
                # 比率统计
                agent_ratio_values.append(ratio.mean().item())
        
        # 记录智能体级别的训练指标
        if topic_enabled("training_step"):
            avg_kl = np.mean(agent_kl_divergences) if agent_kl_divergences else 0.0
            avg_clip = np.mean(agent_clip_fractions) if agent_clip_fractions else 0.0
            avg_entropy = np.mean(agent_entropy_values) if agent_entropy_values else 0.0
            avg_ratio = np.mean(agent_ratio_values) if agent_ratio_values else 0.0
            
            self.logger.info(f"[TRAIN_METRICS] {agent}: "
                           f"actor_loss={total_actor_loss:.4f}, "
                           f"critic_loss={total_critic_loss:.4f}, "
                           f"entropy={avg_entropy:.4f}, "
                           f"kl_div={avg_kl:.4f}, "
                           f"clip_frac={avg_clip:.4f}, "
                           f"ratio_mean={avg_ratio:.4f}")
        
        return {
            'total_loss': total_loss,
            'actor_loss': total_actor_loss,
            'critic_loss': total_critic_loss,
            'entropy_loss': total_entropy_loss
        }
    
    def _train_step_multi(self, agent: str, agent_exps: List[Dict]) -> Dict[str, float]:
        """多动作模式的训练逻辑"""
        # 获取网络
        network = self.selector.actor_networks_multi[agent]
        network.train()
        
        # 🔍 检查1：网络参数是否异常
        if self.current_update < 2:
            param_issues = []
            for name, param in network.named_parameters():
                param_max = param.data.abs().max().item()
                param_mean = param.data.abs().mean().item()
                param_std = param.data.std().item()
                
                # 检查异常大的参数
                if param_max > 100:
                    param_issues.append({
                        'name': name,
                        'max': param_max,
                        'mean': param_mean,
                        'std': param_std
                    })
            
            if param_issues:
                for issue in param_issues:
                    self.logger.warning(
                        f"[NETWORK_PARAM_CHECK] agent={agent} update={self.current_update} "
                        f"param={issue['name']} max_abs={issue['max']:.2f} "
                        f"mean_abs={issue['mean']:.2f} std={issue['std']:.2f} (异常！)"
                    )
        
        # 提取基础数据
        obs = torch.FloatTensor([exp['obs'] for exp in agent_exps]).to(self.device)
        rewards_raw = torch.FloatTensor([exp['reward'] for exp in agent_exps]).to(self.device)
        dones = torch.FloatTensor([1.0 if exp['done'] else 0.0 for exp in agent_exps]).to(self.device)
        values_old = torch.FloatTensor([exp['value'] for exp in agent_exps]).to(self.device)
        logprobs_old = torch.FloatTensor([exp['logprob'] for exp in agent_exps]).to(self.device)
        
        # 修复：对reward进行归一化（仅用于训练，不影响StepLog.reward_terms）
        # 🔍 检查reward值范围
        if self.current_update < 2:
            reward_min = rewards_raw.min().item()
            reward_max = rewards_raw.max().item()
            reward_mean = rewards_raw.mean().item()
            reward_std = rewards_raw.std().item()
            self.logger.warning(
                f"[REWARD_CHECK] agent={agent} update={self.current_update} "
                f"reward_raw: min={reward_min:.2f} max={reward_max:.2f} "
                f"mean={reward_mean:.2f} std={reward_std:.2f}"
            )
            if abs(reward_max) > 10000 or abs(reward_min) > 10000 or reward_std > 5000:
                self.logger.error(
                    f"[REWARD_ABNORMAL] agent={agent} reward值异常！可能需要归一化"
                )
        
        if self.reward_scaling_enabled:
            # 使用配置的reward_scale归一化
            rewards = rewards_raw / self.reward_scale
            # Clip到target_range
            rewards = torch.clamp(rewards, self.reward_target_range[0], self.reward_target_range[1])
            if self.current_update < 2:
                self.logger.warning(
                    f"[REWARD_NORM] agent={agent} 使用配置归一化: "
                    f"scale={self.reward_scale}, range={self.reward_target_range}, "
                    f"reward_norm: min={rewards.min().item():.2f} max={rewards.max().item():.2f}"
                )
        else:
            # 如果没有启用，使用经验统计归一化（防止reward过大）
            reward_mean = rewards_raw.mean().item()
            reward_std = rewards_raw.std().item() + 1e-8
            if abs(reward_mean) > 1000 or reward_std > 1000:
                # reward值过大，使用clip到合理范围
                rewards = torch.clamp(rewards_raw / 3000.0, -1.0, 1.0)
                if self.current_update < 2:
                    self.logger.warning(
                        f"[REWARD_NORM] agent={agent} reward_raw: mean={reward_mean:.2f} std={reward_std:.2f}, "
                        f"使用clip归一化到[-1, 1], "
                        f"reward_norm: min={rewards.min().item():.2f} max={rewards.max().item():.2f}"
                    )
            else:
                rewards = rewards_raw
        
        # 提取多动作详情
        actions_detail_list = [exp['actions_detail'] for exp in agent_exps]
        num_actions_list = [exp.get('num_actions', 0) for exp in agent_exps]
        
        # 🔍 验证：检查读取的actions_detail数量（前10次更新）
        if self.current_update < 10:
            for i, exp in enumerate(agent_exps[:5]):
                actions_detail_count = len(exp.get('actions_detail', []))
                num_actions_stored = exp.get('num_actions', 0)
                trace_id_exp = exp.get('trace_id', None)
                
                if actions_detail_count != num_actions_stored:
                    self.logger.error(
                        f"[ACTIONS_DETAIL_LOAD] agent={agent} sample={i} "
                        f"trace_id={trace_id_exp} "
                        f"len(actions_detail)={actions_detail_count} "
                        f"stored_num_actions={num_actions_stored} "
                        f"⚠️ 不匹配！说明experience中的actions_detail不完整"
                    )
        
        # 引导值
        with torch.no_grad():
            last_next = torch.FloatTensor(agent_exps[-1]['next_obs']).unsqueeze(0).to(self.device)
            next_value = self.selector.critic_networks[agent](last_next).squeeze().item()
        
        # 计算GAE（基于累积logprob，使用归一化后的rewards）
        advantages, returns = self.compute_gae(rewards.tolist(), values_old.tolist(), next_value, dones.tolist())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取网络
        network = self.selector.actor_networks_multi[agent]
        network.train()
        
        # 训练指标
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        agent_kl_divergences = []
        agent_clip_fractions = []
        agent_entropy_values = []
        agent_ratio_values = []
        
        for epoch in range(self.updates_per_iter):
            if self.current_update >= self.max_updates:
                break
            
            # 随机采样批次
            idx = torch.randperm(len(agent_exps))[:min(self.minibatch_size, len(agent_exps))]
            batch_obs = obs[idx]
            batch_adv = advantages[idx]
            batch_ret = returns[idx]
            batch_old_logp = logprobs_old[idx]
            batch_actions_detail = [actions_detail_list[i] for i in idx.tolist()]
            
            # 🔍 调试日志：记录batch采样信息（前10次更新，前5个样本）
            if self.current_update < 10 and epoch == 0:
                for j in range(min(5, len(idx))):
                    batch_idx = idx[j].item() if isinstance(idx[j], torch.Tensor) else idx[j]
                    if batch_idx < len(agent_exps):
                        exp = agent_exps[batch_idx]
                        trace_id_batch = exp.get('trace_id', None)
                        old_logprob_from_exp = exp.get('logprob', 0.0)
                        old_logprob_from_batch = batch_old_logp[j].item() if hasattr(batch_old_logp[j], 'item') else batch_old_logp[j]
                        
                        # 验证是否匹配
                        if abs(old_logprob_from_exp - old_logprob_from_batch) > 0.001:
                            self.logger.error(
                                f"[BATCH_LOGPROB_MISMATCH] agent={agent} update={self.current_update} epoch={epoch} "
                                f"sample={j} batch_idx={batch_idx} trace_id={trace_id_batch} "
                                f"exp['logprob']={old_logprob_from_exp:.6f} "
                                f"batch_old_logp[{j}]={old_logprob_from_batch:.6f} "
                                f"⚠️ 不匹配！batch中的old_logprob与experience不一致"
                            )
            
            # 计算critic loss（与单动作相同）
            with torch.no_grad():
                values_pred = self.selector.critic_networks[agent](batch_obs).squeeze(1)
            critic_loss = F.mse_loss(values_pred, batch_ret)
            
            # 🔍 检查returns和value_pred值范围（用于调试）
            if epoch == 0 and self.current_update < 2:
                ret_min = batch_ret.min().item()
                ret_max = batch_ret.max().item()
                ret_mean = batch_ret.mean().item()
                ret_std = batch_ret.std().item()
                val_min = values_pred.min().item()
                val_max = values_pred.max().item()
                val_mean = values_pred.mean().item()
                val_std = values_pred.std().item()
                self.logger.warning(
                    f"[RETURNS_CHECK] agent={agent} update={self.current_update} epoch={epoch} "
                    f"returns: min={ret_min:.2f} max={ret_max:.2f} mean={ret_mean:.2f} std={ret_std:.2f}, "
                    f"values_pred: min={val_min:.2f} max={val_max:.2f} mean={val_mean:.2f} std={val_std:.2f}"
                )
                if abs(ret_max) > 100 or abs(ret_min) > 100 or ret_std > 50:
                    self.logger.error(
                        f"[RETURNS_ABNORMAL] agent={agent} returns异常！可能是reward归一化未生效"
                    )
            
            # 🔍 检查输入obs是否异常
            if self.current_update < 2:
                obs_min = batch_obs.min().item()
                obs_max = batch_obs.max().item()
                obs_mean = batch_obs.mean().item()
                obs_std = batch_obs.std().item()
                
                self.logger.warning(
                    f"[OBS_CHECK] agent={agent} update={self.current_update} "
                    f"obs: min={obs_min:.2f} max={obs_max:.2f} "
                    f"mean={obs_mean:.2f} std={obs_std:.2f} shape={batch_obs.shape}"
                )
                
                if abs(obs_max) > 100 or abs(obs_min) > 100 or obs_std > 50:
                    self.logger.error(
                        f"[OBS_ABNORMAL] agent={agent} 输入obs异常！可能需要检查环境或特征提取"
                    )
            
            # 获取网络特征
            feat = network.encoder(batch_obs)  # [B, hidden_size]
            
            # 🔍 检查编码器输出是否异常
            if self.current_update < 2:
                feat_min = feat.min().item()
                feat_max = feat.max().item()
                feat_mean = feat.mean().item()
                feat_std = feat.std().item()
                
                self.logger.warning(
                    f"[ENCODER_OUTPUT_CHECK] agent={agent} update={self.current_update} "
                    f"feat (encoder output): min={feat_min:.2f} max={feat_max:.2f} "
                    f"mean={feat_mean:.2f} std={feat_std:.2f} shape={feat.shape}"
                )
                
                if abs(feat_max) > 100 or abs(feat_min) > 100 or feat_std > 50:
                    self.logger.error(
                        f"[ENCODER_OUTPUT_ABNORMAL] agent={agent} 编码器输出异常！"
                        f"如果obs正常，可能是编码器有问题"
                    )
            
            # 为每个样本的每个动作计算actor loss
            actor_losses = []
            total_entropy_batch = 0.0
            ratios = []
            
            for i, actions_detail in enumerate(batch_actions_detail):
                # 🔍 验证：检查batch中的actions_detail数量（前20个样本，前10次更新）
                if i < 20 and self.current_update < 10:
                    # 修复：使用idx[i]来获取正确的experience索引
                    # idx是tensor，需要先转成list或直接索引
                    try:
                        if isinstance(idx, torch.Tensor):
                            actual_exp_idx = int(idx[i].item())
                        else:
                            actual_exp_idx = int(idx[i])
                    except (IndexError, AttributeError, TypeError):
                        actual_exp_idx = -1
                        self.logger.error(
                            f"[BATCH_ACTIONS_DETAIL_CHECK_ERROR] agent={agent} update={self.current_update} sample={i} "
                            f"无法获取actual_exp_idx: idx类型={type(idx)}, idx[i]={idx[i] if isinstance(idx, (list, torch.Tensor)) and i < len(idx) else 'N/A'}"
                        )
                    
                    if actual_exp_idx >= 0 and actual_exp_idx < len(agent_exps):
                        exp = agent_exps[actual_exp_idx]
                        trace_id_batch = exp.get('trace_id', None)
                        stored_num_actions = exp.get('num_actions', 0)
                        
                        # 同时检查actions_detail_list中的值
                        actions_detail_from_list = actions_detail_list[actual_exp_idx] if actual_exp_idx < len(actions_detail_list) else []
                        
                        if len(actions_detail) != stored_num_actions:
                            self.logger.error(
                                f"[BATCH_ACTIONS_DETAIL_CHECK] agent={agent} update={self.current_update} sample={i} "
                                f"actual_exp_idx={actual_exp_idx} trace_id={trace_id_batch} "
                                f"batch中len(actions_detail)={len(actions_detail)} "
                                f"actions_detail_list中len={len(actions_detail_from_list)} "
                                f"experience中stored_num_actions={stored_num_actions} "
                                f"⚠️ 不匹配！batch中的actions_detail数量不对"
                            )
                    else:
                        self.logger.error(
                            f"[BATCH_ACTIONS_DETAIL_CHECK_ERROR] agent={agent} update={self.current_update} sample={i} "
                            f"actual_exp_idx={actual_exp_idx} 超出范围 [0, {len(agent_exps)})"
                        )
                
                # 获取该样本的特征
                feat_i = feat[i]  # [hidden_size]
                
                # 🔍 检查2：输入特征是否异常
                if i < 2 and self.current_update < 2:
                    feat_i_min = feat_i.min().item()
                    feat_i_max = feat_i.max().item()
                    feat_i_mean = feat_i.mean().item()
                    feat_i_std = feat_i.std().item()
                    
                    self.logger.warning(
                        f"[FEAT_CHECK] agent={agent} update={self.current_update} sample={i} "
                        f"feat_i: min={feat_i_min:.2f} max={feat_i_max:.2f} "
                        f"mean={feat_i_mean:.2f} std={feat_i_std:.2f} "
                        f"shape={feat_i.shape}"
                    )
                    
                    # 检查特征是否异常
                    if abs(feat_i_max) > 100 or abs(feat_i_min) > 100 or feat_i_std > 50:
                        self.logger.error(
                            f"[FEAT_ABNORMAL] agent={agent} sample={i} "
                            f"特征值异常！可能需要检查编码器或输入数据"
                        )
                
                # 计算该样本的总logprob（重建决策过程）
                # 修复：初始化为tensor而不是float，确保在计算图中
                sample_logprob = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 获取样本定位信息（优先从experience中读取trace_id）
                # 🔍 修复：使用batch_idx获取正确的experience索引
                batch_idx = idx[i].item() if isinstance(idx[i], torch.Tensor) else idx[i]
                exp = agent_exps[batch_idx] if batch_idx < len(agent_exps) else {}
                trace_id = exp.get('trace_id', None)
                month = exp.get('month', None)
                
                # 🔍 调试日志：验证batch中的old_logprob与experience中的logprob是否匹配（前10次更新，前5个样本）
                if self.current_update < 10 and i < 5:
                    old_logprob_from_exp = exp.get('logprob', 0.0)
                    old_logprob_from_batch = batch_old_logp[i].item() if hasattr(batch_old_logp[i], 'item') else batch_old_logp[i]
                    
                    if abs(old_logprob_from_exp - old_logprob_from_batch) > 0.001:
                        self.logger.error(
                            f"[RATIO_DEBUG_LOGPROB_MISMATCH] agent={agent} update={self.current_update} sample={i} "
                            f"batch_idx={batch_idx} trace_id={trace_id} "
                            f"exp['logprob']={old_logprob_from_exp:.6f} "
                            f"batch_old_logp[{i}]={old_logprob_from_batch:.6f} "
                            f"⚠️ 不匹配！batch中的old_logprob与experience不一致"
                        )
                
                # 如果没有trace_id，尝试从step_log获取
                if not trace_id:
                    step_log = exp.get('step_log', None)
                    if step_log and hasattr(step_log, 't'):
                        trace_id = f"step_{step_log.t}"
                        step_num = step_log.t
                    elif step_log and isinstance(step_log, dict):
                        trace_id = step_log.get('trace_id', f"sample_{i}")
                        step_num = step_log.get('month', step_log.get('t', 'N/A'))
                    else:
                        trace_id = f"sample_{i}"
                        step_num = 'N/A'
                else:
                    # 从trace_id中提取信息，格式为"agent-month-step"
                    # 例如: "COUNCIL-13-2"
                    trace_parts = trace_id.split('-')
                    if len(trace_parts) >= 3:
                        step_num = trace_parts[2]  # step是第三部分
                    else:
                        step_num = trace_id
                
                for j, atomic_action in enumerate(actions_detail):
                    point_idx = atomic_action['point_idx']
                    type_idx = atomic_action['type_idx']  # 修复：这是类型索引，不是action_id
                    available_types = atomic_action.get('available_types', [])  # 该点的类型列表
                    
                    # 修复：从experience中读取mask并应用（与经验收集时一致）
                    initial_point_mask = exp.get('initial_point_mask', None)
                    point_masks_per_action = exp.get('point_masks_per_action', [])
                    cand_idx_points_count = exp.get('cand_idx_points_count', None)
                    
                    # 获取该动作选择前的point_mask
                    if point_masks_per_action and j < len(point_masks_per_action):
                        # 使用存储的mask状态（从numpy数组转换为tensor）
                        mask_data = point_masks_per_action[j]
                        if isinstance(mask_data, np.ndarray):
                            point_mask = torch.from_numpy(mask_data).float().to(self.device)
                        elif isinstance(mask_data, torch.Tensor):
                            point_mask = mask_data.to(self.device)
                        else:
                            point_mask = torch.tensor(mask_data, dtype=torch.float32, device=self.device)
                    elif initial_point_mask is not None:
                        # 如果没有存储每个动作的mask，从初始mask重建
                        if isinstance(initial_point_mask, np.ndarray):
                            point_mask = torch.from_numpy(initial_point_mask).float().to(self.device)
                        elif isinstance(initial_point_mask, torch.Tensor):
                            point_mask = initial_point_mask.clone().to(self.device)
                        else:
                            point_mask = torch.tensor(initial_point_mask, dtype=torch.float32, device=self.device)
                        for prev_action in actions_detail[:j]:
                            prev_point_idx = prev_action['point_idx']
                            if prev_point_idx < len(point_mask):
                                point_mask[prev_point_idx] = 0
                    elif cand_idx_points_count is not None:
                        # 降级：假设所有点都可用（但记录警告）
                        point_mask = torch.ones(cand_idx_points_count, device=self.device)
                        if self.current_update < 10 and j == 0:
                            self.logger.warning(
                                f"[MASK_MISSING] agent={agent} sample={i} trace_id={trace_id} "
                                f"mask信息缺失，使用全1mask（可能与经验收集时不一致）"
                            )
                    else:
                        # 最终降级：先获取网络输出大小，再创建mask
                        # 先获取point_logits以确定大小
                        point_logits_temp = network.forward_point(feat_i.unsqueeze(0))  # [1, max_points]
                        point_mask = torch.ones(point_logits_temp.shape[1], device=self.device)
                        if self.current_update < 10 and j == 0:
                            self.logger.warning(
                                f"[MASK_MISSING_FULL] agent={agent} sample={i} trace_id={trace_id} "
                                f"mask信息完全缺失，使用全1mask"
                            )
                    
                    # Step 1: 计算点选择的logprob（应用mask，与经验收集时一致）
                    point_logits = network.forward_point(feat_i.unsqueeze(0))  # [1, max_points]
                    
                    # 修复：应用mask到logits（与经验收集时一致）
                    num_points = min(len(point_mask), point_logits.shape[1])
                    point_logits_masked = point_logits.clone()
                    point_logits_masked[0, num_points:] = float('-inf')  # 超出部分设为-inf
                    point_logits_masked[0, :num_points] = point_logits_masked[0, :num_points] + \
                        torch.where(point_mask[:num_points] > 0, 
                                   torch.zeros_like(point_mask[:num_points]), 
                                   torch.full_like(point_mask[:num_points], float('-inf')))
                    
                    # 计算概率（应用mask后）
                    temperature = self.config.get("multi_action", {}).get("temperature", 1.0)
                    p_probs = F.softmax(point_logits_masked[0, :num_points] / temperature, dim=-1)
                    p_probs_masked = p_probs * point_mask[:num_points]
                    p_probs_for_logprob = p_probs_masked / (p_probs_masked.sum() + 1e-8) if p_probs_masked.sum() > 0 else p_probs_masked
                    
                    # 计算logprob（与经验收集时一致）
                    point_logprob = torch.log(p_probs_for_logprob[point_idx] + 1e-8)
                    
                    # 不要用clamp掩盖错误，应该assert检查索引范围
                    assert 0 <= point_idx < point_logits.shape[1], (
                        f"[点索引越界] agent={agent} sample={i} action_idx={j} "
                        f"point_idx={point_idx} 超出范围 [0, {point_logits.shape[1]}) "
                        f"候选信息: point_logits.shape={point_logits.shape} "
                        f"actions_detail={actions_detail}"
                    )
                    
                    # 计算统计量（用于logits异常检查）
                    point_logits_min = point_logits.min().item()
                    point_logits_max = point_logits.max().item()
                    point_logits_mean = point_logits.mean().item()
                    point_logits_std = point_logits.std().item()
                    
                    # 调试：输出点logprob和统计量
                    if i < 2 and self.current_update < 2:
                        self.logger.warning(
                            f"[LOGPROB_DEBUG] agent={agent} trace_id={trace_id} step={step_num} "
                            f"sample={i} action_idx={j} "
                            f"point_idx={point_idx} "
                            f"point_logprob={point_logprob.item():.6f} "
                            f"point_logits: min={point_logits_min:.2f} max={point_logits_max:.2f} "
                            f"mean={point_logits_mean:.2f} std={point_logits_std:.2f} "
                            f"shape={point_logits.shape}"
                        )
                    
                    # Step 2: 计算类型选择的logprob
                    type_logits = network.forward_type(feat_i.unsqueeze(0), torch.tensor([point_idx], device=self.device))
                    # 修复：只使用实际可用的类型数量，而不是整个max_types
                    if available_types:
                        available_types_count = len(available_types)
                        type_logits_masked = type_logits[0, :available_types_count]  # 只取有效的类型
                    else:
                        # 如果不知道available_types（理论上不应该发生，因为ratio=0修复已确保available_types存在），使用整个输出
                        available_types_count = type_logits.shape[1]
                        type_logits_masked = type_logits[0]
                    
                    type_logprobs = torch.log_softmax(type_logits_masked, dim=-1)  # [available_types_count]
                    
                    # 不要用clamp掩盖错误，应该assert检查索引范围
                    assert 0 <= type_idx < available_types_count, (
                        f"[类型索引越界] agent={agent} sample={i} action_idx={j} "
                        f"type_idx={type_idx} 超出范围 [0, {available_types_count}) "
                        f"available_types={available_types} "
                        f"type_logits_masked.shape={type_logits_masked.shape}"
                    )
                    type_logprob = type_logprobs[type_idx]  # 直接使用，不再clamp
                    
                    # 计算统计量
                    type_logits_min = type_logits_masked.min().item()
                    type_logits_max = type_logits_masked.max().item()
                    type_logits_mean = type_logits_masked.mean().item()
                    type_logits_std = type_logits_masked.std().item()
                    
                    # 调试：输出类型logprob和统计量
                    if i < 2 and self.current_update < 2:
                        self.logger.warning(
                            f"[LOGPROB_DEBUG] agent={agent} trace_id={trace_id} step={step_num} "
                            f"sample={i} action_idx={j} "
                            f"type_idx={type_idx} available_types_count={available_types_count} "
                            f"type_logprob={type_logprob.item():.6f} "
                            f"type_logits: min={type_logits_min:.2f} max={type_logits_max:.2f} "
                            f"mean={type_logits_mean:.2f} std={type_logits_std:.2f} "
                            f"shape={type_logits_masked.shape}"
                        )
                    
                    # 累计
                    sample_logprob = sample_logprob + point_logprob + type_logprob
                    
                    # 🔍 调试日志：记录每个动作的logprob重建（前20个动作，前50次更新或每10次更新）
                    if j < 20 and (self.current_update < 50 or self.current_update % 10 == 0):
                        self.logger.warning(
                            f"[LOGPROB_REBUILD] agent={agent} trace_id={trace_id} step={step_num} "
                            f"sample={i} action_idx={j} "
                            f"p_idx={point_idx} p_logprob={point_logprob.item():.6f} "
                            f"t_idx={type_idx} t_logprob={type_logprob.item():.6f} "
                            f"accumulated_sample_logprob={sample_logprob.item():.6f} "
                            f"available_types_count={available_types_count}"
                        )
                
                # 与old_logprob比较（累积值）
                # 修复：直接使用tensor计算ratio，确保有grad
                old_logp_tensor = batch_old_logp[i] if isinstance(batch_old_logp[i], torch.Tensor) else torch.tensor(batch_old_logp[i], device=self.device)
                logprob_diff = sample_logprob - old_logp_tensor
                ratio = torch.exp(logprob_diff)
                
                # 先计算ratio_val用于条件判断
                old_logp_val = batch_old_logp[i].item() if hasattr(batch_old_logp[i], 'item') else batch_old_logp[i]
                sample_logp_val = sample_logprob.item()
                diff_val = logprob_diff.item()
                ratio_val = ratio.item()
                
                # 🔍 调试日志：详细的ratio对比（前20个样本，前50次更新或每10次更新或ratio异常时）
                if i < 20 and (self.current_update < 50 or self.current_update % 10 == 0 or ratio_val < 0.1 or ratio_val > 10.0):
                    # 输出更详细的信息
                    point_info = []
                    type_info = []
                    for j, atomic_action in enumerate(actions_detail):
                        point_info.append(f"p{atomic_action['point_idx']}")
                        type_info.append(f"t{atomic_action['type_idx']}({len(atomic_action.get('available_types', []))})")
                    
                    # 🔍 验证：检查actions_detail的数量是否合理
                    if len(actions_detail) != len(point_info):
                        self.logger.error(
                            f"[ACTIONS_DETAIL_MISMATCH] agent={agent} sample={i} "
                            f"len(actions_detail)={len(actions_detail)} len(point_info)={len(point_info)}"
                        )
                    
                    # 检查差异是否异常
                    diff_abs = abs(diff_val)
                    ratio_abnormal = ratio_val < 0.1 or ratio_val > 10.0
                    
                    log_level = "error" if ratio_abnormal or diff_abs > 5.0 else "warning"
                    log_msg = (
                        f"[RATIO_DEBUG] agent={agent} update={self.current_update} sample={i} "
                        f"sample_logprob={sample_logp_val:.6f} old_logprob={old_logp_val:.6f} "
                        f"diff={diff_val:.6f} ratio={ratio_val:.6f} "
                        f"num_actions={len(actions_detail)} "
                        f"actions={'+'.join(point_info)}+{'+'.join(type_info)} "
                        f"trace_id={trace_id} step={step_num}"
                    )
                    
                    if ratio_abnormal:
                        log_msg += f" ⚠️ RATIO_ABNORMAL: ratio={ratio_val:.2f} out of range [0.1, 10.0]"
                    if diff_abs > 5.0:
                        log_msg += f" ⚠️ DIFF_LARGE: diff={diff_val:.2f} > 5.0"
                    
                    if log_level == "error":
                        self.logger.error(log_msg)
                    else:
                        self.logger.warning(log_msg)
                
                surr1 = ratio * batch_adv[i]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv[i]
                actor_loss = -torch.min(surr1, surr2)
                
                actor_losses.append(actor_loss)
                ratios.append(ratio.item())
                
                # 计算entropy（简化：只计算最后一个动作的entropy）
                with torch.no_grad():
                    if actions_detail:
                        last_action = actions_detail[-1]
                        point_idx = last_action['point_idx']
                        available_types = last_action.get('available_types', [])
                        type_logits = network.forward_type(feat_i.unsqueeze(0), torch.tensor([point_idx], device=self.device))
                        if available_types:
                            available_types_count = len(available_types)
                            type_logits_masked = type_logits[0, :available_types_count]
                        else:
                            type_logits_masked = type_logits[0]
                        type_probs = torch.softmax(type_logits_masked, dim=-1)
                        entropy = -(type_probs * torch.log(type_probs + 1e-8)).sum()
                        total_entropy_batch += entropy.item()
            
            # 聚合loss
            total_actor_loss_batch = torch.stack(actor_losses).mean() if actor_losses else torch.tensor(0.0, device=self.device)
            total_entropy_batch = total_entropy_batch / len(batch_actions_detail) if batch_actions_detail else 0.0
            
            # 总loss
            total_loss_batch = total_actor_loss_batch + self.value_loss_coef * critic_loss - self.entropy_coef * total_entropy_batch
            
            # 🔍 检查3：反向传播前的loss是否异常
            if self.current_update < 2:
                loss_value = total_loss_batch.item() if isinstance(total_loss_batch, torch.Tensor) else total_loss_batch
                self.logger.warning(
                    f"[LOSS_CHECK] agent={agent} update={self.current_update} "
                    f"total_loss_batch={loss_value:.6f}"
                )
                
                # 检查loss是否异常大
                if loss_value > 1000:
                    self.logger.error(
                        f"[LOSS_ABNORMAL] agent={agent} loss异常大！可能导致梯度爆炸"
                    )
            
            # 反向传播
            self.optimizers[agent]['actor'].zero_grad()
            self.optimizers[agent]['critic'].zero_grad()
            total_loss_batch.backward()
            
            # 🔍 检查4：梯度是否爆炸
            if self.current_update < 2:
                actor_grad_norm = 0.0
                critic_grad_norm = 0.0
                
                # 计算actor梯度范数
                for param in network.parameters():
                    if param.grad is not None:
                        actor_grad_norm += param.grad.data.norm(2).item() ** 2
                actor_grad_norm = actor_grad_norm ** 0.5
                
                # 计算critic梯度范数
                for param in self.selector.critic_networks[agent].parameters():
                    if param.grad is not None:
                        critic_grad_norm += param.grad.data.norm(2).item() ** 2
                critic_grad_norm = critic_grad_norm ** 0.5
                
                self.logger.warning(
                    f"[GRAD_CHECK] agent={agent} update={self.current_update} "
                    f"actor_grad_norm={actor_grad_norm:.2f} critic_grad_norm={critic_grad_norm:.2f}"
                )
                
                # 检查梯度是否异常
                if actor_grad_norm > 100 or critic_grad_norm > 100:
                    self.logger.error(
                        f"[GRAD_EXPLOSION] agent={agent} 梯度爆炸！actor_norm={actor_grad_norm:.2f} critic_norm={critic_grad_norm:.2f}"
                    )
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                self.selector.critic_networks[agent].parameters(), 
                self.max_grad_norm
            )
            
            # 更新参数
            self.optimizers[agent]['actor'].step()
            self.optimizers[agent]['critic'].step()
            
            self.current_update += 1
            
            total_loss = total_loss_batch.item()
            total_actor_loss += total_actor_loss_batch.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += total_entropy_batch
            
            # 记录指标
            with torch.no_grad():
                if ratios:
                    ratio_tensor = torch.FloatTensor(ratios).to(self.device)
                    kl_div = ((ratio_tensor - 1.0) - torch.log(ratio_tensor + 1e-8)).mean()
                    agent_kl_divergences.append(kl_div.item())
                    
                    clip_fraction = ((ratio_tensor - 1.0).abs() > self.clip_eps).float().mean()
                    agent_clip_fractions.append(clip_fraction.item())
                    
                    agent_ratio_values.append(ratio_tensor.mean().item())
                
                agent_entropy_values.append(total_entropy_batch)
        
        # 记录智能体级别的训练指标
        if topic_enabled("training_step"):
            avg_kl = np.mean(agent_kl_divergences) if agent_kl_divergences else 0.0
            avg_clip = np.mean(agent_clip_fractions) if agent_clip_fractions else 0.0
            avg_entropy = np.mean(agent_entropy_values) if agent_entropy_values else 0.0
            avg_ratio = np.mean(agent_ratio_values) if agent_ratio_values else 0.0
            
            self.logger.info(f"[TRAIN_METRICS] {agent}: "
                           f"actor_loss={total_actor_loss:.4f}, "
                           f"critic_loss={total_critic_loss:.4f}, "
                           f"entropy={avg_entropy:.4f}, "
                           f"kl_div={avg_kl:.4f}, "
                           f"clip_frac={avg_clip:.4f}, "
                           f"ratio_mean={avg_ratio:.4f}")
        
        return {
            'total_loss': total_loss,
            'actor_loss': total_actor_loss,
            'critic_loss': total_critic_loss,
            'entropy_loss': total_entropy_loss
        }
    
    # 旧的简化损失已删除，改用上方标准 PPO 计算
    
    def train(self, num_episodes: int, save_interval: int = 100) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            num_episodes: 训练轮数
            save_interval: 保存间隔
            
        Returns:
            训练结果
        """
        print(f"开始训练 {num_episodes} 轮...")
        print(f"温度退火: {self.initial_temperature} -> {self.final_temperature} (steps={self.anneal_steps})")
        
        for episode in range(num_episodes):
            # 更新温度（线性退火）
            self._update_temperature()
            
            # 收集经验
            experiences = self.collect_experience(self.rollout_horizon)
            
            if not experiences:
                print(f"Episode {episode}: 没有收集到经验")
                continue
            
            # 更新总步数
            self.total_steps += len(experiences)
            
            # 训练
            train_stats = self.train_step(experiences)
            
            # 记录统计信息
            self.training_history.append(train_stats)
            
            # 计算episode奖励
            episode_rewards = {}
            for agent in self.config.get("agents", {}).get("order", []):
                agent_rewards = [exp['reward'] for exp in experiences if exp['agent'] == agent]
                episode_rewards[agent] = sum(agent_rewards) if agent_rewards else 0.0
                self.episode_rewards[agent].append(episode_rewards[agent])
            
            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode}: "
                      f"Total Loss: {train_stats['total_loss']:.4f}, "
                      f"Actor Loss: {train_stats['actor_loss']:.4f}, "
                      f"Critic Loss: {train_stats['critic_loss']:.4f}, "
                      f"Entropy: {train_stats['entropy_loss']:.4f}, "
                      f"Temp: {self.current_temperature:.3f}, "
                      f"Updates: {train_stats['current_update']}")
                
                for agent, reward in episode_rewards.items():
                    print(f"  {agent} Reward: {reward:.2f}")
                
                # 打印训练统计摘要
                print(f"  训练统计: 总步数={train_stats['total_steps']}, "
                      f"智能体数={train_stats['num_agents']}, "
                      f"经验数={train_stats['total_experiences']}")
            
            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(f"checkpoints/v5_0_ppo_episode_{episode}.pth")
        
        print("训练完成!")
        
        return {
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'total_episodes': num_episodes,
            'step_logs': self.env.step_logs,
            'env_states': self.env.env_states
        }
    
    def save_model(self, path: str):
        """保存模型"""
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        multi_enabled = self.config.get("multi_action", {}).get("enabled", False)
        
        model_state = {
            'optimizers': {agent: {opt_name: opt.state_dict() for opt_name, opt in opts.items()} 
                          for agent, opts in self.optimizers.items()},
            'training_step': self.training_step,
            'config': self.config
        }
        
        # 根据模式保存不同的网络
        if multi_enabled:
            model_state['actor_networks_multi'] = {
                agent: net.state_dict() for agent, net in self.selector.actor_networks_multi.items()
            }
        else:
            model_state['actor_networks'] = {
                agent: net.state_dict() for agent, net in self.selector.actor_networks.items()
            }
        
        # Critic网络总是相同
        model_state['critic_networks'] = {
            agent: net.state_dict() for agent, net in self.selector.critic_networks.items()
        }
        
        torch.save(model_state, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return
        
        model_state = torch.load(path, map_location=self.device)
        
        multi_enabled = self.config.get("multi_action", {}).get("enabled", False)
        
        # 根据模式加载不同的网络
        if multi_enabled and 'actor_networks_multi' in model_state:
            for agent, net in self.selector.actor_networks_multi.items():
                if agent in model_state['actor_networks_multi']:
                    net.load_state_dict(model_state['actor_networks_multi'][agent])
        elif 'actor_networks' in model_state:
            for agent, net in self.selector.actor_networks.items():
                if agent in model_state['actor_networks']:
                    net.load_state_dict(model_state['actor_networks'][agent])
        
        # 加载critic网络
        for agent, net in self.selector.critic_networks.items():
            if agent in model_state.get('critic_networks', {}):
                net.load_state_dict(model_state['critic_networks'][agent])
        
        # 加载优化器状态
        for agent, opts in self.optimizers.items():
            if agent in model_state.get('optimizers', {}):
                for opt_name, opt in opts.items():
                    if opt_name in model_state['optimizers'][agent]:
                        opt.load_state_dict(model_state['optimizers'][agent][opt_name])
        
        self.training_step = model_state.get('training_step', 0)
        print(f"模型已从 {path} 加载")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            num_episodes: 评估轮数
            
        Returns:
            评估结果
        """
        print(f"开始评估 {num_episodes} 轮...")
        
        eval_rewards = {agent: [] for agent in self.config.get("agents", {}).get("order", [])}
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = {agent: 0.0 for agent in self.config.get("agents", {}).get("order", [])}
            
            step_count = 0
            max_steps = 1000  # 防止无限循环
            
            while not done and step_count < max_steps:
                current_agent = self.env.current_agent
                
                # 获取动作候选
                candidates = self.env.get_action_candidates(current_agent)
                
                if not candidates:
                    self.env._update_state()
                    step_count += 1
                    # 检查是否结束
                    done = self.env._is_done()
                    continue
                
                # 选择动作序列（使用贪心策略）
                sequence = self.selector.choose_sequence(
                    agent=current_agent,
                    candidates=candidates,
                    state=state,
                    greedy=True
                )
                
                # 执行动作
                next_state, reward, done, info = self.env.step(current_agent, sequence)
                
                # 记录奖励
                episode_rewards[current_agent] += reward
                
                # 更新状态
                state = next_state
                step_count += 1
            
            # 记录episode奖励
            for agent, reward in episode_rewards.items():
                eval_rewards[agent].append(reward)
        
        # 计算平均奖励
        avg_rewards = {}
        for agent, rewards in eval_rewards.items():
            avg_rewards[agent] = np.mean(rewards) if rewards else 0.0
        
        print("评估结果:")
        for agent, avg_reward in avg_rewards.items():
            print(f"  {agent}: {avg_reward:.2f}")
        
        return avg_rewards
