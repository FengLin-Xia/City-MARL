"""
v5.0 RL选择器

基于契约对象和配置的RL策略选择器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts import ActionCandidate, Sequence, EnvironmentState, CandidateIndex, AtomicAction
from config_loader import ConfigLoader
import torch.distributions as D
from utils.logger_factory import get_logger, topic_enabled, sampling_allows


class V5ActorNetwork(nn.Module):
    """v5.0 Actor网络"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 9):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class V5CriticNetwork(nn.Module):
    """v5.0 Critic网络"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class V5ActorNetworkMulti(nn.Module):
    """v5.1 多动作Actor网络（三头：point/type/stop）"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 max_points: int = 200, max_types: int = 9, point_embed_dim: int = 16):
        super().__init__()
        
        # 共享编码器（复用v5.0结构）
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 三个小头
        self.point_head = nn.Linear(hidden_size, max_points)      # 选点
        self.type_head = nn.Linear(hidden_size + point_embed_dim, max_types)  # 选类型
        self.stop_head = nn.Linear(hidden_size, 1)                # STOP
        
        # 点嵌入（用于type_head的条件输入）
        self.point_embed = nn.Embedding(max_points, point_embed_dim)
        
        self.max_points = max_points
        self.max_types = max_types
        self.point_embed_dim = point_embed_dim
    
    def forward(self, x):
        """保持向后兼容：默认只返回点分布"""
        feat = self.encoder(x)
        return self.point_head(feat)
    
    def forward_point(self, feat):
        """前向计算点分布"""
        return self.point_head(feat)
    
    def forward_type(self, feat, point_idx):
        """前向计算类型分布（条件于选定的点）"""
        # point_idx: [B] or scalar
        if not isinstance(point_idx, torch.Tensor):
            point_idx = torch.tensor([point_idx], device=feat.device)
        if point_idx.dim() == 0:
            point_idx = point_idx.unsqueeze(0)
        
        pe = self.point_embed(point_idx)  # [B, E]
        
        # 如果feat是[B, H]且point_idx是[B]，则正常concat
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)  # [1, H]
        
        # 确保维度匹配
        if feat.shape[0] != pe.shape[0]:
            if feat.shape[0] == 1:
                feat = feat.expand(pe.shape[0], -1)
            elif pe.shape[0] == 1:
                pe = pe.expand(feat.shape[0], -1)
        
        combined = torch.cat([feat, pe], dim=-1)  # [B, H+E]
        return self.type_head(combined)
    
    def forward_stop(self, feat):
        """前向计算STOP logit"""
        return self.stop_head(feat)


class V5RLSelector:
    """v5.0 RL选择器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RL选择器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.loader = ConfigLoader()
        
        # 获取智能体配置
        self.agents = config.get("agents", {}).get("order", [])
        
        # 网络参数
        self.obs_size = 64  # 观察空间大小
        self.hidden_size = 128
        self.action_size = 9  # 动作空间大小（0-8）
        
        # 初始化网络
        self.actor_networks = {}
        self.critic_networks = {}
        
        for agent in self.agents:
            self.actor_networks[agent] = V5ActorNetwork(
                input_size=self.obs_size,
                hidden_size=self.hidden_size,
                output_size=self.action_size
            )
            self.critic_networks[agent] = V5CriticNetwork(
                input_size=self.obs_size,
                hidden_size=self.hidden_size
            )
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将网络移到设备
        for agent in self.agents:
            self.actor_networks[agent] = self.actor_networks[agent].to(self.device)
            self.critic_networks[agent] = self.critic_networks[agent].to(self.device)
        self.logger = get_logger("policy")
        
        # v5.1: 初始化多动作网络（如果启用）
        self.multi_action_enabled = config.get("multi_action", {}).get("enabled", False)
        if self.multi_action_enabled:
            self.actor_networks_multi = {}
            max_points = config.get("multi_action", {}).get("candidate_topP", 200)
            for agent in self.agents:
                self.actor_networks_multi[agent] = V5ActorNetworkMulti(
                    input_size=self.obs_size,
                    hidden_size=self.hidden_size,
                    max_points=max_points,
                    max_types=self.action_size
                ).to(self.device)
    
    def _agent_allowed_actions(self, agent: str) -> List[int]:
        agent_config = self.config.get("agents", {}).get("defs", {}).get(agent, {})
        return list(agent_config.get("action_ids", []))

    def _candidate_ids(self, candidates: List[ActionCandidate]) -> List[int]:
        return [c.id for c in candidates]

    def _masked_logits(self, agent: str, state: EnvironmentState, allowed_ids: List[int]) -> torch.Tensor:
        obs = self._encode_state(state)
        logits = self.actor_networks[agent](obs)
        mask = torch.full((self.action_size,), float('-inf'), device=self.device)
        if allowed_ids:
            mask[allowed_ids] = 0.0
        masked_logits = logits + mask
        return masked_logits

    def select_action(self, agent: str, candidates: List[ActionCandidate], state: EnvironmentState, greedy: bool = False) -> Optional[Dict[str, Any]]:
        """基于策略从候选中选择动作，返回包含 logprob/value 的信息"""
        if not candidates:
            return None
        # 候选内索引化：仅对当前候选集合建分布
        obs = self._encode_state(state)
        logits_full = self.actor_networks[agent](obs)
        # 提取对应候选的 logit，按候选顺序组成向量
        cand_ids_list = [c.id for c in candidates]
        logits = logits_full[cand_ids_list]
        # 温度采样（从配置取 mappo.exploration.temperature，若无则1.0）
        temp = float(self.config.get('mappo', {}).get('exploration', {}).get('temperature', 1.0))
        logits = logits / max(temp, 1e-6)
        log_probs_vec = torch.log_softmax(logits, dim=-1)
        probs_vec = torch.softmax(logits, dim=-1)
        if greedy:
            idx = int(torch.argmax(probs_vec).item())
        else:
            dist = D.Categorical(probs_vec)
            idx = int(dist.sample().item())
        action_id = cand_ids_list[idx]
        chosen_logprob = log_probs_vec[idx].detach().item()
        with torch.no_grad():
            value = self.critic_networks[agent](self._encode_state(state)).squeeze().item()

        # 找到对应候选
        chosen_cand = candidates[idx] if 0 <= idx < len(candidates) else None
        if chosen_cand is None:
            return None

        sequence = Sequence(agent=agent, actions=[action_id])

        # 选择日志（受配置开关与采样控制）
        if topic_enabled("policy_select") and sampling_allows(agent, getattr(state, 'month', None), None):
            # 提取允许集合上的 top3 概率
            topk = []
            vals, idxs = torch.topk(probs_vec, k=min(3, probs_vec.numel()))
            for v, ix in zip(vals.tolist(), idxs.tolist()):
                topk.append((int(cand_ids_list[ix]), round(float(v), 4)))
            # 计算熵
            p = probs_vec
            entropy = float(-(p * (p + 1e-8).log()).sum().item())
            self.logger.info(
                f"policy_select agent={agent} month={getattr(state, 'month', '?')} action_id={action_id} logp={round(chosen_logprob,4)} value={round(value,3)} top3={topk} H={round(entropy,4)}")
        return {
            'sequence': sequence,
            'action_id': action_id,
            'logprob': chosen_logprob,
            'value': value,
            'probs': probs_vec.detach().cpu().numpy(),
        }

    def choose_sequence(self, agent: str, candidates: List[ActionCandidate], 
                       state: EnvironmentState, greedy: bool = False) -> Optional[Sequence]:
        """
        选择动作序列
        
        Args:
            agent: 智能体名称
            candidates: 动作候选列表
            state: 环境状态
            greedy: 是否使用贪心策略
            
        Returns:
            选择的序列
        """
        if not candidates:
            return None
        
        # 获取智能体的可用动作ID
        agent_config = self.config.get("agents", {}).get("defs", {}).get(agent, {})
        available_action_ids = agent_config.get("action_ids", [])
        
        # 过滤候选动作
        valid_candidates = [c for c in candidates if c.id in available_action_ids]
        
        if not valid_candidates:
            return None
        
        # 选择动作
        if greedy:
            # 贪心策略：选择第一个有效动作
            chosen_action = valid_candidates[0]
        else:
            # 随机策略：随机选择一个有效动作
            chosen_action = np.random.choice(valid_candidates)
        
        # 创建序列
        sequence = Sequence(
            agent=agent,
            actions=[chosen_action.id]
        )
        
        return sequence
    
    def get_action_probabilities(self, agent: str, candidates: List[ActionCandidate], 
                                state: EnvironmentState) -> torch.Tensor:
        """
        获取动作概率分布
        
        Args:
            agent: 智能体名称
            candidates: 动作候选列表
            state: 环境状态
            
        Returns:
            动作概率分布
        """
        if not candidates:
            return torch.zeros(self.action_size)
        
        # 获取智能体的可用动作ID
        agent_config = self.config.get("agents", {}).get("defs", {}).get(agent, {})
        available_action_ids = agent_config.get("action_ids", [])
        
        # 创建动作掩码
        action_mask = torch.zeros(self.action_size)
        for action_id in available_action_ids:
            action_mask[action_id] = 1.0
        
        # 获取网络输出
        with torch.no_grad():
            obs = self._encode_state(state)
            logits = self.actor_networks[agent](obs)
            
            # 应用掩码
            masked_logits = logits * action_mask
            
            # 计算概率
            probs = F.softmax(masked_logits, dim=-1)
        
        return probs
    
    def get_value(self, agent: str, state: EnvironmentState) -> float:
        """
        获取状态价值
        
        Args:
            agent: 智能体名称
            state: 环境状态
            
        Returns:
            状态价值
        """
        with torch.no_grad():
            obs = self._encode_state(state)
            value = self.critic_networks[agent](obs)
            return value.item()
    
    def select_action_multi(self, agent: str, candidates: List[ActionCandidate], 
                           cand_idx: CandidateIndex, state: EnvironmentState, 
                           max_k: int = 5, greedy: bool = False) -> Optional[Dict[str, Any]]:
        """
        多动作自回归采样（v5.1）
        
        Args:
            agent: 智能体名称
            candidates: 动作候选列表
            cand_idx: 候选索引
            state: 环境状态
            max_k: 最多选择动作数
            greedy: 是否贪心
            
        Returns:
            {sequence, logprob, entropy, value}
        """
        if not self.multi_action_enabled or not cand_idx or len(cand_idx.points) == 0:
            # 降级到单动作
            return self.select_action(agent, candidates, state, greedy)
        
        # 编码器只执行一次
        obs = self._encode_state(state)
        network = self.actor_networks_multi[agent]
        feat = network.encoder(obs)
        
        # 初始化掩码
        point_mask = torch.ones(len(cand_idx.points), device=self.device)
        type_masks = [torch.ones(len(types), device=self.device) 
                     for types in cand_idx.types_per_point]
        
        selected_actions = []
        total_logprob = 0.0
        total_entropy = 0.0
        
        for k in range(max_k):
            # Step 1: 选点（包含STOP）
            p_logits = network.forward_point(feat)  # [hidden_size] -> [max_points]
            
            # 掩码：只保留有效点
            p_logits_masked = p_logits.clone()
            num_points = len(cand_idx.points)
            
            # 如果候选点数量超过网络输出大小，截断
            if num_points > p_logits_masked.shape[0]:
                num_points = p_logits_masked.shape[0]
                point_mask = point_mask[:num_points]
            
            p_logits_masked[num_points:] = float('-inf')  # 超出部分设为-inf
            p_logits_masked[:num_points] = p_logits_masked[:num_points] + \
                torch.where(point_mask > 0, torch.zeros_like(point_mask), 
                           torch.full_like(point_mask, float('-inf')))
            
            # STOP logit
            stop_logit = network.forward_stop(feat).squeeze()
            stop_prob = self._compute_stop_prob(stop_logit, point_mask, k, max_k)
            
            # 合并点分布和STOP
            p_probs = F.softmax(p_logits_masked[:len(cand_idx.points)], dim=-1)
            probs_with_stop = torch.cat([p_probs * (1 - stop_prob), stop_prob.unsqueeze(0)])
            probs_with_stop = probs_with_stop / (probs_with_stop.sum() + 1e-8)
            
            # 采样点
            if greedy:
                choice_idx = torch.argmax(probs_with_stop).item()
            else:
                choice_idx = torch.multinomial(probs_with_stop + 1e-8, 1).item()
            
            # 检查STOP
            if choice_idx == len(p_probs):
                break
            
            p_idx = choice_idx
            
            # Step 2: 在选定的点上选类型
            t_logits = network.forward_type(feat, torch.tensor([p_idx], device=self.device))
            
            # 类型掩码 - 只处理该点实际可用的类型
            available_types = len(cand_idx.types_per_point[p_idx])
            t_logits_masked = t_logits[0, :available_types].clone()  # 只取可用的类型，并去掉batch维度
            
            # 应用类型掩码
            current_type_mask = type_masks[p_idx]
            t_logits_masked = t_logits_masked + \
                torch.where(current_type_mask > 0, torch.zeros_like(current_type_mask),
                           torch.full_like(current_type_mask, float('-inf')))
            
            t_probs = F.softmax(t_logits_masked, dim=-1)
            
            # 采样类型
            if greedy:
                t_idx = torch.argmax(t_probs).item()
            else:
                t_idx = torch.multinomial(t_probs + 1e-8, 1).item()
            
            action_type = cand_idx.types_per_point[p_idx][t_idx]
            
            # 记录
            selected_actions.append(AtomicAction(
                point=p_idx, 
                atype=action_type,
                meta={"action_id": action_type, "point_id": cand_idx.points[p_idx]}
            ))
            total_logprob += torch.log(probs_with_stop[p_idx] + 1e-8).item()
            total_logprob += torch.log(t_probs[t_idx] + 1e-8).item()
            total_entropy += -(p_probs * torch.log(p_probs + 1e-8)).sum().item()
            total_entropy += -(t_probs * torch.log(t_probs + 1e-8)).sum().item()
            
            # Step 3: 更新掩码
            point_mask[p_idx] = 0  # 禁用已选点
            self._update_masks_after_choice(p_idx, t_idx, point_mask, type_masks, 
                                           cand_idx, agent, state)
            
            # 检查是否还有可用点
            if point_mask.sum() == 0:
                break
        
        # 获取价值
        with torch.no_grad():
            value = self.critic_networks[agent](obs).squeeze().item()
        
        # 日志
        if topic_enabled("policy_select") and sampling_allows(agent, getattr(state, 'month', None), None):
            self.logger.info(
                f"policy_select_multi agent={agent} month={getattr(state, 'month', '?')} "
                f"num_actions={len(selected_actions)} logprob_sum={total_logprob:.4f} "
                f"entropy_sum={total_entropy:.4f} value={value:.3f}"
            )
        
        return {
            'sequence': Sequence(agent=agent, actions=selected_actions),
            'logprob': total_logprob,
            'entropy': total_entropy,
            'value': value
        }
    
    def _compute_stop_prob(self, stop_logit: torch.Tensor, point_mask: torch.Tensor, 
                          k: int, max_k: int) -> torch.Tensor:
        """计算STOP概率"""
        # 基础sigmoid
        stop_prob = torch.sigmoid(stop_logit)
        
        # 动态调整：已选越多，越倾向STOP
        if k > 0:
            decay_factor = self.config.get("multi_action", {}).get("stop_bias", 0.0)
            stop_prob = stop_prob + decay_factor * k
        
        # 配置偏置
        stop_bias = self.config.get("multi_action", {}).get("stop_bias", 0.0)
        stop_prob = torch.clamp(stop_prob + stop_bias, 0.0, 1.0)
        
        # 如果没有可用点，强制STOP
        if point_mask.sum() == 0:
            return torch.tensor(1.0, device=self.device)
        
        # 如果达到上限，强制STOP
        if k >= max_k:
            return torch.tensor(1.0, device=self.device)
        
        return stop_prob
    
    def _update_masks_after_choice(self, p_idx: int, t_idx: int, 
                                   point_mask: torch.Tensor, type_masks: List[torch.Tensor],
                                   cand_idx: CandidateIndex, agent: str, state: EnvironmentState):
        """选择(point, type)后更新掩码"""
        # 1. 禁用已选点（去重策略）
        dup_policy = self.config.get("multi_action", {}).get("dup_policy", "no_repeat_point")
        if dup_policy in ['no_repeat_point', 'both']:
            point_mask[p_idx] = 0
            
            # 槽位级别去重：禁用所有使用相同槽位坐标的点
            selected_point_id = cand_idx.points[p_idx]
            selected_slots = set(cand_idx.point_to_slots.get(selected_point_id, []))
            
            # 检查所有其他点，如果有槽位重叠则禁用
            for i in range(min(len(cand_idx.points), point_mask.shape[0])):
                if i != p_idx and point_mask[i] > 0:  # 只检查未禁用的点
                    point_id = cand_idx.points[i]
                    point_slots = set(cand_idx.point_to_slots.get(point_id, []))
                    # 如果有槽位重叠，禁用该点
                    if selected_slots & point_slots:
                        point_mask[i] = 0
        
        # 2. 更新预算约束（简化：不在这里更新，因为需要实际执行才知道）
        # 实际执行会检查预算，这里只是选择阶段
        
        # 3. 槽位冲突检查（简化版本）
        # 后续如果需要更复杂的约束，可以在这里添加
        pass
    
    def _prune_candidates(self, cand_idx: CandidateIndex, topP: int = 128) -> CandidateIndex:
        """裁剪候选（保留Top-P个点）"""
        if len(cand_idx.points) <= topP:
            return cand_idx
        
        # 简化实现：随机选择topP个点
        # 实际应用中可以基于启发式评分
        indices = np.random.choice(len(cand_idx.points), topP, replace=False)
        
        return CandidateIndex(
            points=[cand_idx.points[i] for i in indices],
            types_per_point=[cand_idx.types_per_point[i] for i in indices],
            point_to_slots={cand_idx.points[i]: cand_idx.point_to_slots[cand_idx.points[i]] 
                           for i in indices},
            meta=cand_idx.meta
        )
    
    def _encode_state(self, state: EnvironmentState) -> torch.Tensor:
        """
        编码状态为观察向量
        
        Args:
            state: 环境状态
            
        Returns:
            观察向量
        """
        # 简化实现：返回固定长度的观察向量
        obs = torch.zeros(self.obs_size)
        
        # 填充基础信息
        obs[0] = state.month
        obs[1] = len(state.buildings)
        obs[2] = len(state.slots)
        
        # 填充预算信息
        for i, (agent, budget) in enumerate(state.budgets.items()):
            if i < 3:  # 最多3个智能体
                obs[3 + i] = budget
        
        # 填充地价信息
        if state.land_prices is not None:
            flat_prices = state.land_prices.flatten()
            obs[6:18] = torch.from_numpy(flat_prices[:12]).float()
        
        return obs.to(self.device)
    
    def save_networks(self, path: str):
        """保存网络权重"""
        torch.save({
            'actor_networks': {agent: net.state_dict() for agent, net in self.actor_networks.items()},
            'critic_networks': {agent: net.state_dict() for agent, net in self.critic_networks.items()}
        }, path)
    
    def load_networks(self, path: str):
        """加载网络权重"""
        if not os.path.exists(path):
            print(f"网络权重文件不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载actor网络
        for agent, net in self.actor_networks.items():
            if agent in checkpoint['actor_networks']:
                net.load_state_dict(checkpoint['actor_networks'][agent])
        
        # 加载critic网络
        for agent, net in self.critic_networks.items():
            if agent in checkpoint['critic_networks']:
                net.load_state_dict(checkpoint['critic_networks'][agent])
        
        print(f"网络权重已从 {path} 加载")




