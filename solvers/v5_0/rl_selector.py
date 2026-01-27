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
        self.action_size = 12  # 动作空间大小（0-11）：EDU(0-2), IND(3-5,9-11), COUNCIL(6-8)
        
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
    
    def _encode_state(self, state: EnvironmentState) -> torch.Tensor:
        """
        编码环境状态为神经网络输入
        
        Args:
            state: 环境状态
            
        Returns:
            编码后的状态向量 [obs_size]
        """
        import numpy as np
        
        # 初始化特征向量
        obs = np.zeros(self.obs_size, dtype=np.float32)
        
        # 基础特征 (0-4)
        obs[0] = float(state.month) / 30.0  # 归一化月份
        obs[1] = float(len(state.buildings)) / 100.0  # 归一化建筑数量
        obs[2] = float(len(state.slots)) / 1000.0  # 归一化槽位数量
        
        # 预算特征 (3-5): 如果有预算信息，提取当前agent的预算（简化处理：取第一个agent或平均值）
        if state.budgets:
            budgets_list = list(state.budgets.values())
            obs[3] = float(sum(budgets_list)) / 100000.0  # 总预算归一化
            obs[4] = float(max(budgets_list)) / 100000.0  # 最大预算归一化
            obs[5] = float(min(budgets_list)) / 100000.0  # 最小预算归一化
        
        # 地价特征 (6-17): 从land_prices数组中提取
        if state.land_prices is not None and state.land_prices.size > 0:
            lp_flat = state.land_prices.flatten()
            lp_len = min(len(lp_flat), 12)
            obs[6:6+lp_len] = lp_flat[:lp_len].astype(np.float32)
            # 归一化地价
            if lp_flat.max() > 0:
                obs[6:6+lp_len] = obs[6:6+lp_len] / lp_flat.max()
        
        # 建筑特征 (18-29): 简化的建筑统计
        if state.buildings:
            building_counts = {}
            for b in state.buildings:
                action_id = b.get('action_id', 0)
                building_counts[action_id] = building_counts.get(action_id, 0) + 1
            for i, (aid, count) in enumerate(sorted(building_counts.items())[:12]):
                obs[18 + i] = float(count) / 10.0  # 归一化
        
        # 月度奖励特征 (30-35): 如果有月度奖励信息
        if state.monthly_rewards:
            rewards_list = list(state.monthly_rewards.values())
            if rewards_list:
                obs[30] = float(sum(rewards_list)) / 10000.0
                obs[31] = float(max(rewards_list)) / 10000.0
                obs[32] = float(min(rewards_list)) / 10000.0
        
        # 协同激活特征 (33-38)
        if state.synergy_activations:
            synergy_count = sum(1 for v in state.synergy_activations.values() if v)
            obs[33] = float(synergy_count) / 10.0
        
        # 填充剩余维度（使用随机或零填充）
        # obs[39:64] 保持为零或可以添加其他特征
        
        # 转换为torch.Tensor并移到设备
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        return obs_tensor

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
            self.logger.warning(f"[TYPE_EARLY_STOP] agent={agent} month={getattr(state,'month','?')} reason=no_candidates_single")
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
        """
        # 无候选早停：在进入主逻辑前给出明确诊断
        if (cand_idx is None) or (not getattr(cand_idx, 'points', None)):
            self.logger.warning(f"[TYPE_EARLY_STOP] agent={agent} month={getattr(state,'month','?')} reason=no_candidates_index")
            return None if not candidates else self.select_action(agent, candidates, state, greedy)
        # 入口横幅（确认路径命中与关键统计）
        try:
            month_val = getattr(state, 'month', '?')
            points_cnt = len(cand_idx.points) if cand_idx and cand_idx.points is not None else 0
            types_cnt = sum(len(t) for t in getattr(cand_idx, 'types_per_point', []) or [])
            self.logger.warning(
                f"[TYPE_BANNER] enter select_action_multi agent={agent} month={month_val} points={points_cnt} types_total={types_cnt}"
            )
        except Exception:
            pass
        
        # 提前返回检查前日志（强制输出）
        self.logger.warning(
            f"[TYPE_CHECK] agent={agent} multi_enabled={self.multi_action_enabled} "
            f"cand_idx={cand_idx is not None} points_len={len(cand_idx.points) if cand_idx and cand_idx.points else 0}"
        )
        
        if not self.multi_action_enabled or not cand_idx or len(cand_idx.points) == 0:
            # 降级到单动作
            self.logger.warning(f"[TYPE_FALLBACK] agent={agent} falling back to single action")
            return self.select_action(agent, candidates, state, greedy)
        
        # 编码器只执行一次
        self.logger.warning(f"[TYPE_NET_INIT] agent={agent} initializing network...")
        obs = self._encode_state(state)
        network = self.actor_networks_multi[agent]
        feat = network.encoder(obs)
        self.logger.warning(f"[TYPE_NET_READY] agent={agent} network ready")
        
        # 初始化掩码
        point_mask = torch.ones(len(cand_idx.points), device=self.device)
        type_masks = [torch.ones(len(types), device=self.device) 
                     for types in cand_idx.types_per_point]
        
        selected_actions = []
        action_logprobs = []  # 新增：保存每个动作的logprob
        total_logprob = 0.0
        total_entropy = 0.0
        stop_reason: Optional[str] = None
        
        # 修复：保存初始的point_mask和每个动作选择前的point_mask状态（用于训练时重建mask）
        initial_point_mask = point_mask.clone().detach().cpu()  # 初始mask（全1）
        point_masks_per_action = []  # 每个动作选择前的point_mask状态
        
        # 循环入口确认（强制输出）
        try:
            self.logger.warning(f"[TYPE_LOOP_START] agent={agent} month={getattr(state, 'month', '?')} max_k={max_k}")
        except Exception:
            pass
        
        for k in range(max_k):
            # 每轮迭代确认（不包裹 try/except）
            self.logger.warning(f"[TYPE_ITER] agent={agent} month={getattr(state,'month','?')} step={k}")
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
            
            # 合并点分布和STOP，应用点掩码
            temperature = self.config.get("multi_action", {}).get("temperature", 1.0)
            p_probs = F.softmax(p_logits_masked[:len(cand_idx.points)] / temperature, dim=-1)
            p_probs_masked = p_probs * point_mask[:len(p_probs)]
            probs_with_stop = torch.cat([p_probs_masked * (1 - stop_prob), stop_prob.unsqueeze(0)])
            probs_with_stop = probs_with_stop / (probs_with_stop.sum() + 1e-8)
            
            # 修复：为logprob计算准备纯点分布（不使用STOP影响）
            # 确保与训练时重建的logprob计算方式一致
            p_probs_for_logprob = p_probs_masked / (p_probs_masked.sum() + 1e-8) if p_probs_masked.sum() > 0 else p_probs_masked

            # 点分布摘要（移除 try/except，强制输出）
            valid_indices_tmp = torch.where(point_mask[:len(p_probs)] > 0)[0]
            top_prob = float((p_probs_masked[valid_indices_tmp].max().item()) if valid_indices_tmp.numel() > 0 else 0.0)
            self.logger.warning(
                f"[TYPE_POINT_DISTR] agent={agent} month={getattr(state,'month','?')} step={k} valid_points={int(valid_indices_tmp.numel())} top_point_prob={round(top_prob,4)} stop_prob={round(float(stop_prob.item()),4)}"
            )

            # 类型级可用性统计（移除 try/except，强制输出）
            available_counts: Dict[int, int] = {}
            valid_point_indices = torch.where(point_mask[:len(cand_idx.points)] > 0)[0].tolist()
            for i in valid_point_indices:
                tmask = type_masks[i]
                types_here = cand_idx.types_per_point[i]
                for j, aid in enumerate(types_here):
                    if j < len(tmask) and float(tmask[j].item()) > 0:
                        available_counts[aid] = available_counts.get(aid, 0) + 1
            self.logger.warning(
                f"[TYPE_COUNTS_BEFORE] agent={agent} month={getattr(state, 'month', '?')} step={k} available_by_type={dict(sorted(available_counts.items()))}"
            )
            
            # 采样点
            valid_indices = torch.where(point_mask[:len(p_probs)] > 0)[0]
            if len(valid_indices) == 0:
                choice_idx = len(p_probs)
                stop_reason = "no_valid_points"
            else:
                valid_probs = p_probs[valid_indices]
                valid_probs = valid_probs / valid_probs.sum()
                sampled_idx = torch.multinomial(valid_probs + 1e-8, 1).item()
                choice_idx = valid_indices[sampled_idx].item()
            
            # STOP 前确认（不包裹 try/except）
            self.logger.warning(
                f"[TYPE_BEFORE_STOP] agent={agent} month={getattr(state,'month','?')} step={k} choice_idx={int(choice_idx)} stop_index={int(len(p_probs))}"
            )
            
            # 如果STOP，输出早停信息然后退出
            if choice_idx == len(p_probs):
                self.logger.warning(
                    f"[TYPE_EARLY_STOP] agent={agent} month={getattr(state,'month','?')} step={k} reason=no_valid_point_or_high_stop"
                )
                if stop_reason is None:
                    stop_reason = "stop_probability"
                break
            
            p_idx = choice_idx
            
            # Step 2: 在选定的点上选类型
            t_logits = network.forward_type(feat, torch.tensor([p_idx], device=self.device))
            available_types = len(cand_idx.types_per_point[p_idx])
            t_logits_masked = t_logits[0, :available_types].clone()
            current_type_mask = type_masks[p_idx]
            t_logits_masked = t_logits_masked + \
                torch.where(current_type_mask > 0, torch.zeros_like(current_type_mask),
                           torch.full_like(current_type_mask, float('-inf')))
            t_probs = F.softmax(t_logits_masked, dim=-1)

            # 记录当前选定点上的类型分布（强制输出）
            types_here = cand_idx.types_per_point[p_idx]
            type_prob_map = {int(types_here[j]): float(t_probs[j].detach().cpu().item()) for j in range(min(len(types_here), len(t_probs)))}
            self.logger.warning(
                f"[TYPE_PROBS_POINT] agent={agent} month={getattr(state, 'month', '?')} step={k} point={p_idx} probs={type_prob_map}"
            )

            # EDU 专用：在同一点上，对可选类型做基础项的what-if估算（不执行，仅日志）
            if agent == "EDU":
                try:
                    # 配置提取
                    action_params = self.config.get("action_params", {})
                    rules = self.config.get("rules", {})
                    terms = rules.get("terms", [])
                    term_weights = {t.get("name"): float(t.get("weight", 0.0)) for t in terms}

                    def get_action_field(aid: int, key: str, default_val: float = 0.0) -> float:
                        p = action_params.get(str(aid), {})
                        return float(p.get(key, default_val)) if isinstance(p.get(key, default_val), (int, float)) else default_val

                    estimates = {}
                    for aid in types_here:
                        # 基础近似：仅使用参数表可直接获取的项（不包含地价/河流/邻近等空间项）
                        revenue = get_action_field(aid, "reward", 0.0) + get_action_field(aid, "rent", 0.0)
                        base_cost = get_action_field(aid, "cost", 0.0)
                        opex = get_action_field(aid, "opex", 0.0)
                        prestige = get_action_field(aid, "prestige", 0.0)

                        # 应用权重的“基础估计总分”（忽略空间项与复杂项）
                        total_est = 0.0
                        total_est += term_weights.get("revenue", 0.0) * revenue
                        total_est += term_weights.get("cost", 0.0) * base_cost  # cost权重通常为负
                        total_est += term_weights.get("prestige", 0.0) * prestige
                        # 将opex视作成本的一部分（未单独列项）
                        total_est += term_weights.get("cost", 0.0) * opex

                        estimates[int(aid)] = {
                            "revenue": round(revenue, 3),
                            "cost": round(base_cost, 3),
                            "opex": round(opex, 3),
                            "prestige": round(prestige, 3),
                            "weighted_total_est": round(total_est, 3)
                        }

                    self.logger.warning(
                        f"[TYPE_WHATIF_BASE] agent=EDU month={getattr(state,'month','?')} point={p_idx} whatif={estimates} weights={term_weights}"
                    )

                    # 预算上下文：输出EDU/COUNCIL当前预算与各类型成本
                    budgets = getattr(state, 'budgets', {}) or {}
                    edu_b = float(budgets.get('EDU', 0.0))
                    council_b = float(budgets.get('COUNCIL', 0.0))
                    type_costs = {int(aid): get_action_field(aid, "cost", 0.0) for aid in types_here}
                    self.logger.warning(
                        f"[LEDGER_WHATIF] month={getattr(state,'month','?')} EDU_budget={round(edu_b,2)} COUNCIL_budget={round(council_b,2)} type_costs={type_costs}"
                    )
                except Exception:
                    pass
            
            # 采样类型
            t_idx = torch.multinomial(t_probs + 1e-8, 1).item()
            action_type = cand_idx.types_per_point[p_idx][t_idx]
            
            # 记录
            point_id = cand_idx.points[p_idx]
            slots = cand_idx.point_to_slots.get(point_id, [])
            selected_actions.append(AtomicAction(
                point=p_idx, 
                atype=t_idx,  # 修复：存储类型索引，不是action_id
                meta={"action_id": action_type, "type_idx": t_idx, "point_id": point_id, "slots": slots, "available_types": cand_idx.types_per_point[p_idx]}
            ))
            # 修复：使用纯点分布的logprob，不使用probs_with_stop
            # 确保与训练时重建的logprob计算方式一致
            p_logprob = torch.log(p_probs_for_logprob[p_idx] + 1e-8).item()
            t_logprob = torch.log(t_probs[t_idx] + 1e-8).item()
            
            # 修复：保存该动作选择前的point_mask状态（用于训练时重建mask）
            point_masks_per_action.append(point_mask.clone().detach().cpu())
            
            # 记录累积前的值
            total_logprob_before = total_logprob
            total_logprob += p_logprob + t_logprob
            # 新增：保存该动作的logprob（点位头 + 类型头）
            action_logprob = p_logprob + t_logprob
            action_logprobs.append(action_logprob)
            
            # 🔍 调试日志：记录每个动作的logprob计算（前10步或前50轮）
            if k < 10 or getattr(state, 'month', 0) < 50:
                self.logger.warning(
                    f"[LOGPROB_COLLECT] agent={agent} month={getattr(state, 'month', '?')} step={k} "
                    f"action_idx={len(selected_actions)} "
                    f"p_idx={p_idx} p_logprob={p_logprob:.6f} "
                    f"t_idx={t_idx} t_logprob={t_logprob:.6f} "
                    f"total_logprob_before={total_logprob_before:.6f} "
                    f"total_logprob_after={total_logprob:.6f} "
                    f"num_actions={len(selected_actions)} "
                    f"p_prob={p_probs_for_logprob[p_idx]:.6f} "
                    f"t_prob={t_probs[t_idx]:.6f}"
                )
            total_entropy += -(p_probs * torch.log(p_probs + 1e-8)).sum().item()
            total_entropy += -(t_probs * torch.log(t_probs + 1e-8)).sum().item()
            
            # Step 3: 更新掩码
            point_mask[p_idx] = 0
            self._update_masks_after_choice(p_idx, t_idx, point_mask, type_masks, 
                                           cand_idx, agent, state)

            # 类型级可用性统计（采样后，强制输出）
            available_counts_after: Dict[int, int] = {}
            valid_point_indices_after = torch.where(point_mask[:len(cand_idx.points)] > 0)[0].tolist()
            for i in valid_point_indices_after:
                tmask = type_masks[i]
                types_here = cand_idx.types_per_point[i]
                for j, aid in enumerate(types_here):
                    if j < len(tmask) and float(tmask[j].item()) > 0:
                        available_counts_after[aid] = available_counts_after.get(aid, 0) + 1
            self.logger.warning(
                f"[TYPE_COUNTS_AFTER] agent={agent} month={getattr(state, 'month', '?')} step={k} available_by_type={dict(sorted(available_counts_after.items()))}"
            )
            
            if point_mask.sum() == 0:
                if stop_reason is None:
                    stop_reason = "mask_exhausted"
                break
        
        with torch.no_grad():
            value = self.critic_networks[agent](obs).squeeze().item()
        
        if stop_reason is None:
            stop_reason = "max_k" if len(selected_actions) >= max_k else "loop_completed"
        
        if topic_enabled("policy_select") or topic_enabled("policy_sequence"):
            seq_summary = [
                (
                    a.meta.get("action_id", a.meta.get("legacy_id", a.atype)),
                    a.meta.get("slots", [])
                ) for a in selected_actions
            ]
            self.logger.warning(
                f"[POLICY_SEQUENCE] agent={agent} actions={seq_summary} "
                f"count={len(selected_actions)} stop_reason={stop_reason} value={value:.3f}"
            )
        
        if topic_enabled("policy_select") and sampling_allows(agent, getattr(state, 'month', None), None):
            self.logger.info(
                f"policy_select_multi agent={agent} month={getattr(state, 'month', '?')} "
                f"num_actions={len(selected_actions)} logprob_sum={total_logprob:.4f} "
                f"entropy_sum={total_entropy:.4f} value={value:.3f}"
            )
        
        # 🔍 调试日志：确认logprob存储（前50轮）
        # 🔍 验证：最终total_logprob是否合理
        if getattr(state, 'month', 0) < 50:
            # 估算期望值：每个动作logprob约-4到-5，总和应该在 len(actions) * (-4到-5) 之间
            num_actions = len(selected_actions)
            expected_min = num_actions * -5.5  # 保守估计（每个动作约-5.5）
            expected_max = num_actions * -3.5   # 宽松估计（每个动作约-3.5）
            
            # 检查是否异常
            is_abnormal = False
            if num_actions > 0:
                if total_logprob < expected_min or total_logprob > expected_max:
                    is_abnormal = True
                    self.logger.error(
                        f"[LOGPROB_ABNORMAL] agent={agent} month={getattr(state, 'month', '?')} "
                        f"num_actions={num_actions} total_logprob={total_logprob:.6f} "
                        f"expected_range=[{expected_min:.1f}, {expected_max:.1f}] "
                        f"⚠️ total_logprob异常！可能是累积不完整或计算错误"
                    )
            
            # 正常日志
            log_level = "error" if is_abnormal else "warning"
            log_msg = (
                f"[LOGPROB_STORE] agent={agent} month={getattr(state, 'month', '?')} "
                f"num_actions={num_actions} "
                f"total_logprob={total_logprob:.6f} "
                f"stored_to_sel['logprob']={total_logprob:.6f} "
                f"expected_range=[{expected_min:.1f}, {expected_max:.1f}]"
            )
            
            if log_level == "error":
                self.logger.error(log_msg)
            else:
                self.logger.warning(log_msg)
        
        # 生成可追踪的trace_id，便于与训练阶段日志关联
        trace_id = f"{agent}-{getattr(state, 'month', '?')}-{k}"
        
        # 🔍 追踪日志：记录select_action_multi返回的logprob（前50轮）
        if getattr(state, 'month', 0) < 50:
            self.logger.warning(
                f"[SEL_RETURN] agent={agent} trace_id={trace_id} "
                f"returning_logprob={total_logprob:.6f} "
                f"num_actions={len(selected_actions)} "
                f"sequence_actions_count={len(selected_actions)}"
            )
        
        return {
            'sequence': Sequence(agent=agent, actions=selected_actions),
            'logprob': total_logprob,  # 保留：用于GAE和向后兼容
            'action_logprobs': action_logprobs,  # 新增：每个动作的logprob列表
            'entropy': total_entropy,
            'value': value,
            'trace_id': trace_id,
            'month': getattr(state, 'month', None),
            # 修复：返回mask信息（用于训练时重建mask）
            'initial_point_mask': initial_point_mask,  # 初始point_mask（全1）
            'point_masks_per_action': point_masks_per_action,  # 每个动作选择前的point_mask状态
            'cand_idx_points_count': len(cand_idx.points),  # 点的数量
        }
    
    def _compute_stop_prob(self, stop_logit: torch.Tensor, point_mask: torch.Tensor, 
                          k: int, max_k: int) -> torch.Tensor:
        """计算STOP概率"""
        stop_prob = torch.sigmoid(stop_logit)
        if k > 0:
            decay_factor = self.config.get("multi_action", {}).get("stop_bias", 0.0)
            stop_prob = stop_prob + decay_factor * k
        stop_bias = self.config.get("multi_action", {}).get("stop_bias", 0.0)
        stop_prob = torch.clamp(stop_prob + stop_bias, 0.0, 1.0)
        if point_mask.sum() == 0:
            return torch.tensor(1.0, device=self.device)
        if k >= max_k:
            return torch.tensor(1.0, device=self.device)
        return stop_prob
    
    def _update_masks_after_choice(self, p_idx: int, t_idx: int, 
                                   point_mask: torch.Tensor, type_masks: List[torch.Tensor],
                                   cand_idx: CandidateIndex, agent: str, state: EnvironmentState):
        """选择(point, type)后更新掩码"""
        dup_policy = self.config.get("multi_action", {}).get("dup_policy", "no_repeat_point")
        if dup_policy in ['no_repeat_point', 'both']:
            point_mask[p_idx] = 0
            selected_point_id = cand_idx.points[p_idx]
            selected_slots = set(cand_idx.point_to_slots.get(selected_point_id, []))
            for i in range(min(len(cand_idx.points), point_mask.shape[0])):
                if i != p_idx and point_mask[i] > 0:
                    point_id = cand_idx.points[i]
                    point_slots = set(cand_idx.point_to_slots.get(point_id, []))
                    if selected_slots & point_slots:
                        point_mask[i] = 0
        
        # 其他约束略
        pass





