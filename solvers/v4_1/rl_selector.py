"""
v4.1 RL选择器 - PPO/MAPPO策略选择器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Set, Optional
from logic.v4_enumeration import Action, ActionEnumerator, ActionScorer, Sequence, SequenceSelector


class Actor(nn.Module):
    """完全独立的策略网络 - 只负责动作选择"""
    
    def __init__(self, state_dim: int = 512, hidden_dim: int = 256, max_actions: int = 50):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions),  # 输出每个动作的logit
        )
        
        self.max_actions = max_actions
        self.temperature = 1.0  # 温度参数，控制概率分布锐度
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # 按照1013-9.md建议：重初始化最后一层（提高gain）
        torch.nn.init.orthogonal_(self.network[-1].weight, gain=0.5)
        torch.nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, state_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embed: [B, state_dim]
        Returns:
            logits: [B, max_actions] - 每个动作的logit（应用温度参数）
        """
        raw_logits = self.network(state_embed)
        logits = raw_logits / self.temperature  # 应用温度参数
        
        # 添加额外的正则化：将logits向0收缩
        logits = logits * 0.8  # 进一步缩小logits
        
        logits = torch.clamp(logits, min=-8.0, max=8.0)  # 限制logits范围
        return logits


class Critic(nn.Module):
    """完全独立的价值网络 - 只负责状态价值估计"""
    
    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 输出状态价值
        )
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embed: [B, state_dim]
        Returns:
            value: [B, 1] - 状态价值
        """
        return self.network(state_embed)


class RLPolicySelector:
    """RL策略选择器 - 使用PPO/MAPPO模型进行动作选择"""
    
    def __init__(self, cfg: Dict, model_path: Optional[str] = None, slots: Optional[Dict] = None):
        self.cfg = cfg
        self.rl_cfg = cfg['solver']['rl']
        self.slots = slots  # 保存槽位信息用于获取building_level
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 【MAPPO】为每个agent创建独立的Actor和Critic网络
        self.state_dim = 512  # 简化状态维度
        self.max_actions = 50  # 最大动作数量
        
        agents = self.rl_cfg.get('agents', ['IND', 'EDU'])
        
        # 独立的Actor网络（每个agent一个）
        self.actors = {}
        for agent in agents:
            actor = Actor(state_dim=self.state_dim, max_actions=self.max_actions).to(self.device)
            actor.temperature = self.rl_cfg.get('temperature', 3.0)  # 激进提升到3.0增加动作多样性
            self.actors[agent] = actor
        
        # 独立的Critic网络（每个agent一个）
        self.critics = {}
        for agent in agents:
            critic = Critic(state_dim=self.state_dim).to(self.device)
            self.critics[agent] = critic
        
        # 独立的优化器
        actor_lr = self.rl_cfg.get('actor_lr', 3e-4)
        critic_lr = self.rl_cfg.get('critic_lr', 3e-4)
        
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        for agent in agents:
            # === 按照1013-9.md建议：给最后一层单独设置更高学习率 ===
            actor = self.actors[agent]
            base_params = []
            last_layer_params = []
            
            for name, param in actor.named_parameters():
                if "network.2" in name:  # 最后一层
                    last_layer_params.append(param)
                else:
                    base_params.append(param)
            
            self.actor_optimizers[agent] = torch.optim.Adam([
                {"params": base_params, "lr": actor_lr, "weight_decay": 0.0},
                {"params": last_layer_params, "lr": actor_lr * 3, "weight_decay": 0.0},  # 头部更大一点
            ])
            self.critic_optimizers[agent] = torch.optim.Adam(
                self.critics[agent].parameters(), 
                lr=critic_lr
            )
        
        # 保持向后兼容的接口（使用第一个agent的）
        self.actor = self.actors[agents[0]]
        self.critic = self.critics[agents[0]]
        self.actor_optimizer = self.actor_optimizers[agents[0]]
        self.critic_optimizer = self.critic_optimizers[agents[0]]
        self.optimizer = self.actor_optimizer
        
        # 探索参数 - 激进配置快速见效
        self.epsilon = 0.8  # ε-贪婪探索 (激进提升到0.8)
        self.epsilon_decay = 0.99  # 探索衰减率（快速衰减）
        self.min_epsilon = 0.3  # 最小探索率（保持高探索率）
        self.high_level_epsilon = 0.9  # 高等级槽位的额外探索率
        
        # 保留枚举器和打分器用于生成动作池和特征
        self.enumerator = None
        self.scorer = None
        self.sequence_selector = None
        
        # 初始化扩展策略
        self._init_expansion_policy()
        
        # 初始化ActionScorer（立即初始化，避免延迟加载问题）
        enum_cfg = self.cfg.get('growth_v4_1', {}).get('enumeration', {})
        obj = enum_cfg.get('objective', {})
        objective = {
            'EDU': obj.get('EDU', {'w_r': 0.3, 'w_p': 0.6, 'w_c': 0.1}),
            'IND': obj.get('IND', {'w_r': 0.6, 'w_p': 0.2, 'w_c': 0.2}),
        }
        normalize = str(obj.get('normalize', 'per-month-pool-minmax'))
        eval_params = self.cfg.get('growth_v4_1', {}).get('evaluation', {})
        
        try:
            from logic.v4_enumeration import ActionScorer
            self.scorer = ActionScorer(objective, normalize, eval_params=eval_params, slots=None)
            print("ActionScorer已在__init__中初始化")
        except Exception as e:
            print(f"ActionScorer初始化失败: {e}")
            self.scorer = None
        
        if model_path:
            self.load_model(model_path)
    
    def _init_expansion_policy(self):
        """初始化扩展策略"""
        try:
            from expansion_policy import create_expansion_policy
            
            # 从配置中获取扩展策略设置
            expansion_config = self.rl_cfg.get('expansion_policy', {
                'type': 'nearest_k',
                'temperature': 1.0,
                'rule': 'euclidean',
                'k': 5
            })
            
            # 创建扩展策略实例
            self.expansion_policy = create_expansion_policy(expansion_config)
            self.expansion_k = expansion_config.get('k', 5)
            
            print(f"Initialized expansion policy: {expansion_config['type']} (k={self.expansion_k})")
            
        except ImportError as e:
            print(f"Warning: Could not import expansion_policy: {e}")
            self.expansion_policy = None
            self.expansion_k = 5
        except Exception as e:
            print(f"Warning: Failed to initialize expansion policy: {e}")
            self.expansion_policy = None
            self.expansion_k = 5
    
    def choose_action_sequence(
        self,
        slots: Dict,
        candidates: Set[str],
        occupied: Set[str],
        lp_provider,
        river_distance_provider=None,
        agent_types: Optional[List[str]] = None,
        sizes: Optional[Dict[str, List[str]]] = None,
        buildings: Optional[List[Dict]] = None,
    ) -> Tuple[List[Action], Optional[Sequence]]:
        """使用RL策略选择动作序列（恢复v4.0的序列机制）"""
        
        # 保存当前槽位信息，供扩展策略使用
        self._current_slots = slots
        
        # 设置slots到scorer（用于邻近性奖励计算）
        if self.scorer and self.scorer.slots is None:
            self.scorer.slots = slots
        
        # 1. 枚举合法动作池
        if self.enumerator is None:
            self.enumerator = ActionEnumerator(slots)
        
        actual_sizes = sizes or {'EDU': ['S', 'M', 'L', 'A', 'B', 'C'], 'IND': ['S', 'M', 'L']}
        actual_agent_types = agent_types or self.rl_cfg['agents']
        print(f"[DEBUG] Actual sizes parameter: {actual_sizes}")
        print(f"[DEBUG] Actual agent_types parameter: {actual_agent_types}")
        actions = self.enumerator.enumerate_actions(
            candidates=candidates,
            occupied=occupied,
            agent_types=actual_agent_types,
            sizes=actual_sizes,
            lp_provider=lp_provider,
            adjacency='4-neighbor',
            caps=self.cfg.get('growth_v4_1', {}).get('enumeration', {}).get('caps', {}),
        )
        
        if not actions:
            return [], None
        
        # 调试：记录过滤前的A/B/C动作数量
        abc_before = [a for a in actions if a.size in ['A', 'B', 'C']]
        print(f"[DEBUG] Before filtering: A/B/C actions = {len(abc_before)}")
        
        # 1.5. 激进限制S型建筑数量以强制平衡动作池
        actions = self._limit_s_size_actions(actions, max_s_ratio=0.3)  # 从0.5降到0.3
        
        # 调试：记录S型建筑限制后的A/B/C动作数量
        abc_after_s_limit = [a for a in actions if a.size in ['A', 'B', 'C']]
        print(f"[DEBUG] After S limit: A/B/C actions = {len(abc_after_s_limit)}")
        
        # 1.6. 高等级槽位优先选择M/L型建筑
        actions = self._prioritize_high_level_slots(actions)
        
        # 调试：记录高等级槽位优先后的A/B/C动作数量
        abc_after_prioritize = [a for a in actions if a.size in ['A', 'B', 'C']]
        print(f"[DEBUG] After prioritize: A/B/C actions = {len(abc_after_prioritize)}")
        
        # 2. 计算动作得分（ActionScorer已在__init__中初始化）
        if self.scorer is None:
            print("警告: ActionScorer未初始化，跳过动作打分")
            return [], None
        
        # 计算动作得分
        actions = self.scorer.score_actions(actions, river_distance_provider, buildings=buildings)
        
        # 调试：记录ActionScorer后的A/B/C动作数量
        abc_after_scorer = [a for a in actions if a.size in ['A', 'B', 'C']]
        print(f"[DEBUG] After scorer: A/B/C actions = {len(abc_after_scorer)}")
        
        # 2.5. 给M/L型建筑添加探索奖励
        actions = self._add_exploration_bonus(actions)
        
        # 调试：记录动作池分布
        size_counts = {'S': 0, 'M': 0, 'L': 0, 'A': 0, 'B': 0, 'C': 0}
        for action in actions:
            if action.size in size_counts:
                size_counts[action.size] += 1
        print(f"Action pool distribution: S={size_counts['S']}, M={size_counts['M']}, L={size_counts['L']}, A={size_counts['A']}, B={size_counts['B']}, C={size_counts['C']}, Total={len(actions)}")
        
        # 调试：记录A/B/C动作的得分情况
        abc_actions = [a for a in actions if a.size in ['A', 'B', 'C']]
        if abc_actions:
            print(f"A/B/C actions count: {len(abc_actions)}")
            for action in abc_actions[:3]:  # 显示前3个A/B/C动作
                print(f"  {action.agent}_{action.size}: score={action.score:.3f}, cost={action.cost:.1f}, reward={action.reward:.1f}")
        else:
            print("WARNING: No A/B/C actions found!")
        
        # 3. 初始化序列选择器
        if self.sequence_selector is None:
            enum_cfg = self.cfg.get('growth_v4_1', {}).get('enumeration', {})
            length_max = int(enum_cfg.get('length_max', 5))
            beam_width = int(enum_cfg.get('beam_width', 16))
            max_expansions = int(enum_cfg.get('max_expansions', 2000))
            self.sequence_selector = SequenceSelector(length_max, beam_width, max_expansions)
        
        # 4. 过滤动作（应用环境约束）
        # 这里需要从环境获取约束检查函数，暂时跳过
        # TODO: 添加动作过滤逻辑
        
        # 5. 使用RL策略选择序列
        best_sequence, action_idx = self._rl_choose_sequence(actions)
        
        return actions, best_sequence
    
    def _rl_choose_sequence(self, actions: List[Action]) -> Tuple[Optional[Sequence], int]:
        """使用RL策略选择锚点槽位，然后通过扩展策略生成多槽位序列"""
        if not actions:
            return None, -1
        
        # 【MAPPO】获取当前agent，选择对应的网络
        current_agent = actions[0].agent if actions else 'IND'
        actor = self.actors.get(current_agent, self.actor)
        
        # 限制动作数量
        num_actions = min(len(actions), self.max_actions)
        action_subset = actions[:num_actions]
        
        # 生成状态编码
        state_embed = self._encode_state_for_rl(action_subset)
        
        # 应用归一化（与训练时保持一致）
        state_embed = self._normalize_state_embed(state_embed)
        
        # 初始化变量
        subset_indices = None
        cached_state_embed = None
        
        # ε-贪婪探索
        if np.random.random() < self.epsilon:
            # 探索：随机选择一个动作作为锚点
            selected_idx = np.random.randint(0, num_actions)
            selected_action = action_subset[selected_idx]
            # 探索时不需要log_prob，设为0
            old_log_prob = torch.tensor(0.0, device=self.device)
            # 探索时也设置基本的局部分布语境
            subset_indices = torch.tensor(list(range(num_actions)), device=self.device, dtype=torch.long)
            cached_state_embed = state_embed.detach().clone()
        else:
            # 利用：使用该agent的策略网络选择锚点动作
            with torch.no_grad():
                # 1) 前向，与训练保持一致
                logits = actor(state_embed)  # shape [1, A]

                # 2) 局部动作子集（当前有效动作列表与顺序）
                num_actions = len(action_subset)                       # 子集大小 K
                # 假设action_subset中的动作有某种标识，这里用索引作为全局ID
                subset_indices = torch.tensor(
                    list(range(num_actions)),  # 如果没有global_id，使用局部索引
                    device=self.device, dtype=torch.long
                )

                # 3) 局部 logits（严格按子集顺序）
                valid_logits = logits[0, :num_actions]                # 配合 num_actions 的持久化
                dist = torch.distributions.Categorical(logits=valid_logits)

                # 4) 采样 + 局部索引（0..K-1）
                selected_idx = dist.sample().item()                   # 局部索引
                selected_action = action_subset[selected_idx]

                # 5) old_log_prob（采样时的局部分布 + 局部索引）
                old_log_prob = dist.log_prob(
                    torch.tensor(selected_idx, device=self.device)
                ).detach()

                # 6) 缓存用于重放的一致前向输入
                cached_state_embed = state_embed.detach().clone()
        
        if selected_action:
            # 使用扩展策略生成多槽位序列，传递局部分布语境
            expanded_sequence = self._expand_anchor_to_sequence(
                selected_action, actions, selected_idx, old_log_prob, 
                num_actions, subset_indices, cached_state_embed
            )
            if expanded_sequence:
                return expanded_sequence, selected_idx
        
            return None, -1
    
    def _compute_anchor_log_prob(self, sequence: Sequence, actions: List[Action]) -> torch.Tensor:
        """计算锚点选择的log概率"""
        if not sequence or not hasattr(sequence, 'action_index'):
            return torch.tensor(0.0)
        
        try:
            # 获取锚点动作索引
            anchor_idx = sequence.action_index
            
            # 限制动作数量
            num_actions = min(len(actions), self.max_actions)
            action_subset = actions[:num_actions]
            
            # 生成状态编码
            state_embed = self._encode_state_for_rl(action_subset)
            
            # 【MAPPO】获取当前agent的网络
            current_agent = actions[0].agent if actions else 'IND'
            actor = self.actors.get(current_agent, self.actor)
            
            # 使用该agent的策略网络计算log概率
            with torch.no_grad():
                logits = actor(state_embed)
                valid_logits = logits[0, :num_actions]
                
                # 创建分布并计算log概率
                import torch.nn.functional as F
                action_probs = F.softmax(valid_logits, dim=-1)
                
                # 确保索引在有效范围内（与训练时保持一致）
                valid_anchor_idx = min(anchor_idx, len(action_probs) - 1)
                if 0 <= valid_anchor_idx < len(action_probs):
                    log_prob = torch.log(action_probs[valid_anchor_idx] + 1e-8)
                    return log_prob
                else:
                    return torch.tensor(0.0)
                    
        except Exception as e:
            print(f"Warning: Failed to compute anchor log prob: {e}")
            return torch.tensor(0.0)
    
    def _expand_anchor_to_sequence(self, anchor_action: Action, all_actions: List[Action], anchor_idx: int, 
                                 old_log_prob: torch.Tensor = None, num_actions: int = None, 
                                 subset_indices: torch.Tensor = None, cached_state_embed: torch.Tensor = None) -> Optional[Sequence]:
        """将锚点动作扩展为多槽位序列
        
        注意：不要覆盖ActionScorer计算的score，保持与训练系统的一致性
        """
        """将锚点动作扩展为多槽位序列"""
        if not hasattr(self, 'expansion_policy'):
            # 如果没有设置扩展策略，回退到单动作序列
            from logic.v4_enumeration import Sequence
            sequence = Sequence(
                actions=[anchor_action],
                sum_cost=anchor_action.cost,
                sum_reward=anchor_action.reward,
                sum_prestige=anchor_action.prestige,
                score=anchor_action.score
            )
            sequence.action_index = anchor_idx
            # 保存采样时的log_prob和局部分布语境
            if old_log_prob is not None:
                sequence.old_log_prob = old_log_prob
            if num_actions is not None:
                sequence.num_actions = num_actions
            if subset_indices is not None:
                sequence.subset_indices = subset_indices
            if cached_state_embed is not None:
                sequence.cached_state_embed = cached_state_embed
            return sequence
        
        try:
            # 获取锚点槽位ID
            anchor_slot_id = anchor_action.footprint_slots[0] if anchor_action.footprint_slots else None
            if not anchor_slot_id:
                # 如果锚点动作没有槽位，回退到单动作序列
                from logic.v4_enumeration import Sequence
                sequence = Sequence(
                    actions=[anchor_action],
                    sum_cost=anchor_action.cost,
                    sum_reward=anchor_action.reward,
                    sum_prestige=anchor_action.prestige,
                    score=anchor_action.score
                )
                sequence.action_index = anchor_idx
                # 保存采样时的log_prob和局部分布语境
                if old_log_prob is not None:
                    sequence.old_log_prob = old_log_prob
                if num_actions is not None:
                    sequence.num_actions = num_actions
                if subset_indices is not None:
                    sequence.subset_indices = subset_indices
                if cached_state_embed is not None:
                    sequence.cached_state_embed = cached_state_embed
                return sequence
            
            # 准备扩展策略的状态信息
            expansion_state = {
                'slots': getattr(self, '_current_slots', {}),
                'actions': all_actions,
                'anchor_action': anchor_action
            }
            
            # 获取可用槽位列表
            available_slots = []
            for action in all_actions:
                if action.footprint_slots:
                    for slot_id in action.footprint_slots:
                        if slot_id not in available_slots:
                            available_slots.append(slot_id)
            
            # 使用扩展策略生成多槽位序列
            k = getattr(self, 'expansion_k', 5)  # 默认扩展5个槽位
            expanded_slot_ids, expansion_log_prob = self.expansion_policy.expand(
                expansion_state, anchor_slot_id, available_slots, k=k
            )
            
            # 根据扩展后的槽位ID创建对应的动作
            expanded_actions = []
            total_cost = 0.0
            total_reward = 0.0
            total_prestige = 0.0
            total_score = 0.0
            
            for slot_id in expanded_slot_ids:
                # 找到对应槽位的动作（优先使用锚点动作的类型和参数）
                matching_action = None
                for action in all_actions:
                    if action.footprint_slots and slot_id in action.footprint_slots:
                        matching_action = action
                        break
                
                if matching_action:
                    # 创建新的动作，保持锚点动作的类型和参数，但使用扩展的槽位
                    from logic.v4_enumeration import Action
                    expanded_action = Action(
                        agent=matching_action.agent,
                        size=matching_action.size,
                        footprint_slots=[slot_id],  # 只包含当前槽位
                        zone=matching_action.zone,
                        LP_norm=matching_action.LP_norm,
                        adjacency=matching_action.adjacency,
                        cost=matching_action.cost / len(expanded_slot_ids),  # 平均分配成本
                        reward=matching_action.reward / len(expanded_slot_ids),  # 平均分配奖励
                        prestige=matching_action.prestige / len(expanded_slot_ids)  # 平均分配声望
                    )
                    # 重新计算分数
                    # 保留原始ActionScorer计算的score（按比例分配）
                    expanded_action.score = anchor_action.score / len(expanded_slot_ids)
                    
                    expanded_actions.append(expanded_action)
                    total_cost += expanded_action.cost
                    total_reward += expanded_action.reward
                    total_prestige += expanded_action.prestige
                    total_score += expanded_action.score  # 使用ActionScorer的score
                else:
                    # 如果找不到匹配的动作，创建一个基于锚点动作的新动作
                    from logic.v4_enumeration import Action
                    expanded_action = Action(
                        agent=anchor_action.agent,
                        size=anchor_action.size,
                        footprint_slots=[slot_id],
                        zone=anchor_action.zone,
                        LP_norm=anchor_action.LP_norm,
                        adjacency=anchor_action.adjacency,
                        cost=anchor_action.cost / len(expanded_slot_ids),
                        reward=anchor_action.reward / len(expanded_slot_ids),
                        prestige=anchor_action.prestige / len(expanded_slot_ids)
                    )
                    # 保留原始ActionScorer计算的score（按比例分配）
                    expanded_action.score = anchor_action.score / len(expanded_slot_ids)
                    
                    expanded_actions.append(expanded_action)
                    total_cost += expanded_action.cost
                    total_reward += expanded_action.reward
                    total_prestige += expanded_action.prestige
                    total_score += expanded_action.score  # 使用ActionScorer的score
            
            # 创建扩展后的序列
            from logic.v4_enumeration import Sequence
            expanded_sequence = Sequence(
                actions=expanded_actions,
                sum_cost=total_cost,
                sum_reward=total_reward,
                sum_prestige=total_prestige,
                score=total_score
            )
            
            # 手动设置action_index属性
            expanded_sequence.action_index = anchor_idx
            # 保存采样时的log_prob和局部分布语境
            if old_log_prob is not None:
                expanded_sequence.old_log_prob = old_log_prob
            if num_actions is not None:
                expanded_sequence.num_actions = num_actions
            if subset_indices is not None:
                expanded_sequence.subset_indices = subset_indices
            if cached_state_embed is not None:
                expanded_sequence.cached_state_embed = cached_state_embed
            
            # 存储扩展信息（用于调试）
            expanded_sequence.expansion_log_prob = expansion_log_prob
            expanded_sequence.anchor_slot_id = anchor_slot_id
            expanded_sequence.expanded_slot_ids = expanded_slot_ids
            
            return expanded_sequence
            
        except Exception as e:
            print(f"Warning: Expansion failed: {e}, falling back to single action")
            # 扩展失败时回退到单动作序列
            from logic.v4_enumeration import Sequence
            sequence = Sequence(
                actions=[anchor_action],
                sum_cost=anchor_action.cost,
                sum_reward=anchor_action.reward,
                sum_prestige=anchor_action.prestige,
                score=anchor_action.score
            )
            sequence.action_index = anchor_idx
            # 保存采样时的log_prob和局部分布语境
            if old_log_prob is not None:
                sequence.old_log_prob = old_log_prob
            if num_actions is not None:
                sequence.num_actions = num_actions
            if subset_indices is not None:
                sequence.subset_indices = subset_indices
            if cached_state_embed is not None:
                sequence.cached_state_embed = cached_state_embed
            return sequence
    
    def _encode_state_for_rl(self, actions: List[Action]) -> torch.Tensor:
        """为RL生成增强的状态编码"""
        # 增强状态编码：包含更多变化的信息
        if not actions:
            return torch.zeros(1, self.state_dim, device=self.device)
        
        # 计算动作池的统计特征
        scores = [action.score for action in actions if action.score is not None]
        costs = [action.cost for action in actions if action.cost is not None]
        rewards = [action.reward for action in actions if action.reward is not None]
        prestiges = [action.prestige for action in actions if action.prestige is not None]
        
        # 构建增强特征向量
        features = []
        
        # 1. 基本统计特征
        features.append(len(actions))  # 动作数量
        features.append(np.mean(scores) if scores else 0.0)  # 平均得分
        features.append(np.std(scores) if scores else 0.0)   # 得分标准差
        features.append(np.mean(costs) if costs else 0.0)    # 平均成本
        features.append(np.mean(rewards) if rewards else 0.0)  # 平均奖励
        features.append(np.mean(prestiges) if prestiges else 0.0)  # 平均声望
        
        # 2. 添加更多变化特征
        features.append(np.max(scores) if scores else 0.0)   # 最高得分
        features.append(np.min(scores) if scores else 0.0)   # 最低得分
        features.append(np.max(costs) if costs else 0.0)     # 最高成本
        features.append(np.min(costs) if costs else 0.0)     # 最低成本
        
        # 3. 添加随机噪声以增加变化（临时解决方案）
        import time
        import random
        random.seed(int(time.time() * 1000) % 10000)
        features.append(random.random())  # 随机特征1
        features.append(random.random())  # 随机特征2
        features.append(random.random())  # 随机特征3
        
        # 4. 动作多样性特征
        if len(actions) > 1:
            # 计算动作得分的变异系数
            score_cv = np.std(scores) / (np.mean(scores) + 1e-8) if scores and np.mean(scores) > 0 else 0.0
            features.append(score_cv)
            
            # 计算动作成本的变异系数
            cost_cv = np.std(costs) / (np.mean(costs) + 1e-8) if costs and np.mean(costs) > 0 else 0.0
            features.append(cost_cv)
        else:
            features.extend([0.0, 0.0])
        
        # 扩展到固定维度
        while len(features) < self.state_dim:
            features.append(0.0)
        
        features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _normalize_state_embed(self, state_embed: torch.Tensor) -> torch.Tensor:
        """状态编码归一化（与训练时保持一致）"""
        if not hasattr(self, '_embed_mean') or not hasattr(self, '_embed_std'):
            # 如果没有running stats，使用当前batch的stats
            mean = state_embed.mean()
            std = state_embed.std()
        else:
            mean = self._embed_mean
            std = self._embed_std
        
        normalized = (state_embed - mean) / (std + 1e-5)
        return normalized.clamp(-5, 5)
    
    def save_model(self, path: str):
        """保存模型权重（MAPPO：保存所有agent的网络）"""
        import os
        import datetime
        import json
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 【MAPPO】保存所有agent的网络
        model_data = {
            'model_version': 'v4.1_mappo',
            'timestamp': datetime.datetime.now().isoformat(),
            'rl_config': self.rl_cfg,
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'max_actions': self.max_actions,
        }
        
        # 保存各agent的网络
        for agent in self.actors.keys():
            model_data[f'actor_{agent}_state_dict'] = self.actors[agent].state_dict()
            model_data[f'critic_{agent}_state_dict'] = self.critics[agent].state_dict()
            model_data[f'actor_{agent}_optimizer_state'] = self.actor_optimizers[agent].state_dict()
            model_data[f'critic_{agent}_optimizer_state'] = self.critic_optimizers[agent].state_dict()
            model_data[f'actor_{agent}_temperature'] = self.actors[agent].temperature
        
        torch.save(model_data, path)
        print(f"MAPPO模型权重已保存到: {path} (包含{len(self.actors)}个agent的网络)")
    
    def load_model(self, path: str):
        """加载模型权重（MAPPO：加载所有agent的网络）"""
        if os.path.exists(path):
            model_data = torch.load(path, map_location=self.device)
            
            model_version = model_data.get('model_version', 'unknown')
            
            # 【MAPPO】检查是否是MAPPO模型
            if 'v4.1_mappo' in model_version or any(f'actor_{ag}_state_dict' in model_data for ag in self.actors.keys()):
                # 加载MAPPO模型（多个agent）
                for agent in self.actors.keys():
                    if f'actor_{agent}_state_dict' in model_data:
                        self.actors[agent].load_state_dict(model_data[f'actor_{agent}_state_dict'])
                        self.critics[agent].load_state_dict(model_data[f'critic_{agent}_state_dict'])
                        
                        if f'actor_{agent}_optimizer_state' in model_data:
                            self.actor_optimizers[agent].load_state_dict(model_data[f'actor_{agent}_optimizer_state'])
                        if f'critic_{agent}_optimizer_state' in model_data:
                            self.critic_optimizers[agent].load_state_dict(model_data[f'critic_{agent}_optimizer_state'])
                        
                        if f'actor_{agent}_temperature' in model_data:
                            self.actors[agent].temperature = model_data[f'actor_{agent}_temperature']
                
                print(f"MAPPO模型权重已从 {path} 加载 (包含{len(self.actors)}个agent的网络)")
            else:
                # 向后兼容：加载旧的共享网络模型
                print(f"警告：加载旧的共享网络模型，将复制到所有agent")
                if 'actor_state_dict' in model_data:
                    for agent in self.actors.keys():
                        self.actors[agent].load_state_dict(model_data['actor_state_dict'])
                        self.critics[agent].load_state_dict(model_data['critic_state_dict'])
                print(f"旧模型权重已加载并复制到{len(self.actors)}个agent")
            
            # 恢复其他参数
            self.epsilon = model_data.get('epsilon', 0.1)
            
            return model_data
        else:
            print(f"模型文件不存在: {path}")
            return None
    
    def _extract_action_features(self, actions: List[Action], slots: Dict, lp_provider) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取动作特征"""
        if not actions:
            return torch.tensor([]), torch.tensor([])
        
        features = []
        for action in actions:
            # 基础特征
            feat = [
                float(action.score),  # 动作得分
                float(action.cost),   # 成本
                float(action.reward), # 奖励
                float(action.prestige), # 声望
                len(action.footprint_slots),  # 占用槽位数量
            ]
            
            # 位置特征
            if action.footprint_slots:
                first_slot = slots.get(action.footprint_slots[0])
                if first_slot:
                    feat.extend([
                        first_slot.x / 200.0,  # 归一化x坐标
                        first_slot.y / 200.0,  # 归一化y坐标
                    ])
                else:
                    feat.extend([0.0, 0.0])
            else:
                feat.extend([0.0, 0.0])
            
            # 地价特征
            if action.footprint_slots:
                prices = []
                for slot_id in action.footprint_slots:
                    price = lp_provider(slot_id)
                    prices.append(price)
                feat.extend([
                    np.mean(prices),  # 平均地价
                    np.std(prices),   # 地价标准差
                    np.min(prices),   # 最低地价
                    np.max(prices),   # 最高地价
                ])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])
            
            # 填充到固定长度
            while len(feat) < 20:
                feat.append(0.0)
            
            features.append(feat[:20])  # 截断到20维
        
        # 转换为张量
        action_feats = torch.tensor(features, dtype=torch.float32)
        mask = torch.ones(len(actions), dtype=torch.bool)
        
        return action_feats, mask
    
    def _encode_state(self, slots: Dict, candidates: Set[str], occupied: Set[str], lp_provider) -> torch.Tensor:
        """编码当前状态"""
        # 简化的状态编码实现
        # 实际应该使用更复杂的CNN编码
        
        # 创建栅格表示
        grid_size = 200
        occupancy_map = torch.zeros(grid_size, grid_size)
        land_price_map = torch.zeros(grid_size, grid_size)
        
        # 填充占用信息
        for slot_id, slot in slots.items():
            x, y = slot.x, slot.y
            if 0 <= x < grid_size and 0 <= y < grid_size:
                if slot_id in occupied:
                    occupancy_map[y, x] = 1.0
                land_price_map[y, x] = lp_provider(slot_id)
        
        # 简化为全局特征（实际应该用CNN）
        state_features = [
            len(candidates) / 1000.0,  # 候选槽位比例
            len(occupied) / 1000.0,    # 占用槽位比例
            torch.mean(land_price_map).item(),  # 平均地价
            torch.std(land_price_map).item(),   # 地价标准差
        ]
        
        # 填充到512维
        while len(state_features) < 512:
            state_features.append(0.0)
        
        return torch.tensor(state_features[:512], dtype=torch.float32).unsqueeze(0)
    
    def _masked_sample(self, logits: torch.Tensor, mask: torch.Tensor) -> int:
        """带掩码的采样"""
        from rl.v4_1.utils import masked_sample
        
        # 确保logits和mask在同一设备上
        if logits.device != mask.device:
            mask = mask.to(logits.device)
        
        # 使用工具函数进行采样
        action_idx, _ = masked_sample(logits, mask)
        
        return action_idx.item()
    
    def update_exploration(self, episode: int):
        """更新探索率（训练过程中逐步降低）"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # 同步更新所有actor的温度参数
        for agent, actor in self.actors.items():
           # 温度参数也可以随探索率衰减
           actor.temperature = max(1.5, 3.0 * self.epsilon / 0.8)
    
    def _limit_s_size_actions(self, actions: List[Action], max_s_ratio: float = 0.5) -> List[Action]:
        """限制S型建筑在动作池中的数量以平衡分布"""
        if not actions:
            return actions
        
        # 按尺寸分组
        s_actions = [a for a in actions if a.size == 'S']
        m_actions = [a for a in actions if a.size == 'M']
        l_actions = [a for a in actions if a.size == 'L']
        a_actions = [a for a in actions if a.size == 'A']
        b_actions = [a for a in actions if a.size == 'B']
        c_actions = [a for a in actions if a.size == 'C']
        
        total_actions = len(actions)
        max_s_count = int(total_actions * max_s_ratio)
        
        # 如果S型建筑数量超过限制
        if len(s_actions) > max_s_count:
            # 按得分排序，保留最好的S型建筑
            s_actions.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
            s_actions = s_actions[:max_s_count]
            print(f"限制S型建筑数量: {len(s_actions)}/{total_actions} (比例: {len(s_actions)/total_actions:.2f})")
        
        # 重新组合动作列表
        balanced_actions = s_actions + m_actions + l_actions + a_actions + b_actions + c_actions
        return balanced_actions
    
    def _add_exploration_bonus(self, actions: List[Action]) -> List[Action]:
        """给M/L型建筑添加探索奖励，特别强化高等级槽位上的M/L建筑"""
        if not actions:
            return actions
        
        for action in actions:
            # 获取槽位等级信息
            slot_level = self._get_slot_level(action.footprint_slots[0]) if action.footprint_slots else 3
            
            if action.size == 'M':
                # M型建筑额外奖励
                base_bonus = self.epsilon * 0.5
                
                # 在高等级槽位上给更多奖励
                if slot_level >= 4:
                    level_bonus = base_bonus * 5.0  # 激进：高等级槽位奖励5倍
                    action.score += level_bonus
                    action.reward += level_bonus * 0.5
                    print(f"高等级槽位M型建筑奖励: slot_level={slot_level}, bonus={level_bonus:.3f}")
                else:
                    action.score += base_bonus
                    action.reward += base_bonus * 0.1
                    
            elif action.size == 'L':
                # L型建筑更多奖励
                base_bonus = self.epsilon * 1.0
                
                # 在高等级槽位上给更多奖励
                if slot_level >= 5:
                    level_bonus = base_bonus * 10.0  # 激进：Level 5槽位奖励10倍
                    action.score += level_bonus
                    action.reward += level_bonus * 1.0
                    print(f"Level 5槽位L型建筑奖励: slot_level={slot_level}, bonus={level_bonus:.3f}")
                elif slot_level >= 4:
                    level_bonus = base_bonus * 7.0  # 激进：Level 4槽位奖励7倍
                    action.score += level_bonus
                    action.reward += level_bonus * 0.7
                    print(f"Level 4槽位L型建筑奖励: slot_level={slot_level}, bonus={level_bonus:.3f}")
                else:
                    action.score += base_bonus
                    action.reward += base_bonus * 0.1
        
        return actions
    
    def _get_slot_level(self, slot_id: str) -> int:
        """获取槽位的建筑等级"""
        if self.slots and slot_id in self.slots:
            slot = self.slots[slot_id]
            if hasattr(slot, 'building_level'):
                return slot.building_level
        return 3  # 默认Level 3
    
    def _prioritize_high_level_slots(self, actions: List[Action]) -> List[Action]:
        """在高等级槽位上优先选择M/L型建筑"""
        if not actions:
            return actions
        
        # 按槽位等级和建筑尺寸重新排序
        def action_priority(action):
            slot_level = self._get_slot_level(action.footprint_slots[0]) if action.footprint_slots else 3
            size_priority = {'L': 3, 'M': 2, 'S': 1, 'A': 1, 'B': 2, 'C': 3}.get(action.size, 1)
            
            # 优先级计算：高等级槽位 + 大尺寸建筑 = 高优先级
            return (slot_level * 10 + size_priority, getattr(action, 'score', 0))
        
        # 重新排序，高等级槽位的大尺寸建筑排在前面
        prioritized_actions = sorted(actions, key=action_priority, reverse=True)
        
        # 统计调整效果
        high_level_m_l = sum(1 for a in prioritized_actions[:20] 
                           if self._get_slot_level(a.footprint_slots[0]) >= 4 and a.size in ['M', 'L'])
        if high_level_m_l > 0:
            print(f"高等级槽位M/L型建筑优先排序: 前20个动作中有{high_level_m_l}个高等级M/L建筑")
        
        return prioritized_actions
    
