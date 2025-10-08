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
        
        # 对输出层使用保守的初始化
        nn.init.normal_(self.network[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.network[-1].bias, 0)
    
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
    
    def __init__(self, cfg: Dict, model_path: Optional[str] = None):
        self.cfg = cfg
        self.rl_cfg = cfg['solver']['rl']
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化独立的Actor和Critic网络
        self.state_dim = 512  # 简化状态维度
        self.max_actions = 50  # 最大动作数量
        
        self.actor = Actor(state_dim=self.state_dim, max_actions=self.max_actions).to(self.device)
        self.critic = Critic(state_dim=self.state_dim).to(self.device)
        
        # 设置初始温度参数（可以调节）
        self.actor.temperature = self.rl_cfg.get('temperature', 1.2)  # 调试温度1.2
        
        # 独立的优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.rl_cfg.get('actor_lr', 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.rl_cfg.get('critic_lr', 3e-4)
        )
        
        # 保持向后兼容的优化器接口
        self.optimizer = self.actor_optimizer
        
        # 探索参数
        self.epsilon = 0.1  # ε-贪婪探索
        
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
            self.scorer = ActionScorer(objective, normalize, eval_params=eval_params)
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
    ) -> Tuple[List[Action], Optional[Sequence]]:
        """使用RL策略选择动作序列（恢复v4.0的序列机制）"""
        
        # 保存当前槽位信息，供扩展策略使用
        self._current_slots = slots
        
        # 1. 枚举合法动作池
        if self.enumerator is None:
            self.enumerator = ActionEnumerator(slots)
        
        actions = self.enumerator.enumerate_actions(
            candidates=candidates,
            occupied=occupied,
            agent_types=agent_types or self.rl_cfg['agents'],
            sizes=sizes or {'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']},
            lp_provider=lp_provider,
            adjacency='4-neighbor',
            caps=self.cfg.get('growth_v4_1', {}).get('enumeration', {}).get('caps', {}),
        )
        
        if not actions:
            return [], None
        
        # 2. 计算动作得分（ActionScorer已在__init__中初始化）
        if self.scorer is None:
            print("警告: ActionScorer未初始化，跳过动作打分")
            return [], None
        
        # 计算动作得分
        actions = self.scorer.score_actions(actions, river_distance_provider)
        
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
        
        # 限制动作数量
        num_actions = min(len(actions), self.max_actions)
        action_subset = actions[:num_actions]
        
        # 生成状态编码
        state_embed = self._encode_state_for_rl(action_subset)
        
        # ε-贪婪探索
        if np.random.random() < self.epsilon:
            # 探索：随机选择一个动作作为锚点
            selected_idx = np.random.randint(0, num_actions)
            selected_action = action_subset[selected_idx]
        else:
            # 利用：使用策略网络选择锚点动作
            with torch.no_grad():
                logits = self.actor(state_embed)
                # 只使用有效动作数量的logits
                valid_logits = logits[0, :num_actions]
                action_probs = F.softmax(valid_logits, dim=-1)
                selected_idx = torch.multinomial(action_probs, 1).item()
                selected_action = action_subset[selected_idx]
        
        if selected_action:
            # 使用扩展策略生成多槽位序列
            expanded_sequence = self._expand_anchor_to_sequence(selected_action, actions, selected_idx)
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
            
            # 使用当前策略计算log概率
            with torch.no_grad():
                logits = self.actor(state_embed)
                valid_logits = logits[0, :num_actions]
                
                # 创建分布并计算log概率
                import torch.nn.functional as F
                action_probs = F.softmax(valid_logits, dim=-1)
                
                if 0 <= anchor_idx < len(action_probs):
                    log_prob = torch.log(action_probs[anchor_idx] + 1e-8)
                    return log_prob
                else:
                    return torch.tensor(0.0)
                    
        except Exception as e:
            print(f"Warning: Failed to compute anchor log prob: {e}")
            return torch.tensor(0.0)
    
    def _expand_anchor_to_sequence(self, anchor_action: Action, all_actions: List[Action], anchor_idx: int) -> Optional[Sequence]:
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
            return sequence
    
    def _encode_state_for_rl(self, actions: List[Action]) -> torch.Tensor:
        """为RL生成简化的状态编码"""
        # 简化状态编码：基于动作池的特征
        if not actions:
            return torch.zeros(1, self.state_dim, device=self.device)
        
        # 计算动作池的统计特征
        scores = [action.score for action in actions if action.score is not None]
        costs = [action.cost for action in actions if action.cost is not None]
        rewards = [action.reward for action in actions if action.reward is not None]
        prestiges = [action.prestige for action in actions if action.prestige is not None]
        
        # 构建特征向量
        features = []
        features.append(len(actions))  # 动作数量
        features.append(np.mean(scores) if scores else 0.0)  # 平均得分
        features.append(np.std(scores) if scores else 0.0)   # 得分标准差
        features.append(np.mean(costs) if costs else 0.0)    # 平均成本
        features.append(np.mean(rewards) if rewards else 0.0)  # 平均奖励
        features.append(np.mean(prestiges) if prestiges else 0.0)  # 平均声望
        
        # 扩展到固定维度
        while len(features) < self.state_dim:
            features.append(0.0)
        
        features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def save_model(self, path: str):
        """保存模型权重"""
        import os
        import datetime
        import json
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'rl_config': self.rl_cfg,
            'model_version': 'v4.1',
            'timestamp': datetime.datetime.now().isoformat(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'max_actions': self.max_actions,
            'actor_temperature': self.actor.temperature,
        }
        torch.save(model_data, path)
        print(f"模型权重已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型权重"""
        if os.path.exists(path):
            model_data = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(model_data['actor_state_dict'])
            self.critic.load_state_dict(model_data['critic_state_dict'])
            
            # 加载优化器状态（如果存在）
            if 'actor_optimizer_state' in model_data:
                self.actor_optimizer.load_state_dict(model_data['actor_optimizer_state'])
            if 'critic_optimizer_state' in model_data:
                self.critic_optimizer.load_state_dict(model_data['critic_optimizer_state'])
            
            # 恢复其他参数
            self.epsilon = model_data.get('epsilon', 0.1)
            self.actor.temperature = model_data.get('actor_temperature', 1.0)
            
            print(f"模型权重已从 {path} 加载")
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
    
