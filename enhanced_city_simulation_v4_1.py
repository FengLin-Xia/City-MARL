#!/usr/bin/env python3
"""
v4.1 增强城市模拟系统
支持RL和参数化两种求解模式
"""

import json
import argparse
import time
import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入v4.1模块
from solvers.v4_1 import ParamSelector, RLPolicySelector
from rl.v4_1 import PPOTrainer, MAPPOTrainer


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_param_mode(cfg: Dict, eval_only: bool = False) -> Dict:
    """运行参数化模式"""
    print("=" * 60)
    print("运行参数化模式 (Parametric Mode)")
    print("=" * 60)
    
    # 创建参数化选择器
    selector = ParamSelector(cfg)
    
    # 运行模拟（这里需要调用原有的模拟逻辑）
    # TODO: 集成现有的城市模拟逻辑
    
    results = {
        'mode': 'param',
        'total_return': 0.0,
        'edu_return': 0.0,
        'ind_return': 0.0,
        'steps_per_second': 0.0,
        'final_layout': {},
        'metrics': {}
    }
    
    return results


def run_rl_mode(cfg: Dict, eval_only: bool = False, model_path: str = None) -> Dict:
    """运行RL模式"""
    print("=" * 60)
    print("运行RL模式 (Reinforcement Learning Mode)")
    print("=" * 60)
    
    rl_cfg = cfg['solver']['rl']
    
    if eval_only:
        # 评估模式：加载预训练模型进行推理
        if model_path is None:
            raise ValueError("评估模式需要指定模型路径")
        
        selector = RLPolicySelector(cfg, model_path=model_path)
        
        # 运行评估
        results = evaluate_rl_model(selector, cfg)
        
    else:
        # 训练模式：训练新的RL模型
        print(f"使用算法: {rl_cfg['algo']}")
        
        # 导入PPO训练器
        from trainers.v4_1.ppo_trainer import PPOTrainer
        from enhanced_training_logger import initialize_training_logger, get_training_logger
        
        # 初始化增强训练记录器
        model_save_path = rl_cfg.get('model_save_path', 'models/v4_1_rl/')
        logger = initialize_training_logger(model_save_path, "ppo_training_v4_1")
        
        # 创建PPO训练器
        trainer = PPOTrainer(cfg)
        
        # 训练模型
        results = train_rl_model(trainer, cfg, logger)
    
    return results


def evaluate_rl_model(selector: RLPolicySelector, cfg: Dict) -> Dict:
    """评估RL模型"""
    print("开始RL模型评估...")
    
    from envs.v4_1.city_env import CityEnvironment
    
    # 初始化环境
    env = CityEnvironment(cfg)
    
    eval_seeds = cfg['solver']['rl'].get('eval_seed_set', [42, 123, 456, 789, 999])
    total_returns = []
    edu_returns = []
    ind_returns = []
    
    # 槽位选择历史记录
    slot_selection_history = {
        'episodes': [],
        'total_selections': 0
    }
    
    for seed_idx, seed in enumerate(eval_seeds):
        print(f"评估种子: {seed}")
        
        # 重置环境
        state = env.reset(seed=seed)
        
        episode_rewards = {'EDU': 0.0, 'IND': 0.0}
        steps = 0
        
        # 记录当前Episode的槽位选择历史
        episode_slot_history = {
            'episode_id': seed_idx,
            'episode_return': 0.0,  # 稍后更新
            'steps': [],
            'summary': {
                'total_selections': 0,
                'unique_slots_selected': set(),
                'edu_selections': 0,
                'ind_selections': 0,
                'avg_action_score': 0.0,
                'avg_sequence_score': 0.0
            }
        }
        
        while True:
            # 获取当前智能体（使用环境的当前智能体）
            current_agent = env.current_agent  # 使用环境当前智能体，而不是状态中的
            
            # 获取动作池
            actions, action_feats, mask = env.get_action_pool(current_agent)
            
            if not actions:
                # 没有可用动作，结束回合
                break
            
            # 使用RL选择器选择动作序列
            all_buildings = env.buildings.get('public', []) + env.buildings.get('industrial', [])
            _, selected_sequence = selector.choose_action_sequence(
                slots=env.slots,
                candidates=set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots),
                occupied=env._get_occupied_slots(),
                lp_provider=env._create_lp_provider(),
                agent_types=[current_agent],
                sizes={current_agent: ['S', 'M', 'L']},
                buildings=all_buildings
            )
            
            if selected_sequence is None:
                break
            
            # 应用环境约束过滤序列中的动作
            if selected_sequence and selected_sequence.actions:
                filtered_actions = []
                for action in selected_sequence.actions:
                    if env.action_allowed(action):
                        filtered_actions.append(action)
                
                # 如果过滤后为空，结束Episode（避免无限循环）
                if not filtered_actions:
                    break
                
                # 创建过滤后的序列
                from logic.v4_enumeration import Sequence
                # 保存原始的action_index
                original_action_index = getattr(selected_sequence, 'action_index', -1)
                
                # 安全地计算序列属性
                sum_cost = sum(getattr(a, 'cost', 0.0) for a in filtered_actions)
                sum_reward = sum(getattr(a, 'reward', 0.0) for a in filtered_actions)
                sum_prestige = sum(getattr(a, 'prestige', 0.0) for a in filtered_actions)
                total_score = sum(getattr(a, 'score', 0.0) for a in filtered_actions)
                
                filtered_sequence = Sequence(
                    actions=filtered_actions,
                    sum_cost=sum_cost,
                    sum_reward=sum_reward,
                    sum_prestige=sum_prestige,
                    score=total_score
                )
                # 恢复action_index属性
                filtered_sequence.action_index = original_action_index
                selected_sequence = filtered_sequence
            
            # 执行动作序列
            next_state, reward, done, info = env.step(current_agent, selected_sequence)
            
            # 累积奖励
            episode_rewards[current_agent] += reward
            steps += 1
            
            # 记录槽位选择信息
            step_slot_info = {
                'month': env.current_month,
                'agent': current_agent,
                'selected_slots': [action.footprint_slots for action in selected_sequence.actions],
                'action_scores': [action.score for action in selected_sequence.actions],
                'sequence_score': selected_sequence.score,
                'available_actions_count': len(actions),
                'candidate_slots_count': len(set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots)),
                'detailed_actions': []
            }
            
            # 记录详细动作信息
            for action in selected_sequence.actions:
                detailed_action = {
                    'agent': action.agent,
                    'size': action.size,
                    'slot_id': action.footprint_slots[0] if action.footprint_slots else '',
                    'footprint_slots': action.footprint_slots,
                    'cost': action.cost,
                    'reward': action.reward,
                    'prestige': action.prestige,
                    'score': action.score,
                    'slot_positions': []
                }
                
                # 添加槽位位置信息
                for slot_id in action.footprint_slots:
                    if slot_id in env.slots:
                        slot = env.slots[slot_id]
                        slot_pos = {
                            'slot_id': slot_id,
                            'x': getattr(slot, 'fx', slot.x),
                            'y': getattr(slot, 'fy', slot.y),
                            'angle': getattr(slot, 'angle', 0.0)
                        }
                        detailed_action['slot_positions'].append(slot_pos)
                
                step_slot_info['detailed_actions'].append(detailed_action)
            
            episode_slot_history['steps'].append(step_slot_info)
            
            # 更新状态
            state = next_state
            
            # 检查是否结束
            if done:
                break
        
        # 记录本轮结果
        total_return = episode_rewards['EDU'] + episode_rewards['IND']
        total_returns.append(total_return)
        edu_returns.append(episode_rewards['EDU'])
        ind_returns.append(episode_rewards['IND'])
        
        # 更新Episode槽位选择历史
        episode_slot_history['episode_return'] = total_return
        
        # 计算Episode统计信息
        total_sequence_scores = []
        all_action_scores = []
        unique_slots = set()
        edu_count = 0
        ind_count = 0
        
        for step in episode_slot_history['steps']:
            total_sequence_scores.append(step['sequence_score'])
            all_action_scores.extend(step['action_scores'])
            
            for detailed_action in step['detailed_actions']:
                unique_slots.update(detailed_action['footprint_slots'])
                if detailed_action['agent'] == 'EDU':
                    edu_count += 1
                elif detailed_action['agent'] == 'IND':
                    ind_count += 1
        
        # 更新Episode摘要
        episode_slot_history['summary']['total_selections'] = len(episode_slot_history['steps'])
        episode_slot_history['summary']['unique_slots_selected'] = unique_slots
        episode_slot_history['summary']['edu_selections'] = edu_count
        episode_slot_history['summary']['ind_selections'] = ind_count
        episode_slot_history['summary']['avg_action_score'] = np.mean(all_action_scores) if all_action_scores else 0.0
        episode_slot_history['summary']['avg_sequence_score'] = np.mean(total_sequence_scores) if total_sequence_scores else 0.0
        
        # 转换set为list以便JSON序列化
        episode_slot_history['summary']['unique_slots_selected'] = list(episode_slot_history['summary']['unique_slots_selected'])
        
        # 添加到总历史
        slot_selection_history['episodes'].append(episode_slot_history)
        slot_selection_history['total_selections'] += episode_slot_history['summary']['total_selections']
        
        print(f"  种子 {seed}: 总奖励={total_return:.3f}, EDU={episode_rewards['EDU']:.3f}, IND={episode_rewards['IND']:.3f}, 步数={steps}")
    
    # 计算平均结果
    avg_total_return = np.mean(total_returns) if total_returns else 0.0
    avg_edu_return = np.mean(edu_returns) if edu_returns else 0.0
    avg_ind_return = np.mean(ind_returns) if ind_returns else 0.0
    
    results = {
        'mode': 'rl_eval',
        'total_return': avg_total_return,
        'edu_return': avg_edu_return,
        'ind_return': avg_ind_return,
        'steps_per_second': 0.0,  # TODO: 计算速度
        'final_layout': {},
        'metrics': {
            'total_returns': total_returns,
            'edu_returns': edu_returns,
            'ind_returns': ind_returns,
            'std_total': np.std(total_returns) if total_returns else 0.0,
            'std_edu': np.std(edu_returns) if edu_returns else 0.0,
            'std_ind': np.std(ind_returns) if ind_returns else 0.0,
        },
        'eval_seeds': eval_seeds
    }
    
    # 保存槽位选择历史
    rl_cfg = cfg['solver']['rl']
    model_save_path = rl_cfg.get('model_save_path', 'models/v4_1_rl/')
    os.makedirs(model_save_path, exist_ok=True)
    
    history_save_path = os.path.join(model_save_path, 'slot_selection_history.json')
    with open(history_save_path, 'w', encoding='utf-8') as f:
        json.dump(slot_selection_history, f, indent=2, ensure_ascii=False)
    print(f"槽位选择历史已保存到: {history_save_path}")
    
    # 打印槽位选择统计
    print(f"\n槽位选择历史统计:")
    print(f"  总Episode数: {len(slot_selection_history['episodes'])}")
    print(f"  总选择次数: {slot_selection_history['total_selections']}")
    
    if slot_selection_history['episodes']:
        # 计算总体统计
        all_unique_slots = set()
        all_avg_scores = []
        edu_total = 0
        ind_total = 0
        
        for episode in slot_selection_history['episodes']:
            all_unique_slots.update(episode['summary']['unique_slots_selected'])
            all_avg_scores.append(episode['summary']['avg_action_score'])
            edu_total += episode['summary']['edu_selections']
            ind_total += episode['summary']['ind_selections']
        
        print(f"  唯一槽位选择数: {len(all_unique_slots)}")
        print(f"  EDU选择次数: {edu_total}")
        print(f"  IND选择次数: {ind_total}")
        print(f"  平均动作得分: {np.mean(all_avg_scores):.3f}")
    
    # 收集Budget历史
    budget_history = {}
    if env.budget_history is not None:
        budget_history = {agent: history.copy() for agent, history in env.budget_history.items()}
    
    # 更新结果
    results['slot_selection_history_path'] = history_save_path
    results['slot_selection_stats'] = {
        'total_episodes': len(slot_selection_history['episodes']),
        'total_selections': slot_selection_history['total_selections'],
        'unique_slots_count': len(all_unique_slots) if slot_selection_history['episodes'] else 0,
        'edu_selections': edu_total,
        'ind_selections': ind_total,
        'avg_action_score': np.mean(all_avg_scores) if all_avg_scores else 0.0
    }
    results['budget_history'] = budget_history
    
    print(f"评估完成: 平均总奖励={avg_total_return:.3f}, EDU={avg_edu_return:.3f}, IND={avg_ind_return:.3f}")
    
    return results


def run_single_episode(env, selector, seed: Optional[int] = None) -> Tuple[List[Dict], float]:
    """运行单个Episode并收集经验"""
    # 重置环境
    state = env.reset(seed=seed)
    
    experiences = []
    episode_rewards = {'EDU': 0.0, 'IND': 0.0}
    steps = 0
    
    while True:
        # 获取当前智能体（使用环境的当前智能体）
        current_agent = env.current_agent  # 使用环境当前智能体，而不是状态中的
        current_month = env.current_month
        
        print(f"  Episode step {steps + 1}: month={current_month}, agent={current_agent}")
        
        # 获取动作池
        actions, action_feats, mask = env.get_action_pool(current_agent)
        
        if not actions:
            # 没有可用动作，结束回合
            print(f"    No available actions, ending episode")
            break
        
        # 使用RL选择器选择动作序列
        all_buildings = env.buildings.get('public', []) + env.buildings.get('industrial', [])
        _, selected_sequence = selector.choose_action_sequence(
            slots=env.slots,
            candidates=set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots),
            occupied=env._get_occupied_slots(),
            buildings=all_buildings,
            lp_provider=env._create_lp_provider(),
            agent_types=[current_agent],
            sizes={current_agent: ['S', 'M', 'L']}
        )
        
        if selected_sequence is None:
            print(f"    No sequence selected, ending episode")
            break
        
        # 应用环境约束过滤序列中的动作
        if selected_sequence and selected_sequence.actions:
            filtered_actions = []
            for action in selected_sequence.actions:
                if env.action_allowed(action):
                    filtered_actions.append(action)
            
            # 如果过滤后为空，结束Episode（避免无限循环）
            if not filtered_actions:
                print(f"    All actions filtered out, ending episode")
                break
            
            # 创建过滤后的序列
            from logic.v4_enumeration import Sequence
            # 保存原始的action_index
            original_action_index = getattr(selected_sequence, 'action_index', -1)
            
            # 安全地计算序列属性
            sum_cost = sum(getattr(a, 'cost', 0.0) for a in filtered_actions)
            sum_reward = sum(getattr(a, 'reward', 0.0) for a in filtered_actions)
            sum_prestige = sum(getattr(a, 'prestige', 0.0) for a in filtered_actions)
            total_score = sum(getattr(a, 'score', 0.0) for a in filtered_actions)
            
            filtered_sequence = Sequence(
                actions=filtered_actions,
                sum_cost=sum_cost,
                sum_reward=sum_reward,
                sum_prestige=sum_prestige,
                score=total_score
            )
            # 恢复action_index属性
            filtered_sequence.action_index = original_action_index
            selected_sequence = filtered_sequence
        
        # 记录旧策略的动作概率（用于PPO训练）
        # 计算锚点选择的log概率
        old_log_prob = selector._compute_anchor_log_prob(selected_sequence, actions)
        
        # 记录详细的动作信息用于TXT导出
        detailed_actions = []
        if selected_sequence and selected_sequence.actions:
            for action in selected_sequence.actions:
                # 获取槽位的位置信息
                slot_positions = []
                for slot_id in action.footprint_slots:
                    slot = env.slots.get(slot_id)
                    if slot:
                        x = float(getattr(slot, 'fx', getattr(slot, 'x', 0.0)))
                        y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
                        angle = float(getattr(slot, 'angle', 0.0))
                        slot_positions.append({
                            'slot_id': slot_id,
                            'x': x,
                            'y': y,
                            'angle': angle
                        })
                
                detailed_action = {
                    'agent': action.agent,
                    'size': action.size,
                    'slot_id': action.footprint_slots[0] if action.footprint_slots else '',
                    'footprint_slots': action.footprint_slots,
                    'cost': action.cost,
                    'reward': action.reward,
                    'prestige': action.prestige,
                    'score': action.score,
                    'slot_positions': slot_positions
                }
                detailed_actions.append(detailed_action)
        
        # 记录经验
        experience = {
            'state': state.copy(),
            'action': selected_sequence,
            'agent': current_agent,
            'month': env.current_month,
            
            # 业务字段（保持原有）
            'selected_slots': [action.footprint_slots for action in selected_sequence.actions] if selected_sequence and selected_sequence.actions else [],
            'action_scores': [action.score for action in selected_sequence.actions] if selected_sequence and selected_sequence.actions else [],
            'action_costs': [action.cost for action in selected_sequence.actions] if selected_sequence and selected_sequence.actions else [],
            'sequence_score': selected_sequence.score if selected_sequence else 0.0,
            'available_actions_count': len(actions),
            'candidate_slots_count': len(set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots)) if actions else 0,
            'detailed_actions': detailed_actions
        }
        
        # 确保写入基本类型的关键字段
        experience.update({
            'action_index': int(getattr(selected_sequence, 'action_index', -1)),  # 局部索引
            'num_actions': int(getattr(selected_sequence, 'num_actions', len(actions))),  # 子集大小
            'old_log_prob': float(getattr(selected_sequence, 'old_log_prob', old_log_prob).detach().cpu().item() if torch.is_tensor(getattr(selected_sequence, 'old_log_prob', old_log_prob)) else getattr(selected_sequence, 'old_log_prob', old_log_prob)),
            # 'subset_indices': [int(x) for x in subset_indices.detach().cpu().tolist()]  # 可选：暂时不写也行
            # 'state_embed': state_embed.detach().cpu().numpy(),  # 可选：如果你训练端仍重编码，可先不写
        })
        
        # 执行动作序列
        next_state, reward, done, info = env.step(current_agent, selected_sequence)
        
        # 完成经验记录
        experience['reward'] = reward
        experience['next_state'] = next_state.copy()
        experience['done'] = done
        experience['info'] = info
        
        experiences.append(experience)
        episode_rewards[current_agent] += reward
        
        # 更新状态
        state = next_state
        
        # 检查是否结束
        if done:
            print(f"    Episode completed: done={done}")
            break
        
        steps += 1
    
    total_return = episode_rewards['EDU'] + episode_rewards['IND']
    return experiences, total_return


def train_rl_model(trainer, cfg: Dict, logger=None) -> Dict:
    """训练RL模型"""
    print("开始RL模型训练...")
    
    rl_cfg = cfg['solver']['rl']
    
    # 创建环境和选择器
    from envs.v4_1.city_env import CityEnvironment
    from solvers.v4_1.rl_selector import RLPolicySelector
    
    env = CityEnvironment(cfg)
    selector = RLPolicySelector(cfg, slots=env.slots)  # 传递槽位信息
    
    # 设置随机种子
    trainer.set_seed(rl_cfg['seed'])
    
    # 训练统计
    training_metrics = {
        'episode_returns': [],
        'edu_returns': [],
        'ind_returns': [],
        'episode_lengths': []
    }
    
    # 槽位选择历史记录
    slot_selection_history = {
        'episodes': [],
        'total_selections': 0
    }
    
    # 训练循环
    for update in range(rl_cfg['max_updates']):
        print(f"训练更新 {update + 1}/{rl_cfg['max_updates']}")
        
        # 更新探索率（逐步降低）
        selector.update_exploration(update)
        print(f"当前探索率: {selector.epsilon:.3f}")
        
        # 开始记录训练更新
        if logger:
            logger.start_training_update(update, rl_cfg['rollout_steps'])
        
        # 1. 收集经验 (Rollout)
        all_experiences = []
        episode_returns = []
        edu_returns = []
        ind_returns = []
        episode_lengths = []
        
        rollout_steps = rl_cfg['rollout_steps']
        steps_collected = 0
        
        while steps_collected < rollout_steps:
            # 运行一个Episode
            experiences, episode_return = run_single_episode(env, selector, seed=None)
            
            if experiences:  # 如果Episode成功完成
                all_experiences.extend(experiences)
                episode_returns.append(episode_return)
                
                # 计算各智能体奖励
                edu_return = sum(exp['reward'] for exp in experiences if exp['agent'] == 'EDU')
                ind_return = sum(exp['reward'] for exp in experiences if exp['agent'] == 'IND')
                edu_returns.append(edu_return)
                ind_returns.append(ind_return)
                
                episode_lengths.append(len(experiences))
                steps_collected += len(experiences)
                
                # 记录episode到增强记录器
                if logger:
                    episode_id = len(training_metrics['episode_returns'])
                    logger.start_episode(episode_id)
                    
                    # 记录每个step
                    for i, exp in enumerate(experiences):
                        logger.record_step(
                            agent=exp['agent'],
                            month=i,
                            reward=exp['reward'],
                            selected_slots=exp.get('selected_slots', []),
                            action_score=exp.get('action_score', 0.0),
                            available_actions=exp.get('available_actions', 0),
                            candidate_slots=exp.get('candidate_slots', 0)
                        )
                    
                    # 完成episode记录
                    episode_metrics = {
                        'total_return': episode_return,
                        'edu_return': edu_return,
                        'ind_return': ind_return,
                        'episode_length': len(experiences)
                    }
                    logger.finish_episode(episode_metrics)
                
                # 收集槽位选择历史
                episode_slot_history = {
                    'episode_id': len(slot_selection_history['episodes']),
                    'update': update,
                    'episode_return': episode_return,
                    'steps': [],
                    'summary': {
                        'total_selections': 0,
                        'edu_selections': 0,
                        'ind_selections': 0,
                        'unique_slots_selected': set(),
                        'avg_action_score': 0.0,
                        'avg_sequence_score': 0.0
                    }
                }
                
                total_scores = []
                total_sequence_scores = []
                
                for exp in experiences:
                    step_info = {
                        'agent': exp['agent'],
                        'month': exp['month'],
                        'selected_slots': exp['selected_slots'],
                        'action_scores': exp['action_scores'],
                        'sequence_score': exp['sequence_score'],
                        'reward': exp['reward'],
                        'available_actions_count': exp['available_actions_count'],
                        'candidate_slots_count': exp['candidate_slots_count']
                    }
                    episode_slot_history['steps'].append(step_info)
                    
                    # 更新统计信息
                    episode_slot_history['summary']['total_selections'] += 1
                    if exp['agent'] == 'EDU':
                        episode_slot_history['summary']['edu_selections'] += 1
                    else:
                        episode_slot_history['summary']['ind_selections'] += 1
                    
                    # 收集选择的槽位
                    for slot_list in exp['selected_slots']:
                        for slot in slot_list:
                            episode_slot_history['summary']['unique_slots_selected'].add(slot)
                    
                    # 收集得分信息
                    total_scores.extend(exp['action_scores'])
                    total_sequence_scores.append(exp['sequence_score'])
                
                # 计算平均得分
                if total_scores:
                    episode_slot_history['summary']['avg_action_score'] = np.mean(total_scores)
                if total_sequence_scores:
                    episode_slot_history['summary']['avg_sequence_score'] = np.mean(total_sequence_scores)
                
                # 转换set为list以便JSON序列化
                episode_slot_history['summary']['unique_slots_selected'] = list(episode_slot_history['summary']['unique_slots_selected'])
                
                slot_selection_history['episodes'].append(episode_slot_history)
                slot_selection_history['total_selections'] += episode_slot_history['summary']['total_selections']
        
        # 记录训练统计
        if episode_returns:
            avg_return = np.mean(episode_returns)
            avg_edu = np.mean(edu_returns)
            avg_ind = np.mean(ind_returns)
            avg_length = np.mean(episode_lengths)
            
            training_metrics['episode_returns'].append(avg_return)
            training_metrics['edu_returns'].append(avg_edu)
            training_metrics['ind_returns'].append(avg_ind)
            training_metrics['episode_lengths'].append(avg_length)
            
            print(f"  收集了 {len(all_experiences)} 步经验，{len(episode_returns)} 个Episode")
            print(f"  平均Episode奖励: {avg_return:.3f} (EDU: {avg_edu:.3f}, IND: {avg_ind:.3f})")
            print(f"  平均Episode长度: {avg_length:.1f} 步")
        
        # 2. 训练策略网络 (如果收集到足够经验)
        if all_experiences:
            print(f"  使用PPO训练器更新策略网络...")
            # 使用PPO训练器进行策略更新
            loss_stats = trainer.update_policy(all_experiences)
            
            # 完成训练更新记录
            if logger:
                update_summary = {
                    'final_policy_loss': loss_stats['policy_loss'],
                    'final_value_loss': loss_stats['value_loss'],
                    'final_entropy': loss_stats['entropy'],
                    'final_total_loss': loss_stats['total_loss'],
                    'final_kl_divergence': loss_stats['kl_divergence'],
                    'final_clip_fraction': loss_stats['clip_fraction']
                }
                logger.finish_training_update(update_summary)
            
            # 记录损失统计
            print(f"  训练损失: policy={loss_stats['policy_loss']:.4f}, "
                  f"value={loss_stats['value_loss']:.4f}, "
                  f"entropy={loss_stats['entropy']:.4f}, "
                  f"total={loss_stats['total_loss']:.4f}")
            print(f"  KL散度: {loss_stats['kl_divergence']:.4f}, "
                  f"裁剪比例: {loss_stats['clip_fraction']:.4f}")
        
        # 定期评估
        if (update + 1) % rl_cfg['eval_every'] == 0:
            print(f"  进行评估...")
            # 运行评估Episode
            eval_returns = []
            for seed in rl_cfg['eval_seed_set']:
                _, episode_return = run_single_episode(env, selector, seed=seed)
                eval_returns.append(episode_return)
            
            avg_eval_return = np.mean(eval_returns) if eval_returns else 0.0
            print(f"  评估结果: 平均奖励={avg_eval_return:.3f}")
        
        # 定期保存模型
        if (update + 1) % rl_cfg['save_every'] == 0:
            model_path, training_state_path = trainer.save_model_with_versioning(
                base_path=rl_cfg['model_save_path'],
                update=update + 1,
                cfg=cfg,
                is_final=False
            )
    
    # 保存最终模型
    final_model_path, final_training_state_path = trainer.save_model_with_versioning(
        base_path=rl_cfg['model_save_path'],
        update=rl_cfg['max_updates'],
        cfg=cfg,
        is_final=True
    )
    
    # 保存增强训练日志
    if logger:
        final_summary = {
            'total_training_updates': rl_cfg['max_updates'],
            'final_model_path': final_model_path,
            'total_episodes_run': len(training_metrics['episode_returns']),
            'final_avg_return': training_metrics['episode_returns'][-1] if training_metrics['episode_returns'] else 0.0,
            'best_return': max(training_metrics['episode_returns']) if training_metrics['episode_returns'] else 0.0,
            'training_completed': True
        }
        logger.set_final_summary(final_summary)
        detailed_log_path = logger.save_log()
        logger.save_summary_csv()
        logger.print_training_summary()
        print(f"详细训练日志已保存: {detailed_log_path}")
    
    # 保存槽位选择历史
    history_save_path = os.path.join(rl_cfg['model_save_path'], 'slot_selection_history.json')
    with open(history_save_path, 'w', encoding='utf-8') as f:
        json.dump(slot_selection_history, f, indent=2, ensure_ascii=False)
    print(f"槽位选择历史已保存到: {history_save_path}")
    
    # 打印槽位选择统计
    print(f"\n槽位选择历史统计:")
    print(f"  总Episode数: {len(slot_selection_history['episodes'])}")
    print(f"  总选择次数: {slot_selection_history['total_selections']}")
    
    if slot_selection_history['episodes']:
        # 计算总体统计
        all_unique_slots = set()
        all_avg_scores = []
        edu_total = 0
        ind_total = 0
        
        for episode in slot_selection_history['episodes']:
            all_unique_slots.update(episode['summary']['unique_slots_selected'])
            all_avg_scores.append(episode['summary']['avg_action_score'])
            edu_total += episode['summary']['edu_selections']
            ind_total += episode['summary']['ind_selections']
        
        print(f"  唯一槽位选择数: {len(all_unique_slots)}")
        print(f"  EDU选择次数: {edu_total}")
        print(f"  IND选择次数: {ind_total}")
        print(f"  平均动作得分: {np.mean(all_avg_scores):.3f}")
    
    # 收集Budget历史
    budget_history = {}
    if env.budget_history is not None:
        budget_history = {agent: history.copy() for agent, history in env.budget_history.items()}
    
    results = {
        'mode': 'rl_train',
        'training_updates': rl_cfg['max_updates'],
        'final_model_path': final_model_path,
        'training_metrics': training_metrics,
        'total_episodes_run': len(training_metrics['episode_returns']),
        'final_avg_return': training_metrics['episode_returns'][-1] if training_metrics['episode_returns'] else 0.0,
        'slot_selection_history_path': history_save_path,
        'slot_selection_stats': {
            'total_episodes': len(slot_selection_history['episodes']),
            'total_selections': slot_selection_history['total_selections'],
            'unique_slots_count': len(all_unique_slots) if slot_selection_history['episodes'] else 0,
            'edu_selections': edu_total,
            'ind_selections': ind_total,
            'avg_action_score': np.mean(all_avg_scores) if all_avg_scores else 0.0
        },
        'budget_history': budget_history
    }
    
    return results


def compare_modes(param_results: Dict, rl_results: Dict) -> Dict:
    """对比两种模式的结果"""
    print("=" * 60)
    print("模式对比结果")
    print("=" * 60)
    
    comparison = {
        'param_results': param_results,
        'rl_results': rl_results,
        'improvements': {
            'total_return': rl_results.get('total_return', 0) - param_results.get('total_return', 0),
            'edu_return': rl_results.get('edu_return', 0) - param_results.get('edu_return', 0),
            'ind_return': rl_results.get('ind_return', 0) - param_results.get('ind_return', 0),
        }
    }
    
    # 打印对比结果
    print(f"总收益对比:")
    print(f"  参数化: {param_results.get('total_return', 0):.4f}")
    print(f"  RL:     {rl_results.get('total_return', 0):.4f}")
    print(f"  改进:   {comparison['improvements']['total_return']:+.4f}")
    
    print(f"\\nEDU收益对比:")
    print(f"  参数化: {param_results.get('edu_return', 0):.4f}")
    print(f"  RL:     {rl_results.get('edu_return', 0):.4f}")
    print(f"  改进:   {comparison['improvements']['edu_return']:+.4f}")
    
    print(f"\\nIND收益对比:")
    print(f"  参数化: {param_results.get('ind_return', 0):.4f}")
    print(f"  RL:     {rl_results.get('ind_return', 0):.4f}")
    print(f"  改进:   {comparison['improvements']['ind_return']:+.4f}")
    
    return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='v4.1 增强城市模拟系统')
    parser.add_argument('--config', type=str, default='configs/city_config_v4_1.json',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['param', 'rl'], default=None,
                       help='求解模式 (覆盖配置文件设置)')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--model_path', type=str, default=None,
                       help='RL模型路径 (评估模式必需)')
    parser.add_argument('--compare', action='store_true',
                       help='对比两种模式')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 覆盖模式设置
    if args.mode:
        cfg['solver']['mode'] = args.mode
    
    print(f"配置加载完成: {args.config}")
    print(f"求解模式: {cfg['solver']['mode']}")
    print(f"评估模式: {args.eval_only}")
    
    start_time = time.time()
    
    if args.compare:
        # 对比模式：运行两种模式并对比
        print("\\n运行对比实验...")
        
        # 运行参数化模式
        param_results = run_param_mode(cfg, eval_only=True)
        
        # 运行RL模式
        rl_results = run_rl_mode(cfg, eval_only=True, model_path=args.model_path)
        
        # 对比结果
        comparison = compare_modes(param_results, rl_results)
        
        # 保存对比结果
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/v4_1_rl/comparison_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"\\n对比报告已保存: {report_path}")
        
    else:
        # 单一模式运行
        if cfg['solver']['mode'] == 'param':
            results = run_param_mode(cfg, eval_only=args.eval_only)
        elif cfg['solver']['mode'] == 'rl':
            results = run_rl_mode(cfg, eval_only=args.eval_only, model_path=args.model_path)
        else:
            raise ValueError(f"不支持的求解模式: {cfg['solver']['mode']}")
        
        print(f"\\n运行完成，结果: {results}")
        
        # 保存训练结果（包含budget历史）
        if 'budget_history' in results:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            rl_cfg = cfg['solver']['rl']
            results_path = os.path.join(rl_cfg['model_save_path'], f'training_results_{timestamp}.json')
            
            # 将set转换为list以便JSON序列化
            results_copy = results.copy()
            if 'slot_selection_stats' in results_copy:
                for key, value in results_copy['slot_selection_stats'].items():
                    if isinstance(value, set):
                        results_copy['slot_selection_stats'][key] = list(value)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, ensure_ascii=False)
            print(f"训练结果已保存: {results_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\\n总运行时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()
