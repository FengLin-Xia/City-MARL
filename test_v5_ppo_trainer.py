"""
测试v5.0 PPO训练器
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from trainers.v5_0.ppo_trainer import V5PPOTrainer


def test_v5_ppo_trainer():
    """测试v5.0 PPO训练器"""
    print("Testing v5.0 PPO trainer...")
    
    try:
        # 创建训练器
        trainer = V5PPOTrainer("configs/city_config_v5_0.json")
        print("Trainer created successfully")
        
        # 测试经验收集
        print("\\nTesting experience collection...")
        experiences = trainer.collect_experience(num_steps=10)
        print(f"Collected {len(experiences)} experiences")
        
        if experiences:
            # 显示第一个经验
            first_exp = experiences[0]
            print(f"First experience: agent={first_exp['agent']}, reward={first_exp['reward']}")
            print(f"Sequence: {first_exp['sequence']}")
            print(f"Step log: {first_exp['step_log']}")
        
        # 测试训练步骤
        print("\\nTesting training step...")
        if experiences:
            train_stats = trainer.train_step(experiences)
            print(f"Training stats: {train_stats}")
        
        # 测试网络保存和加载
        print("\\nTesting network save/load...")
        trainer.save_model("test_v5_0_model.pth")
        trainer.load_model("test_v5_0_model.pth")
        
        # 测试评估
        print("\\nTesting evaluation...")
        eval_results = trainer.evaluate(num_episodes=2)
        print(f"Evaluation results: {eval_results}")
        
        # 测试完整训练（小规模）
        print("\\nTesting mini training...")
        training_results = trainer.train(num_episodes=5, save_interval=2)
        print(f"Training results: {training_results}")
        
        print("\\nPPO trainer test passed!")
        return True
        
    except Exception as e:
        print(f"PPO trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing v5.0 PPO trainer...")
    
    success = test_v5_ppo_trainer()
    
    if success:
        print("\\nAll v5.0 PPO trainer tests passed!")
    else:
        print("\\nSome v5.0 PPO trainer tests failed!")
