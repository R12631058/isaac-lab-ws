"""
比较不同训练阶段的 checkpoint 性能
测试 2025-11-12_13-12-31 训练的不同模型
"""

import subprocess
import sys
from pathlib import Path

# 定义要测试的 checkpoints
TRAIN_DIR = "logs/rsl_rl/tm5_reach_stable_v2/2025-11-12_13-12-31"
CHECKPOINTS = [
    ("早期", "model_500.pt", 500),
    ("中期", "model_1500.pt", 1500),
    ("后期", "model_2500.pt", 2500),
    ("最终", "model_2999.pt", 2999),
]

def test_checkpoint(stage, model_file, iterations):
    """测试单个 checkpoint"""
    print(f"\n{'='*80}")
    print(f"测试 {stage} 模型 (Iteration {iterations}): {model_file}")
    print(f"{'='*80}\n")
    
    checkpoint_path = f"{TRAIN_DIR}/{model_file}"
    cmd = [
        "powershell.exe",
        "-Command",
        f".\\isaaclab_conda.bat -p scripts\\reinforcement_learning\\rsl_rl\\play.py --task Isaac-Reach-TM5-SurgeryRoom-v0 --num_envs 4 --checkpoint {checkpoint_path}"
    ]
    
    print(f"执行命令: {' '.join(cmd[2:])}")
    print(f"\n按 Ctrl+C 可关闭视窗并继续下一个测试...")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print(f"\n用户中断 {stage} 模型测试")
    except Exception as e:
        print(f"\n测试 {stage} 模型时出错: {e}")

def main():
    print("\n" + "="*80)
    print("TM5 Reach Surgery Room - Checkpoint 比较测试")
    print("训练目录: 2025-11-12_13-12-31 (3000 iterations, V3 progressive penalty)")
    print("="*80)
    
    print("\n测试计划:")
    for i, (stage, model, iters) in enumerate(CHECKPOINTS, 1):
        print(f"  {i}. {stage:6} - {model:16} (Iteration {iters:4})")
    
    print("\n" + "="*80)
    input("按 Enter 开始测试...")
    
    # 测试每个 checkpoint
    for stage, model, iters in CHECKPOINTS:
        test_checkpoint(stage, model, iters)
        print(f"\n{stage} 模型测试完成!")
        
        # 询问是否继续
        if stage != CHECKPOINTS[-1][0]:  # 不是最后一个
            response = input(f"\n继续测试下一个模型? (Y/n): ").strip().lower()
            if response == 'n':
                print("测试中止")
                break
    
    print("\n" + "="*80)
    print("所有测试完成!")
    print("="*80)
    
    print("\n观察要点:")
    print("  1. 早期模型 (500): 是否已学会基本的避碰行为?")
    print("  2. 中期模型 (1500): 避碰与到达的平衡如何?")
    print("  3. 后期模型 (2500): 运动是否更流畅?")
    print("  4. 最终模型 (2999): 整体性能是否最佳?")
    print("\n请根据观察结果决定是否需要:")
    print("  - 继续训练 (如果还在改善)")
    print("  - 调整奖励权重 (如果有明显问题)")
    print("  - 使用新配置重新训练 (如果需要大幅改变)")

if __name__ == "__main__":
    main()
