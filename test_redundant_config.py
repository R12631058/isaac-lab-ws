"""測試 Needle Tip Redundant Control 配置"""

from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5NeedleTipRedundantReachEnvCfg,
    NeedleTipRedundantRewardsCfg,
    NeedleTipRedundantObservationsCfg,
)

def main():
    print("\n" + "="*60)
    print("🧪 測試 Needle Tip Redundant Control 配置")
    print("="*60)
    
    # 1. 測試配置類別可以實例化
    try:
        cfg = TM5NeedleTipRedundantReachEnvCfg()
        print("\n✅ TM5NeedleTipRedundantReachEnvCfg 實例化成功")
    except Exception as e:
        print(f"\n❌ TM5NeedleTipRedundantReachEnvCfg 實例化失敗: {e}")
        return
    
    # 2. 檢查場景配置
    print(f"\n📦 場景類型: {type(cfg.scene).__name__}")
    print(f"   - 環境數量: {cfg.scene.num_envs}")
    print(f"   - 環境間距: {cfg.scene.env_spacing}m")
    
    # 3. 檢查獎勵配置
    print(f"\n🎯 獎勵配置:")
    rewards = cfg.rewards
    if hasattr(rewards, 'needle_tip_position_tracking'):
        print(f"   ✓ needle_tip_position_tracking: weight={rewards.needle_tip_position_tracking.weight}")
    if hasattr(rewards, 'needle_tip_position_tracking_fine_grained'):
        print(f"   ✓ needle_tip_position_tracking_fine_grained: weight={rewards.needle_tip_position_tracking_fine_grained.weight}")
    if hasattr(rewards, 'ee_orientation_keep_preferred'):
        print(f"   ✓ ee_orientation_keep_preferred: weight={rewards.ee_orientation_keep_preferred.weight}")
        print(f"     - preferred_quat: {rewards.ee_orientation_keep_preferred.params.get('preferred_quat', 'N/A')}")
    
    # 4. 檢查觀察配置
    print(f"\n👁️  觀察配置:")
    obs_policy = cfg.observations.policy
    print(f"   - 包含條款數: {len([k for k in dir(obs_policy) if not k.startswith('_')])}")
    
    if hasattr(obs_policy, 'needle_tip_position'):
        print(f"   ✓ needle_tip_position (針尖位置)")
    if hasattr(obs_policy, 'position_command'):
        print(f"   ✓ position_command (位置命令)")
    if hasattr(obs_policy, 'joint_pos'):
        print(f"   ✓ joint_pos (關節位置)")
    if hasattr(obs_policy, 'joint_vel'):
        print(f"   ✓ joint_vel (關節速度)")
    
    # 5. 檢查命令配置
    print(f"\n🎮 命令配置:")
    if hasattr(cfg, 'commands') and hasattr(cfg.commands, 'ee_pose'):
        cmd = cfg.commands.ee_pose
        print(f"   - body_name: {cmd.body_name}")
        print(f"   - resampling_time: {cmd.resampling_time_range}")
        print(f"   - debug_vis: {cmd.debug_vis}")
    
    print("\n" + "="*60)
    print("✅ 所有檢查完成!")
    print("="*60 + "\n")
    
    print("📝 使用方式:")
    print("   訓練: isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-TM5-NeedleTip-Redundant-v0")
    print("   播放: isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-TM5-NeedleTip-Redundant-Play-v0 --checkpoint <path>")
    print()

if __name__ == "__main__":
    main()
