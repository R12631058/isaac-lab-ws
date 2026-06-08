import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper

from surgery_rl_env import SurgeryRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

class IsaacLabVecEnvWrapper(VecEnvWrapper):
    """Isaac Lab 環境的 Stable Baselines3 包裝器"""
    
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        super().__init__(env, env.observation_space, env.action_space)
    
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self._actions)
        return obs, rewards, terminated | truncated, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def close(self):
        self.env.close()

def main():
    """使用 PPO 訓練手術機器人"""
    
    # 創建環境
    env_cfg = SurgeryRLEnvCfg()
    env_cfg.scene.num_envs = 512
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = IsaacLabVecEnvWrapper(env)
    
    # 創建 PPO 模型
    model = PPO(
        "MultiInputPolicy",
        wrapped_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./surgery_robot_tensorboard/",
    )
    
    # 開始訓練
    print("Starting PPO training...")
    model.learn(
        total_timesteps=1000000,
        progress_bar=True,
        tb_log_name="surgery_robot_ppo",
    )
    
    # 保存模型
    model.save("surgery_robot_ppo_model")
    print("Training completed and model saved!")
    
    # 測試訓練好的模型
    print("Testing trained model...")
    obs = wrapped_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = wrapped_env.step(action)
        
        if _ % 100 == 0:
            mean_reward = torch.mean(rewards) if isinstance(rewards, torch.Tensor) else torch.mean(torch.tensor(rewards))
            print(f"Test step {_}, mean reward: {mean_reward:.4f}")
    
    wrapped_env.close()

if __name__ == "__main__":
    main()