import gym
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from fly_drone.envs import fly_drone_env

# ---- 추가 import (그래프/CSV) ----
import torch
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 서버에서도 그림 저장되게
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------- Reward Plot Callback ----------------
class RewardPlotCallback(BaseCallback):
    """
    VecEnv(dummmy)에서 dones/rewards를 읽어 에피소드 리턴을 csv로 저장하고,
    N 에피소드마다 reward_plot.png를 갱신한다.
    """
    def __init__(self, log_dir: Path, plot_every_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "episode_returns.csv"
        self.plot_path = self.log_dir / "reward_plot.png"
        self.plot_every = plot_every_episodes

    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.ep_returns = np.zeros(self.n_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(self.n_envs, dtype=np.int32)
        self.total_episodes = 0

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "return", "length", "timesteps"])

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.ep_returns += rewards
        self.ep_lengths += 1

        for i, done in enumerate(dones):
            if done:
                self.total_episodes += 1
                # CSV 기록
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self.total_episodes, float(self.ep_returns[i]), int(self.ep_lengths[i]), self.num_timesteps]
                    )
                # 리셋
                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0

                # 플롯 저장
                if self.total_episodes % self.plot_every == 0:
                    self._plot()

        return True

    def _plot(self):
        try:
            data = np.loadtxt(self.csv_path, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            ep = data[:, 0]
            rets = data[:, 1]

            plt.figure()
            plt.plot(ep, rets, label="return")
            if len(rets) > 1:
                win = min(50, len(rets))
                ma = np.convolve(rets, np.ones(win) / win, mode="valid")
                plt.plot(ep[win - 1 :], ma, label=f"MA({win})")
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()
        except Exception as e:
            if self.verbose:
                print(f"[RewardPlotCallback] plot failed: {e}")


render = True
env_name = 'fly_drone-v0'
start_time = datetime.now().replace(microsecond=0)

# ==== 로그 디렉토리 (체크포인트/CSV/플롯 저장용) ====
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = Path("PPO_trained_model") / env_name / run_id
log_dir.mkdir(parents=True, exist_ok=True)

env = gym.make(env_name)
env.settings(rend = render, train = True)
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO("MlpPolicy", vec_env, verbose=1)

checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path=log_dir,
  name_prefix=env_name,
  save_replay_buffer=True,
  save_vecnormalize=True,
)

reward_plot_cb = RewardPlotCallback(log_dir, plot_every_episodes=20, verbose=1)

model.learn(total_timesteps=50_000, callback=[checkpoint_callback, reward_plot_cb])


log_dir = "PPO_trained_model"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name+ '/'
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

model.save(log_dir + "ppo_" + env_name)
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
vec_env.save(stats_path)

print("modelsaved at" + log_dir)

end_time = datetime.now().replace(microsecond=0)

print("--------------------------------------------------------------------------------------------")
print("Started training at : ", start_time)
print("Finished training at : ", end_time)
print("Total training time : ", end_time - start_time)
print("--------------------------------------------------------------------------------------------")

