import gym
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, ProgressBarCallback
from fly_drone.envs import fly_drone_env_og_ver
import torch
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from run_manager import create_session
import yaml

# run manager에서 실행할 부분들
cfg = dict(
    algo="PPO",
    env_name="FlyDrone-v0",
    total_timesteps=10_000,#실행시 2_000_000, 테스트시 10_000,
    log_every_steps=20,
    checkpoint_freq=10_000,
    lr=3e-4,
    #lr=0.00013296321722176043,
    n_steps=2048,
    #n_steps=1024,
    gamma=0.995,
    #gamma=0.9541215081998161,
    etc="…"
)

SESSION_DIR = create_session(cfg)
LOG_DIR      = SESSION_DIR / "logs"
PLOT_DIR     = SESSION_DIR / "plots"
MODEL_DIR    = SESSION_DIR / "models"
TB_DIR       = SESSION_DIR / "tensorboard"

print('> 1')
# gym 등록(한 번만)
import gym
ENV_ID = "FlyDrone-v0-custom"           # 새 이름
if ENV_ID not in gym.envs.registry:
    gym.register(
        id=ENV_ID,
        entry_point="fly_drone.envs.fly_drone_env:Fly_drone",
        kwargs=dict(log_dir=LOG_DIR, plot_dir=PLOT_DIR)   # ← 키워드 전달
    )
print('> 2')


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        from tqdm import tqdm
        self.pbar = tqdm(total=total_timesteps, desc="Timesteps", unit="ts")

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


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
        self.total_episodes = -1 # 이렇게 해야 0부터 시작함

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "return", "length", "timesteps"])

    
    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        #dones = self.locals["dones"]
        terminated = self.locals.get("terminated", self.locals.get("dones"))
        truncated  = self.locals.get("truncated",  [False] * len(terminated))
        #dones = [t or r for t, r in zip(terminated, truncated)]

        self.ep_returns += rewards
        #raw_rew = self.training_env.get_original_reward()
        #self.ep_returns += raw_rew

        self.ep_lengths += 1

        for i, (term, trunc) in enumerate(zip(terminated, truncated)):
            if term:
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
    '''
    def _on_step(self):
        self.returns += self.locals["rewards"]
        self.lengths += 1
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.ep_cnt += 1
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self.ep_cnt, float(self.returns[i]), int(self.lengths[i]), self.num_timesteps]
                    )
                self.returns[i] = 0.0
                self.lengths[i] = 0
                if self.ep_cnt % self.plot_every == 0:
                    self._plot()
        return True
    '''

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

#env_name = 'fly_drone-v0'
#start_time = datetime.now().replace(microsecond=0)

#run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
#log_dir = Path("PPO_trained_model") / env_name / run_id
#log_dir.mkdir(parents=True, exist_ok=True)

#env = gym.make(env_name)
env = gym.make(ENV_ID)
env.settings(rend = True, train = True)
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

print('> 3')

#model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(TB_DIR))
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=cfg["lr"],
    n_steps=cfg["n_steps"],
    gamma=cfg["gamma"],
    verbose=1,
    tensorboard_log=str(TB_DIR),
    device="cpu",
)

checkpoint_cb = CheckpointCallback(
  save_freq=cfg["checkpoint_freq"],
  save_path=str(MODEL_DIR),
  name_prefix="ppo_step",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

progress_cb = ProgressBarCallback(cfg["total_timesteps"])
reward_plot_cb = RewardPlotCallback(PLOT_DIR, plot_every_episodes=20, verbose=1)

print('> 학습 곧 시작')
model.learn(total_timesteps=cfg["total_timesteps"], callback=[checkpoint_cb, reward_plot_cb, progress_cb])

model.save(MODEL_DIR / "final_model")
vec_env.save(MODEL_DIR / "vec_normalize.pkl")

#print(f"✅  Training finished.  All artifacts are in  {SESSION}")


'''
log_dir = "PPO_trained_model"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name+ '/'
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

model.save(MODEL_DIR / "final_model")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
vec_env.save(stats_path)

print("modelsaved at" + log_dir)

end_time = datetime.now().replace(microsecond=0)

print("--------------------------------------------------------------------------------------------")
print("Started training at : ", start_time)
print("Finished training at : ", end_time)
print("Total training time : ", end_time - start_time)
print("--------------------------------------------------------------------------------------------")
'''

