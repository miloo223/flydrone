#!/usr/bin/env python3
# train_optuna.py  ─ Optuna 최적값으로 학습 & 모든 콜백 포함
# ----------------------------------------------------------
import json, csv, gym, yaml, os
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, ProgressBarCallback
)

from run_manager import create_session   # ← 기존 세션 관리자

# ──────────────────────────────────────────────────────────
# 1. Optuna 결과 로드 (없으면 기본값으로 대체)
# ──────────────────────────────────────────────────────────
best_path = Path("best_reward_weights.json")
if best_path.exists():
    best = json.load(best_path.open())
    print("✅  Optuna best 파라미터 로드:", best_path)
else:
    print("⚠️  best_reward_weights.json 을 찾지 못했습니다. 기본값으로 학습합니다.")
    best = dict(
        w_area=1.0, w_alt=1.0, w_energy=1.0,
        lr=3e-4, gamma=0.99, n_steps=2048,
    )

# 환경 kwargs + PPO kwargs 로 분리
ENV_KWARGS  = {k: best[k] for k in ("w_area", "w_alt", "w_energy")}
PPO_KWARGS  = {k: best[k] for k in ("lr", "gamma", "n_steps")}
PPO_KWARGS["learning_rate"] = PPO_KWARGS.pop("lr")
PPO_KWARGS["verbose"]       = 1

# ──────────────────────────────────────────────────────────
# 2. 세션 디렉터리 생성
# ──────────────────────────────────────────────────────────
cfg = dict(
    algo="PPO-optuna",
    total_timesteps=2_000_000,
    log_every_steps=20,
    checkpoint_freq=10_000,
)
SESSION = create_session(cfg)
LOG_DIR   = SESSION / "logs"
PLOT_DIR  = SESSION / "plots"
MODEL_DIR = SESSION / "models"
TB_DIR    = SESSION / "tensorboard"

# ──────────────────────────────────────────────────────────
# 3. RewardPlotCallback  (원본 보상*만* 기록)
# ──────────────────────────────────────────────────────────
class RewardPlotCallback(BaseCallback):
    def __init__(self, log_dir: Path, plot_every_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir   = Path(log_dir)
        self.csv_path  = self.log_dir / "episode_returns.csv"
        self.plot_path = self.log_dir / "reward_plot.png"
        self.plot_every = plot_every_episodes

    def _on_training_start(self):
        self.n_envs = self.training_env.num_envs
        self.ep_returns = [0.0] * self.n_envs
        self.ep_lengths = [0]   * self.n_envs
        self.total_episodes = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "return", "length", "timesteps"])

    def _on_step(self):
        rewards = self.locals["rewards"]      # 정규화 보상
        raw     = self.training_env.get_original_reward(rewards)
        term    = self.locals.get("terminated", self.locals.get("dones"))
        trunc   = self.locals.get("truncated",  [False]*len(term))

        for i in range(self.n_envs):
            self.ep_returns[i] += raw[i]
            self.ep_lengths[i] += 1
            if term[i]:                         # truncated 는 무시
                self.total_episodes += 1
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self.total_episodes,
                         float(self.ep_returns[i]),
                         int(self.ep_lengths[i]),
                         self.num_timesteps]
                    )
                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0

        return True

# ──────────────────────────────────────────────────────────
# 4. Gym 환경 등록 (딱 한 번)
# ──────────────────────────────────────────────────────────
ENV_ID = "FlyDrone-v0-custom"
if ENV_ID not in gym.envs.registry:
    gym.register(
        id=ENV_ID,
        entry_point="fly_drone.envs.fly_drone_env:Fly_drone",
        kwargs={**ENV_KWARGS, "log_dir": LOG_DIR, "plot_dir": PLOT_DIR},
    )

def make_env():
    env = gym.make(ENV_ID)
    env.settings(rend=False, train=True)
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# ──────────────────────────────────────────────────────────
# 5. 모델 생성 & 콜백
# ──────────────────────────────────────────────────────────
model = PPO("MlpPolicy", vec_env,
            tensorboard_log=str(TB_DIR),
            **PPO_KWARGS)

checkpoint_cb = CheckpointCallback(save_freq=cfg["checkpoint_freq"],
                                   save_path=MODEL_DIR,
                                   name_prefix="ckpt",
                                   save_replay_buffer=False,
                                   save_vecnormalize=True)
rewardplot_cb = RewardPlotCallback(PLOT_DIR, plot_every_episodes=20)
progress_cb   = ProgressBarCallback()

# ──────────────────────────────────────────────────────────
# 6. 학습
# ──────────────────────────────────────────────────────────
model.learn(total_timesteps=cfg["total_timesteps"],
            callback=[checkpoint_cb, rewardplot_cb, progress_cb])

model.save(MODEL_DIR / "final_model")
vec_env.save(MODEL_DIR / "vec_normalize.pkl")
print("🏁  Training finished:", SESSION)
