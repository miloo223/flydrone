import gym, yaml, optuna, json, torch, warnings
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ── 0. 환경 한 번만 등록 ──────────────────────────────
ENV_ID = "FlyDrone-v0-custom"
if ENV_ID not in gym.envs.registry:
    gym.register(
        id=ENV_ID,
        entry_point="fly_drone.envs.fly_drone_env:Fly_drone",
        kwargs=dict(log_dir=Path("/tmp"), plot_dir=Path("/tmp"))  # 필수 인자만
    )

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "experiments"

def make_env(w_area, w_alt, w_energy):
    def _init():
        env = gym.make(
            ENV_ID,
            log_dir=Path("/tmp"), plot_dir=Path("/tmp"),
            w_area=w_area, w_alt=w_alt, w_energy=w_energy,
        )
        env.settings(rend=False, train=True)
        return env
    return _init
def objective(trial: optuna.Trial):

    # ── 1. 샘플링 공간 ─────────────────────
    w_area   = trial.suggest_float("w_area",   0.1, 10.0, log=True)
    w_alt    = trial.suggest_float("w_alt",    0.01, 5.0, log=True)
    w_energy = trial.suggest_float("w_energy", 0.001, 1.0, log=True)

    lr       = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma    = trial.suggest_float("gamma", 0.93, 0.999)

    n_steps  = trial.suggest_categorical("n_steps", [512, 1024, 2048])

    # ── 2. 환경 및 모델 ───────────────────
    vec_env = DummyVecEnv([make_env(w_area, w_alt, w_energy)])
    model = PPO("MlpPolicy",
                vec_env,
                learning_rate=lr,
                gamma=gamma,
                n_steps=n_steps,
                verbose=0,
                seed=0,
                device="cpu")

    # ── 3. 짧은 미니-학습 & 성능 평가 ─────
    model.learn(total_timesteps=25_000)
    rewards = []
    obs = vec_env.reset()
    for _ in range(3):              # 3 에피소드 평가
        done, ep_ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_ret += reward[0]
        rewards.append(ep_ret)
        obs = vec_env.reset()

    mean_ret = sum(rewards) / len(rewards)
    return mean_ret     # Optuna가 maximize

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("★ Best trial")
    print(study.best_trial.params)
    # 결과 저장
    with open("best_reward_weights.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
