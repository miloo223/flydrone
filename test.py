from pathlib import Path
import sys, yaml, gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ▶ 스크립트가 있는 폴더를 기준으로 experiments 찾기
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS       = SCRIPT_DIR / "experiments"

sessions = sorted(RUNS.glob("20??????_??????"))
if not sessions:
    sys.exit(f"[TEST] '{RUNS}' 안에 세션 폴더가 없습니다!")

latest = sessions[-1]
print(f"[TEST] 최신 세션 →  {latest.name}")


# 2) config, env, 모델 로드 (이전과 동일) ─────────────
with open(latest / "config.yaml") as f:
    cfg = yaml.safe_load(f)

ENV_ID = "FlyDrone-v0-custom"
if ENV_ID not in gym.envs.registry:
    gym.register(
        id=ENV_ID,
        entry_point="fly_drone.envs.fly_drone_env:Fly_drone",
        kwargs=dict(
            log_dir = latest / "logs",
            plot_dir = latest / "plots",
            #log_every = cfg["log_every_steps"],
        ),
    )

def make_env():
    env = gym.make(ENV_ID)
    env.settings(rend=False, train=False)
    return env

dummy   = DummyVecEnv([make_env])
vec_env = VecNormalize.load(latest / "models/vec_normalize.pkl", dummy)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(latest / "models/final_model.zip", env=vec_env, device="cpu")
print("[TEST] 모델 로드 완료")

# 3) 한 에피소드 실행 ────────────────────────────────
obs = vec_env.reset()
done, ret, steps = False, 0.0, 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = vec_env.step(action)
    ret += reward[0]; steps += 1
print(f"[TEST] 완료  |  steps={steps}  return={ret:.2f}")
