import numpy as np
from drone_env import DroneEnv
from stable_baselines3 import PPO

if __name__ == '__main__':
    dem_file = 'data/mountain_dem.tif'
    env = DroneEnv(dem_file)
    model = PPO.load('drone_ppo')

    obs = env.reset()
    done = False
    total_area = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_area += reward
    print(f"Total explored area: {total_area:.2f} m²")