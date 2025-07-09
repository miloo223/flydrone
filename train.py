import gymnasium as gym
import numpy as np
from drone_env import DroneEnv
from drone_agent import PPOAgent
import torch

def train():
    ############## Hyperparameters ##############
    env_name = "DroneEnv-v0"
    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 4         # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e4)          # save model frequency (in num timesteps)
    
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std
    min_action_std = 0.1                # minimum action_std
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    
    update_timestep = max_ep_len * 4    # update policy every n timesteps
    K_epochs = 40                       # update policy for K epochs
    eps_clip = 0.2                      # clip parameter for PPO
    gamma = 0.99                        # discount factor

    lr_actor = 0.0003                   # learning rate for actor network
    lr_critic = 0.001                   # learning rate for critic network
    #############################################

    # 1. Environment Setup
    env = DroneEnv(dem_path='dem_data/image_cut.tif', render_mode='human')
    # For simplicity, we flatten the observation space
    state_dim = env.observation_space['position'].shape[0] + env.observation_space['velocity'].shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Agent Setup
    agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # 3. Training Loop
    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # Flatten state
            state_flat = np.concatenate([state['position'], state['velocity']])
            
            # Select action with policy
            action = agent.select_action(state_flat)
            state, reward, terminated, truncated, info = env.step(action)

            # Saving reward and is_terminals
            agent.buffer_rewards.append((reward, terminated or truncated))

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()

            # printing average reward
            if time_step % print_freq == 0:
                avg_reward = current_ep_reward / (t+1)
                print(f"Episode: {i_episode}, Timestep: {time_step}, Average Reward: {avg_reward:.3f}")

            if terminated or truncated:
                print(f"Episode {i_episode} finished. Total True Explored Area: {info['total_true_explored_area']:.2f}")
                break
        
        i_episode += 1

    env.close()

if __name__ == '__main__':
    train()
