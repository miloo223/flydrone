import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_dem_loader import load_dem
from calculate_area import calculate_view_area

class DroneEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, dem_path, render_mode=None):
        super().__init__()

        # 1. Load DEM data
        self.dem = load_dem(dem_path)
        if self.dem is None:
            raise ValueError("DEM could not be loaded.")
        
        self.grid_size_y, self.grid_size_x = self.dem.shape
        self.start_pos = np.array([self.grid_size_x / 2, self.grid_size_y / 2, np.max(self.dem) + 100], dtype=np.float32)

        # 2. Define action and observation space
        # Action: 3D acceleration vector (ax, ay, az)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: Drone's relative position, velocity, attitude, and explored map
        # For simplicity, we start with position and velocity.
        # The explored map will be added later.
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset drone state
        self.drone_pos = self.start_pos.copy()
        self.drone_vel = np.zeros(3, dtype=np.float32)
        self.explored_mask = np.zeros_like(self.dem, dtype=bool)
        self.total_steps = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # --- Physics Simulation (Simplified) ---
        # Apply action (acceleration)
        self.drone_vel += action
        # Apply gravity
        self.drone_vel[2] -= 9.8 * 0.1 # dt=0.1s
        # Update position
        self.drone_pos += self.drone_vel * 0.1 # dt=0.1s

        # --- Collision Detection (Simplified) ---
        x, y, z = self.drone_pos
        ix, iy = int(x), int(y)
        terminated = False
        if not (0 <= ix < self.grid_size_x and 0 <= iy < self.grid_size_y):
            terminated = True # Out of bounds
        elif z < self.dem[iy, ix]:
            terminated = True # Crashed

        # --- Reward Calculation (Simplified) ---
        # For now, reward is based on staying alive and moving
        reward = 1.0 - 0.1 * np.linalg.norm(action) # Survive and conserve energy

        # --- Update explored area (Placeholder) ---
        # This will use calculate_view_area later
        self.explored_mask[iy, ix] = True

        # --- Termination & Truncation ---
        self.total_steps += 1
        truncated = self.total_steps >= 1000

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "position": self.drone_pos - self.start_pos, # Relative position
            "velocity": self.drone_vel
        }

    def _get_info(self):
        return {
            "explored_ratio": np.mean(self.explored_mask)
        }

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.total_steps}, Pos: {self.drone_pos}, Reward: ...")

    def close(self):
        pass
