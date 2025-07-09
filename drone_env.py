import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

from drone_dem_loader import load_dem
from drone_viewshed import compute_viewshed, calculate_true_visible_area

class DroneEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, dem_path, render_mode=None):
        super().__init__()

        # 1. Load DEM data
        self.dem, self.resolution = load_dem(dem_path)
        if self.dem is None or self.resolution is None:
            raise ValueError("DEM could not be loaded or resolution not found.")
        
        self.grid_size_y, self.grid_size_x = self.dem.shape
        self.start_pos = np.array([self.grid_size_x / 2, self.grid_size_y / 2, np.max(self.dem) + 100], dtype=np.float32)

        # Viewshed parameters
        self.fov_h = math.radians(60) # Horizontal Field of View
        self.max_viewshed_distance = 500 # Max distance for viewshed calculation

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
        self.total_true_explored_area = 0.0 # Initialize total true explored area
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
        ix, iy = int(x / self.resolution), int(y / self.resolution) # Convert world coords to grid coords
        
        terminated = False
        if not (0 <= ix < self.grid_size_x and 0 <= iy < self.grid_size_y):
            terminated = True # Out of bounds
        elif z < self.dem[iy, ix]: # Note: DEM is (rows, cols) -> (y, x)
            terminated = True # Crashed

        # --- Viewshed and True Area Calculation ---
        # Assuming origin_x, origin_y are 0 for now. Yaw is fixed to 0 for simplicity.
        # TODO: Integrate drone's yaw into observation/action space for realistic viewshed
        visible_mask, _ = compute_viewshed(
            dem=self.dem,
            resolution=self.resolution,
            origin_x=0, # Assuming DEM origin is 0,0
            origin_y=0, # Assuming DEM origin is 0,0
            obs_x=self.drone_pos[0],
            obs_y=self.drone_pos[1],
            obs_height=self.drone_pos[2],
            yaw=0, # Placeholder: fixed yaw for now
            fov_h=self.fov_h,
            max_distance=self.max_viewshed_distance
        )
        
        # Calculate true explored area for the current step
        current_true_explored_area = calculate_true_visible_area(
            mask=visible_mask,
            dem=self.dem,
            resolution=self.resolution
        )
        self.total_true_explored_area += current_true_explored_area


        # --- Reward Calculation (Simplified) ---
        # For now, reward is based on staying alive and moving, and explored area
        reward = 1.0 - 0.1 * np.linalg.norm(action) + (current_true_explored_area / 1000.0) # Scale area reward
        self.last_reward = reward # Store reward for rendering

        # --- Termination & Truncation ---
        self.total_steps += 1
        truncated = self.total_steps >= 1000

        observation = self._get_obs()
        info = self._get_info()

        # Render if in human mode
        self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "position": self.drone_pos - self.start_pos, # Relative position
            "velocity": self.drone_vel
        }

    def _get_info(self):
        return {
            "total_true_explored_area": self.total_true_explored_area
        }

    def render(self):
        if self.render_mode == 'human':
            print("-" * 40)
            print(f"STEP: {self.total_steps}")
            print(f"  Drone Position (X, Y, Z): ({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f})")
            print(f"  Drone Velocity (Vx, Vy, Vz): ({self.drone_vel[0]:.2f}, {self.drone_vel[1]:.2f}, {self.drone_vel[2]:.2f})")
            print(f"  Total True Explored Area: {self.total_true_explored_area:.2f} sq. units")
            print(f"  Current Step Reward: {self.last_reward:.2f}")
            print("-" * 40)

    def close(self):
        pass
