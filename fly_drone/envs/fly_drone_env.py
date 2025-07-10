import os
import gym
from gym import spaces
import numpy as np
import math
import pylab
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

env_name = "fly_drone-v0"

scores, episodes = [], []
all_polygons = []
drone_path = []
tif_path = "flydrone/fly_drone/envs/image_cut.tif"

time = 0

FOV_DEG = 60
ASPECT_RATIO = 4 / 3
FOV_RAD_X = np.deg2rad(FOV_DEG)
FOV_RAD_Y = FOV_RAD_X / ASPECT_RATIO

drone_xy = np.array([264000.0, 309500.0], dtype=np.float32)
drone_xy_velocity = np.array([0.0, 0.0], dtype=np.float32)
roll, pitch, yaw = 0, 0, 0
explored_area = 0

N = 30
STEP_SIZE = 1.0
MAX_STEPS = 200

def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

with rasterio.open(tif_path) as src:
    dem = src.read(1)
    transform = src.transform
    bounds = src.bounds
    width = src.width
    height = src.height

x_res, y_res = transform.a, -transform.e
x0, y0 = transform.c, transform.f

def world_to_pixel(x, y):
    px = ((x - x0) / x_res).astype(int)
    py = ((y0 - y) / y_res).astype(int)
    return px, py

class Fly_drone(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(3, ), dtype="float32") #set action space size, range
        self.observation_space = spaces.Box(low=-10**2, high=10**2, shape=(8,), dtype="float32") #set observation space size, range
        self.done = False
        self.episode = 0
        self.train = True
        self.rend = True
        self.total_return = 0
        self.score_avg = 0

    def step(self, action):
        global time
        global drone_xy, drone_xy_velocity, roll, pitch, yaw, explored_area
        time += 0.1
        drone_xy[0] += action[0] * 0.5 * 0.1**2 + drone_xy_velocity[0] * 0.1
        drone_xy[1] += action[1] * 0.5 * 0.1**2 + drone_xy_velocity[1] * 0.1
        drone_xy_velocity[0] += action[0] * 0.1
        drone_xy_velocity[1] += action[1] * 0.1
        yaw = action[2] #set action
        px, py = world_to_pixel(np.array([drone_xy[0]]), np.array([drone_xy[1]])) #면적 계산
        drone_alt = dem[py[0], px[0]] + 10
        u = np.linspace(-np.tan(FOV_RAD_X / 2), np.tan(FOV_RAD_X / 2), N)
        v = np.linspace(-np.tan(FOV_RAD_Y / 2), np.tan(FOV_RAD_Y / 2), N)
        uu, vv = np.meshgrid(u, v)
        dirs_local = np.stack([uu, vv, -np.ones_like(uu)], axis=-1).reshape(-1, 3)
        dirs_local /= np.linalg.norm(dirs_local, axis=1, keepdims=True)

        R = rpy_to_matrix(roll, pitch, yaw)
        dirs_world = dirs_local @ R.T
        intersections = []
        for d in dirs_world:
            ray_pos = np.array([*drone_xy, drone_alt], dtype=np.float32)
            for _ in range(MAX_STEPS):
                ray_pos += d * STEP_SIZE
                px, py = world_to_pixel(ray_pos[0], ray_pos[1])
                if 0 <= px < width and 0 <= py < height:
                    ground_z = dem[py, px]
                    if ray_pos[2] <= ground_z:
                        intersections.append((ray_pos[0], ray_pos[1]))
                        break
                else:
                    break
        poly = MultiPoint(intersections).convex_hull
        if all_polygons:
            total_area_poly = unary_union(all_polygons)
            new_polygon = poly.difference(total_area_poly)
            new_area = new_polygon.area
            explored_area = total_area_poly.union(poly).area
        else:
            new_polygon = poly
            new_area = poly.area
            explored_area = poly.area

        all_polygons.append(poly)

        state1 = [drone_xy[i] for i in range(2)]
        state2 = [drone_xy_velocity[i] for i in range(2)]
        state3 = [roll, pitch, yaw, explored_area]
        state = state1 + state2 + state3

        reward = new_area #reward function
        self.total_return  = self.total_return + reward

        if time >= 20: #end episode
            self.done = True
            self.plot(self.train)
            self.episode = self.episode + 1    
        return state, reward, self.done, {}


    def reset(self):
        global time, drone_xy, drone_xy_velocity, roll, pitch, yaw, explored_area, all_polygons
        all_polygons = []
        self.total_return = 0
        self.done = False
        time = 0
        drone_xy = np.array([264000.0, 309500.0])
        drone_xy_velocity = np.array([0.0, 0.0])
        roll, pitch, yaw, explored_area = 0, 0, 0, 0
        state1 = [drone_xy[i] for i in range(2)]
        state2 = [drone_xy_velocity[i] for i in range(2)]
        state3 = [roll, pitch, yaw, explored_area]
        state = state1 + state2 + state3
        return state
    
    def settings(self, rend, train):
        self.train = train
        self.rend = rend

    def plot(self, enable):
        global all_polygons, drone_path
        if enable:
            self.score_avg = 0.9 * self.score_avg + 0.1 * self.total_return if self.episode != 0 else self.total_return 
            scores.append(self.score_avg)
            episodes.append(self.episode)
            fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
            ax.imshow(dem, cmap='terrain', extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), interpolation='none')

            if all_polygons:
                total_poly = unary_union(all_polygons)
                if total_poly.geom_type == 'Polygon':
                    x, y = total_poly.exterior.xy
                    ax.fill(x, y, color='cyan', alpha=0.4, label='Explored Area')
                elif total_poly.geom_type == 'MultiPolygon':
                    for part in total_poly.geoms:
                        if not part.is_empty:
                            x, y = part.exterior.xy
                            ax.fill(x, y, color='cyan', alpha=0.4)

            if drone_path:
                path_x, path_y = zip(*drone_path)
                ax.plot(path_x, path_y, 'r-', linewidth=2, label='Drone Path')
                ax.plot(path_x[-1], path_y[-1], 'ro', label='Final Position')

            ax.set_title(f"Episode {self.episode}: Explored Area and Path")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"flydrone/Area_fig/episode_{self.episode}_map.svg")
            plt.close()

            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig("PPO_reward.png")