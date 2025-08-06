import os, math, rasterio, numpy as np, mujoco, glfw
from mujoco import viewer
import time
import pickle
import numpy as np
from simple_pid import PID
import pandas as pd

xml = "flydrone/fly_drone/envs/skydio_x2/scene.xml"

def save_data(filename, positions, velocities):
    data = {'positions': positions, 'velocities': velocities}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def pid_to_thrust(input: np.array):
  c_to_F =np.array([
      [-0.25, 0.25, 0.25, -0.25],
      [0.25, 0.25, -0.25, -0.25],
      [-0.25, 0.25, -0.25, 0.25]
  ]).transpose()
  return np.dot((c_to_F*input),np.array([1,1,1]))

def outer_pid_to_thrust(input: np.array):
  c_to_F =np.array([
      [0.25, 0.25, -0.25, -0.25],
      [0.25, -0.25, -0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
  ]).transpose()
  return np.dot((c_to_F*input),np.array([1,1,1]))

class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output

class PIDController:
  def __init__(self, kp, ki, kd, setpoint):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    self.prev_error = error
    return output

class dummyPlanner:
  def __init__(self, target, vel_limit = 2) -> None:
    self.target = target  
    self.vel_limit = vel_limit
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit),)
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))
  
  def __call__(self, loc: np.array):
    velocites = np.array([0,0,0])
    velocites[0] = self.pid_x(loc[0])
    velocites[1] = self.pid_y(loc[1])
    return velocites

  def get_velocities(self,loc: np.array, target: np.array,
                     time_to_target: float = None,
                     flight_speed: float = 0.5) -> np.array:
    direction = target - loc
    distance = np.linalg.norm(direction)
    if distance > 1:
        velocities = flight_speed * direction / distance
    else:
        velocities =  direction * distance
    return velocities

  def get_alt_setpoint(self, loc: np.array) -> float:
    target = self.target
    distance = target[2] - loc[2]
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        delta_alt = distance / number_steps
        alt_set = loc[2] + 2 * delta_alt
    else:
        alt_set = target[2]
    return alt_set

  def update_target(self, target):
    self.target = target  
    self.pid_x.setpoint = self.target[0]
    self.pid_y.setpoint = self.target[1]

class dummySensor:
  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel
    self.acceleration = d.qacc

  def get_position(self):
    return self.position
  
  def get_velocity(self):
    return self.velocity
  
  def get_acceleration(self):
    return self.acceleration

class drone:
  def __init__(self, target=np.array((0,0,0))):
    self.positions_log = []
    self.m = mujoco.MjModel.from_xml_path(xml)
    TIF_PATH = "flydrone/fly_drone/envs/image_cut_1km.tif"
    with rasterio.open(TIF_PATH) as src:
        dem = src.read(1).astype(np.float32)          
        res_x, res_y = src.res                        
        nodata = src.nodata if src.nodata is not None else np.nan
    dem = np.nan_to_num(dem, nan=nodata)
    dem -= dem.min()
    dem /= dem.max() if dem.max() else 1.0
    dem = np.flipud(dem)
    ta = 200
    if dem.shape != (ta, ta):
        from skimage.transform import resize
        dem = resize(dem, (ta, ta), order=1, anti_aliasing=True).astype(np.float32)
    adr = self.m.hfield_adr[0]
    self.m.hfield_data[adr: adr + 200*200] = dem.flatten()
    self.d = mujoco.MjData(self.m)

    self.planner = dummyPlanner(target=target)
    self.sensor = dummySensor(self.d)

    self.pid_v_x = PID(0.03, 0.002, 0.01, setpoint=0, output_limits=(-0.1, 0.1))
    self.pid_v_y = PID(0.03, 0.002, 0.01, setpoint=0, output_limits=(-0.1, 0.1))

    # 고도
    self.pid_alt = PID(3, 0.3, 0.7, setpoint=0)

    # 자세
    self.pid_roll = PID(1, 0.2, 0.6, setpoint=0, output_limits=(-1,1))
    self.pid_pitch = PID(1, 0.2, 0.6, setpoint=0, output_limits=(-1,1))
    self.pid_yaw = PID(0.3, 0.0, 3.5, setpoint=0, output_limits=(-3,3))

  def update_outer_conrol(self):
    v = self.sensor.get_velocity()
    location = self.sensor.get_position()[:3]
    velocites = self.planner(loc=location)
    self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
    self.pid_v_x.setpoint = velocites[0]
    self.pid_v_y.setpoint = velocites[1]
    angle_pitch = self.pid_v_x(v[0])
    angle_roll = - self.pid_v_y(v[1])
    self.pid_pitch.setpoint= angle_pitch
    self.pid_roll.setpoint = angle_roll
    self.pid_yaw.setpoint = 0.0

  def update_inner_control(self):
    alt = self.sensor.get_position()[2]
    angles = self.sensor.get_position()[3:]
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    self.d.ctrl[:4] = out
    pos = self.sensor.get_position()[:3].copy()
    self.positions_log.append(pos)
    if len(self.positions_log) > 300:
        self.positions_log.pop(0)

  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control
df = pd.read_csv("flydrone/fly_drone/envs/test.csv", header=None, skiprows=1, names=["time", "x", "y", "z"],
                 dtype={"time": float, "x": float, "y": float, "z": float})
df.columns = ["time", "x", "y", "z"]
df["x"] = (df["x"] - 264274.10)/10
df["y"] = (df["y"] - 309327.43)/10
df["z"] = (df["z"] - 493)/432*25 + 3
waypoints = df.values

my_drone = drone(target=np.array((0, 0, 22)))
waypoint_index = 0
timestep = my_drone.m.opt.timestep

velocity_buffer = []
buffer_size = 100
fig = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(fig)

with mujoco.viewer.launch_passive(my_drone.m, my_drone.d) as viewer:
  start = time.time()
  step = 1
  viewer.user_scn.ngeom = 0

  while viewer.is_running():
    step_start = time.time()
    current_time = step * timestep

    if waypoint_index < len(waypoints):
      waypoint_time = 0.1 * waypoint_index
      if current_time >= waypoint_time:
        target_xyz = waypoints[waypoint_index][1:]
        my_drone.planner.update_target(np.array(target_xyz))
        waypoint_index += 1

    if step % 10 == 0:
      my_drone.update_outer_conrol()
    my_drone.update_inner_control()
    mujoco.mj_step(my_drone.m, my_drone.d)

    velocity = my_drone.sensor.get_velocity()[:3].copy()
    velocity_buffer.append(velocity)
    if len(velocity_buffer) > buffer_size:
      velocity_buffer.pop(0)

    fig.linepnt[0] = fig.linepnt[1] = fig.linepnt[2] = len(velocity_buffer)
    fig.linename[0] = b"Vx"
    fig.linename[1] = b"Vy"
    fig.linename[2] = b"Vz"
    fig.flg[0] = 1
    for i in range(len(velocity_buffer)):
      mujoco.mjv_addLine(fig, 0, i, velocity_buffer[i][0])
      mujoco.mjv_addLine(fig, 1, i, velocity_buffer[i][1])
      mujoco.mjv_addLine(fig, 2, i, velocity_buffer[i][2])

    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(my_drone.d.time % 2)

      pos = my_drone.sensor.get_position()[:3]
      coord_text = f"x: {pos[0]:.2f}\ny: {pos[1]:.2f}\nz: {pos[2]:.2f}"
      viewer.overlay(mujoco.mjtOverlay.mjOVERLAY_TOPLEFT, "Drone Position", coord_text)

      mujoco.mjv_figure(viewer.viewport, fig, viewer.ctx)


    if step % 50 == 0:
      mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.2, 0, 0],
        pos=my_drone.positions_log[-1],
        mat=np.eye(3).flatten(),
        rgba=0.5*np.array([1, 0, 0, 2])
      )
      viewer.user_scn.ngeom += 1

    viewer.sync()
    step += 1
    time_until_next_step = timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)