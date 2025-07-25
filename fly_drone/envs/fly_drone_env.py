import os
import gym
from gym import spaces
import numpy as np
import math
import pylab
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
import numpy as np
from dataclasses import dataclass
from shapely.validation import make_valid
from shapely.errors import GEOSException


env_name = "fly_drone-v0"

scores, episodes = [], []
all_polygons = []
drone_path = []
tif_path = "fly_drone/envs/image_cut.tif"

time = 0


 # 카메라 관련 부분
FOV_DEG = 60
ASPECT_RATIO = 4 / 3
FOV_RAD_X = np.deg2rad(FOV_DEG)
FOV_RAD_Y = FOV_RAD_X / ASPECT_RATIO

drone_xy = np.array([264000.0, 309500.0], dtype=np.float32)
drone_xy_velocity = np.array([0.0, 0.0], dtype=np.float32)
drone_alt = 0.0  # Initial altitude, will be updated in reset
drone_z_velocity = 0.0
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
    return Rz @ Ry @ Rx # 행렬연산

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

# 폴리곤 관련 뭐시기들

# ---- polygon helper -------------------------------------------------
def fix_polygon(poly):
    """
    shapely가 invalid polygon으로 던지는 TopologyException을 방지하기 위해
    make_valid + buffer(0)로 항상 valid하게 만든다.
    """
    if poly.is_empty:
        return poly
    try:
        return make_valid(poly).buffer(0)
    except Exception:
        # shapely < 2.0 환경 등 make_valid가 없다면 buffer(0)만으로도 대부분 복구됨
        return poly.buffer(0)

def force_vertex_count(poly, n):
    """
    exterior를 길이 기준으로 균등 샘플링해서 꼭짓점 개수를 n(닫힌 경계 포함 시 n+1)개로 맞춘다.
    -> observation dimension이 항상 일정해짐.
    """
    if poly.is_empty:
        return poly
    ring = poly.exterior
    total_len = ring.length
    if total_len == 0:
        return poly

    # n개의 점을 perimeter 길이에 따라 균등 간격으로 찍어서 재구성
    dists = np.linspace(0.0, total_len, n, endpoint=False)
    pts = [ring.interpolate(d).coords[0] for d in dists]
    return Polygon(pts).buffer(0)  # buffer(0)로 한 번 더 topology 정리

def as_fixed_length_coords(poly, n):
    """
    obs 벡터에 넣기 위해 (n+1, 2)짜리 닫힌 좌표 배열을 돌려준다.
    shapely는 마지막 점을 다시 한 번 반복해주므로, coords 그대로 써도 되고
    여기선 force_vertex_count를 먼저 적용했으니 n개의 점 + 자동으로 닫힌 점 1개가 생긴다.
    """
    p = force_vertex_count(fix_polygon(poly), n)
    return np.asarray(p.exterior.coords)  # 길이 = n+1


# ------------------------------------------------------------
# 1) 파라미터 정의
# ------------------------------------------------------------
@dataclass
class QuadParams:
    # Rigid-body / aero
    m: float = 1.0          # kg
    g: float = 9.81         # m/s^2
    rho: float = 1.225      # air density (kg/m^3)
    S_F: float = 0.0        # flat-plate equivalent area for drag term (m^2). 0이면 끔.

    # Geometry & prop constants
    l: float = 0.2          # arm length (m)
    k_f: float = 1e-5       # thrust coeff  [N / (rad/s)^2]
    k_tau: float = 2e-7     # moment coeff  [N*m / (rad/s)^2]

    # Motor/electrical
    Rm: float = 0.2         # motor internal resistance (Ohm)
    K_T: float = 0.02       # torque constant (N*m/A)
    K_E: float = 0.02       # back-EMF constant (V/(rad/s))
    D_f: float = 1e-6       # viscous friction coeff (N*m/(rad/s))

    # Reward scaling
    lambda_energy: float = 1e-6  # energy penalty scale (you tune this)

# ------------------------------------------------------------
# 2) 유틸
# ------------------------------------------------------------
DT = 0.1               # 당신이 step에서 쓰는 고정 dt
ATT_KP = 0.5           # (간단) 자세 제어 P 이득. 필요시 튜닝
quad_params = QuadParams()  # 에너지 계산용 파라미터

def attitude_torque_from_acc(acc_world, roll, pitch, params: QuadParams, kp=ATT_KP):
    """
    yaw = 0 가정.
    원하는 선형가속도(acc_world) -> 필요한 thrust 방향(b3_des) -> 목표 roll/pitch ->
    간단 P 제어로 Mx, My 생성.
    (실제로 roll/pitch를 적분/업데이트하지 않아도, 전력 패널티 계산용으로 충분)
    """
    # 원하는 힘
    F_des = params.m * (acc_world + np.array([0.0, 0.0, -params.g], dtype=np.float64))
    normF = np.linalg.norm(F_des) + 1e-9
    b3_des = F_des / normF

    # yaw=0에서 흔히 쓰는 근사식
    pitch_des = np.arctan2(b3_des[0], b3_des[2])      # x로 가려면 pitch(+)
    roll_des  = np.arctan2(-b3_des[1], b3_des[2])     # y로 가려면 roll(-)

    err_roll  = roll_des  - roll
    err_pitch = pitch_des - pitch

    # 아주 간단히 P 제어만 사용 (Mx ~ Kp * error)
    # (관성행렬 J를 모르면, 여기서 나온 값은 '토크 스케일' 정도로만 쓰고, 에너지 패널티에 그대로 넣어도 됨)
    Mx = kp * err_roll
    My = kp * err_pitch
    return Mx, My

def allocation_matrix(params: QuadParams) -> np.ndarray:
    """
    표준 쿼드콥터(구성)에서 ω_i^2 -> [T, Mx, My, Mz] 로 가는 4x4 매트릭스 A
    (w1: front-left, w2: front-right, w3: rear-right, w4: rear-left 라고 가정)
    """
    kf = params.k_f
    kt = params.k_tau
    l  = params.l

    # T     = kf*(w1^2 + w2^2 + w3^2 + w4^2)
    # Mx    = l*kf*(   0*w1^2 - w2^2 + 0*w3^2 + w4^2)  (roll)
    # My    = l*kf*(-w1^2 + 0*w2^2 + w3^2 + 0*w4^2)    (pitch)
    # Mz    = kt*(-w1^2 + w2^2 - w3^2 + w4^2)          (yaw)
    A = np.array([
        [kf,   kf,   kf,   kf  ],
        [0.0, -l*kf, 0.0,  l*kf],
        [-l*kf,0.0,  l*kf, 0.0 ],
        [-kt,   kt,  -kt,   kt ]
    ], dtype=np.float64)
    return A

def thrust_from_acc(acc_world: np.ndarray,
                    vel_world: np.ndarray,
                    params: QuadParams) -> float:
    """
    T (N) = || m * (a + g*e3) + 0.5*rho*S_F*||v||*v ||  (식 그대로)
    yaw=0 가정 하에서 "크기"만 필요하다고 보고, 배치 시엔 이 스칼라 T만 사용.
    """
    g_vec = np.array([0.0, 0.0, -params.g], dtype=np.float64)
    F_req = params.m * (acc_world + g_vec)
    if params.S_F > 0.0:
        F_req += 0.5 * params.rho * params.S_F * np.linalg.norm(vel_world) * vel_world
    return float(np.linalg.norm(F_req))


# ------------------------------------------------------------
# 3) (T, Mx, My, Mz=0) -> ω_i, τ_i, P_i
# ------------------------------------------------------------
def allocate_omegas_closed_form(T: float,
                                Mx: float,
                                My: float,
                                params: QuadParams,
                                Mz: float = 0.0):
    """
    고전적 A^-1 * [T, Mx, My, Mz]^T 방식(ω^2에 대한 선형 시스템).
    -> yaw=0 이면 Mz=0 으로 넣는다.
    음수가 나오는 ω^2는 0으로 클리핑(물론 물리적으로는 saturation/실패로 다뤄야).
    """
    A = allocation_matrix(params)
    b = np.array([T, Mx, My, Mz], dtype=np.float64)

    # ω^2
    w2 = np.linalg.solve(A, b)
    w2 = np.clip(w2, 0.0, None)
    w = np.sqrt(w2)

    # 각 로터 부하 토크(가장 단순한 근사) τ_load = k_tau * ω^2
    tau_load = params.k_tau * w2

    # 각 모터 전력
    P = motor_power(w, tau_load, params)

    return w, tau_load, P

def motor_power(omega: np.ndarray,
                tau_load: np.ndarray,
                params: QuadParams,
                m_L_fn=None) -> np.ndarray:
    """
    p_i = (R/K_T^2) * (D_f*ω_i + m_L(ω_i))^2 + (K_E*ω_i / K_T) * (D_f*ω_i + m_L(ω_i))
    기본은 m_L(ω) = tau_load 를 사용(= k_tau*ω^2 근사).
    원하면 m_L_fn(omega) 를 외부에서 넘겨서 더 정교한 모델(D1,D2,...)을 쓸 수 있음.
    """
    if m_L_fn is None:
        mL = tau_load  # 기본 근사
    else:
        mL = m_L_fn(omega)

    term = params.D_f * omega + mL

    resist = (params.Rm / (params.K_T ** 2)) * (term ** 2)
    backemf = (params.K_E * omega / params.K_T) * term

    return resist + backemf

# ------------------------------------------------------------
# 4) (선택) w1+w2=w3+w4 제약까지 넣은 비선형 4식 풀이 (Newton)
#     - 식(수식 블록)에 있는 C1,C2,C3 / D1,D2,D3 를 쓰고 싶을 때 쓸 수 있게 일반화
# ------------------------------------------------------------
def solve_omegas_newton(
        T, Mx, My,
        params: QuadParams,
        C1=None, C2=None, C3=None,   # thrust side coeffs
        D1=None, D2=None, D3=None,   # yaw side coeffs (미사용 가능)
        w_init=None, max_iter=100, tol=1e-8):
    """
    아래 4개의 식(네가 보낸 일반식)을 직접 풀고 싶을 때 사용하는 solver.
    (1) ∑ (C1 w_i^2 + C2 w_i + C3) = T
    (2) lC1(w4^2 - w2^2) + lC2(w4 - w2) = Mx
    (3) lC1(w3^2 - w1^2) + lC2(w3 - w1) = My
    (4) w1 + w2 - w3 - w4 = 0   ← yaw=0을 '합각속도' 제약으로 표현

    * 기본값(C1=k_f, C2=0, C3=0)로 두면 "선형 A^-1 방식"과 거의 같은 해를 준다.
    * SciPy 없이 뉴턴-랩슨 (수치 jacobian)으로 구현.
    """
    if C1 is None: C1 = params.k_f
    if C2 is None: C2 = 0.0
    if C3 is None: C3 = 0.0

    # yaw 방정식에 들어가는 계수 D1..D3는 여기선 쓰지 않지만, 필요하면 확장 가능
    l = params.l

    if w_init is None:
        # 닫힌형 해로 얻은 값으로 초기화하면 수렴 아주 빨라짐
        w0, _, _ = allocate_omegas_closed_form(T, Mx, My, params, Mz=0.0)
        w = w0.copy()
    else:
        w = np.asarray(w_init, dtype=np.float64)

    def F(w):
        w1, w2, w3, w4 = w
        eq1 = C1*(w1**2 + w2**2 + w3**2 + w4**2) + C2*(w1 + w2 + w3 + w4) + 4*C3 - T
        eq2 = l*(C1*(w4**2 - w2**2) + C2*(w4 - w2)) - Mx
        eq3 = l*(C1*(w3**2 - w1**2) + C2*(w3 - w1)) - My
        eq4 = (w1 + w2) - (w3 + w4)  # yaw=0 제약(합각속도 동일)
        return np.array([eq1, eq2, eq3, eq4], dtype=np.float64)

    def jacobian(w, eps=1e-6):
        J = np.zeros((4,4), dtype=np.float64)
        f0 = F(w)
        for i in range(4):
            w_eps = w.copy()
            w_eps[i] += eps
            fi = F(w_eps)
            J[:, i] = (fi - f0) / eps
        return J

    for _ in range(max_iter):
        f = F(w)
        if np.linalg.norm(f, ord=np.inf) < tol:
            break
        J = jacobian(w)
        try:
            delta = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            break
        w += delta
        w = np.clip(w, 0.0, None)  # 음수 방지 (필요에 따라 제거)

    # 최종 토크/전력
    tau_load = params.k_tau * (w ** 2)
    P = motor_power(w, tau_load, params)
    return w, tau_load, P


# ------------------------------------------------------------
# 5) 리워드에 바로 넣는 helper
# ------------------------------------------------------------
def energy_penalty_from_action(acc_world: np.ndarray,
                               vel_world: np.ndarray,
                               Mx: float,
                               My: float,
                               params: QuadParams,
                               dt: float,
                               use_newton: bool = False):
    """
    (ax,ay,az) + 현재 속도 -> 필요한 총추력 T 계산 -> (Mx, My, yaw=0) 로 분배 -> 각 모터 전력 -> dt 곱해서 에너지
    반환:
        penalty  : -lambda_energy * E
        omegas   : 각 모터 각속도
        power    : 각 모터 전력
        T, Mx, My: 디버깅용
    """
    T = thrust_from_acc(acc_world, vel_world, params)

    if not use_newton:
        w, tau_load, P = allocate_omegas_closed_form(T, Mx, My, params, Mz=0.0)
    else:
        w, tau_load, P = solve_omegas_newton(
            T, Mx, My, params,
            C1=params.k_f, C2=0.0, C3=0.0,
            D1=None, D2=None, D3=None
        )

    total_power = float(np.sum(P))
    energy = total_power * dt
    penalty = -params.lambda_energy * energy
    return penalty, w, P, dict(T=T, Mx=Mx, My=My, total_power=total_power, energy=energy)




class Fly_drone(gym.Env):
    MAX_VERTICES = 8

    def __init__(self):
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(4, ), dtype="float32") #set action space size, range
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9 + (self.MAX_VERTICES + 1) * 2 + 1,), dtype="float32") #set observation space size, range
        self.done = False
        self.episode = 0
        self.train = True
        self.rend = True
        self.total_return = 0
        self.score_avg = 0
        self.target_area = None
        self.POLY_ENLARGE = 3.0

    def step(self, action):
        global time
        global drone_xy, drone_xy_velocity, drone_alt, drone_z_velocity
        global roll, pitch, yaw, explored_area, all_polygons, drone_path

        # --------- 1) 액션 파싱 & yaw=0 강제 ---------
        ax, ay, az, _ = action.astype(np.float32)
        yaw = 0.0  # yaw는 쓰지 않으므로 0 고정  ### <--

        # --------- 2) 상태 적분 ---------
        time += DT
        # pos
        drone_xy[0] += ax * 0.5 * DT**2 + drone_xy_velocity[0] * DT
        drone_xy[1] += ay * 0.5 * DT**2 + drone_xy_velocity[1] * DT
        drone_alt   += az * 0.5 * DT**2 + drone_z_velocity      * DT
        # vel
        drone_xy_velocity[0] += ax * DT
        drone_xy_velocity[1] += ay * DT
        drone_z_velocity     += az * DT

        # --------- 3) DEM/ground alt ---------
        px, py = world_to_pixel(np.array([drone_xy[0]]), np.array([drone_xy[1]]))
        px, py = px[0], py[0]
        if not (0 <= px < width and 0 <= py < height):
            # out of bounds -> 끝내고 큰 페널티
            reward = -1000.0
            self.total_return += reward
            self.done = True
            return self._build_state(), reward, self.done, {}

        ground_alt = dem[py, px]

        # --------- 4) 시야 레이캐스팅 (기존) ---------
        u = np.linspace(-np.tan(FOV_RAD_X / 2), np.tan(FOV_RAD_X / 2), N)
        v = np.linspace(-np.tan(FOV_RAD_Y / 2), np.tan(FOV_RAD_Y / 2), N)
        uu, vv = np.meshgrid(u, v)
        dirs_local = np.stack([uu, vv, -np.ones_like(uu)], axis=-1).reshape(-1, 3)
        dirs_local /= np.linalg.norm(dirs_local, axis=1, keepdims=True)

        R = rpy_to_matrix(roll, pitch, yaw)
        dirs_world = dirs_local @ R.T
        intersections = []

        drone_path.append((drone_xy[0], drone_xy[1]))

        # state (관측) 구성용 (기존)
        state1 = [drone_xy[i] for i in range(2)]
        state2 = [drone_xy_velocity[i] for i in range(2)]
        state3 = [roll, pitch, yaw, drone_z_velocity, explored_area]
        target_vertices = np.array(self.target_area.exterior.coords)
        state = state1 + state2 + state3 + [drone_alt] + list(target_vertices.flatten())

        for d in dirs_world:
            ray_pos = np.array([*drone_xy, drone_alt], dtype=np.float32)
            for _ in range(MAX_STEPS):
                ray_pos += d * STEP_SIZE
                px_r, py_r = world_to_pixel(ray_pos[0], ray_pos[1])
                px_r, py_r = int(px_r), int(py_r)
                if 0 <= px_r < width and 0 <= py_r < height:
                    ground_z = dem[py_r, px_r]
                    if ray_pos[2] <= ground_z:
                        intersections.append((ray_pos[0], ray_pos[1]))
                        break
                else:
                    break

        # --------- 5) 면적 보상 (patched & robust) ---------
        area_reward = 0.0
        if intersections:
            # 1) 레이캐스팅 점 -> convex_hull
            hull = MultiPoint(intersections).convex_hull

            # 2) 고정된 꼭짓점 수 + valid 보정 + 조금 키우기
            hull_fixed = force_vertex_count(
                fix_polygon(hull).buffer(self.POLY_ENLARGE),
                self.MAX_VERTICES
            )

            # 3) target_area와 교차하는 부분만 계산
            if  self.target_area is not None:
                try:
                    effective_poly = fix_polygon(hull_fixed).intersection(self.target_area)
                    effective_poly = fix_polygon(effective_poly)
                except GEOSException:
                    # 매우 드물게 또 터질 수 있으니 fallback
                    effective_poly = fix_polygon(hull_fixed.buffer(0)).intersection(
                        fix_polygon(self.target_area.buffer(0))
                    )
                    effective_poly = fix_polygon(effective_poly)

            if not effective_poly.is_empty:
                if all_polygons:
                    total_area_poly = unary_union(all_polygons)
                    new_polygon = effective_poly.difference(total_area_poly)
                    area_reward = new_polygon.area
                else:
                    area_reward = effective_poly.area

                explored_area += area_reward
                # 이제부터는 invalid 가능성을 낮추기 위해 effective_poly(= 실제로 카운트한 면적)만 저장
                all_polygons.append(effective_poly)


        # --------- 6) 고도 유지 보상 ---------
        target_altitude = ground_alt + 10
        altitude_error = abs(drone_alt - target_altitude)
        altitude_reward = -0.1 * altitude_error
        if drone_alt < ground_alt:
            altitude_reward -= 1000
        elif drone_alt > ground_alt + 20:
            altitude_reward -= 500

        # --------- 7) 에너지 패널티 계산 ---------
        #   - 여기서 Mx, My를 "원하는 선형가속도"로부터 만들고,
        #   - yaw=0을 유지한다고 가정.
        acc_world = np.array([ax, ay, az], dtype=np.float64)
        vel_world = np.array([drone_xy_velocity[0], drone_xy_velocity[1], drone_z_velocity], dtype=np.float64)

        Mx, My = attitude_torque_from_acc(acc_world, roll, pitch, quad_params, kp=ATT_KP)  #  <-- 핵심
        penalty_E, omegas, powers, info_E = energy_penalty_from_action(
            acc_world, vel_world, Mx, My, quad_params, dt=DT
        )

        reward = area_reward + altitude_reward + penalty_E
        self.total_return += reward

        # --------- 8) 종료 조건 ---------
        self._check_done(ground_alt)
        if self.done:
            self.plot(self.train)
            self.episode += 1

        # info에 에너지 디버깅 정보 넣어두면 학습 중 로깅하기 좋음
        info = {"energy": info_E["energy"], "power_total": info_E["total_power"], "Mx": Mx, "My": My}
        return state, reward, self.done, info



    def reset(self):
        global time, drone_xy, drone_xy_velocity, drone_alt, drone_z_velocity, roll, pitch, yaw, explored_area, all_polygons, drone_path
        all_polygons = []
        self.total_return = 0
        self.done = False
        time = 0
        drone_xy = np.array([264000.0, 309500.0])
        drone_xy_velocity = np.array([0.0, 0.0])
        drone_z_velocity = 0.0
        roll, pitch, yaw, explored_area = 0, 0, 0, 0

        drone_path = []
        drone_path.append((drone_xy[0], drone_xy[1]))
        px, py = world_to_pixel(np.array([drone_xy[0]]), np.array([drone_xy[1]]))
        ground_alt = dem[py[0], px[0]]
        drone_alt = ground_alt + 10 #리셋 높이

        # ---- Create a random *valid* target area with fixed number of vertices ----
        center_x = drone_xy[0] + np.random.uniform(-50, 50)
        center_y = drone_xy[1] + np.random.uniform(-50, 50)

        num_points = self.MAX_VERTICES
        # “조금 더 큰 범위” -> 반지름 범위를 키움
        radius = np.random.uniform(500, 1000)

        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
        points = []
        for angle in angles:
            x = center_x + radius * np.cos(angle) + np.random.uniform(-10, 10)
            y = center_y + radius * np.sin(angle) + np.random.uniform(-10, 10)
            points.append((x, y))

        raw_target = Polygon(points)
        # 1) topology fix  2) 살짝 키우기(buffer)  3) 꼭짓점 수 강제
        self.target_area = force_vertex_count(
            fix_polygon(raw_target).buffer(self.POLY_ENLARGE),
            self.MAX_VERTICES
        )


        state1 = [drone_xy[i] for i in range(2)]
        state2 = [drone_xy_velocity[i] for i in range(2)]
        state3 = [roll, pitch, yaw, drone_z_velocity, explored_area]

        target_vertices = as_fixed_length_coords(self.target_area, self.MAX_VERTICES)
        state = state1 + state2 + state3 + [drone_alt] + list(target_vertices.flatten())

        return state
    
    def _check_done(self, ground_alt):
        # 1. Time limit
        if time >= 2000:
            self.done = True

        # 2. Collision with ground
        if drone_alt <= ground_alt:
            self.done = True
            self.total_return -= 1000 # Large penalty for collision

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

            # Plot the target area
            if self.target_area:
                x, y = self.target_area.exterior.xy
                ax.plot(x, y, 'y--', linewidth=2, label='Target Area')

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
            plt.savefig(f"Area_fig/episode_{self.episode}_map.svg")
            plt.close()

            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig("PPO_reward.png")