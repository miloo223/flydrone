import os
import gym
from gym import spaces
import numpy as np
import math
import pylab
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from matplotlib.patches import Polygon as MplPolygon
from shapely.ops import unary_union
import numpy as np
from dataclasses import dataclass
from shapely.validation import make_valid
from shapely.errors import GEOSException
from typing import Optional

env_name = "fly_drone-v0"

scores, episodes = [], []
all_polygons = []
drone_path = []
tif_path = "fly_drone/envs/image_cut_1km.tif"
#tif_path = "fly_drone/envs/image_cut.tif"

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

# 광선 개수 조절
N = 16 # 원래 30이었는데 16으로 줄임
STEP_SIZE = 1.0
MAX_STEPS = 100 # 원래 200이었음

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


# 현재 시간 불러와서 그걸로 폴더 저장
from pathlib import Path
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

class Fly_drone(gym.Env):
    MAX_VERTICES = 8
    # 튜닝값 쓸거임
    #"w_area": 2.560835061994321,
    #"w_alt": 1.5048324796298398,
    #"w_energy": 0.005251185724086418,
    def __init__(self, log_dir: Path, plot_dir: Path, w_area : float = 0.02, w_alt : float = 0.5, w_energy: float = 0.1, **kwargs):
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(2, ), dtype="float32") #set action space size, range
        self.RADAR_N = 24
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12+self.RADAR_N,), dtype="float32") #set observation space size, range
        self.done = False
        self.episode = 0
        self.train = True
        self.rend = True
        self.total_return = 0
        self.score_avg = 0
        #self.target_area = None
        self.POLY_ENLARGE = 3.0
        self.max_xy_speed = 8.0   # 수평(xy) 최대 속도 [m/s]
        self.max_z_speed = 5.0     # 수직(z) 최대 속도 [m/s]
        self._log_dir  = log_dir
        self._plot_dir = plot_dir
        self.LOG_EVERY_STEPS = 20 # 에피소드 중 특정 스텝마다 로그 
        self._step_idx = 0
        self._log_buf  = []
        # 하이퍼 파라미터
        self.w_area   = w_area
        self.w_alt    = w_alt
        self.w_energy = w_energy
        self.idle_counter = 0       
        self._prev_area   = 0.0     

        (self._plot_dir / "maps").mkdir(parents=True, exist_ok=True)

        # ── 표준화용 상수 (DEM bounds 활용)
        self.map_w = float(bounds.right - bounds.left)
        self.map_h = float(bounds.top   - bounds.bottom)
        self.cx    = float((bounds.left + bounds.right)  * 0.5)
        self.cy    = float((bounds.bottom + bounds.top)  * 0.5)
        # 월드 경계 사각형(라인 교차에 사용) + 라이다 최대 사거리
        self.world_poly = Polygon([
            (bounds.left,  bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left,  bounds.top)
        ])
        self.radar_R = max(self.map_w, self.map_h) + 1e-6  # 최대 레인지


        # --- (에너지 정규화용 Hover 에너지 계산 & 람다 설정) ---
    acc0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vel0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    Mx0, My0 = 0.0, 0.0

    # hover 에너지 한 스텝치 (lambda_energy는 계산에 영향 없음, info의 energy만 사용)
    _, _, _, info0 = energy_penalty_from_action(acc0, vel0, Mx0, My0, quad_params, dt=DT)
    E_hover = max(float(info0["energy"]), 1e-12)  # 0 division 방지

    # 한 스텝에 원하는 '가중치 전' 패널티 규모 (예: -0.1)
    target_penalty_per_step = 0.1
    # energy_penalty_from_action 이 반환하는 penalty = -lambda_energy * energy 이므로
    # lambda를 "목표 패널티 / Hover 에너지"로 두면, 정지/hover 시 약 -0.1이 됨
    quad_params.lambda_energy = target_penalty_per_step / E_hover


    def _flush_log(self):
        """20스텝마다 호출되어 _log_buf → CSV 로 쓴다."""
        if not self._log_buf:
            return

        fname = self._log_dir / f"episode_{self.episode}.csv"

        # ── (A) 새 에피소드면 헤더부터 쓰기 ───────────────────────
        write_header = not fname.exists()
        mode = "a"    # append, 헤더가 필요하면 아래에서 w+a 처리

        with open(fname, "a") as f:
            if write_header:
                f.write("time,x,y,z\n")        # 헤더 한 줄
            for t, x, y, z in self._log_buf:
                f.write(f"{t:.2f},{x:.2f},{y:.2f},{z:.2f}\n")

        self._log_buf.clear()

    

    def _limit_speed(self):
        global drone_xy_velocity, drone_z_velocity
        # XY 평면 속도 제한 (벡터 노름 기준)
        v_xy = np.linalg.norm(drone_xy_velocity)
        if v_xy > self.max_xy_speed:
            drone_xy_velocity *= (self.max_xy_speed / (v_xy + 1e-8))
        # Z 속도 제한 (스칼라이므로 clip)
        #drone_z_velocity = float(np.clip(drone_z_velocity, -self.max_z_speed, self.max_z_speed))

    def seed(self, seed: Optional[int] = None): 
        #옵튜나 때문에 만든 함수임
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        # 환경 안에서 np.random 대신 self.np_random 사용 권장
        return [seed]
    
    def _radar_intersect_dists_(self, ray: LineString, geom):
        """
        ray ∩ geom에서 '시작점→교차점'까지의 거리 목록을 반환.
        geom은 Point/MultiPoint/LineString/MultiLineString/GeometryCollection 모두 처리.
        """
        if geom is None:
            return []
        inter = ray.intersection(geom)
        if inter.is_empty:
            return []
        pts = []
        gt = inter.geom_type
        if gt == "Point":
            pts = [inter]
        elif gt == "MultiPoint":
            pts = list(inter.geoms)
        elif gt in ("LineString", "MultiLineString"):
            # 겹치는 경우: 선분의 양 끝을 후보로 사용 (0 거리가 나올 수 있음)
            geoms = [inter] if gt == "LineString" else list(inter.geoms)
            for g in geoms:
                c = list(g.coords)
                if len(c) >= 2:
                    pts.append(Point(c[0]))
                    pts.append(Point(c[-1]))
        elif gt == "GeometryCollection":
            for g in inter.geoms:
                if g.geom_type == "Point":
                    pts.append(g)
                elif g.geom_type == "LineString":
                    c = list(g.coords)
                    if len(c) >= 2:
                        pts.append(Point(c[0])); pts.append(Point(c[-1]))
        # ray.project는 선분 거리(0..길이). 지금 ray는 직선 한 구간이라 유클리드 거리와 동일.
        return [ray.project(p) for p in pts]


    def _frontier_radar(self, x: float, y: float) -> np.ndarray:
        """
        드론 (x,y)에서 RADAR_N개 방향으로 레이를 쏴
        '탐색 경계(frontier) 또는 월드 경계'까지의 최단 거리를 [0..1]로 정규화하여 반환.
        """
        # 후보 경계: (1) 이미 탐색한 영역의 경계, (2) 월드 경계
        frontier = None
        if all_polygons:
            try:
                union_poly = unary_union(all_polygons)
                if not union_poly.is_empty:
                    frontier = union_poly.boundary
            except Exception:
                frontier = None
        world_boundary = self.world_poly.boundary

        # 레이들 생성
        R = self.radar_R
        angles = np.linspace(0.0, 2.0*np.pi, num=self.RADAR_N, endpoint=False)
        dists = np.empty(self.RADAR_N, dtype=np.float32)
        for i, th in enumerate(angles):
            x2 = x + R * np.cos(th)
            y2 = y + R * np.sin(th)
            ray = LineString([(x, y), (x2, y2)])

            cand = []
            if frontier is not None and not frontier.is_empty:
                cand += self._radar_intersect_dists_(ray, frontier)
            cand += self._radar_intersect_dists_(ray, world_boundary)

            # 유효 후보 없으면 최대 사거리로 간주
            d = min(cand) if len(cand) > 0 else R
            #dists[i] = np.clip(d / R, 0.0, 1.0)
            scale = getattr(self, "RADAR_SCALE", 50.0)  # 50m 스케일(원하면 조정)
            dists[i] = float(1.0 - np.exp(-d / scale))

        return dists


    def _build_state(self) -> np.ndarray:
        # 1) 위치·속도·자세·면적
        state = [
            drone_xy[0], drone_xy[1],
            drone_xy_velocity[0], drone_xy_velocity[1], drone_z_velocity,
            roll, pitch, yaw,
            explored_area,
        ]

        # 2) 정규화 좌표: 중심 (cx, cy) 기준 [-1, 1]
        nx = np.clip((drone_xy[0] - self.cx) / (self.map_w * 0.5 + 1e-9), -1.0, 1.0)
        ny = np.clip((drone_xy[1] - self.cy) / (self.map_h * 0.5 + 1e-9), -1.0, 1.0)

        # 3) 경계 근접도: 0(중앙) → 1(벽 바로 앞)
        sx = (drone_xy[0] - bounds.left)   / (self.map_w + 1e-9)  # 0..1
        sy = (drone_xy[1] - bounds.bottom) / (self.map_h + 1e-9)  # 0..1
        margin_ratio = min(sx, 1.0 - sx, sy, 1.0 - sy)            # 0(벽)~0.5(중앙)
        edge_prox = float(np.clip(1.0 - 2.0 * margin_ratio, 0.0, 1.0))

        state += [nx, ny, edge_prox]

        radar = self._frontier_radar(drone_xy[0], drone_xy[1])
        state += radar.tolist()
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        global time
        global drone_xy, drone_xy_velocity, drone_alt, drone_z_velocity
        global roll, pitch, yaw, explored_area, all_polygons, drone_path

        # --------- 1) 액션 파싱 & yaw=0 강제 ---------
        ax, ay = action.astype(np.float32)
        yaw = 0.0  # yaw는 쓰지 않으므로 0 고정  ### <--

        # --------- 3) DEM/ground alt ---------
        px, py = world_to_pixel(np.array([drone_xy[0]]), np.array([drone_xy[1]]))
        px, py = px[0], py[0]

        if not (0 <= px < width and 0 <= py < height):
            # out of bounds -> 끝내고 큰 페널티
            reward = -1000.0
            self.total_return += reward
            self.done = True
            ground_alt = None
        else:
            ground_alt = dem[py, px]
            drone_alt = ground_alt + 10.0

        # --------- 2) 상태 적분 ---------
        time += DT
        self._step_idx += 1

        # pos
        drone_xy[0] += ax * 0.5 * DT**2 + drone_xy_velocity[0] * DT
        drone_xy[1] += ay * 0.5 * DT**2 + drone_xy_velocity[1] * DT
        #drone_alt   = dem[py, px] + 10
        # vel
        drone_xy_velocity[0] += ax * DT
        drone_xy_velocity[1] += ay * DT
        drone_z_velocity = 0.0
        self._limit_speed()


        '''
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
        '''
        # ------------------------------------------------------------------
        # 4) 시야 레이캐스팅 (벡터화 버전)
        # ------------------------------------------------------------------
        u = np.linspace(-np.tan(FOV_RAD_X / 2), np.tan(FOV_RAD_X / 2), N)
        v = np.linspace(-np.tan(FOV_RAD_Y / 2), np.tan(FOV_RAD_Y / 2), N)
        dirs_local = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)        # (M,2)
        dirs_local = np.c_[dirs_local, -np.ones(len(dirs_local))]               # (M,3)
        dirs_local /= np.linalg.norm(dirs_local, axis=1, keepdims=True)

        R          = rpy_to_matrix(roll, pitch, yaw)
        dirs_world = dirs_local @ R.T                                           # (M,3)

        drone_xyz  = np.array([*drone_xy, drone_alt], dtype=np.float32)
        M,  S      = len(dirs_world), MAX_STEPS
        dists      = np.arange(1, S + 1, dtype=np.float32) * STEP_SIZE          # (S,)

        # ▶ 모든 (ray × step) 좌표 한꺼번에 계산
        pts = (drone_xyz + dirs_world[:, None, :] * dists[None, :, None])       # (M,S,3)
        pts2d = pts[..., :2].reshape(-1, 2)                                     # (M·S,2)

        # 픽셀 인덱스
        px = ((pts2d[:, 0] - x0) / x_res).astype(int)
        py = ((y0 - pts2d[:, 1]) / y_res).astype(int)
        valid = (0 <= px) & (px < width) & (0 <= py) & (py < height)

        # 지형 고도 vs 광선 z 비교
        z_ground = np.empty_like(px, dtype=np.float32)
        z_ground[valid] = dem[py[valid], px[valid]]
        z_ray = pts[..., 2].reshape(-1)
        hit = valid & (z_ray <= z_ground)

        # ▶ 각 레이에 대해 '가장 처음' 맞은 인덱스 추출
        hit_idx_flat = np.where(hit)[0]
        ray_idx      = hit_idx_flat // S
        step_idx     = hit_idx_flat %  S
        first_hit_mask = np.zeros_like(hit, dtype=bool)
        first_hit_mask[np.minimum.reduceat(hit_idx_flat, np.unique(ray_idx, return_index=True)[1])] = True

        intersections = pts2d[first_hit_mask].tolist()          # [(x,y), …]

        drone_path.append((*drone_xy, drone_alt))               # z 포함 저장 권장

                # state (관측) 구성용 (기존)
        state1 = [drone_xy[i] for i in range(2)]
        state2 = [drone_xy_velocity[i] for i in range(2)]
        state3 = [roll, pitch, yaw, drone_z_velocity, explored_area]
        state = state1 + state2 + state3

        nx = np.clip((drone_xy[0] - self.cx) / (self.map_w * 0.5 + 1e-9), -1.0, 1.0)
        ny = np.clip((drone_xy[1] - self.cy) / (self.map_h * 0.5 + 1e-9), -1.0, 1.0)

        # 3) 경계 근접도: 0(중앙) → 1(벽 바로 앞)
        sx = (drone_xy[0] - bounds.left)   / (self.map_w + 1e-9)  # 0..1
        sy = (drone_xy[1] - bounds.bottom) / (self.map_h + 1e-9)  # 0..1
        margin_ratio = min(sx, 1.0 - sx, sy, 1.0 - sy)            # 0(벽)~0.5(중앙)
        edge_prox = float(np.clip(1.0 - 2.0 * margin_ratio, 0.0, 1.0))

        state += [nx, ny, edge_prox]
        radar = self._frontier_radar(drone_xy[0], drone_xy[1])
        state += radar.tolist()


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

            '''
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
            '''
            effective_poly = fix_polygon(hull_fixed)
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
        if area_reward < 1.0:              # 새 면적 < 1 m²
            self.idle_counter += 1
        else:
            self.idle_counter  = 0

        area_reward *= self.w_area

        '''
        # --------- 6) 고도 유지 보상 ---------
        target_altitude = ground_alt + 10
        altitude_error = abs(drone_alt - target_altitude)
        #altitude_reward = -0.02 * altitude_error
        #if drone_alt < ground_alt:
        #    altitude_reward -= 1000
        alt_diff = abs(drone_alt - (ground_alt + 10.0)) #타겟과의 거리
        #altitude_reward = -0.05 * (alt_diff ** 2)            
        altitude_reward = -max(0.0, np.exp(alt_diff* 0.05))
        altitude_reward *= self.w_alt
        '''

        # --------- 7) 에너지 패널티 계산 ---------
        #   - 여기서 Mx, My를 "원하는 선형가속도"로부터 만들고,
        #   - yaw=0을 유지한다고 가정.
        acc_world = np.array([ax, ay, 0.0], dtype=np.float64) # 목표 az는 0으로 고정
        vel_world = np.array([drone_xy_velocity[0], drone_xy_velocity[1], 0.0], dtype=np.float64)

        Mx, My = attitude_torque_from_acc(acc_world, roll, pitch, quad_params, kp=ATT_KP)  #  <-- 핵심
        penalty_E, omegas, powers, info_E = energy_penalty_from_action(
            acc_world, vel_world, Mx, My, quad_params, dt=DT
        )
        penalty_E  *= self.w_energy

        # 전체 리워드
        reward = area_reward  + penalty_E
        self.total_return += reward

        # --------- 8) 종료 조건 ---------
        px, py = world_to_pixel(np.array([drone_xy[0]]), np.array([drone_xy[1]]))
        px, py = px[0], py[0]
        self._check_done()

        if self.done:
            self.plot(self.train)
            self.episode += 1

        self._log_buf.append((time, drone_xy[0], drone_xy[1], drone_alt))
        if self._step_idx % self.LOG_EVERY_STEPS == 0:
            self._flush_log()
        

        # info에 에너지 디버깅 정보 넣어두면 학습 중 로깅하기 좋음
        info = {"energy": info_E["energy"], "power_total": info_E["total_power"], "Mx": Mx, "My": My}


        # 100번마다 로그 나옴
        '''
        if self._step_idx % 100 == 0:      # 100스텝마다
            #print(f"[env] ep{self.episode} step{self._step_idx}"
            #    f"  t={time:.1f}s  reward={reward:.2f}")
            v_xy = np.linalg.norm(drone_xy_velocity)
            print((
                f"[env] ep{self.episode:04d}  step{self._step_idx:05d}  t={time:6.1f}s\n"
                f"       ▸ area_r={area_reward:8.2f}   alt_r={altitude_reward:8.2f}   "
                f"eng_r={penalty_E:8.2f}   total_r={reward:8.2f}\n"
                f"       ▸ explored={explored_area:8.1f} m²   "
                f"pos=({drone_xy[0]:.1f}, {drone_xy[1]:.1f}, {drone_alt:.1f})   "
                f"v_xy={v_xy:.2f} m/s   v_z={drone_z_velocity:.2f} m/s"
            ))
            '''
        
        if self._step_idx % 100 == 0:
            delta_area = explored_area - getattr(self, "_prev_area", 0)
            self._prev_area = explored_area
            radar = self._frontier_radar(drone_xy[0], drone_xy[1])
            v_xy = np.linalg.norm(drone_xy_velocity)
            print(f"[{self.episode:04d}|{self._step_idx:05d}]"
                f" Δarea={delta_area:6.1f}  explored={explored_area:8.1f}"
                f" | r_area={area_reward:+6.2f}" 
                f" r_E={penalty_E:+6.2f}"
                f" | z={drone_alt:7.1f}"
                f" | v_xy={v_xy:.2f} m/s   v_z={drone_z_velocity:.2f} m/s \n"
                f"radar[:]={radar[:]}"
                )

        # ---------------------------------------------------------

        return state, reward, self.done, info



    def reset(self):
        global time, drone_xy, drone_xy_velocity, drone_alt, drone_z_velocity, roll, pitch, yaw, explored_area, all_polygons, drone_path
        all_polygons = []
        self.total_return = 0
        self.idle_counter = 0
        self.done = False
        time = 0
        self._prev_area = 0.0
        drone_xy = np.array([264300.0, 309370.0])
        drone_xy_velocity = np.array([0.0, 0.0])
        #drone_z_velocity = 0.0
        roll, pitch, yaw, explored_area = 0, 0, 0, 0
        dem_minx, dem_miny, dem_maxx, dem_maxy = bounds

        drone_path = []
        drone_path.append((drone_xy[0], drone_xy[1]))
        px, py = world_to_pixel(np.array([drone_xy[0]]), np.array([drone_xy[1]]))
        ground_alt = dem[py[0], px[0]]
        drone_alt = ground_alt + 10 #리셋 높이
        #alt_error  = drone_alt - (ground_alt + 10.0)


        # ---- Create a random *valid* target area with fixed number of vertices ----
        #center_x = np.clip(drone_xy[0] + np.random.uniform(-150, 150), dem_minx + 100, dem_maxx - 100)
        #center_y = np.clip(drone_xy[1] + np.random.uniform(-150, 150), dem_miny + 100, dem_maxy - 100)


        #num_points = self.MAX_VERTICES
        #radius = np.random.uniform(100, 200)
        '''
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
        '''

        state1 = [drone_xy[i] for i in range(2)]
        state2 = [drone_xy_velocity[i] for i in range(2)]
        state3 = [roll, pitch, yaw, drone_z_velocity, explored_area]
        state = state1 + state2 + state3 

        nx = np.clip((drone_xy[0] - self.cx) / (self.map_w * 0.5 + 1e-9), -1.0, 1.0)
        ny = np.clip((drone_xy[1] - self.cy) / (self.map_h * 0.5 + 1e-9), -1.0, 1.0)

        # 3) 경계 근접도: 0(중앙) → 1(벽 바로 앞)
        sx = (drone_xy[0] - bounds.left)   / (self.map_w + 1e-9)  # 0..1
        sy = (drone_xy[1] - bounds.bottom) / (self.map_h + 1e-9)  # 0..1
        margin_ratio = min(sx, 1.0 - sx, sy, 1.0 - sy)            # 0(벽)~0.5(중앙)
        edge_prox = float(np.clip(1.0 - 2.0 * margin_ratio, 0.0, 1.0))

        state += [nx, ny, edge_prox]
        radar = self._frontier_radar(drone_xy[0], drone_xy[1])
        state += radar.tolist()

        return state
    
    def _check_done(self):
        # 1. Time limit
        if time >= 300:
            self.done = True

        '''
        # 2. Collision with ground
        if drone_alt <= ground_alt:
            self.done = True
            self.total_return -= 50 # Large penalty for collision
        '''

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

            self._plot_dir.mkdir(exist_ok=True)
            ax.set_title(f"Episode {self.episode}: Explored Area and Path")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(self._plot_dir / "maps" / f"episode_{self.episode}_map.svg")
            plt.close()

            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig(self._plot_dir /"reward_curve.png")
