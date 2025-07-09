import numpy as np
import math


def compute_viewshed(dem, resolution, origin_x, origin_y,
                     obs_x, obs_y, obs_height,
                     yaw, fov_h,
                     num_directions=120, max_distance=None):
    """
    주어진 드론 위치·자세에서 시야 내 보이는 DEM 셀 마스크를 계산합니다.
    returns: visible_mask (bool 2D), (obs_i, obs_j, z_obs)
    """
    n_rows, n_cols = dem.shape
    cell = resolution
    if max_distance is None:
        max_distance = max(n_rows, n_cols) * cell

    # 드론 그리드 인덱스 및 절대 고도
    obs_i = int((obs_y - origin_y) / cell)
    obs_j = int((obs_x - origin_x) / cell)
    z_obs = dem[obs_i, obs_j] + obs_height

    visible = np.zeros_like(dem, dtype=bool)
    angles = np.linspace(yaw - fov_h/2, yaw + fov_h/2, num_directions)

    for theta in angles:
        max_angle = -np.inf
        step = 1
        while True:
            d = step * cell
            if d > max_distance:
                break
            di = int(round(step * math.sin(theta)))
            dj = int(round(step * math.cos(theta)))
            i, j = obs_i + di, obs_j + dj
            if not (0 <= i < n_rows and 0 <= j < n_cols):
                break
            angle = math.atan2(dem[i,j] - z_obs, d)
            if angle > max_angle:
                max_angle = angle
                visible[i,j] = True
            step += 1

    return visible, (obs_i, obs_j, z_obs)


def calculate_visible_area(mask, resolution):
    """
    bool 마스크를 받아 총 투영 면적을 계산
    """
    cell_area = resolution ** 2
    return np.sum(mask) * cell_area

# drone_viewshed.py (추가)

import numpy as np

def cell_true_area(i, j, dem, dx, dy):
    """그리드 셀 (i,j)의 실제 표면 면적을 삼각형 분할로 계산."""
    n, m = dem.shape
    # 가장자리 셀은 근사치(투영 면적)로 처리
    if i+1>=n or j+1>=m:
        return dx * dy

    z00 = dem[i  , j  ]
    z10 = dem[i  , j+1]
    z01 = dem[i+1, j  ]
    z11 = dem[i+1, j+1]

    # 3D 좌표
    P00 = np.array([0.0,  0.0,  z00])
    P10 = np.array([dx,   0.0,  z10])
    P01 = np.array([0.0,  dy,   z01])
    P11 = np.array([dx,   dy,   z11])

    # 삼각형 A: P00,P10,P01
    v1 = P10 - P00
    v2 = P01 - P00
    areaA = 0.5 * np.linalg.norm(np.cross(v1, v2))

    # 삼각형 B: P11,P10,P01
    v3 = P10 - P11
    v4 = P01 - P11
    areaB = 0.5 * np.linalg.norm(np.cross(v3, v4))

    return areaA + areaB

def calculate_true_visible_area(mask, dem, resolution):
    """mask가 True인 셀들의 실제 표면 면적 합."""
    dx = dy = resolution
    total = 0.0
    # mask[i,j] == True 셀에 대해 실제 면적 계산
    it = np.argwhere(mask)
    for (i,j) in it:
        total += cell_true_area(i, j, dem, dx, dy)
    return total
