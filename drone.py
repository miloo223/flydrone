import numpy as np

def compute_sloped_area_vectorized(mask: np.ndarray,
                                   dem: np.ndarray,
                                   dx: float = 1.0,
                                   dy: float = None) -> float:
    """
    Vectorized computation of sloped surface area for large grids.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (H, W)
        True for explored cells.
    dem : np.ndarray of float, shape (H+1, W+1)
        Elevation values at grid points.
    dx : float
        Grid spacing in x-direction (default 1.0).
    dy : float, optional
        Grid spacing in y-direction (default same as dx).

    Returns
    -------
    float
        Total sloped surface area of explored cells.
    """
    if dy is None:
        dy = dx

    # 1) 탐색된 셀 인덱스 추출
    i_idx, j_idx = np.nonzero(mask)
    
    # 2) 네 꼭짓점 고도
    z00 = dem[i_idx, j_idx]
    z10 = dem[i_idx, j_idx + 1]
    z11 = dem[i_idx + 1, j_idx + 1]
    z01 = dem[i_idx + 1, j_idx]
    
    # 3) 네 꼭짓점 좌표
    x00 = j_idx * dx; y00 = i_idx * dy
    x10 = (j_idx + 1) * dx; y10 = y00
    x11 = x10;            y11 = (i_idx + 1) * dy
    x01 = x00;            y01 = y11
    
    p00 = np.stack([x00, y00, z00], axis=1)
    p10 = np.stack([x10, y10, z10], axis=1)
    p11 = np.stack([x11, y11, z11], axis=1)
    p01 = np.stack([x01, y01, z01], axis=1)
    
    # 4) 두 개의 삼각형 면적 계산
    v1 = p10 - p00
    v2 = p11 - p00
    area1 = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    
    v3 = p11 - p00
    v4 = p01 - p00
    area2 = 0.5 * np.linalg.norm(np.cross(v3, v4), axis=1)
    
    # 5) 전체 면적 합산
    return np.sum(area1 + area2)


if __name__ == "__main__":
    # 간단한 테스트
    dem = np.array([
        [100.0, 101.0, 100.5],
        [100.2, 100.8, 100.3],
        [ 99.9, 100.1, 100.0]
    ])
    mask = np.array([
        [True, False],
        [ True, True]
    ])
    area = compute_sloped_area_vectorized(mask, dem, dx=1.0)
    print(f"실제 탐색 면적: {area:.3f} m²")
