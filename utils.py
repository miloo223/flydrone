# 좌표계 변환 등 기타 함수 모음
import numpy as np
from affine import Affine


def world_to_grid(x, y, transform):
    """
    실제(world) 좌표 → DEM 그리드 인덱스로 변환
    """
    col, row = ~transform * (x, y)
    return int(round(row)), int(round(col))