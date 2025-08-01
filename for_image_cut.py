#!/usr/bin/env python3
"""
make_small_dem.py

5 km  5 km 원본 DEM(@fly_drone/envs/image_cut.tif)을
정확히 1 km  1 km(가운데 부분)만 잘라
@fly_drone/envs/image_cut_1km.tif 으로 저장.
"""

import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

SRC_PATH = "fly_drone/envs/image_cut.tif"
DST_PATH = "fly_drone/envs/image_cut_1km.tif"

# ----------------------------------------------------------------------
def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"{SRC_PATH} 가 없습니다.")

    with rasterio.open(SRC_PATH) as src:
        # ── ① 해상도(픽셀당 m) 구하기 ───────────────────────────────
        #   transform.a  : +x 방향 해상도 (m/pixel)
        #   transform.e  : -y 방향 해상도 (m/pixel; 음수이므로 부호만 뒤집음)
        x_res = src.transform.a
        y_res = -src.transform.e

        # ── ② 1 km 에 해당하는 픽셀 수 산출 ─────────────────────────
        nx = int(round(1000 / x_res))      # 가로 픽셀
        ny = int(round(1000 / y_res))      # 세로 픽셀
        if nx > src.width or ny > src.height:
            raise ValueError("원본 DEM 이 1 km × 1 km 보다 작습니다.")

        # ── ③ 이미지 중앙을 기준으로 crop 윈도 계산 ─────────────────
        cx, cy = src.width // 2, src.height // 2
        x0 = cx - nx // 2
        y0 = cy - ny // 2
        win = Window(x0, y0, nx, ny)

        # ── ④ 데이터 읽고 새 GeoTIFF 작성 ───────────────────────────
        data = src.read(1, window=win)
        new_transform: Affine = src.window_transform(win)

        profile = src.profile
        profile.update({
            "height": ny,
            "width": nx,
            "transform": new_transform,
            "compress": "lzw"
        })

    os.makedirs(os.path.dirname(DST_PATH), exist_ok=True)
    with rasterio.open(DST_PATH, "w", **profile) as dst:
        dst.write(data, 1)

    print(f"✅ 1 km DEM 저장 완료 →  {DST_PATH}")
    print(f"   shape: {data.shape}, transform: {new_transform}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
