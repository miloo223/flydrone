import os, math, rasterio, numpy as np, mujoco, glfw
from mujoco import viewer
import time


TIF_PATH = "flydrone/fly_drone/envs/image_cut.tif"
with rasterio.open(TIF_PATH) as src:
    dem = src.read(1).astype(np.float32)          
    res_x, res_y = src.res                        
    nodata = src.nodata if src.nodata is not None else np.nan
dem = np.nan_to_num(dem, nan=nodata) #결측치 처리
dem -= dem.min()
dem /= dem.max() if dem.max() else 1.0 #정규화
dem = np.flipud(dem) #행 뒤집기

target = 500
if dem.shape != (target, target):
    from skimage.transform import resize
    dem = resize(dem, (target, target), order=1, anti_aliasing=True).astype(np.float32)


nrow, ncol = dem.shape
print(f"DEM ready : {nrow}×{ncol}")


SX = SY = 5000 #이게 5키로였나? 근데 5000으로 하면 걍 평평해보임
SZ = 1200             

'''xml = f"""
<mujoco>
  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}"
            size="{SX} {SY} {SZ} 0.05"/>
  </asset>

  <worldbody>
    <geom type="hfield" hfield="terrain"
          pos="0 0 0"
          rgba="0.6 0.5 0.4 1"/>
    <camera name="view" pos="491.262 -11127.187 9767.584" xyaxes="1.000 0.029 -0.000 -0.020 0.709 0.705"/>
  </worldbody>
</mujoco>
""" '''
xml = "flydrone/fly_drone/envs/skydio_x2/scene.xml"

model = mujoco.MjModel.from_xml_path(xml)
adr = model.hfield_adr[0]
model.hfield_data[adr: adr + nrow*ncol] = dem.flatten()
data = mujoco.MjData(model)     


with viewer.launch_passive(model, data) as v:
    start = time.time()
    while v.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        with v.lock():
          v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
        v.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)