import os, math, rasterio, numpy as np, mujoco, glfw
from mujoco import viewer


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


SX = SY = 5 #이게 5키로였나? 근데 5000으로 하면 걍 평평해보임
SZ = 1.2              

xml = f"""
<mujoco>
  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}"
            size="{SX} {SY} {SZ} 0.05"/>
  </asset>

  <worldbody>
    <geom type="hfield" hfield="terrain"
          pos="0 0 0"
          rgba="0.6 0.5 0.4 1"/>
    <camera name="angled_view" pos="0.073 -10.819 10.398" xyaxes="1.000 -0.000 0.000 0.000 0.707 0.707"/>
  </worldbody>
</mujoco>
""" 

model = mujoco.MjModel.from_xml_string(xml)
adr = model.hfield_adr[0]
model.hfield_data[adr: adr + nrow*ncol] = dem.flatten()
data = mujoco.MjData(model)     


with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)