import rasterio
import numpy as np

def load_dem(tif_path: str) -> np.ndarray:
    """
    Loads a DEM from a GeoTIFF file.

    Args:
        tif_path: Path to the .tif file.

    Returns:
        A 2D numpy array containing the elevation data.
    """
    try:
        with rasterio.open(tif_path) as dataset:
            # Read the first band
            dem_data = dataset.read(1)
            resolution = dataset.res[0] # Assuming square pixels
            print(f"Successfully loaded DEM data from {tif_path}. Shape: {dem_data.shape}, Resolution: {resolution}")
            return dem_data, resolution
    except Exception as e:
        print(f"Error loading DEM file: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage:
    dem = load_dem('dem_data/image.tif')
    if dem is not None:
        print(f"DEM data loaded. Min elevation: {np.min(dem)}, Max elevation: {np.max(dem)}")
