import numpy as np
import rasterio

def detect_waterbody(ndwi_array):
    # Agafa la matriu NDWI i genera un array de 1 i 0: 1 per aigua, 0 per terra.
   
    waterbody = (ndwi_array > 0) & (~np.isnan(ndwi_array))
    return waterbody.astype(np.uint8)

def save_as_geotiff(output_path, array, reference_profile):
    # Desa la matriu processada com un archiu GeoTIFF preservant informació geogràfica.
 
    profile = reference_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array, 1)