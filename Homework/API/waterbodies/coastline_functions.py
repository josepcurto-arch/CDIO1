import numpy as np
import scipy.ndimage as ndimage
import rasterio

def detect_waterbody(ndwi_array: np.ndarray) -> np.ndarray:
    """Agafa la matriu NDWI i genera un array de 1 i 0: 1 per aigua, 0 per terra."""
    waterbody = (ndwi_array > 0) & (~np.isnan(ndwi_array))
    return waterbody.astype(np.uint8)


def estimate_coastline(waterbody_array: np.ndarray) -> np.ndarray:
    """Obté la línia de costa mitjançant el gradient morfològic (dilatació - erosió)."""
    kernel = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(waterbody_array, structure=kernel)
    eroded = ndimage.binary_erosion(waterbody_array, structure=kernel)

    coastline = dilated.astype(int) - eroded.astype(int)
    return coastline.astype(np.uint8)


def save_as_geotiff(output_path: str, array: np.ndarray, reference_profile: dict) -> None:
    """Desa la matriu processada com un arxiu GeoTIFF preservant informació geogràfica."""
    profile = reference_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array, 1)