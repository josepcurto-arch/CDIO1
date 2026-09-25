from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Files
script_dir = Path(__file__).parent
green_path = script_dir / "green.tif"
nir_path = script_dir / "nir.tif"

# 1. Open files
with rasterio.open(green_path) as src_green:
    green = src_green.read(1).astype(np.float32)

with rasterio.open(nir_path) as src_nir:
    nir = src_nir.read(1).astype(np.float32)

# 2. NDWI = (Green - NIR) / (Green + NIR) with error control
with np.errstate(divide="ignore", invalid="ignore"): # Doesn't show a error each time it divides by zero
    ndwi = (green - nir) / (green + nir)
    ndwi = np.nan_to_num(ndwi, nan=0.0)

# 3. Visualise
plt.figure(figsize=(10, 8))
plt.imshow(ndwi, cmap="Blues")
plt.colorbar(label="Índex NDWI")
plt.title("Normalized Difference Water Index (NDWI  )")
plt.show()
