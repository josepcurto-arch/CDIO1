import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from detect_water import detect_waterbody, save_as_geotiff
from coastline import estimate_coastline

# PAS 0: Definir la ruta base del projecte (~/CDIO1) per evitar errors de directori
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

project_name = "castelldefels_h1_2025"
date_str = "20250115" 

# Directori de sortida a coastline_estimator amb ruta absoluta
project_output_dir = os.path.join(
    BASE_DIR, f"coastline_estimator/projects/{project_name}/output/estimated_waterbodies_images"
)
os.makedirs(project_output_dir, exist_ok=True)

# PAS 1: Carregar dades i calcular/carregar NDWI utilitzant rutes absolutes
ndwi_path = os.path.join(BASE_DIR, "Homework/API/ndwi/ndwi.tif")
green_path = os.path.join(BASE_DIR, "Homework/API/green.tif")
nir_path = os.path.join(BASE_DIR, "Homework/API/nir.tif")

if os.path.exists(ndwi_path):
    with rasterio.open(ndwi_path) as src:
        ndwi_data = src.read(1)
        profile = src.profile
else:
    with rasterio.open(green_path) as src_green, rasterio.open(nir_path) as src_nir:
        green = src_green.read(1).astype(float)
        nir = src_nir.read(1).astype(float)
        profile = src_green.profile
        
        denominator = green + nir
        ndwi_data = np.where(denominator == 0, np.nan, (green - nir) / denominator)

# PAS 2: Crear la màscara d'aigua i estimar la línia de costa
waterbody_array = detect_waterbody(ndwi_data)
coastline_array = estimate_coastline(waterbody_array)

# PAS 3: Guardar el GeoTIFF oficial
waterbody_filename = f"waterbody_{date_str}.tif"
save_as_geotiff(os.path.join(project_output_dir, waterbody_filename), waterbody_array, profile)

# PAS 4: Generar i guardar les imatges PNG de comprovació
plt.figure()
plt.imshow(waterbody_array, cmap='Blues', vmin=0, vmax=1)
plt.title(f"Waterbody Detection ({date_str})")
plt.axis('off')
plt.savefig(os.path.join(project_output_dir, f"Resultats_waterbody_{date_str}.png"), bbox_inches='tight')
plt.close()

plt.figure()
plt.imshow(coastline_array, cmap='Reds', vmin=0, vmax=1)
plt.title(f"Coastline Estimation ({date_str})")
plt.axis('off')
plt.savefig(os.path.join(project_output_dir, f"Resultats_coastline_{date_str}.png"), bbox_inches='tight')
plt.close()

print("¡Procés completat amb èxit!")
print(f"Tots els fitxers s'han guardat a: {project_output_dir}")