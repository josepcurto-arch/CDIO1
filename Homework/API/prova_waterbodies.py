import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from detect_water import detect_waterbody, save_as_geotiff
from coastline import estimate_coastline

# PAS 0: Crear la carpeta de sortida dins de Homework/API
output_dir = "Homework/API/outputs"
os.makedirs(output_dir, exist_ok=True)

# PAS 1: Carregar dades i calcular/carregar NDWI
# Intentem obrir el fitxer NDWI si existeix; si no, el calculem amb green i nir
ndwi_path = "Homework/API/ndwi/ndwi.tif"

if os.path.exists(ndwi_path):
    with rasterio.open(ndwi_path) as src:
        ndwi_data = src.read(1)
        profile = src.profile
else:
    # Si no tenim el NDWI generat, el calculem a partir de les bandes green i nir
    with rasterio.open("Homework/API/green.tif") as src_green, rasterio.open("Homework/API/nir.tif") as src_nir:
        green = src_green.read(1).astype(float)
        nir = src_nir.read(1).astype(float)
        profile = src_green.profile
        
        # Fórmula NDWI = (Green - NIR) / (Green + NIR)
        denominator = green + nir
        ndwi_data = np.where(denominator == 0, np.nan, (green - nir) / denominator)

# PAS 2: Crear la màscara d'aigua i estimar la línia de costa
waterbody_array = detect_waterbody(ndwi_data)
coastline_array = estimate_coastline(waterbody_array)

# PAS 3: Guardar els GeoTIFF a la carpeta outputs
save_as_geotiff(os.path.join(output_dir, "waterbody.tif"), waterbody_array, profile)
save_as_geotiff(os.path.join(output_dir, "coastline.tif"), coastline_array, profile)

# PAS 4: Generar i guardar les imatges PNG amb el contrast correcte (vmin=0, vmax=1)
plt.figure()
plt.imshow(waterbody_array, cmap='Blues', vmin=0, vmax=1)
plt.title("Waterbody Detection")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "Resultats_waterbody.png"), bbox_inches='tight')
plt.close()

plt.figure()
plt.imshow(coastline_array, cmap='Reds', vmin=0, vmax=1)
plt.title("Coastline Estimation")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "Resultats_coastline.png"), bbox_inches='tight')
plt.close()

print("¡Procés completat amb èxit!")
print(f"Fitxers guardats correctament a: {output_dir}")