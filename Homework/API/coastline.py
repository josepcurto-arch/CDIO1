import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import yaml
import geopandas as gpd
import pystac_client
import stackstac
import numpy as np
import xarray as xr
import rioxarray
import rasterio
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib


# --- CONFIGURACIÓ DE MATPLOTLIB PER A MULTITHREADING ---
matplotlib.use('Agg')  # Evita el UserWarning de Matplotlib GUI
import matplotlib.pyplot as plt


print("Iniciant programa!")
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ------------------------------------------------------------------------------
# FUNCIONS DE PROCESSAMENT D'IMATGES (WATERBODY I COASTLINE)
# ------------------------------------------------------------------------------
def detect_waterbody(ndwi_array):
    """Agafa la matriu NDWI i genera un array de 1 i 0: 1 per aigua, 0 per terra."""
    waterbody = (ndwi_array > 0) & (~np.isnan(ndwi_array))
    return waterbody.astype(np.uint8)


def estimate_coastline(waterbody_array):
    """Obté la línia de costa mitjançant el gradient morfològic (dilatació - erosió)."""
    kernel = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(waterbody_array, structure=kernel)
    eroded = ndimage.binary_erosion(waterbody_array, structure=kernel)

    coastline = dilated.astype(int) - eroded.astype(int)
    return coastline.astype(np.uint8)


def save_as_geotiff(output_path, array, reference_profile):
    """Desa la matriu processada com un arxiu GeoTIFF preservant informació geogràfica."""
    profile = reference_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array, 1)


# ------------------------------------------------------------------------------
# 1. CARREGAR CONFIGURACIÓ I RUTES
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directori del script
project_root = os.path.dirname(script_dir)  # Directori arrel del projecte

config_path = os.path.join(project_root, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Polígon GeoJSON definit a la configuració
geojson_path = os.path.join(script_dir, config["geojson_file"])

try:
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    geometry = gdf.geometry.iloc[0].__geo_interface__
    bbox = list(gdf.total_bounds)
    print("Arxiu de configuració i GeoJSON carregats correctament.")
except Exception as e:
    print(f"Error en carregar el GeoJSON: {e}")
    exit(1)

# ------------------------------------------------------------------------------
# 2. PARÀMETRES DE CERCA AL CATÀLEG STAC
# ------------------------------------------------------------------------------
BANDS = config["bands"]
MAX_CLOUD_COVER = config["search"]["max_cloud_cover"]
MAX_WORKERS = config.get("performance", {}).get("max_workers", 4)
CHUNK_SIZE = config.get("performance", {}).get("chunk_size", 2048)

try:
    print("Cercant elements al catàleg que compleixin els filtres...")
    catalog = pystac_client.Client.open(config["search"]["catalog_url"])
    search = catalog.search(
        collections=[config["search"]["collection"]],
        datetime=f"{config['search']['start_date']}/{config['search']['end_date']}",
        intersects=geometry,
        query=[f"eo:cloud_cover<{MAX_CLOUD_COVER}"],
    )
    items = search.item_collection()
    print(
        f"{len(items)} elements trobats amb menys del {MAX_CLOUD_COVER}% de núvols."
    )
except Exception as e:
    print(f"Error durant la cerca al catàleg STAC: {e}")
    exit(1)

if len(items) == 0:
    print("No s'han trobat imatges per al període especificat.")
    exit(0)

# ------------------------------------------------------------------------------
# 3. PREPARACIÓ DE CARPETES DE SORTIDA UTILITZANT EL CONFIG
# ------------------------------------------------------------------------------
output_cfg = config.get("outputs", {})

folders = {
    "red": os.path.join(project_root, "banda_red"),
    "green": os.path.join(project_root, "banda_green"),
    "blue": os.path.join(project_root, "banda_blue"),
    "nir": os.path.join(project_root, "banda_nir"),
    "ndwi": os.path.join(project_root, "ndwi"),
    "rgb": os.path.join(project_root, "rgb"),
    "waterbody": os.path.join(
        project_root, output_cfg.get("waterbody_dir", "outputs/waterbody")
    ),
    "coastline": os.path.join(
        project_root, output_cfg.get("coastline_dir", "outputs/coastline")
    ),
    "plots": os.path.join(
        project_root, output_cfg.get("plots_dir", "outputs/plots")
    ),
}

for folder_path in folders.values():
    os.makedirs(folder_path, exist_ok=True)

# ------------------------------------------------------------------------------
# 4. CREACIÓ DE L'STACK
# ------------------------------------------------------------------------------
stack = stackstac.stack(
    items, assets=BANDS, bounds_latlon=bbox, epsg=4326, chunksize=CHUNK_SIZE
)


# ------------------------------------------------------------------------------
# 5. FUNCIÓ MULTITHREAD DE PROCESSAMENT, ANÀLISI I DESAR
# ------------------------------------------------------------------------------
def process_single_date(t_idx):
    try:
        data_date = stack.isel(time=t_idx)
        raw_time = str(data_date.time.values)

        # Extreure la data en format AAAAMMDD (e.g. 20250315)
        date_str = raw_time[:10].replace("-", "")

        # 1. Exportar bandes individuals a GeoTIFF
        for band_name in BANDS:
            band_data = data_date.sel(band=band_name)
            output_path = os.path.join(
                folders[band_name], f"{band_name}_{date_str}.tif"
            )
            band_data.rio.write_crs("EPSG:4326").rio.to_raster(output_path)

        # 2. Exportar composició RGB a GeoTIFF
        rgb_stack = data_date.sel(band=["red", "green", "blue"])
        rgb_output_path = os.path.join(folders["rgb"], f"rgb_{date_str}.tif")
        rgb_stack.rio.write_crs("EPSG:4326").rio.to_raster(rgb_output_path)

        # 3. Càlcul i exportació del NDWI a GeoTIFF
        green = data_date.sel(band="green")
        nir = data_date.sel(band="nir")

        denominator = green + nir
        ndwi = xr.where(
            np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0
        )
        ndwi = ndwi.fillna(0.0)

        ndwi_output_path = os.path.join(folders["ndwi"], f"ndwi_{date_str}.tif")
        ndwi.rio.write_crs("EPSG:4326").rio.to_raster(ndwi_output_path)

        # 4. EXECUTAR DETECCIÓ DIRECTA D'AIGUA I LÍNIES DE COSTA
        ndwi_array = ndwi.values
        waterbody_array = detect_waterbody(ndwi_array)
        coastline_array = estimate_coastline(waterbody_array)

        # Convertir a DataArray mantenint les coordenades originals de l'stack
        water_da = xr.DataArray(
            waterbody_array, coords=ndwi.coords, dims=ndwi.dims
        )
        water_da.rio.write_crs("EPSG:4326").rio.to_raster(
            os.path.join(folders["waterbody"], f"waterbody_{date_str}.tif")
        )

        coast_da = xr.DataArray(
            coastline_array, coords=ndwi.coords, dims=ndwi.dims
        )
        coast_da.rio.write_crs("EPSG:4326").rio.to_raster(
            os.path.join(folders["coastline"], f"coastline_{date_str}.tif")
        )

        # 5. Generar i guardar les imatges PNG de comprovaició
        plt.figure()
        plt.imshow(waterbody_array, cmap="Blues", vmin=0, vmax=1)
        plt.title(f"Waterbody Detection ({date_str})")
        plt.axis("off")
        plt.savefig(
            os.path.join(folders["plots"], f"waterbody_{date_str}.png"),
            bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        plt.imshow(coastline_array, cmap="Reds", vmin=0, vmax=1)
        plt.title(f"Coastline Estimation ({date_str})")
        plt.axis("off")
        plt.savefig(
            os.path.join(folders["plots"], f"coastline_{date_str}.png"),
            bbox_inches="tight",
        )
        plt.close()

        print(f"Completada la data: {date_str}")
        return date_str

    except Exception as e:
        print(f"Error processant t_idx {t_idx}: {e}")
        return None


# ------------------------------------------------------------------------------
# 6. EXECUCIÓ PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    total_times = len(stack.time)

    print(
        f"\n--- INICIANT PROCESSAMENT I ANÀLISI AMB MULTITHREADING ({MAX_WORKERS} WORKERS) ---"
    )
    start_multi = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_multi = list(
            executor.map(process_single_date, range(total_times))
        )

    end_multi = time.time()
    print(
        f"\nProcés finalitzat amb èxit en {end_multi - start_multi:.2f} segons!"
    )