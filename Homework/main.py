import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib
# Configurar backend no interactiu ABANS d'importar pyplot (crític per multithreading)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml
import pandas as pd
import geopandas as gpd
import pystac_client
import stackstac
import numpy as np
import xarray as xr
import rioxarray

print("Iniciant programa!")
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ------------------------------------------------------------------------------
# 1. CARREGAR CONFIGURACIÓ, RUTES I MÒDULS LOCALS
# ------------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))  # Homework/

api_dir = os.path.join(project_root, 'API')
waterbodies_dir = os.path.join(api_dir, 'waterbodies')

# Afegir directoris locals al sys.path per permetre les importacions
for d in [api_dir, waterbodies_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Importació de les funcions des de coastline_functions.py
from coastline_functions import detect_waterbody, estimate_coastline

config_path = os.path.join(project_root, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

geojson_path = os.path.join(project_root, 'API', config['geojson_file'])

try:
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    geometry = gdf.geometry.iloc[0].__geo_interface__
    bbox = list(gdf.total_bounds)
    print("Arxiu de configuració i GeoJSON carregats correctament.")
except Exception as e:
    print(f"Error en carregar el GeoJSON ({geojson_path}): {e}")
    exit(1)

# ------------------------------------------------------------------------------
# 2. CERCA AL CATÀLEG STAC
# ------------------------------------------------------------------------------
BANDS = config['bands']
MAX_CLOUD_COVER = config['search']['max_cloud_cover']
MAX_WORKERS = config.get('performance', {}).get('max_workers', 4)
CHUNK_SIZE = config.get('performance', {}).get('chunk_size', 2048)

try:
    print("Cercant elements al catàleg que compleixin els filtres...")
    catalog = pystac_client.Client.open(config['search']['catalog_url'])
    search = catalog.search(
        collections=[config['search']['collection']],
        datetime=f"{config['search']['start_date']}/{config['search']['end_date']}",
        intersects=geometry,
        query=[f'eo:cloud_cover<{MAX_CLOUD_COVER}']
    )
    items = search.item_collection()
    print(f'{len(items)} elements trobats amb menys del {MAX_CLOUD_COVER}% de núvols.')
except Exception as e:
    print(f"Error durant la cerca al catàleg STAC: {e}")
    exit(1)

if len(items) == 0:
    print("No s'han trobat imatges per al període especificat.")
    exit(0)

# ------------------------------------------------------------------------------
# 3. PREPARACIÓ DE CARPETES DE SORTIDA
# ------------------------------------------------------------------------------
output_cfg = config.get('outputs', {})

folders = {
    'red': os.path.join(project_root, 'banda_red'),
    'green': os.path.join(project_root, 'banda_green'),
    'blue': os.path.join(project_root, 'banda_blue'),
    'nir': os.path.join(project_root, 'banda_nir'),
    'ndwi': os.path.join(project_root, 'ndwi'),
    'rgb': os.path.join(project_root, 'rgb'),
    'waterbody': os.path.join(project_root, output_cfg.get('waterbody_dir', 'outputs/waterbody')),
    'coastline': os.path.join(project_root, output_cfg.get('coastline_dir', 'outputs/coastline')),
    'plots': os.path.join(project_root, output_cfg.get('plots_dir', 'outputs/plots'))
}

for folder_path in folders.values():
    os.makedirs(folder_path, exist_ok=True)

# ------------------------------------------------------------------------------
# 4. CREACIÓ DE L'STACK
# ------------------------------------------------------------------------------
stack = stackstac.stack(items, assets=BANDS, bounds_latlon=bbox, epsg=4326, chunksize=CHUNK_SIZE)
# ------------------------------------------------------------------------------
# 5. FUNCIÓ MULTITHREAD DE PROCESSAMENT I ANÀLISI
# ------------------------------------------------------------------------------
def process_single_date(t_idx):
    try:
        data_date = stack.isel(time=t_idx)
        # Format de data segur: AAAAMMDD
        date_str = pd.to_datetime(data_date.time.values).strftime('%Y%m%d')

        # 1. Exportar bandes individuals
        for band_name in BANDS:
            band_data = data_date.sel(band=band_name)
            output_path = os.path.join(folders[band_name], f'{band_name}_{date_str}.tif')
            band_data.rio.write_crs("EPSG:4326").rio.to_raster(output_path)

        # 2. Exportar composició RGB
        rgb_stack = data_date.sel(band=['red', 'green', 'blue'])
        rgb_output_path = os.path.join(folders['rgb'], f'rgb_{date_str}.tif')
        rgb_stack.rio.write_crs("EPSG:4326").rio.to_raster(rgb_output_path)

        # 3. Càlcul i exportació del NDWI
        green = data_date.sel(band='green')
        nir = data_date.sel(band='nir')
        
        denominator = green + nir
        ndwi = xr.where(np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0)
        ndwi = ndwi.fillna(0.0)

        ndwi_output_path = os.path.join(folders['ndwi'], f'ndwi_{date_str}.tif')
        ndwi_rio = ndwi.rio.write_crs("EPSG:4326")
        ndwi_rio.rio.to_raster(ndwi_output_path)

        # 4. EXECUTAR DETECCIÓ DIRECTA D'AIGUA I LÍNIES DE COSTA
        ndwi_array = ndwi.values
        waterbody_array = detect_waterbody(ndwi_array)
        coastline_array = estimate_coastline(waterbody_array)

        # Preservar metadades geogràfiques originals amb copy(data=...)
        water_da = ndwi.copy(data=waterbody_array)
        water_da.rio.write_crs("EPSG:4326").rio.to_raster(
            os.path.join(folders['waterbody'], f'waterbody_{date_str}.tif')
        )

        coast_da = ndwi.copy(data=coastline_array)
        coast_da.rio.write_crs("EPSG:4326").rio.to_raster(
            os.path.join(folders['coastline'], f'coastline_{date_str}.tif')
        )

        # 5. Generar i guardar les gràfiques PNG (Thread-Safe + aspect='auto' per evitar imatges aplanades)
        height, width = waterbody_array.shape
        aspect_ratio = width / max(height, 1)
        fig_height = max(3, 10 / aspect_ratio)

        # Plot Waterbody
        fig_w, ax_w = plt.subplots(figsize=(10, fig_height))
        ax_w.imshow(waterbody_array, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax_w.set_title(f"Waterbody Detection ({date_str})")
        ax_w.axis('off')
        fig_w.savefig(os.path.join(folders['plots'], f'waterbody_{date_str}.png'), bbox_inches='tight', dpi=150)
        plt.close(fig_w)

        # Plot Coastline
        fig_c, ax_c = plt.subplots(figsize=(10, fig_height))
        ax_c.imshow(coastline_array, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        ax_c.set_title(f"Coastline Estimation ({date_str})")
        ax_c.axis('off')
        fig_c.savefig(os.path.join(folders['plots'], f'coastline_{date_str}.png'), bbox_inches='tight', dpi=150)
        plt.close(fig_c)

        print(f"Completada la data: {date_str}")
        return date_str

    except Exception as e:
        print(f"Error processant t_idx {t_idx}: {e}")
        return None

# ------------------------------------------------------------------------------
# 6. EXECUCIÓ PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    total_times = len(stack.time)

    print(f"\n--- INICIANT PROCESSAMENT I ANÀLISI AMB MULTITHREADING ({MAX_WORKERS} WORKERS) ---")
    start_multi = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_multi = list(executor.map(process_single_date, range(total_times)))

    end_multi = time.time()
    print(f"\nProcés finalitzat amb èxit en {end_multi - start_multi:.2f} segons!")