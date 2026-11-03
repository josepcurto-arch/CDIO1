import os
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import geopandas as gpd
import pystac_client
import stackstac
import matplotlib.pyplot as plt
import numpy as np

# 1. Rutes del projecte
script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_path = os.path.join(script_dir, "polygon.geojson")

try:
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    geometry = gdf.geometry.iloc[0].__geo_interface__
    bbox = list(gdf.total_bounds)
except Exception as e:
    print(f"Error en carregar el GeoJSON: {e}")
    exit(1)

# 2. Paràmetres de la cerca (1er Semestre de 2025 i Filtre < 10% de núvols)
MAX_CLOUD_COVER = 10
BANDS = ['red', 'green', 'blue', 'nir']

try:
    catalog = pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        datetime='2025-01-01/2025-06-30',
        intersects=geometry,
        query=[f'eo:cloud_cover<{MAX_CLOUD_COVER}']
    )
    items = search.item_collection()
    print(f'{len(items)} elements trobats amb menys del {MAX_CLOUD_COVER}% de núvols per al 2025.')
except Exception as e:
    print(f"Error durant la cerca al catàleg STAC: {e}")
    exit(1)

if len(items) == 0:
    print("No s'han trobat imatges per al període especificat.")
    exit(0)

# 3. Preparació de les carpetes de sortida
folders = {
    'red': os.path.join(script_dir, 'banda_red'),
    'green': os.path.join(script_dir, 'banda_green'),
    'blue': os.path.join(script_dir, 'banda_blue'),
    'nir': os.path.join(script_dir, 'banda_nir'),
    'ndwi': os.path.join(script_dir, 'ndwi'),
    'rgb': os.path.join(script_dir, 'rgb')
}
for folder_path in folders.values():
    os.makedirs(folder_path, exist_ok=True)

# 4. Creació de l'stack
stack = stackstac.stack(items, assets=BANDS, bounds_latlon=bbox, epsg=4326, chunksize=2048)

# --- FUNCIÓ AUXILIAR PER A LA DESCÀRREGA D'UNA IMATGE / DATA ---
def process_single_date(t_idx):
    try:
        data_date = stack.isel(time=t_idx)
        raw_time = str(data_date.time.values)
        date_str = raw_time[:19].replace(':', '').replace('-', '').replace('T', '_')

        for band_name in BANDS:
            band_data = data_date.sel(band=band_name)
            plt.figure()
            band_data.plot.imshow(cmap='gray')
            plt.title(f'Banda {band_name.upper()} - {date_str}')
            plt.savefig(os.path.join(folders[band_name], f'{band_name}_{date_str}.png'))
            plt.close()
        
        return t_idx, date_str, data_date
    except Exception as e:
        print(f"Error processant t_idx {t_idx}: {e}")
        return None

# --- FUNCIÓ AUXILIAR PER AL CÀLCUL DE NDWI (MULTIPROCESSAMENT) ---
def compute_ndwi_for_item(item_data):
    if item_data is None:
        return
    t_idx, date_str, data_date = item_data
    try:
        green = data_date.sel(band='green').values
        nir = data_date.sel(band='nir').values
        denominator = green + nir
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi_arr = np.where(np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0)
            ndwi_arr = np.nan_to_num(ndwi_arr, nan=0.0, posinf=1.0, neginf=-1.0)

        plt.figure()
        plt.imshow(ndwi_arr, cmap='BrBG', vmin=-1, vmax=1)
        plt.colorbar(label='NDWI')
        plt.title(f'NDWI - {date_str}')
        plt.savefig(os.path.join(folders['ndwi'], f'ndwi_{date_str}.png'))
        plt.close()
    except Exception as e:
        print(f"Error NDWI a {date_str}: {e}")

if __name__ == '__main__':
    total_times = len(stack.time)
    print(f"\n--- INICIANT DESCÀRREGA SENSE MULTITHREADING (1 FIL) ---")
    start_seq = time.time()
    results = [process_single_date(i) for i in range(total_times)]
    end_seq = time.time()
    t_single = end_seq - start_seq
    print(f"Temps de descàrrega amb 1 fil: {t_single:.2f} segons")

    print(f"\n--- INICIANT DESCÀRREGA AMB MULTITHREADING (4 FILS) ---")
    start_multi = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_multi = list(executor.map(process_single_date, range(total_times)))
    end_multi = time.time()
    t_multi = end_multi - start_multi
    print(f"Temps de descàrrega amb 4 fils: {t_multi:.2f} segons")

    print(f"\n--- INICIANT CÀLCUL NDWI AMB MULTIPROCESSAMENT ---")
    start_mp = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(compute_ndwi_for_item, results_multi)
    end_mp = time.time()
    print(f"Temps de càlcul NDWI multiprocés: {end_mp - start_mp:.2f} segons")