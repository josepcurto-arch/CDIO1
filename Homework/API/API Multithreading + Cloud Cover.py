import os
import time
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import pystac_client
import stackstac
import numpy as np
import xarray as xr
import rioxarray  # Necessari per exportar a GeoTIFF (.tif)
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

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

# 2. Paràmetres de la cerca
BANDS = ['red', 'green', 'blue', 'nir']
MAX_CLOUD_COVER = 40  # %

try:
    catalog = pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        datetime='2025-01-01/2025-06-30',
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


# --- FUNCIÓ PER A LA DESCÀRREGA I PROCESSAMENT D'UNA DATA EN GEOTIFF ---
def process_single_date(t_idx):
    try:
        data_date = stack.isel(time=t_idx)
        raw_time = str(data_date.time.values)
        date_str = raw_time[:19].replace(':', '').replace('-', '').replace('T', '_')

        # 1. Exportar bandes individuals a GeoTIFF (.tif)
        for band_name in BANDS:
            band_data = data_date.sel(band=band_name)
            output_path = os.path.join(folders[band_name], f'{band_name}_{date_str}.tif')
            band_data = band_data.rio.write_crs("EPSG:4326")
            band_data.rio.to_raster(output_path)

        # 2. Exportar composició RGB a GeoTIFF (.tif)
        rgb_stack = data_date.sel(band=['red', 'green', 'blue'])
        rgb_output_path = os.path.join(folders['rgb'], f'rgb_{date_str}.tif')
        rgb_stack = rgb_stack.rio.write_crs("EPSG:4326")
        rgb_stack.rio.to_raster(rgb_output_path)

        # 3. Càlcul i exportació del NDWI a GeoTIFF (.tif) (AMB CONTROL DE WARNINGS)
        green = data_date.sel(band='green')
        nir = data_date.sel(band='nir')
        
        denominator = green + nir
        ndwi = xr.where(np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0)
        ndwi = ndwi.fillna(0.0)
        ndwi_output_path = os.path.join(folders['ndwi'], f'ndwi_{date_str}.tif')
        ndwi = ndwi.rio.write_crs("EPSG:4326")
        ndwi.rio.to_raster(ndwi_output_path)

        print(f"Completada la data: {date_str}")
        return date_str
    except Exception as e:
        print(f"Error processant t_idx {t_idx}: {e}")
        return None

# --- EXECUCIÓ PRINCIPAL (NOMÉS MULTITHREADING) ---
if __name__ == '__main__':
    total_times = len(stack.time)

    print(f"\n--- INICIANT PROCESSAMENT D'EXPORTACIÓ GEOTIFF AMB MULTITHREADING (4 FILS) ---")
    start_multi = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results_multi = list(executor.map(process_single_date, range(total_times)))

    end_multi = time.time()
    print(f"\nProcés finalitzat amb èxit en {end_multi - start_multi:.2f} segons!")