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

# Silence runtime warnings for array division
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ------------------------------------------------------------------------------
# 1. PATH RESOLUTION & LOAD CONFIG FROM PARENT DIRECTORY
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # Homework/API
project_root = os.path.dirname(script_dir)             # Homework/

config_path = os.path.join(project_root, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extract config values (resolve geojson_file relative to Homework/ root)
GEOJSON_PATH = os.path.join(project_root, config["geojson_file"])
CATALOG_URL = config["search"]["catalog_url"]
COLLECTION = config["search"]["collection"]
DATE_RANGE = f"{config['search']['start_date']}/{config['search']['end_date']}"
MAX_CLOUD_COVER = config["search"]["max_cloud_cover"]
EPSG = config["search"]["target_epsg"]
BANDS = config["bands"]
MAX_WORKERS = config["performance"]["max_workers"]
CHUNK_SIZE = config["performance"]["chunk_size"]

# Output folders created inside Homework/
folders = {
    'red': os.path.join(project_root, 'banda_red'),
    'green': os.path.join(project_root, 'banda_green'),
    'blue': os.path.join(project_root, 'banda_blue'),
    'nir': os.path.join(project_root, 'banda_nir'),
    'ndwi': os.path.join(project_root, 'ndwi'),
    'rgb': os.path.join(project_root, 'rgb')
}
for folder_path in folders.values():
    os.makedirs(folder_path, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. LOAD AOI & SEARCH STAC CATALOG
# ------------------------------------------------------------------------------
try:
    gdf = gpd.read_file(GEOJSON_PATH).to_crs(epsg=4326)
    geometry = gdf.geometry.iloc[0].__geo_interface__
    bbox = list(gdf.total_bounds)
except Exception as e:
    print(f"Error loading GeoJSON file ({GEOJSON_PATH}): {e}")
    exit(1)

try:
    catalog = pystac_client.Client.open(CATALOG_URL)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=DATE_RANGE,
        intersects=geometry,
        query=[f'eo:cloud_cover<{MAX_CLOUD_COVER}']
    )
    items = search.item_collection()
    print(f'{len(items)} items found with <{MAX_CLOUD_COVER}% cloud cover.')
except Exception as e:
    print(f"Error during STAC catalog search: {e}")
    exit(1)

if len(items) == 0:
    print("No images found matching criteria.")
    exit(0)

# ------------------------------------------------------------------------------
# 3. STACK STAC DATA
# ------------------------------------------------------------------------------
stack = stackstac.stack(
    items, 
    assets=BANDS, 
    bounds_latlon=bbox, 
    epsg=4326, 
    chunksize=CHUNK_SIZE
)

# ------------------------------------------------------------------------------
# 4. MULTITHREADED PROCESSING FUNCTION
# ------------------------------------------------------------------------------
def process_single_date(t_idx):
    try:
        data_date = stack.isel(time=t_idx)
        raw_time = str(data_date.time.values)
        
        # Formats date into AAAAMMDD (e.g. "20250315")
        date_str = raw_time[:10].replace('-', '')

        # Export individual bands (e.g. red_20250315.tif)
        for band_name in BANDS:
            band_data = data_date.sel(band=band_name)
            output_path = os.path.join(folders[band_name], f'{band_name}_{date_str}.tif')
            band_data.rio.write_crs(EPSG).rio.to_raster(output_path)

        # Export RGB composition (e.g. rgb_20250315.tif)
        rgb_stack = data_date.sel(band=['red', 'green', 'blue'])
        rgb_output_path = os.path.join(folders['rgb'], f'rgb_{date_str}.tif')
        rgb_stack.rio.write_crs(EPSG).rio.to_raster(rgb_output_path)

        # Export NDWI calculation (e.g. ndwi_20250315.tif)
        green = data_date.sel(band='green')
        nir = data_date.sel(band='nir')
        denominator = green + nir

        ndwi = xr.where(np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0)
        ndwi = ndwi.fillna(0.0)

        ndwi_output_path = os.path.join(folders['ndwi'], f'ndwi_{date_str}.tif')
        ndwi.rio.write_crs(EPSG).rio.to_raster(ndwi_output_path)

        print(f"Completed date: {date_str}")
        return date_str

    except Exception as e:
        print(f"Error processing t_idx {t_idx}: {e}")
        return None

# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    total_times = len(stack.time)

    print(f"\n--- PROCESSING GEOTIFF EXPORT WITH MULTITHREADING ({MAX_WORKERS} WORKERS) ---")
    start_multi = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_multi = list(executor.map(process_single_date, range(total_times)))

    end_multi = time.time()
    print(f"\nExecution successfully finished in {end_multi - start_multi:.2f} seconds!")