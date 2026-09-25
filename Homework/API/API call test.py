import os
import geopandas as gpd
import pystac_client
import stackstac
import matplotlib.pyplot as plt
import numpy as np

# 1. Definir la ruta de la carpeta on està guardat aquest script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Carregar la geometria del GeoJSON
geojson_path = os.path.join(script_dir, "polygon.geojson")

try:
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    geometry = gdf.geometry.iloc[0].__geo_interface__
    bbox = list(gdf.total_bounds)
except Exception as e:
    print(f"Error en carregar el fitxer GeoJSON ({geojson_path}): {e}")
    exit(1)

# 3. Cerca al catàleg STAC amb filtre de cobertura de núvols
MAX_CLOUD_COVER = 10

try:
    catalog = pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        datetime='2020-07-01/2020-08-15',
        intersects=geometry,
        query=[f'eo:cloud_cover<{MAX_CLOUD_COVER}']  # Filtre per núvols de Codi 2
    )
    items = search.item_collection()
    print(f'{len(items)} items trobats amb menys del {MAX_CLOUD_COVER}% de núvols')
except Exception as e:
    print(f"Error durant la cerca al catàleg STAC: {e}")
    exit(1)

if len(items) == 0:
    print(f"No s'han trobat imatges amb menys del {MAX_CLOUD_COVER}% de núvols per als criteris especificats.")
    exit(0)

# 4. Sol·licitar les 4 bandes desitjades
bands = ['red', 'green', 'blue', 'nir']

try:
    stack = stackstac.stack(
        items,
        assets=bands,
        bounds_latlon=bbox,
        epsg=4326,
        chunksize=2048
    )
except Exception as e:
    print(f"Error en crear l'stack amb stackstac: {e}")
    exit(1)

# 5. Crear les carpetes de sortida per a cada producte
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

# 6. Iterar per TOTES les dates trobades amb filtre i control d'errors
for t_idx in range(len(stack.time)):
    try:
        data_date = stack.isel(time=t_idx)
        
        # Formatejar la data per al nom del fitxer (YYYY-MM-DD_HHMMSS)
        raw_time = str(data_date.time.values)
        date_str = raw_time[:19].replace(':', '').replace('-', '').replace('T', '_')
        
        # --- 6.1 Desar les 4 bandes individuals ---
        for band_name in bands:
            try:
                band_data = data_date.sel(band=band_name)
                
                plt.figure()
                band_data.plot.imshow(cmap='gray')
                plt.title(f'Banda {band_name.upper()} - {date_str} (<{MAX_CLOUD_COVER}% núvols)')
                
                output_path = os.path.join(folders[band_name], f'{band_name}_{date_str}.png')
                plt.savefig(output_path)
                plt.close()
            except Exception as e:
                plt.close()
                print(f"  [ERROR] No s'ha pogut desar la banda {band_name} ({date_str}): {e}")
            
        # --- 6.2 Càlcul i desat del NDWI (Gestió segura de divisió per zero) ---
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
            plt.title(f'NDWI Sentinel-2 - {date_str} (<{MAX_CLOUD_COVER}% núvols)')

            ndwi_output_path = os.path.join(folders['ndwi'], f'ndwi_{date_str}.png')
            plt.savefig(ndwi_output_path)
            plt.close()
        except Exception as e:
            plt.close()
            print(f"  [ERROR] No s'ha pogut calcular o desar el NDWI ({date_str}): {e}")

        # --- 6.3 Composició RGB (Color Real) ---
        try:
            red_b = data_date.sel(band='red').values
            green_b = data_date.sel(band='green').values
            blue_b = data_date.sel(band='blue').values

            rgb_data = np.stack([red_b, green_b, blue_b], axis=-1)
            rgb_clean = np.nan_to_num(rgb_data, nan=0.0, posinf=3000.0, neginf=0.0)
            rgb_normalized = np.clip(rgb_clean / 3000.0, 0, 1)

            plt.figure()
            plt.imshow(rgb_normalized)
            plt.title(f'RGB Color Real - {date_str} (<{MAX_CLOUD_COVER}% núvols)')
            plt.axis('off')

            rgb_output_path = os.path.join(folders['rgb'], f'rgb_{date_str}.png')
            plt.savefig(rgb_output_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            plt.close()
            print(f"  [ERROR] No s'ha pogut generar el RGB ({date_str}): {e}")

        print(f'Completat el processament de la data: {date_str}')

    except Exception as e:
        print(f"[ERROR CRÍTIC] Falla general en l'ítem {t_idx}: {e}")

print("Procés finalitzat amb èxit!")