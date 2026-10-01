# SCRIPT DE DESCÀRREGA, PROCESSAMENT I EXPORTACIÓ A GEOTIFF D'IMATGES SENTINEL-2

# Aquest script llegeix un polígon des d'un fitxer GeoJSON, realitza
# una cerca d'imatges Sentinel-2 L2A en un catàleg STAC públic, 
# calcula l'índex hídric NDWI, genera composicions RGB i exporta
# tots els resultats georeferenciats en format GeoTIFF (.tif).

# 1. IMPORTACIÓ DE LLIBRERIES NECESSÀRIES
import os  # Per a la gestió de rutes del sistema de fitxers i creació de carpetes
import geopandas as gpd  # Per carregar, llegir i manipular dades vectorials (GeoJSON)
import pystac_client  # Client per fer consultes a la API del catàleg STAC (Earth Search)
import stackstac  # Per convertir ítems STAC en un DataArray multidimensional de xarray
import numpy as np  # Per dur a terme operacions matemàtiques i control d'errors numèrics
import xarray as xr  # Gestió d'arrays multidimensionals amb etiquetes de coordenades
import rioxarray  # Extensió de xarray per exportar dades en formats ràster com GeoTIFF (.tif)
import warnings

# Evitar els warnings als càlculs (bàsicament el divir per zero)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 2. CONFIGURACIÓ DE DIRECTORIS I LECTURA DE LA GEOMETRIA (GEOJSON)
# Obté la ruta absoluta del directori on està guardat aquest mateix fitxer de Python
script_dir = os.path.dirname(os.path.abspath(__file__))

# Defineix la ruta completa cap al fitxer GeoJSON que conté la nostra àrea d'interès (AOI)
geojson_path = os.path.join(script_dir, "polygon.geojson")

try:
    # Llegeix el fitxer GeoJSON utilitzant GeoPandas
    gdf = gpd.read_file(geojson_path)
    
    # Assegura que les coordenades estiguin en el sistema WGS84 / EPSG:4326 (Latitud/Longitud)
    gdf = gdf.to_crs(epsg=4326)
    
    # Extreu la primera geometria en format diccionari GeoJSON (__geo_interface__)
    geometry = gdf.geometry.iloc[0].__geo_interface__
    
    # Obté els límits rectangles de la geometria [min_lon, min_lat, max_lon, max_lat]
    bbox = list(gdf.total_bounds)

except Exception as e:
    # Si falla la lectura del fitxer o la conversió de coordenades, mostra l'error i atura l'script
    print(f"[ERROR CRÍTIC] No s'ha pogut carregar el fitxer GeoJSON ({geojson_path}): {e}")
    exit(1)


# 3. CERCA D'IMATGES DE SATÈL·LIT AL CATÀLEG STAC (AWS EARTH SEARCH)
try:
    # Connecta amb el catàleg STAC públic mantingut per Element 84 a AWS
    catalog = pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
    
    # Defineix la cerca segons la col·lecció de Sentinel-2 L2A, rang de dates i la geometria
    search = catalog.search(
        collections=['sentinel-2-l2a'],  # Col·lecció amb correcció atmosfèrica aplicada
        datetime='2025-01-01/2025-02-01', # Rang temporal de cerca (YYYY-MM-DD)
        intersects=geometry              # Filtra les imatges que toquen el nostre polígon
    )
    
    # Executa la cerca i obté la col·lecció d'elements (ítems)
    items = search.item_collection()
    print(f'[INFO] {len(items)} imatges trobades per al període especificat.')

except Exception as e:
    # Si la connexió a la API falla o la cerca és invàlida, mostra l'error i atura
    print(f"[ERROR CRÍTIC] Falla durant la cerca al catàleg STAC: {e}")
    exit(1)

# Comprova si la cerca no ha retornat cap resultat
if len(items) == 0:
    print("[AVÍS] No s'han trobat imatges que compleixin els criteris. Finalitzant executabilitat.")
    exit(0)


# 4. CARREGADA DE DADES ESPECTRALS MITJANÇANT STACKSTAC
# Llista amb les 4 bandes espectrals que volem carregar
bands = ['red', 'green', 'blue', 'nir']

try:
    # Apila totes les imatges trobades en una estructura xarray multidimensional (time, band, y, x)
    stack = stackstac.stack(
        items,                  # Col·lecció d'ítems trobats al catàleg STAC
        assets=bands,           # Selecciona només les 4 bandes especificades
        bounds_latlon=bbox,     # Retalla la descàrrega al recinte del nostre polígon
        epsg=4326,              # Força la projecció de sortida a EPSG:4326
        chunksize=2048          # Mida dels blocs en memòria utilitzats per Dask
    )
except Exception as e:
    print(f"[ERROR CRÍTIC] No s'ha pogut crear l'estructura de dades amb stackstac: {e}")
    exit(1)


# 5. CREACIÓ AUTOMÀTICA DE CARPETES PER ALS RESULTATS
# Diccionari on la clau és el tipus de producte i el valor és la ruta de la carpeta
folders = {
    'red': os.path.join(script_dir, 'banda_red'),
    'green': os.path.join(script_dir, 'banda_green'),
    'blue': os.path.join(script_dir, 'banda_blue'),
    'nir': os.path.join(script_dir, 'banda_nir'),
    'ndwi': os.path.join(script_dir, 'ndwi'),
    'rgb': os.path.join(script_dir, 'rgb')
}

# Itera sobre cada ruta i crea la carpeta física si aquesta encara no existeix
for folder_path in folders.values():
    os.makedirs(folder_path, exist_ok=True)


# 6. BUCLE PRINCIPAL: PROCESSAMENT I EXPORTACIÓ GEOTIFF PER CADA DATA
# Recorrem totes les capes temporals que conté l'stack de dades
for t_idx in range(len(stack.time)):
    try:
        # Extreu el DataArray corresponent a l'índex temporal actual
        data_date = stack.isel(time=t_idx)
        
        # Extreu la data en text (string) i la formata per utilitzar-la com a nom de fitxer
        raw_time = str(data_date.time.values)
        date_str = raw_time[:19].replace(':', '').replace('-', '').replace('T', '_')
        
        print(f"\n[PROCESSANT] Data ({t_idx + 1}/{len(stack.time)}): {date_str}")
        
        # 6.1 EXPORTACIÓ DE LES 4 BANDES INDIVIDUALS A GEOTIFF
        for band_name in bands:
            try:
                # Selecciona la dada de la banda espectral específica
                band_data = data_date.sel(band=band_name)
                
                # Assigna el nom del fitxer de sortida GeoTIFF
                output_path = os.path.join(folders[band_name], f'{band_name}_{date_str}.tif')
                
                # Defineix explícitament el sistema de referència de coordenades (CRS)
                band_data = band_data.rio.write_crs("EPSG:4326")
                
                # Exporta el DataArray directament a un fitxer GeoTIFF
                band_data.rio.to_raster(output_path)
                print(f"Banda {band_name.upper()} exportada correctament.")
            
            except Exception as e:
                print(f"[ERROR] Error en exportar la banda {band_name} ({date_str}): {e}")
            
        # 6.2 CÀLCUL I EXPORTACIÓ DE L'ÍNDEX NDWI (EVALUACIÓ D'AIGUA)
        try:
            # Formula NDWI: (GREEN - NIR) / (GREEN + NIR)
            green = data_date.sel(band='green')
            nir = data_date.sel(band='nir')

            # Denominador per al càlcul de l'índex
            denominator = green + nir
            
            # Lògica d'evasió de divisió per zero i valors nuls:
            # Si el valor absolut del denominador és menor a 1e-6, assigna 0.0 per evitar dividir per zero
            ndwi = xr.where(np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0)
            
            # Substitueix qualsevol valor invàlid restant (NaN) per 0.0
            ndwi = ndwi.fillna(0.0)

            # Assigna el camí del fitxer de sortida
            ndwi_output_path = os.path.join(folders['ndwi'], f'ndwi_{date_str}.tif')
            
            # Configura el CRS i desa el resultat a disc
            ndwi = ndwi.rio.write_crs("EPSG:4326")
            ndwi.rio.to_raster(ndwi_output_path)
            print(f"Càlcul i exportació de l'NDWI completat.")
        
        except Exception as e:
            print(f"[ERROR] No s'ha pogut generar el fitxer NDWI ({date_str}): {e}")

        # 6.3 EXPORTACIÓ DE LA COMPOSICIÓ MULTIBANDA RGB (COLOR REAL)
        try:
            # Extreu un DataArray 3D que conté les 3 bandes de color en l'ordre Red, Green, Blue
            rgb_stack = data_date.sel(band=['red', 'green', 'blue'])
            
            # Defineix la ruta del GeoTIFF multibanda
            rgb_output_path = os.path.join(folders['rgb'], f'rgb_{date_str}.tif')
            
            # Assigna el CRS espacial i guarda la imatge amb les 3 capes
            rgb_stack = rgb_stack.rio.write_crs("EPSG:4326")
            rgb_stack.rio.to_raster(rgb_output_path)
            print(f"  └─ Composició RGB (3 capes) exportada a GeoTIFF.")
        
        except Exception as e:
            print(f"  └─ [ERROR] Falla en la generació del GeoTIFF RGB ({date_str}): {e}")

    except Exception as e:
        # En cas de falla general en la captura d'una data, l'script no s'atura i passa a la següent
        print(f"[ERROR CRÍTIC] Falla en processar la data d'índex {t_idx}: {e}")

# 7. FINALITZACIÓ DE L'EXECUCIÓ
print("[OK]")