import os
import geopandas as gpd
import pystac_client
import stackstac
import matplotlib.pyplot as plt

# 1. Definir la ruta de la carpeta on està guardat aquest script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Carregar la geometria del GeoJSON
geojson_path = os.path.join(script_dir, "polygon.geojson")
gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)

geometry = gdf.geometry.iloc[0].__geo_interface__
bbox = list(gdf.total_bounds)

# 3. Cerca al catàleg STAC (Sentinel-2 L2A)
catalog = pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
search = catalog.search(
    collections=['sentinel-2-l2a'],
    datetime='2020-07-01/2020-08-15',
    intersects=geometry
)

items = search.item_collection()
print(f'{len(items)} items trobats')

# 4. Sol·licitar les 4 bandes desitjades
bands = ['red', 'green', 'blue', 'nir']

stack = stackstac.stack(
    items,
    assets=bands,
    bounds_latlon=bbox,
    epsg=4326,
    chunksize=2048
)

# Seleccionem la primera data disponible per generar les imatges
data_first_date = stack.isel(time=0)

# 5. Desar cada banda individualment
for band_name in bands:
    band_data = data_first_date.sel(band=band_name)
    
    plt.figure() # Crea una nova figura neta
    band_data.plot.imshow(cmap='gray') # Escala de grisos per a bandes individuals
    plt.title(f'Banda: {band_name.upper()}')
    
    output_path = os.path.join(script_dir, f'banda_{band_name}.png')
    plt.savefig(output_path)
    plt.close() # Tanca la figura per no carregar la memòria
    print(f'Desada la banda: {output_path}')

# 6. Càlcul i desat del NDVI
green = data_first_date.sel(band='green')
nir = data_first_date.sel(band='nir')
ndwi = (green - nir) / (green + nir)

plt.figure()
ndwi.plot.imshow(cmap='BrBG', vmin=-1, vmax=1)
plt.title('NDVI Sentinel-2')

ndwi_output_path = os.path.join(script_dir, 'ndwi.png')
print(ndwi_output_path)
plt.savefig(ndwi_output_path)
plt.close()

print(f'Desat el NDWI: {ndwi_output_path}')