from pystac_client import Client
import geopandas as gpd
import json
from tqdm import tqdm
import pprint


pp = pprint.PrettyPrinter(indent=4)



def search_catalog(catalog):
    if catalog == "copernicus":
        catalog_url = "https://stac.dataspace.copernicus.eu/v1"
    elif catalog == "element84":
        catalog_url = "https://earth-search.aws.element84.com/v1"
    else:
        raise ValueError("Invalid catalog name. Choose 'copernicus' or 'element84'.")

    gdf = gpd.read_file(f"../../coastline_estimator/projects/castelldefels_h1_2025/input/polygon.geojson")
    aoi = gdf.geometry[0].__geo_interface__
    config = json.load(open(f"../../coastline_estimator/projects/castelldefels_h1_2025/input/config.json"))

    catalog = Client.open(catalog_url)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=f"{config['START_DATE']}/{config['END_DATE']}",
    )
    items = list(search.items())
    pp.pprint(items[0].to_dict()['assets'].keys())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STAC Catalog Demo")
    parser.add_argument("--catalog", type=str, default="copernicus", choices=["copernicus", "element84"], help="STAC catalog to use")
    args = parser.parse_args()
    search_catalog(args.catalog)
