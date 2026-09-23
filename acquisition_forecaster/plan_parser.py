import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import glob


def main(project):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    config = json.load(open(f"acquisition_forecaster/projects/{project}/input/config.json"))
    rows = []

    pt = Point(config["POINT"][0], config["POINT"][1])

    kmls = glob.glob(f"acquisition_forecaster/projects/{project}/input/sentinel_plans/*.kml")

    for kml in kmls:
        satellite = kml.split("/")[-1].split("_")[0]

        tree = ET.parse(kml)
        root = tree.getroot()

        for pm in root.findall(".//kml:Placemark", ns):
            # Time information
            begin_el = pm.find(".//kml:TimeSpan/kml:begin", ns)
            end_el = pm.find(".//kml:TimeSpan/kml:end", ns)
            coords_el = pm.find(".//kml:Polygon//kml:coordinates", ns)

            # Skip if any of the pieces are missing
            if begin_el is None or end_el is None or coords_el is None:
                continue

            capture_start = pd.to_datetime(begin_el.text.strip())
            capture_end = pd.to_datetime(end_el.text.strip())

            mode = None
            timeliness = None
            for data in pm.findall(".//kml:Data", ns):
                name = data.attrib.get("name")
                val_el = data.find("kml:value", ns)
                if val_el is None or not val_el.text:
                    continue
                val = val_el.text.strip()
                if name == "Mode":
                    mode = val  # e.g. 'NOBS', 'DARK-O', 'VIC'
                elif name == "Timeliness":
                    timeliness = val  # e.g. 'NOMINAL'

            # e.g. 'NOBS NOMINAL', 'DARK-O NOMINAL', 'VIC NOMINAL'
            if mode and timeliness:
                acquisition_type = f"{mode} {timeliness}"
            else:
                acquisition_type = mode or timeliness  # fallback

            coord_text = coords_el.text.strip()
            points = []
            for triplet in coord_text.replace("\n", " ").split():
                parts = triplet.split(",")
                if len(parts) < 2:
                    continue
                lon = float(parts[0])
                lat = float(parts[1])
                points.append((lon, lat))

            if len(points) < 3:
                print(f"Skipping invalid polygon with less than 3 points: {points}")
                continue

            if points[0] != points[-1]:
                print("Closing polygon by appending first point to the end.")
                points.append(points[0])

            poly = Polygon(points)

            rows.append(
                {
                    "capture_start": capture_start,
                    "capture_end": capture_end,
                    "polygon": poly,
                    "satellite": satellite,
                    "acquisition_type": acquisition_type,
                }
            )

        # Build GeoDataFrame; use "polygon" as the geometry column name
        gdf = gpd.GeoDataFrame(rows, geometry="polygon", crs="EPSG:4326")

    # Filter to only those acquisitions that cover the point of interest
    gdf = gdf[gdf.geometry.apply(lambda geom: geom.contains(pt))]

    # Filter to the ones in between config["START_DATE"] (at 00:00:00) and config["END_DATE"] (at 23:59:59)
    start_date = pd.to_datetime(config["START_DATE"])
    end_date = pd.to_datetime(config["END_DATE"]) + pd.Timedelta(days=1)
    gdf = gdf[(gdf["capture_start"] >= start_date) & (gdf["capture_end"] < end_date)]
    return gdf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse Sentinel acquisition KML to GeoDataFrame and filter by"
                                                 " point and date range")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    args = parser.parse_args()
    gdf = main(args.project)
    print(gdf)
