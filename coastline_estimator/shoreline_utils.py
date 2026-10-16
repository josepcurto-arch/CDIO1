import multiprocessing as mp
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely.geometry import LineString


DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def extract_date(name: str):
    match = DATE_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def export_coords_to_kmz(coords, output_kmz_path):
    doc = Document()
    kml = doc.createElement("kml")
    kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
    doc.appendChild(kml)

    document = doc.createElement("Document")
    kml.appendChild(document)

    for lon, lat in coords:
        placemark = doc.createElement("Placemark")
        point = doc.createElement("Point")
        coord = doc.createElement("coordinates")
        coord.appendChild(doc.createTextNode(f"{lon},{lat},0"))
        point.appendChild(coord)
        placemark.appendChild(point)
        document.appendChild(placemark)

    kml_str = doc.toprettyxml(indent="  ", encoding="UTF-8")

    with zipfile.ZipFile(output_kmz_path, "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml_str)


def load_coords_csv(csv_path, source_crs=32631, target_crs=4326):
    df = pd.read_csv(
        csv_path,
        sep=r"\s+|,",
        engine="python",
        comment="%",
        names=["x", "y"],
        usecols=[0, 1],
        header=None,
    )
    df = df.dropna(subset=["x", "y"])

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lons, lats = transformer.transform(df["x"].to_numpy(), df["y"].to_numpy())
    return list(zip(lons, lats))


def build_ordered_linestring(coords, ordered=False):
    """
    Ordena eficientment els punts de la línia de costa mitjançant KDTree
    per evitar problemes de memòria (MemoryError) amb matrius grans.
    """
    if ordered:
        return LineString(coords)

    coords = np.unique(np.asarray(coords), axis=0)
    n = len(coords)
    if n < 2:
        return LineString(coords)

    print(f"Número de punts a ordenar: {n}")

    # Cercador espacial basat en KDTree
    tree = KDTree(coords)
    unvisited = set(range(n))

    # Comencem pel punt situat més a l'oest (X mínima)
    start_idx = int(np.argmin(coords[:, 0]))
    ordered_idx = [start_idx]
    unvisited.remove(start_idx)

    current_idx = start_idx
    while unvisited:
        k_search = min(15, len(coords))
        distances, indices = tree.query(coords[current_idx], k=k_search)

        if isinstance(indices, (int, np.integer)):
            indices = [indices]

        next_idx = None
        for idx in indices:
            if idx in unvisited:
                next_idx = idx
                break

        # Si no troba un veí proper no visitat, selecciona el més proper globalment
        if next_idx is None:
            unvisited_list = list(unvisited)
            sub_tree = KDTree(coords[unvisited_list])
            _, sub_idx = sub_tree.query(coords[current_idx])
            next_idx = unvisited_list[sub_idx]

        ordered_idx.append(next_idx)
        unvisited.remove(next_idx)
        current_idx = next_idx

    return LineString(coords[ordered_idx])


def project_path(project):
    return Path(__file__).resolve().parent / "projects" / project
