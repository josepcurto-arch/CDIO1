import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, MultiPoint, Point

import shoreline_utils as utils

ESPAIAMENT_ESPATIAL = 10   # Distància entre transectes (en metres)
LONGITUD_TRANSECTE = 100  # Longitud de cada transecte (en metres)


def distancia_interseccio_amb_signe(interseccio, px, py, normal):
    """Calcula la distància amb signe des d'un punt del transecte fins a la intersecció amb la costa."""
    if isinstance(interseccio, MultiPoint):
        distancies = [Point(px, py).distance(punt) for punt in interseccio.geoms]
        return distancies[int(np.argmin(distancies))]

    if isinstance(interseccio, Point):
        vector_signe = np.array([interseccio.x - px, interseccio.y - py])
        distancia = np.linalg.norm(vector_signe)
        if np.dot(vector_signe, normal) < 0:
            distancia = -distancia
        return distancia

    return np.nan


def main(projecte, mostrar_grafic=True):
    directori_projecte = utils.project_path(projecte)
    directori_sortida = directori_projecte / "output" / "estimated_erosion_accretion"
    directori_sortida.mkdir(parents=True, exist_ok=True)

    ruta_csv = directori_projecte / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    if not ruta_csv.exists():
        print(f"Error: No s'ha trobat el fitxer {ruta_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(ruta_csv)
    linies_costa = {data: grup[["x", "y"]].values for data, grup in df.groupby("date")}

    if len(linies_costa) < 2:
        print("Calen almenys 2 dates a all_shorelines.csv per calcular l'erosió/acreditació.", file=sys.stderr)
        sys.exit(1)

    # 1. Utilitzem la primera data com a línia de base per construir la referència
    data_referencia = sorted(linies_costa.keys())[0]
    coordenades_referencia = linies_costa[data_referencia]

    print(f"Generant línia de costa base utilitzant la data: {data_referencia}")
    x = coordenades_referencia[:, 0]
    y = coordenades_referencia[:, 1]
    xy_nics = np.unique(np.column_stack((x, y)), axis=0)
    x_nics = xy_nics[:, 0]
    y_nics = xy_nics[:, 1]

    linia_referencia = utils.build_ordered_linestring(coordenades_referencia, ordered=False)
    longitud_costa = linia_referencia.length
    n_punts = int(longitud_costa / ESPAIAMENT_ESPATIAL)

    # Ajust de Spline suavitzat per projectar transectes normals
    tck, _ = splprep([x_nics, y_nics], s=0)
    unew = np.linspace(0, 1, num=n_punts)
    x_suau, y_suau = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    totes_distancies = []

    # 2. Calcular distàncies a cada transecte per a totes les dates
    for data, coordenades in linies_costa.items():
        print(f"Processant data: {data}...")
        linia = utils.build_ordered_linestring(coordenades, ordered=False)

        for i in range(len(unew)):
            px, py = x_suau[i], y_suau[i]
            tx, ty = dx[i], dy[i]
            normal = np.array([-ty, tx])
            norm_val = np.linalg.norm(normal)
            if norm_val > 0:
                normal /= norm_val

            inici = (px - normal[0] * LONGITUD_TRANSECTE / 2, py - normal[1] * LONGITUD_TRANSECTE / 2)
            fi = (px + normal[0] * LONGITUD_TRANSECTE / 2, py + normal[1] * LONGITUD_TRANSECTE / 2)
            transecte = LineString([inici, fi])

            dist = distancia_interseccio_amb_signe(transecte.intersection(linia), px, py, normal)
            
            totes_distancies.append({
                "date": data,
                "transect_id": i,
                "reference_x": px,
                "reference_y": py,
                "distance_m": dist
            })

    # Desar el CSV de distàncies per data i transecte
    df_distancies = pd.DataFrame(totes_distancies)
    csv_sortida = directori_sortida / "shoreline_distances.csv"
    df_distancies.to_csv(csv_sortida, index=False)
    print(f"\nResultats desats correctament a: {csv_sortida}")

    # Gràfic explicatiu si no es desactiva
    if mostrar_grafic:
        plt.figure(figsize=(10, 5))
        df_mitjana = df_distancies.groupby("date")["distance_m"].mean().reset_index()
        plt.plot(df_mitjana["date"], df_mitjana["distance_m"], marker="o", linestyle="-", color="tab:blue")
        plt.xlabel("Data")
        plt.ylabel("Distància mitjana a la costa base (m)")
        plt.title("Evolució de la Línia de Costa (Erosió / Acreditació)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula l'erosió/acreditació basant-se en transectes perpendiculars.")
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Nom del projecte ubicat a coastline_estimator/projects.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Desar els resultats sense mostrar els gràfics.")
    args = parser.parse_args()
    main(args.project, mostrar_grafic=not args.no_plot)
