import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

import shoreline_utils as utils

TARGET_GT_DATES = ["2022-05-18", "2025-01-24", "2025-03-30"]
MAX_DAYS_GAP = 60  # Umbral màxim de dies permès entre Ground Truth i Sentinel-2


def closest_ground_truth_file(ground_truth_dir, target_date):
    """Cerca el fitxer de Ground Truth amb la data més propera, de manera recursiva."""
    base = Path(ground_truth_dir).expanduser().resolve()
    if not base.is_dir():
        print(f"Error: {base} no és un directori vàlid", file=sys.stderr)
        sys.exit(1)

    dated_files = []
    for csv_file in base.rglob("*.csv"):
        date = utils.extract_date(csv_file.name)
        if date is not None:
            dated_files.append((date, csv_file))

    if not dated_files:
        raise FileNotFoundError(f"No s'han trobat fitxers CSV amb data a {base}")

    return min(dated_files, key=lambda row: abs(row[0] - target_date))


def closest_estimated_date(available_dates, gt_date):
    """Busca la data estimada de Sentinel-2 més propera a la data de Ground Truth."""
    dates_parsed = [datetime.strptime(d, "%Y-%m-%d").date() for d in available_dates]
    closest_dt = min(dates_parsed, key=lambda d: abs(d - gt_date))
    return closest_dt.strftime("%Y-%m-%d"), abs((closest_dt - gt_date).days)


def process_single_comparison(project_dir, gt_csv_path, s2_date, show_plot=False):
    """Calcula la distància perpendicular directa i el RMSE filtrant per la mateixa finestra espacial."""
    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    df = pd.read_csv(csv_path)

    if s2_date not in df["date"].values:
        raise ValueError(f"La data {s2_date} no està present a {csv_path}")

    # 1. Carregar coordenades Sentinel-2 per a la data seleccionada
    s2_group = df[df["date"] == s2_date]
    s2_raw = s2_group[["x", "y"]].values.astype(float)

    # 2. Carregar Ground Truth (fitxer d'ortofoto)
    gt_coords = np.asarray(utils.load_coords_csv(gt_csv_path), dtype=float)
    if gt_coords.size == 0 or gt_coords.ndim != 2:
        raise ValueError(f"No s'han pogut carregar coordenades vàlides de {gt_csv_path}")

    utils.export_coords_to_kmz(gt_coords, gt_csv_path.with_suffix(".kmz"))

    # 3. Detectar i reprojectar Sentinel-2 a UTM (EPSG:32631) si està en graus geogràfics
    if np.abs(s2_raw[0, 0]) < 1000:
        # A Castelldefels/Catalunya: Lon ~ 1.9 - 2.0, Lat ~ 41.2 - 41.3
        if s2_raw[0, 0] < 10:  # És (Lon, Lat)
            lon, lat = s2_raw[:, 0], s2_raw[:, 1]
        else:  # És (Lat, Lon)
            lat, lon = s2_raw[:, 0], s2_raw[:, 1]

        transformer = Transformer.from_crs(4326, 32631, always_xy=True)
        x_m, y_m = transformer.transform(lon, lat)
        s2_utm = np.column_stack((x_m, y_m))
    else:
        s2_utm = s2_raw

    # 4. Assegurar que Ground Truth també estigui en UTM 31N
    if np.abs(gt_coords[0, 0]) < 1000:
        if gt_coords[0, 0] < 10:
            lon, lat = gt_coords[:, 0], gt_coords[:, 1]
        else:
            lat, lon = gt_coords[:, 0], gt_coords[:, 1]

        transformer = Transformer.from_crs(4326, 32631, always_xy=True)
        x_m, y_m = transformer.transform(lon, lat)
        gt_utm = np.column_stack((x_m, y_m))
    else:
        gt_utm = gt_coords

    # ------------------------------------------------------------------
    # 5. FILTRE ESPATIAL: Retallar el Ground Truth al mateix rang X que Sentinel-2
    # ------------------------------------------------------------------
    min_x, max_x = np.min(s2_utm[:, 0]) - 100, np.max(s2_utm[:, 0]) + 100
    mask_spatial = (gt_utm[:, 0] >= min_x) & (gt_utm[:, 0] <= max_x)
    gt_utm_filtered = gt_utm[mask_spatial]

    if len(gt_utm_filtered) < 2:
        print(f"Avís: El Ground Truth {gt_csv_path.name} no se superposa espacialment amb el rang del projecte.")
        return np.nan

    # 6. Geometries Shapely en metres
    reference_line = utils.build_ordered_linestring(gt_utm_filtered, ordered=False)
    s2_line = utils.build_ordered_linestring(s2_utm, ordered=False)

    # 7. Càlcul de distàncies d'error punt a línia en metres
    distances = [Point(pt[0], pt[1]).distance(s2_line) for pt in gt_utm_filtered]
    distances = np.array(distances)
    valid_distances = distances[~np.isnan(distances)]

    # 8. RMSE en metres reals
    rmse = np.sqrt((valid_distances ** 2).mean()) if len(valid_distances) > 0 else np.nan

    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        x_ref, y_ref = reference_line.xy
        ax.plot(x_ref, y_ref, "k--", label="Ground Truth (Retallat)", alpha=0.7)
        x_line, y_line = s2_line.xy
        ax.plot(x_line, y_line, "r-", label=f"Sentinel-2 ({s2_date})", alpha=0.7)
        ax.set_title(f"Validació Línia de Costa vs Ground Truth ({s2_date}) - RMSE: {rmse:.2f} m")
        ax.set_xlabel("UTM X (m)")
        ax.set_ylabel("UTM Y (m)")
        ax.legend()
        ax.axis("equal")
        plt.grid(True)
        plt.show()

    return rmse


def main(project, ground_truth, date=None, show_plot=True):
    project_dir = utils.project_path(project)
    output_dir = project_dir / "output" / "estimated_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No s'ha trobat el fitxer {csv_path}")

    df = pd.read_csv(csv_path)
    available_dates = df["date"].unique().tolist()

    validation_summary = []
    gt_dates_to_process = [date] if date else TARGET_GT_DATES

    for gt_date_str in gt_dates_to_process:
        gt_date = datetime.strptime(gt_date_str, "%Y-%m-%d").date()
        try:
            ref_date, gt_csv_path = closest_ground_truth_file(ground_truth, gt_date)
            s2_date, time_gap = closest_estimated_date(available_dates, gt_date)

            if time_gap > MAX_DAYS_GAP:
                print(
                    f"Avís: S'omet la data GT {gt_date_str} per un desfase temporal excessiu ({time_gap} dies > {MAX_DAYS_GAP} dies)."
                )
                rmse_val = np.nan
            else:
                print(f"Calculant RMSE per Ground Truth {gt_date_str} vs Sentinel-2 {s2_date}...")
                rmse_val = process_single_comparison(
                    project_dir, gt_csv_path, s2_date, show_plot=show_plot
                )

            validation_summary.append(
                {
                    "Ground truth date": gt_date_str,
                    "Sentinel 2 date": s2_date if time_gap <= MAX_DAYS_GAP else "N/A",
                    "Time gap (days)": time_gap,
                    "RMSE (meters)": round(rmse_val, 2) if not np.isnan(rmse_val) else "N/A",
                }
            )
        except Exception as e:
            print(f"Avís: No s'ha pogut processar la data {gt_date_str}: {e}")

    results_df = pd.DataFrame(validation_summary)
    results_df.to_csv(output_dir / "rmse_validation_summary.csv", index=False)

    print("\n" + "=" * 65)
    print(" RESULTATS DE LA VALIDACIÓ AMB GROUND TRUTH (RMSE)")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate shoreline error against ground truth data.")
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Nom del projecte a coastline_estimator/projects.",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        required=True,
        help="Ruta a la carpeta amb els fitxers CSV de ground truth.",
    )
    parser.add_argument("-d", "--date", required=False, help="Data concreta de Ground Truth a comparar (YYYY-MM-DD).")
    parser.add_argument("--no-plot", action="store_true", help="No mostrar gràfics visuals.")

    args = parser.parse_args()
    main(args.project, args.ground_truth, args.date, show_plot=not args.no_plot)

# python coastline_estimator/error_estimator.py -p castelldefels_h1_2025 -g coastline_estimator/projects/castelldefels_h1_2025/input/ground_truth --no-plot
