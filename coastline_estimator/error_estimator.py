import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, MultiPoint, Point

import shoreline_utils as utils


SPATIAL_SPACING = 10
TRANSECT_LENGTH = 100
TARGET_GT_DATES = ["2022-05-18", "2025-01-24", "2025-03-30"]


def closest_ground_truth_file(ground_truth_dir, target_date):
    base = Path(ground_truth_dir).expanduser().resolve()
    if not base.is_dir():
        print(f"Error: {base} no és un directori vàlid", file=sys.stderr)
        sys.exit(1)

    dated_files = []
    for csv_file in base.glob("*.csv"):
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


def signed_intersection_distance(intersection, px, py, normal):
    if isinstance(intersection, MultiPoint):
        distances = [Point(px, py).distance(point) for point in intersection.geoms]
        return distances[int(np.argmin(distances))]

    if isinstance(intersection, Point):
        signed_vector = np.array([intersection.x - px, intersection.y - py])
        distance = np.linalg.norm(signed_vector)
        if np.dot(signed_vector, normal) < 0:
            distance = -distance
        return distance

    return np.nan


def process_single_comparison(project_dir, gt_csv_path, s2_date, show_plot=False):
    """Calcula la distància dels transectes i el RMSE per a una parella (GT, Sentinel-2)."""
    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    df = pd.read_csv(csv_path)
    shorelines = {date: group[["x", "y"]].values for date, group in df.groupby("date")}

    if s2_date not in shorelines:
        raise ValueError(f"La data {s2_date} no està present a {csv_path}")

    # Carregar Ground Truth
    try:
        reference_coords = np.loadtxt(gt_csv_path, delimiter="\t", skiprows=1)
    except Exception:
        reference_coords = np.loadtxt(gt_csv_path, delimiter=",", skiprows=1)

    gt_coords = utils.load_coords_csv(gt_csv_path)
    utils.export_coords_to_kmz(gt_coords, gt_csv_path.with_suffix(".kmz"))

    x = reference_coords[:, 0]
    y = reference_coords[:, 1]
    xy_unique = np.unique(np.column_stack((x, y)), axis=0)
    x_unique = xy_unique[:, 0]
    y_unique = xy_unique[:, 1]

    reference_line = utils.build_ordered_linestring(np.column_stack((x, y)), ordered=True)
    shoreline_length = reference_line.length
    n_points = int(shoreline_length / SPATIAL_SPACING)

    tck, _ = splprep([x_unique, y_unique], s=0)
    unew = np.linspace(0, 1, num=n_points)
    x_smooth, y_smooth = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    line = utils.build_ordered_linestring(shorelines[s2_date], False)

    distance_results = []
    for i in range(len(unew)):
        px, py = x_smooth[i], y_smooth[i]
        tx, ty = dx[i], dy[i]
        normal = np.array([-ty, tx])
        normal_len = np.linalg.norm(normal)
        if normal_len > 0:
            normal /= normal_len

        start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
        end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
        transect = LineString([start, end])

        distance = signed_intersection_distance(transect.intersection(line), px, py, normal)
        distance_results.append(
            {
                "date": s2_date,
                "transect_id": i,
                "reference_x": px,
                "reference_y": py,
                "distance_m": distance,
            }
        )

    distance_df = pd.DataFrame(distance_results)
    valid_distances = distance_df["distance_m"].dropna()
    
    # RMSE formula: sqrt(mean((x_i - x_hat_i)^2 + (y_i - y_hat_i)^2))
    rmse = np.sqrt((valid_distances ** 2).mean()) if not valid_distances.empty else np.nan

    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        x_ref, y_ref = reference_line.xy
        ax.plot(x_ref, y_ref, "k--", label=f"Línia de referència GT", alpha=0.5)
        ax.plot(x_smooth, y_smooth, "b", label=f"Línia suavitzada GT", alpha=0.7)
        x_line, y_line = line.xy
        ax.plot(x_line, y_line, "r-", label=f"Estimació Sentinel-2 ({s2_date})", alpha=0.7)
        ax.set_title(f"Validació Línia de Costa vs Ground Truth ({s2_date})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
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

            print(f"Calculant RMSE per Ground Truth {gt_date_str} vs Sentinel-2 {s2_date}...")
            rmse_val = process_single_comparison(
                project_dir, gt_csv_path, s2_date, show_plot=show_plot
            )

            validation_summary.append(
                {
                    "Ground truth date": gt_date_str,
                    "Sentinel 2 date": s2_date,
                    "Time gap (days)": time_gap,
                    "RMSE (meters)": round(rmse_val, 2) if not np.isnan(rmse_val) else "N/A",
                }
            )
        except Exception as e:
            print(f"Avís: No s'ha pogut processar la data {gt_date_str}: {e}")

    # Generar i guardar la taula de resultats finals
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

# Per donar-li run:
# python coastline_estimator/error_estimator.py -p castelldefels_h1_2025 -g coastline_estimator/projects/castelldefels_h1_2025/input/ground_truth --no-plot
