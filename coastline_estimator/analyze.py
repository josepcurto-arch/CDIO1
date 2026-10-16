import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd

import shoreline_utils as utils


def main(project):
    csv_path = (
        utils.project_path(project)
        / "output"
        / "estimated_erosion_accretion"
        / "shoreline_distances.csv"
    )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["distance_m"])

    colors = plt.cm.tab10.colors
    markers = itertools.cycle(("o", "s", "^", "D", "v", "<", ">", "p", "*", "h"))

    # 1. Gràfic de perfil espacial per transecte
    plt.figure(figsize=(12, 6))
    for i, (date, group) in enumerate(df.groupby("date")):
        plt.plot(
            group["transect_id"],
            group["distance_m"],
            label=date.strftime("%Y-%m-%d"),
            marker=next(markers),
            color=colors[i % len(colors)],
            linestyle="-",
        )

    plt.xlabel("ID del Transecte")
    plt.ylabel("Distància (m)")
    plt.title("Distància de la Línia de Costa per Transecte i Data")
    plt.grid(True)
    plt.legend(title="Data")
    plt.tight_layout()
    plt.show()

    # 2. Gràfic d'evolució temporal mitjana
    avg_df = df.groupby("date")["distance_m"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(avg_df["date"], avg_df["distance_m"], marker="o", linestyle="-", color="tab:blue")
    plt.xlabel("Data")
    plt.ylabel("Distància Mitjana (m)")
    plt.title("Distància Mitjana de la Línia de Costa al Llarg del Temps")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representa els resultats d'erosió i acreditació.")
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Nom del projecte ubicat a coastline_estimator/projects.",
    )
    args = parser.parse_args()
    main(args.project)
