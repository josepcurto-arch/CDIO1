from pystac_client import Client
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta  # for month intervals
import json
import os
from shapely.geometry import Point, mapping

CATALOG = Client.open("https://earth-search.aws.element84.com/v1")


def acquire(project):
    config = json.load(open(f"acquisition_forecaster/projects/{project}/input/config.json"))
    pt = Point(config["POINT"][0], config["POINT"][1])

    start_date = datetime.strptime(config["START_DATE"], "%Y-%m-%d")
    end_date = datetime.strptime(config["END_DATE"], "%Y-%m-%d")

    chunk_size = relativedelta(months=3)    # Chunk the query to prevent overloading the server

    current_start = start_date
    all_items = []

    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        date_range = f"{current_start.date()}/{current_end.date()}"
        print(f"Querying: {date_range}")

        search = CATALOG.search(
            collections=["sentinel-2-l2a"],
            intersects=mapping(pt),
            datetime=date_range,
            limit=1000,
        )
        try:
            items = list(search.items())
            all_items.extend(items)
        except Exception as e:
            print(f"Error on {date_range}: {e}")

        current_start = current_end

    data = []
    for item in all_items:
        dt = item.datetime
        data.append({
            'satellite': item.id[:3],
            'datetime': dt,
            'day_of_week': dt.strftime('%A'),
            'hour': dt.hour + dt.minute / 60
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by='datetime')  # ascending by default
    df['days_since_last'] = df['datetime'].diff().dt.total_seconds() / (3600 * 24)

    # Create empty columns to hold results
    df['days_since_last_s2a'] = None
    df['days_since_last_s2b'] = None
    df['days_since_last_s2c'] = None

    # Compute per-satellite differences
    for sat in ['S2A', 'S2B', 'S2C']:
        mask = df['satellite'] == sat
        diffs = df.loc[mask, 'datetime'].diff().dt.total_seconds() / (3600 * 24)
        df.loc[mask, f'days_since_last_{sat.lower()}'] = diffs

    print(df)
    os.makedirs(f"acquisition_forecaster/projects/{project}/output/", exist_ok=True)
    df.to_pickle(f"acquisition_forecaster/projects/{project}/output/sentinel2_acquisitions.pkl")


def plot(project):
    df = pd.read_pickle(f"acquisition_forecaster/projects/{project}/output/sentinel2_acquisitions.pkl")

    plt.plot(df['datetime'], df['days_since_last'], marker='o', linestyle='-', label='All Satellites')
    plt.plot(df['datetime'], df['days_since_last_s2a'], marker='o', linestyle='-', label='S2A')
    plt.plot(df['datetime'], df['days_since_last_s2b'], marker='x', linestyle='-', label='S2B')
    plt.plot(df['datetime'], df['days_since_last_s2c'], marker='s', linestyle='-', label='S2C')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Acquire and plot Sentinel-2 acquisition dates")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--action", type=str, choices=["acquire", "plot"], required=True, help="Action to perform")
    args = parser.parse_args()

    if args.action == "acquire":
        acquire(args.project)
    elif args.action == "plot":
        plot(args.project)
