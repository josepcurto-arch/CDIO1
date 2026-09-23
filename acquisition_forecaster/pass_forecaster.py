from datetime import datetime, timedelta
from skyfield.api import EarthSatellite, load, wgs84
import json

ELEVATION_MIN_DEG = 0
STEP_SECONDS = 30

def overflight_times(project):
    config = json.load(open(f"acquisition_forecaster/projects/{project}/input/config.json"))
    tle_path = f"acquisition_forecaster/projects/{project}/input/tle"
    with open(tle_path, "r") as f:
        tle_lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(tle_lines) < 2:
        raise ValueError("TLE file must contain at least two non-empty lines")

    ts = load.timescale()
    sat = EarthSatellite(tle_lines[0], tle_lines[1], "SAT", ts)
    observer = wgs84.latlon(config["POINT"][1], config["POINT"][0], 0.0)

    start_utc = datetime.strptime(config["START_DATE"], "%Y-%m-%d")
    end_utc = datetime.strptime(config["END_DATE"], "%Y-%m-%d") + timedelta(days=1)

    passes = []
    current_pass_start = None

    t = start_utc
    while t < end_utc:
        tf = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)
        alt_now = (sat - observer).at(tf).altaz()[0].degrees

        t_next = t + timedelta(seconds=STEP_SECONDS)
        if t_next > end_utc:
            t_next = end_utc

        tf_next = ts.utc(
            t_next.year, t_next.month, t_next.day,
            t_next.hour, t_next.minute, t_next.second
        )
        alt_next = (sat - observer).at(tf_next).altaz()[0].degrees

        # ---- PASS START: below -> above ----
        if alt_now < ELEVATION_MIN_DEG <= alt_next and current_pass_start is None:
            frac = (ELEVATION_MIN_DEG - alt_now) / (alt_next - alt_now)
            dt_sec = (t_next - t).total_seconds()
            t_cross = t + timedelta(seconds=frac * dt_sec)
            current_pass_start = t_cross

        # ---- PASS END: above -> below ----
        if alt_now >= ELEVATION_MIN_DEG > alt_next and current_pass_start is not None:
            frac = (ELEVATION_MIN_DEG - alt_now) / (alt_next - alt_now)
            dt_sec = (t_next - t).total_seconds()
            t_cross = t + timedelta(seconds=frac * dt_sec)
            passes.append((current_pass_start, t_cross))
            current_pass_start = None

        t = t_next

    # If the satellite was above at the end of window, close the pass
    if current_pass_start is not None:
        passes.append((current_pass_start, end_utc))

    return passes



# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate satellite pass times")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    args = parser.parse_args()

    pass_list = overflight_times(args.project)

    if not pass_list:
        print(f"No overflights above {ELEVATION_MIN_DEG}° in the given interval.")
    else:
        print(f"Overflights above {ELEVATION_MIN_DEG}°:")
        for i, (start, end) in enumerate(pass_list, start=1):
            print(f"  #{i}: {start}  →  {end} UTC")
