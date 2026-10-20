"""
Save the raw best-track data (time, position, wind, pressure) for every storm
that passed through a region and reached at least `min_category` between
`y0` and `y1` -- the same set of storms plotted by
``hurricane_history_by_years.py`` -- to CSV files so the tracks can be reused
without needing tropycal installed.

Writes one CSV per storm plus a combined CSV of every storm concatenated
together.
"""
import numpy as np
import pandas as pd
from tropycal import tracks

import ioos_model_comparisons.configs as conf

# Minimum sustained wind speed (knots) for each Saffir-Simpson category.
CATEGORY_MIN_KT = {1: 64, 2: 83, 3: 96, 4: 113, 5: 137}


def _storm_in_region(lons, lats, region_extent, buffer=0.0):
    """Return True if any track point falls within region_extent (+buffer)."""
    lon_min, lon_max, lat_min, lat_max = region_extent
    in_lon = (lons >= lon_min - buffer) & (lons <= lon_max + buffer)
    in_lat = (lats >= lat_min - buffer) & (lats <= lat_max + buffer)
    return bool(np.any(in_lon & in_lat))


def _collect_storms(basin, year, wind_threshold, region_extent=None):
    """Return tropycal Storm objects for every storm in `year` that reached
    >=wind_threshold kt and, if `region_extent` is given, entered that region."""
    season = basin.get_season(year)
    storm_ids = season.summary()["id"]
    storms = []

    for storm_id in storm_ids:
        storm = basin.get_storm(storm_id)
        vmax = np.asarray(storm.dict["vmax"], dtype=float)
        if np.isnan(vmax).all() or np.nanmax(vmax) < wind_threshold:
            continue

        lons = np.asarray(storm.dict["lon"], dtype=float)
        lats = np.asarray(storm.dict["lat"], dtype=float)
        if region_extent is not None and not _storm_in_region(lons, lats, region_extent):
            continue

        storms.append(storm)

    return storms


def _track_dataframe(storm):
    """Build a tidy dataframe of a storm's full best-track history."""
    d = storm.dict
    df = pd.DataFrame({
        "time": pd.to_datetime(d["time"]),
        "lat": d["lat"],
        "lon": d["lon"],
        "vmax_kt": d["vmax"],
        "mslp_mb": d["mslp"],
        "type": d["type"],
    })
    df.insert(0, "name", getattr(storm, "name", d.get("id", "")))
    df.insert(0, "storm_id", d.get("id", ""))
    return df


def main():
    y0 = 2016
    y1 = 2025
    min_category = 4  # Minimum Saffir-Simpson category to include (1-5)
    extent = [-98.25, -80.75, 17.75, 30.25]  # Gulf of Mexico
    wind_threshold = CATEGORY_MIN_KT[min_category]

    out_dir = conf.path_plots / "maps" / "atlantic_hurricanes" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    basin = tracks.TrackDataset(basin="north_atlantic", include_btk=True)

    all_tracks = []
    for year in range(y0, y1 + 1):
        try:
            storms = _collect_storms(basin, year, wind_threshold, region_extent=extent)
        except Exception as exc:
            print(f"Warning: Could not load season {year}: {exc}")
            continue

        for storm in storms:
            df = _track_dataframe(storm)
            storm_id = df["storm_id"].iloc[0]
            name = df["name"].iloc[0]
            display_name = name.upper() if name and name.upper() != "UNNAMED" else storm_id

            fname = f"{storm_id}_{display_name}.csv".replace(" ", "_")
            df.to_csv(out_dir / fname, index=False)
            print(f"  - saved {fname}")
            all_tracks.append(df)

    if not all_tracks:
        print(f"No Gulf of Mexico category {min_category}+ hurricanes found between {y0} and {y1}.")
        return

    combined = pd.concat(all_tracks, ignore_index=True)
    combined_fname = f"gulf_of_mexico_category{min_category}plus_hurricanes_{y0}_to_{y1}_tracks.csv"
    combined.to_csv(out_dir / combined_fname, index=False)
    print(f"Saved combined track file: {combined_fname}")
    print(f"All files saved to {out_dir}")


if __name__ == "__main__":
    main()
