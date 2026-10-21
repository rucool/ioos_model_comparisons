#!/usr/bin/env python
"""
Lightweight storm-track map for the Guam region — no gridded model data,
just bathymetry + JTWC storm track(s) + the two currently-deployed gliders
(ng252, ng738). Built because the live-published Guam OHC maps don't carry
a storm overlay yet (the server's ocean_heat_content.py predates the
jtwc=True fix — see that script and rtofs_binary_ohc_comparisons.py).

Usage:
    python guam_storm_track_map.py
"""
import datetime as dt
import types
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from cool_maps.plot import create

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.platforms import get_active_gliders
from ioos_model_comparisons.plotting_hurricanes import plot_storms
from ioos_model_comparisons.regions import region_config

REGION = "guam"
GLIDERS = ["ng252", "ng738"]


def main():
    extent = region_config(REGION)["extent"]
    today = dt.date.today()

    from tropycal import realtime as trealtime
    rt = trealtime.Realtime(jtwc=True)
    active_ids = rt.list_active_storms(basin="all")

    storms, forecasts = [], []
    for sid in active_ids:
        s = rt.get_storm(sid)
        lon = s.lon[-1] if hasattr(s.lon, "__len__") else s.lon
        lat = s.lat[-1] if hasattr(s.lat, "__len__") else s.lat
        # Only keep storms inside (or just outside) the Guam map extent.
        if not (extent[0] - 10 <= lon <= extent[1] + 10 and extent[2] - 10 <= lat <= extent[3] + 10):
            continue
        fct = {}
        if not s.invest:
            try:
                fct = s.get_forecast_realtime()
            except Exception as e:
                print(f"  No forecast cone for {sid} ({s.name}): {e} -- track only")
        storms.append(s)
        forecasts.append(fct)
        print(f"  Including {sid} {s.name}: lon={lon:.1f} lat={lat:.1f} vmax={s.vmax[-1]}kt")

    if not storms:
        print("No active storms found near Guam.")

    glider_data = get_active_gliders(
        [extent[0] - 5, extent[1] + 5, extent[2] - 5, extent[3] + 5],
        today - dt.timedelta(days=2), today, parallel=False,
    )

    fig, ax = create(
        extent, bathymetry=True, figsize=(12, 9),
        title=f"Guam Region — Storm Tracks & Active Gliders — {today.isoformat()}",
    )

    plot_storms(ax, storms, forecasts, zorder=5)

    # Guam itself
    ax.plot(144.7937, 13.4443, marker="*", markersize=20, markeredgecolor="black",
            color="red", transform=ccrs.PlateCarree(), zorder=20)
    ax.text(144.7937, 13.1, "Guam", transform=ccrs.PlateCarree(), ha="center",
            va="top", fontsize=12, fontweight="bold", zorder=20)

    colors = {"ng252": "dodgerblue", "ng738": "darkorange"}
    if not glider_data.empty:
        for gid in GLIDERS:
            match = [g for g in glider_data.index.get_level_values("glider").unique() if g.startswith(gid)]
            if not match:
                print(f"  {gid}: not found in active-glider query")
                continue
            df = glider_data.loc[match[0]].sort_index()
            ax.plot(df["lon"], df["lat"], "-", color=colors.get(gid, "black"),
                    linewidth=2, transform=ccrs.PlateCarree(), zorder=15)
            last = df.iloc[-1]
            ax.plot(last["lon"], last["lat"], marker="^", markersize=14,
                    markeredgecolor="black", color=colors.get(gid, "black"),
                    transform=ccrs.PlateCarree(), zorder=16, label=gid)
            ax.text(last["lon"], last["lat"] + 0.3, gid, transform=ccrs.PlateCarree(),
                    ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=16)

    ax.legend(loc="lower left", fontsize=10, framealpha=0.85)

    out = conf.path_plots / "maps" / "guam" / "storm_tracks"
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"guam_storm_tracks_{today.isoformat()}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved -> {outfile}")


if __name__ == "__main__":
    main()
