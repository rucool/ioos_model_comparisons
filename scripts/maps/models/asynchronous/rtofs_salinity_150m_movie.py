"""
RTOFS-only 150 m salinity maps (+ optional movie) for the Gulf of Mexico, with
gliders and Argo floats overlaid. Built to show the gliders crossing the
Yucatan Channel and the Argo floats in the Gulf that bring the salinity max
back into the model.

Same look as the standard rtofs-gofs-cmems-amseas.py maps, but a single RTOFS
panel, one timestamp per day (12Z by default) so it plots quickly.

Usage:
    python3 scripts/maps/models/asynchronous/rtofs_salinity_150m_movie.py

    # Different window / region / depth, no movie
    python3 scripts/maps/models/asynchronous/rtofs_salinity_150m_movie.py \
        --start 2024-01-15T12 --end 2024-05-15T12 --regions gom --depth 150 --no-movie
"""
import argparse
import datetime as dt
import logging
import subprocess
import time

import matplotlib
import numpy as np
import pandas as pd

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.db import apply_colorbar_overrides
from ioos_model_comparisons.models import rtofs
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
from ioos_model_comparisons.plotting import plot_model_region_single, region_transform
from ioos_model_comparisons.regions import region_config
from cool_maps.plot import get_bathymetry

matplotlib.use('agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tstr = '%Y-%m-%d %H:%M:%S'
path_save = conf.path_plots / "maps"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions", nargs="*", default=["gom"])
    p.add_argument("--start", type=pd.Timestamp, default=pd.Timestamp("2024-01-15T12"))
    p.add_argument("--end", type=pd.Timestamp, default=pd.Timestamp("2024-05-15T12"))
    p.add_argument("--freq", default="24H", help="Timestamp spacing (default: 24H, i.e. one 12Z map per day)")
    p.add_argument("--depth", type=int, default=150)
    p.add_argument("--variable", default="salinity")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--no-movie", action="store_true")
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def load_assets(extent, t0, t1):
    """Argo + glider tracks for the whole window (subset per-frame later)."""
    argo = pd.DataFrame()
    gliders = pd.DataFrame()
    if conf.argo:
        try:
            argo = get_argo_floats_by_time(extent, t0, t1)
            logger.info(f"Argo data loaded ({len(argo)} records).")
        except Exception as e:
            logger.error(f"Failed to load Argo data: {e}")
    if conf.gliders:
        try:
            gliders = get_active_gliders(extent, t0, t1, parallel=False, timeout=60)
            logger.info(f"Glider data loaded ({len(gliders)} records).")
            # Same handling as the standard script: keep the latest deployment
            # per glider and drop the deployment suffix from the name.
            dep_ids = gliders.index.get_level_values('glider').unique()
            latest = {}
            for dep_id in dep_ids:
                short = dep_id.rsplit("-", 1)[0]
                if short not in latest or dep_id > latest[short]:
                    latest[short] = dep_id
            gliders = gliders[gliders.index.get_level_values('glider').isin(set(latest.values()))]
            gliders.index = gliders.index.remove_unused_levels()
            gliders.index = gliders.index.set_levels(
                gliders.index.levels[0].str.rsplit("-", n=1).str[0], level="glider"
            )
        except Exception as e:
            logger.error(f"Failed to load Glider data: {e}")
            gliders = pd.DataFrame()
    return argo, gliders


def subset_rtofs(ds, extent):
    lons_ind = np.interp(extent[:2], ds.lon.values[0, :], ds.x.values)
    lats_ind = np.interp(extent[2:], ds.lat.values[:, 0], ds.y.values)
    return ds.isel(
        x=slice(int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1]))),
        y=slice(int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1]))),
    )


def make_movie(frames_dir, pattern_files, out_file, fps):
    """Stitch PNGs into an H.264 mp4 (pads to even dimensions for yuv420p)."""
    list_file = frames_dir / "frames.txt"
    list_file.write_text("".join(f"file '{f}'\nduration {1 / fps}\n" for f in pattern_files))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-i", str(list_file),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
           "-r", str(fps), "-c:v", "libx264", str(out_file)]
    subprocess.run(cmd, check=True)
    list_file.unlink()
    logger.info(f"Movie written: {out_file}")


def main():
    args = parse_args()
    start_time = time.time()

    date_list = pd.date_range(args.start, args.end, freq=args.freq)
    regions = {name: apply_colorbar_overrides(name, region_config(name)) for name in args.regions}

    # Restrict each region to the one variable/depth we care about.
    for name, region in regions.items():
        entries = [e for e in region['variables'].get(args.variable, []) if e['depth'] == args.depth]
        if not entries:
            raise SystemExit(f"{name}: no {args.variable} @ {args.depth} m in region config")
        region['variables'] = {args.variable: entries}

    extents = np.array([r['extent'] for r in regions.values()])
    global_extent = [extents[:, 0].min(), extents[:, 1].max(), extents[:, 2].min(), extents[:, 3].max()]
    logger.info(f"{len(date_list)} timestamps {date_list[0]} → {date_list[-1]}, regions={list(regions)}")

    search_start = date_list[0] - dt.timedelta(hours=conf.search_hours)
    argo_data, glider_data = load_assets(global_extent, search_start, date_list[-1])
    bathy = get_bathymetry(global_extent) if conf.bathy else None

    rds = rtofs()
    frames = {name: [] for name in regions}

    for ctime in date_list:
        try:
            rds_time = rds.sel(time=ctime)
        except KeyError:
            logger.warning(f"RTOFS: no data for {ctime}, skipping")
            continue

        t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
        t1 = ctime.strftime(tstr)

        for name, region in regions.items():
            extent = region['extent']
            extended = np.add(extent, [-1, 1, -1, 1]).tolist()
            rds_sub = subset_rtofs(rds_time, extended).set_coords(['u', 'v'])

            kw = dict(
                bathy=bathy,
                path_save=path_save / region['folder'],
                transform=region_transform(extent),
                dpi=conf.dpi,
                overwrite=not args.no_overwrite,
                eez=region.get('eez', False),
            )
            if 'legend' in region.get('figure', {}):
                kw['cols'] = region['figure']['legend']['columns']

            if not argo_data.empty:
                lon, lat = argo_data['lon'], argo_data['lat']
                mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
                argo_region = argo_data[mask].sort_index()
                kw['argo'] = argo_region.loc[pd.IndexSlice[:, t0:t1], :]

            if not glider_data.empty:
                lon, lat = glider_data['lon'], glider_data['lat']
                mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
                g = glider_data[mask]
                times = g.index.get_level_values('time')
                kw['gliders'] = g[(pd.Timestamp(t0) < times) & (times < pd.Timestamp(t1))]

            try:
                plot_model_region_single(rds_sub, region, **kw)
                fname = (f'{"-".join(region["folder"].split("_"))}_{ctime.strftime("%Y-%m-%dT%H%M%SZ")}'
                         f'_{args.variable}-{args.depth}m_{rds_sub.model.lower()}.png')
                frames[name].append(
                    kw['path_save'] / f"{args.variable}_{args.depth}m" / ctime.strftime('%Y/%m') / fname)
                logger.info(f"Plotted {name} {ctime}")
            except Exception as e:
                logger.error(f"Failed {name} at {ctime}: {e}", exc_info=True)

    if not args.no_movie:
        for name, files in frames.items():
            files = [f for f in files if f.is_file()]
            if not files:
                continue
            region = regions[name]
            tag = f"{args.variable}_{args.depth}m"
            out = path_save / region['folder'] / tag / (
                f"{region['folder']}_{tag}_rtofs_{args.start:%Y%m%d}-{args.end:%Y%m%d}.mp4")
            make_movie(path_save / region['folder'] / tag, files, out, args.fps)

    logger.info(f"Done in {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
