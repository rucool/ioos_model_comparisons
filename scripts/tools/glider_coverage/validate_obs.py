#!/usr/bin/env python3
"""
Observation check of the glider radius of influence: is a model more accurate
near recent glider data, and at what distance does that advantage disappear?

A set of independent "referee" profiles is compared with the model field from
BEFORE each profile (00Z RTOFS snapshot / CMEMS daily mean, --lead-days
earlier), so the model has not yet assimilated it. Each referee profile is
tagged with the distance to the nearest glider profile in the --window-days
before that model time (only glider data the model could already have used),
and errors are binned by that distance.

Variables (whichever the model files and the referee provide):
    heat, t100, t200   temperature: 0-200 m mean, ~100 m, ~200 m      (fields.nc)
    salt, s100, s200   salinity: 0-200 m mean, ~100 m, ~200 m         (fields_salt.nc)
    u, v               glider depth-averaged current vs the model's mean current
                       over the glider's dive depth (0-200, 0-500 or 0-1000 m,
                       picked from the profile's maximum depth)       (fields_uv.nc)
                       Gliders only: Argo floats do not measure currents.

Referees (--referee):
    argo     good-QC Argo profiles. Independent of the gliders; the clean test.
    gliders  glider profiles themselves, with distance measured to the nearest
             profile from a DIFFERENT glider mission. The referee glider's own
             earlier profiles are already in the model, so this measures what a
             neighbouring glider adds on top of a glider's own data -- the
             question for fleet spacing. Profiles are thinned to one per mission
             per 6 h.

Gliders are deployed where the ocean is lively, where any model struggles, so
raw errors are confounded. Each error is therefore also divided by the local
natural variability (anomaly std of that model field at that grid point).
Confidence intervals resample whole platforms (floats or glider missions).

Use RTOFS or ESPC (check ESPC with --lead-days first). The CMEMS archive is revised weekly with observations from the same
week (likely including the referee profiles), so short leads are not a fair test.

Inputs: <workdir>/<region>/<model>/fields*.nc and masks.npz (download.py, scales.py).
Observations are fetched month by month and cached in <workdir>/obs/<region>/.

Writes to <workdir>/<region>/<model>/validation_<referee>_lead<N>d/:
    matchups.csv          one row per referee profile: per-variable errors, nearest-glider distance and age
    binned.csv            per variable, season and distance bin: RMS error (raw and normalized), bias,
                          centered RMS (bias removed), counts, each with a 90% CI
    validation_<var>.png  raw and normalized RMS error and bias vs distance to the nearest glider

Usage:
    python scripts/tools/glider_coverage/validate_obs.py --model rtofs
    python scripts/tools/glider_coverage/validate_obs.py --model rtofs --region caribbean_pr --referee gliders
    python scripts/tools/glider_coverage/validate_obs.py --model rtofs --until 2025-07-31   # RTOFS v2.4 only
    python scripts/tools/glider_coverage/validate_obs.py --model rtofs --since 2025-08-01   # RTOFS v2.5 only
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gc_regions import REGIONS
from ioos_model_comparisons import configs
from ioos_model_comparisons.platforms import (ARGO_GOOD_QC_FLAGS, filter_argo_by_qc, get_active_gliders,
                                              get_argo_floats_by_time)

R_E = 6371.0
# Model depths (m) that the stored fields correspond to: levels for t100/t200 (and s100/s200)
# and the depth-mean limit for heat/salt.
DEPTHS = {
    "rtofs": dict(d100=100.0, d200=200.0, h=200.0),
    "cmems": dict(d100=109.7, d200=186.1, h=186.1),
    "espc": dict(d100=100.0, d200=200.0, h=200.0),
}


# Depth-averaged currents faster than this are bad data: gliders cannot fly in them, and real
# 0-1000 m mean currents stay well below it even in the Loop Current.
DAC_MAX = 1.5  # m/s


def _dac(r, c):
    x = r.get(f"{c}_dac")
    return x if x is not None and np.isfinite(x) and abs(x) <= DAC_MAX else None


def _dac_band(r):
    """Model depth-mean current that matches a glider's dive depth."""
    z = r.get("dive_depth", np.nan)
    return 1000 if z >= 900 else 500 if z >= 450 else 200 if z >= 180 else None


# label, unit, observed value from a referee row, and the model field it is compared with
VARS = {
    "heat": dict(label="Mean temperature, surface to ~200 m", unit="°C", obs=lambda r, d: r.get(f"heat{d['h']}"), field=lambda r: "heat"),
    "t100": dict(label="Temperature near 100 m", unit="°C", obs=lambda r, d: r.get(f"t{d['d100']:g}"), field=lambda r: "t100"),
    "t200": dict(label="Temperature near 200 m", unit="°C", obs=lambda r, d: r.get(f"t{d['d200']:g}"), field=lambda r: "t200"),
    "salt": dict(label="Mean salinity, surface to ~200 m", unit="", obs=lambda r, d: r.get(f"salt{d['h']}"), field=lambda r: "salt"),
    "s100": dict(label="Salinity near 100 m", unit="", obs=lambda r, d: r.get(f"s{d['d100']:g}"), field=lambda r: "s100"),
    "s200": dict(label="Salinity near 200 m", unit="", obs=lambda r, d: r.get(f"s{d['d200']:g}"), field=lambda r: "s200"),
    "u": dict(label="Depth-averaged eastward current", unit="m/s", obs=lambda r, d: _dac(r, "u"),
              field=lambda r: f"u{_dac_band(r)}" if _dac_band(r) else None),
    "v": dict(label="Depth-averaged northward current", unit="m/s", obs=lambda r, d: _dac(r, "v"),
              field=lambda r: f"v{_dac_band(r)}" if _dac_band(r) else None),
}
BINS = [0, 50, 100, 150, 250, 500]
LOOKBACK_PAD_DAYS = 30  # glider cache starts this long before the model period; caps lead + window


def hav(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.deg2rad, (lo1, la1, lo2, la2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_E * np.arcsin(np.sqrt(a))


def months(start, end):
    edges = list(pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end), freq="MS"))
    edges = [pd.Timestamp(start)] + [e for e in edges if e > pd.Timestamp(start)] + [pd.Timestamp(end)]
    return list(zip(edges[:-1], edges[1:]))


# Plausible ranges: samples outside are dropped before summarizing a profile, and referee summaries
# outside are rejected at matchup time (glider salinity in particular has occasional spikes/fill values).
VALID = {"t": (-2.0, 40.0), "s": (10.0, 42.0)}
SUMMARY_VALID = {"heat": (5, 35), "t100": (5, 35), "t200": (5, 35), "salt": (30, 40), "s100": (33, 39), "s200": (33, 39)}


def _summarize(z_obs, x_obs, prefix, mean_prefix):
    """Level values and 0-H depth means of one variable from one profile (needs 0-10 m to >=200 m)."""
    lo, hi = VALID[prefix]
    ok = np.isfinite(z_obs) & np.isfinite(x_obs) & (x_obs >= lo) & (x_obs <= hi)
    z_obs, x_obs = z_obs[ok], x_obs[ok]
    o = np.argsort(z_obs)
    z_obs, k = np.unique(z_obs[o], return_index=True)
    x_obs = x_obs[o][k]
    if len(z_obs) < 10 or z_obs[0] > 10 or z_obs[-1] < 200:
        return {}
    z = np.arange(0, 201)
    x = np.interp(z, z_obs, x_obs)  # constant above the shallowest sample
    out = {f"{prefix}{d:g}": float(np.interp(d, z, x)) for d in (100.0, 109.7, 186.1, 200.0)}
    for H in (186.1, 200.0):
        zz = np.append(z[z < H], H)
        out[f"{mean_prefix}{H}"] = float(np.trapezoid(np.interp(zz, z, x), zz) / H)
    return out


def profile_stats(z, t, s=None):
    """Temperature (and salinity, if given) summaries; None if temperature is unusable."""
    out = _summarize(z, t, "t", "heat")
    if not out:
        return None
    if s is not None:
        out.update(_summarize(z, s, "s", "salt"))
    out["dive_depth"] = float(np.nanmax(z))
    return out


# --- observations ---------------------------------------------------------
def fetch_gliders(region, start, end, cache):
    """One row per glider profile: glider, time, lon, lat (inside the region). Positions only."""
    f = cache / f"gliders_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    R = REGIONS[region]
    parts = []
    for t0, t1 in months(start, end):
        df = get_active_gliders(bbox=R["bbox"], t0=t0.to_pydatetime(), t1=t1.to_pydatetime(),
                                variables=["time", "longitude", "latitude", "profile_id"], timeout=120)
        if df.empty:
            continue
        df = df.reset_index()
        df = df.groupby(["glider", "profile_id"], as_index=False).agg(time=("time", "min"), lon=("lon", "mean"), lat=("lat", "mean"))
        parts.append(df)
        print(f"  gliders {t0:%Y-%m}: {len(df)} profiles, {df.glider.nunique()} missions", flush=True)
    g = pd.concat(parts).drop_duplicates(["glider", "profile_id"])
    g = g[R["inside"](g.lon.values, g.lat.values)].sort_values("time").reset_index(drop=True)
    g.to_pickle(f)
    return g


def fetch_argo(region, start, end, cache):
    """Referee table from good-QC Argo profiles: id, time, lon, lat + temperature and salinity summaries.
    Temperature needs good position/pressure/temperature QC; salinity is kept only where psal QC is good."""
    f = cache / f"argo_ts_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    R = REGIONS[region]
    good = {str(x) for x in ARGO_GOOD_QC_FLAGS}
    rows = []
    for t0, t1 in months(start, end):
        df = get_argo_floats_by_time(bbox=R["bbox"], time_start=t0, time_end=t1, variables=["pres", "temp", "psal"],
                                     include_qc=["position_qc", "pres_qc", "temp_qc", "psal_qc"])
        if df.empty:
            continue
        df = filter_argo_by_qc(df, qc_columns=["position_qc", "pres_qc", "temp_qc"]).reset_index()
        df = df.rename(columns={"pres (decibar)": "pres", "temp (degree_Celsius)": "temp", "psal (PSU)": "psal"})
        if "psal" in df:
            df.loc[~df.psal_qc.astype(str).str.strip().isin(good), "psal"] = np.nan
        n0 = len(rows)
        for (wmo, t), p in df.groupby(["argo", "time"]):
            s = profile_stats(p.pres.values.astype(float), p.temp.values.astype(float),
                              p.psal.values.astype(float) if "psal" in p else None)
            if s:
                rows.append(dict(argo=str(wmo), time=t, lon=p.lon.iloc[0], lat=p.lat.iloc[0], **s))
        print(f"  argo {t0:%Y-%m}: {len(rows) - n0} usable profiles", flush=True)
    a = pd.DataFrame(rows)
    a = a[R["inside"](a.lon.values, a.lat.values)].sort_values("time").reset_index(drop=True)
    a.to_pickle(f)
    return a


def fetch_glider_profiles(region, start, end, cache, thin_hours=6):
    """Referee table from glider profiles, thinned to one per mission per thin_hours: temperature and
    salinity summaries, dive depth, and the depth-averaged current (u_dac, v_dac) of a dive in the slot."""
    f = cache / f"glider_profiles_ts_uv_{start:%Y%m%d}_{end:%Y%m%d}_{thin_hours}h.pkl"
    if f.exists():
        return pd.read_pickle(f)
    R = REGIONS[region]
    rows = []
    for t0, t1 in months(start, end):
        kw = dict(bbox=R["bbox"], t0=t0.to_pydatetime(), t1=t1.to_pydatetime(), timeout=300)
        df = get_active_gliders(variables=["time", "longitude", "latitude", "profile_id", "depth", "temperature", "salinity"], **kw)
        if df.empty:
            continue
        # Depth-averaged currents are per-dive (profile-level) and often missing on one profile of
        # each dive, so they come in a separate request to keep profiles without them.
        dac = get_active_gliders(variables=["time", "profile_id", "u", "v"], **kw)
        dac = (dac.reset_index().groupby(["glider", "profile_id"], as_index=False)[["u", "v"]].first()
               if not dac.empty else pd.DataFrame(columns=["glider", "profile_id", "u", "v"]))
        df = df.reset_index()
        df = df[R["inside"](df.lon.values, df.lat.values)]
        df["slot"] = df.time.dt.floor(f"{thin_hours}h")
        n0 = len(rows)
        for (gl, slot), d in df.groupby(["glider", "slot"]):
            stats = None
            cur = dict(u_dac=np.nan, v_dac=np.nan, dac_depth=np.nan)
            for pid, p in d.groupby("profile_id"):
                z = p.depth.values.astype(float)
                if stats is None:
                    s = profile_stats(z, p.temperature.values.astype(float), p.salinity.values.astype(float))
                    if s:
                        stats = dict(argo=gl, time=p.time.min(), lon=p.lon.mean(), lat=p.lat.mean(), **s)
                if np.isnan(cur["u_dac"]):
                    m = dac[(dac.glider == gl) & (dac.profile_id == pid)]
                    if len(m) and np.isfinite(m.u.iloc[0]) and np.isfinite(m.v.iloc[0]):
                        cur = dict(u_dac=float(m.u.iloc[0]), v_dac=float(m.v.iloc[0]), dac_depth=float(np.nanmax(z)))
            if stats:
                if np.isfinite(cur["u_dac"]):  # the current applies to its own dive's depth
                    stats.update(cur)
                    stats["dive_depth"] = cur["dac_depth"]
                rows.append(stats)
        print(f"  glider profiles {t0:%Y-%m}: {len(rows) - n0} usable (1 per mission per {thin_hours} h)", flush=True)
    g = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    g.to_pickle(f)
    return g


# --- matchups -------------------------------------------------------------
def local_sd(F, var):
    """Per-grid-point std of the anomaly (mean + annual + semiannual removed): how much the ocean
    normally varies there. Errors are divided by it so calm and lively areas compare fairly."""
    tt = (F.time.values - F.time.values[0]) / np.timedelta64(1, "D")
    w = 2 * np.pi * tt / 365.25
    H = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    X = F[var].values.reshape(len(tt), -1)
    good = np.all(np.isfinite(X), axis=0)
    sd = np.full(X.shape[1], np.nan)
    beta, *_ = np.linalg.lstsq(H, X[:, good], rcond=None)
    sd[good] = (X[:, good] - H @ beta).std(0)
    return sd.reshape(F[var].shape[1:])


def load_fields(W):
    """fields.nc (temperature) plus fields_salt.nc / fields_uv.nc when present."""
    return xr.merge([xr.open_dataset(f) for f in sorted(W.glob("fields*.nc"))], combine_attrs="override")


def matchups(model, refs, gliders, W, lead_days, window_days, exclude_same_platform=False):
    """One row per referee profile, with err_/nerr_/sd_/obs_ columns per variable (NaN where that
    variable is unavailable). With exclude_same_platform, helper gliders with the referee's own id
    (its own mission) are ignored when finding the nearest glider."""
    F = load_fields(W)
    m = np.load(W / "masks.npz")
    deep = m["deep200"]
    lon1, lat1 = F.longitude.values, F.latitude.values
    times = F.time.values.astype("datetime64[D]")
    dep = DEPTHS[model]
    names = sorted({n for n in F.data_vars})
    fields = {n: F[n].values for n in names}
    sds = {n: local_sd(F, n) for n in names}
    gt, glon, glat, gid = gliders.time.values, gliders.lon.values, gliders.lat.values, gliders.glider.values

    rows = []
    for r in refs.to_dict("records"):
        mt = np.datetime64(pd.Timestamp(r["time"]).normalize() - pd.Timedelta(days=lead_days), "D")
        k = np.searchsorted(times, mt)
        if k >= len(times) or times[k] != mt:
            continue
        i, j = np.abs(lat1 - r["lat"]).argmin(), np.abs(lon1 - r["lon"]).argmin()
        if not deep[i, j]:
            continue
        vals = {}
        for v, spec in VARS.items():
            name, obs = spec["field"](r), spec["obs"](r, dep)
            if name not in fields or obs is None or not np.isfinite(obs):
                continue
            if v in SUMMARY_VALID and not SUMMARY_VALID[v][0] <= obs <= SUMMARY_VALID[v][1]:
                continue
            mod, sd = float(fields[name][k, i, j]), float(sds[name][i, j])
            if np.isfinite(mod) and np.isfinite(sd) and sd > 0:
                vals.update({f"err_{v}": mod - obs, f"nerr_{v}": (mod - obs) / sd, f"sd_{v}": sd, f"obs_{v}": obs})
        if not vals:
            continue
        tm = pd.Timestamp(mt).to_datetime64()
        sel = (gt <= tm) & (gt >= tm - np.timedelta64(window_days, "D"))
        if exclude_same_platform:
            sel &= gid != r["argo"]
        if sel.any():
            d = hav(r["lon"], r["lat"], glon[sel], glat[sel])
            n = int(np.argmin(d))
            near = dict(glider_km=float(d[n]), glider_age_days=float((tm - gt[sel][n]) / np.timedelta64(1, "h") / 24),
                        glider=gid[sel][n], n_glider_profiles_100km=int((d < 100).sum()))
        else:
            near = dict(glider_km=np.inf, glider_age_days=np.nan, glider="", n_glider_profiles_100km=0)
        rows.append(dict(platform=r["argo"], time=r["time"], lon=r["lon"], lat=r["lat"], model_time=str(mt),
                         dive_depth=r.get("dive_depth", np.nan), **near, **vals))
    M = pd.DataFrame(rows)
    for v in VARS:  # every variable gets a column, even if never available
        for p in ("err_", "nerr_", "sd_", "obs_"):
            if f"{p}{v}" not in M:
                M[f"{p}{v}"] = np.nan
    M["season"] = np.where(pd.to_datetime(M.time).dt.month.isin(range(6, 12)), "Jun-Nov", "Dec-May")
    return M


def binned(M, v, rng, n_boot=2000):
    """For one variable: RMS error per distance bin, raw and normalized, bias, and centered RMS (bias
    removed), with 90% intervals from resampling whole platforms. Per-platform sums make each bootstrap
    draw a cheap weighted sum."""
    M = M[np.isfinite(M[f"err_{v}"].values)]
    labels = [f"{a}-{b} km" for a, b in zip(BINS[:-1], BINS[1:])] + [f">{BINS[-1]} km or none"]
    nb = len(labels)
    out = [dict(var=v, bin=lab, n=0, n_platforms=0, median_sd=np.nan) for lab in labels]
    if len(M) == 0:
        return pd.DataFrame(out)
    idx = np.digitize(np.nan_to_num(M.glider_km.values, posinf=1e12), BINS) - 1
    plats, pi = np.unique(M.platform.values, return_inverse=True)

    def sums(x):
        a = np.zeros((len(plats), nb))
        np.add.at(a, (pi, idx), x)
        return a

    counts = sums(np.ones(len(M)))
    weight = np.zeros((n_boot, len(plats)))
    for b, draw in enumerate(rng.integers(0, len(plats), size=(n_boot, len(plats)))):
        np.add.at(weight[b], draw, 1)
    n_boot_bin = weight @ counts
    for k in range(nb):
        s = idx == k
        out[k].update(n=int(s.sum()), n_platforms=int(M.platform[s].nunique()),
                      median_sd=float(M[f"sd_{v}"][s].median()) if s.any() else np.nan)
    for col in ("err_", "nerr_"):
        e = M[f"{col}{v}"].values
        s1, s2 = sums(e), sums(e ** 2)
        n = counts.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1, m2 = (weight @ s1) / n_boot_bin, (weight @ s2) / n_boot_bin
            stats = {
                ("rmse" if col == "err_" else "nrmse"): (np.sqrt(s2.sum(0) / n), np.sqrt(m2)),
                ("crmse" if col == "err_" else "ncrmse"): (np.sqrt(np.maximum(s2.sum(0) / n - (s1.sum(0) / n) ** 2, 0)),
                                                           np.sqrt(np.maximum(m2 - m1 ** 2, 0))),
            }
            if col == "err_":
                stats["bias"] = (s1.sum(0) / n, m1)
        for p, (est, boot) in stats.items():
            boot = np.where(n_boot_bin < 5, np.nan, boot)
            for k in range(nb):
                ok = out[k]["n"] >= 5
                out[k][p] = float(est[k]) if ok else np.nan
                lo, hi = np.nanpercentile(boot[:, k], [5, 95]) if ok else (np.nan, np.nan)
                out[k][f"{p}_p05"], out[k][f"{p}_p95"] = float(lo), float(hi)
    return pd.DataFrame(out)


def figure(B, Bs, out, meta, v):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    INK2, GRID = "#52514e", "#e4e3df"
    spec = VARS[v]
    unit = f" ({spec['unit']})" if spec["unit"] else ""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), facecolor="#fcfcfb", sharex=True)
    x = np.arange(len(B))

    def line(ax, D, col, lab, off, color, ls="-"):
        y = D[col].values
        ax.errorbar(x + off, y, yerr=[y - D[f"{col}_p05"].values, D[f"{col}_p95"].values - y],
                    color=color, marker="o", ms=5, lw=1.8, ls=ls, capsize=3, label=lab)

    ax = axs[0]
    line(ax, B, "rmse", f"Raw RMS error{unit}", -0.08, "#8a8984")
    ax2 = ax.twinx()
    ax2.plot(x, B.median_sd.values, color="#eb6834", lw=1.4, ls=":", marker="s", ms=4)
    ax2.set_ylabel(f"Natural variability at the referee site{unit}", color="#eb6834")
    ax2.spines[["top"]].set_visible(False)
    ax.set_ylabel(f"RMS error, model − {meta['referee_label']}{unit}")
    ax.set_title("a) Raw error, and how lively the water is (dotted)", loc="left", fontsize=11)

    ax = axs[1]
    line(ax, B, "nrmse", "All seasons", -0.1, "#2a78d6")
    if Bs is not None and Bs.n.sum():
        line(ax, Bs, "nrmse", "Jun–Nov only", 0.1, "#1baf7a", "--")
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("RMS error ÷ local natural variability")
    ax.set_title("b) Error relative to local variability: lower = better", loc="left", fontsize=11)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.12)  # headroom so the legend clears the error bars
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

    ax = axs[2]
    line(ax, B, "bias", "All seasons", -0.1, "#2a78d6")
    if Bs is not None and Bs.n.sum():
        line(ax, Bs, "bias", "Jun–Nov only", 0.1, "#1baf7a", "--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(f"Mean error, model − {meta['referee_label']}{unit}")
    ax.set_title("c) Bias: a steady offset gliders can correct", loc="left", fontsize=11)
    for a in axs:
        a.set_xticks(x, B.bin.values, rotation=30, ha="right")
        a.set_xlabel(meta["distance_label"])
        a.spines[["top"]].set_visible(False) if a is axs[0] else a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)
    for xi, n, nf in zip(x, B.n.values, B.n_platforms.values):
        axs[1].text(xi, 0.02, f"{n}\n{nf} pl", transform=axs[1].get_xaxis_transform(), ha="center", fontsize=7, color=INK2)
    fig.suptitle(f"{meta['model']} vs {meta['referee_label']}, {meta['region_name']}: {spec['label'].lower()}, "
                 f"by distance to recent glider data", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, f"{meta['period']}; {int(B.n.sum())} referee profiles, {meta['n_glider_missions']} glider missions. "
             f"Model value from {meta['lead_days']} day(s) before each profile; gliders within the {meta['window_days']} days before that.\n"
             "Bars: 90% interval from resampling whole platforms. Labels in b: profiles / platforms per bin.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cmems", "rtofs", "espc"], required=True)
    ap.add_argument("--region", choices=list(REGIONS), default="gulf")
    ap.add_argument("--referee", choices=["argo", "gliders"], default="argo")
    ap.add_argument("--workdir", type=Path, default=configs.path_data / "glider_coverage")
    ap.add_argument("--lead-days", type=int, default=1, help="days between the model field and the referee profile's day")
    ap.add_argument("--window-days", type=int, default=20, help="how far back glider profiles count")
    ap.add_argument("--since", help="only model times on/after this date (e.g. 2025-08-01, the RTOFS v2.5 start)")
    ap.add_argument("--until", help="only model times on/before this date")
    args = ap.parse_args()
    if args.lead_days + args.window_days > LOOKBACK_PAD_DAYS:
        ap.error(f"--lead-days + --window-days must be <= {LOOKBACK_PAD_DAYS} (the glider cache's look-back pad)")

    W = args.workdir / args.region / args.model
    F = load_fields(W)
    start, end = pd.Timestamp(F.time.values[0]), pd.Timestamp(F.time.values[-1]) + pd.Timedelta(days=1)
    cache = args.workdir / "obs" / args.region
    cache.mkdir(parents=True, exist_ok=True)
    gliders = fetch_gliders(args.region, start - pd.Timedelta(days=LOOKBACK_PAD_DAYS), end, cache)
    if args.referee == "argo":
        refs = fetch_argo(args.region, start, end, cache)
        referee_label, distance_label = "Argo", "Distance to the nearest glider profile in the preceding days"
    else:
        refs = fetch_glider_profiles(args.region, start, end, cache)
        referee_label, distance_label = "gliders", "Distance to the nearest profile from a different glider in the preceding days"
    print(f"{len(gliders)} glider profiles from {gliders.glider.nunique()} missions; "
          f"{len(refs)} usable {referee_label} referee profiles from {refs.argo.nunique()} platforms")

    M = matchups(args.model, refs, gliders, W, args.lead_days, args.window_days, exclude_same_platform=args.referee == "gliders")
    mt = pd.to_datetime(M.model_time)
    if args.since:
        M = M[mt >= pd.Timestamp(args.since)].reset_index(drop=True)
    if args.until:
        M = M[pd.to_datetime(M.model_time) <= pd.Timestamp(args.until)].reset_index(drop=True)
    tag = "".join(f"_{k}{v.replace('-', '')}" for k, v in (("from", args.since), ("to", args.until)) if v)
    out = W / f"validation_{args.referee}_lead{args.lead_days}d{tag}"
    out.mkdir(exist_ok=True)
    M.to_csv(out / "matchups.csv", index=False)
    meta = dict(model=F.attrs.get("model", args.model), region=args.region, region_name=REGIONS[args.region]["name"],
                referee=args.referee, referee_label=referee_label, distance_label=distance_label,
                period=f"{args.since or f'{start:%Y-%m-%d}'} to {args.until or f'{end - pd.Timedelta(days=1):%Y-%m-%d}'}",
                lead_days=args.lead_days, window_days=args.window_days, n_matchups=len(M),
                n_platforms=int(M.platform.nunique()), n_glider_missions=int(gliders.glider.nunique()))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)

    rng = np.random.default_rng(0)
    tables = []
    cols = ["bin", "n", "n_platforms", "median_sd", "rmse", "nrmse", "nrmse_p05", "nrmse_p95", "bias", "bias_p05", "bias_p95", "ncrmse"]
    for v in VARS:
        if not np.isfinite(M[f"err_{v}"]).any():
            continue
        B = binned(M, v, rng).assign(season="All")
        Bs = binned(M[M.season == "Jun-Nov"], v, rng).assign(season="Jun-Nov")
        tables += [B, Bs]
        figure(B, Bs, out / f"validation_{v}.png", meta, v)
        print(f"\n{v}: {VARS[v]['label']} ({int(B.n.sum())} matchups, all seasons)")
        print(B[cols].round(3).to_string(index=False))
    pd.concat(tables).to_csv(out / "binned.csv", index=False)
    print(f"\n{len(M)} matchups from {M.platform.nunique()} platforms -> {out}")


if __name__ == "__main__":
    main()
