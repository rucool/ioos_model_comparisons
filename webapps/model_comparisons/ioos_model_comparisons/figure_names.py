"""
figure_names.py — parse published plot paths into structured metadata.

The plot archive's file names *are* its metadata (region folder, variable,
depth, valid time, compared models). This module is the single place that
knows those naming conventions, so the figure catalog (catalog.py), the
backfill indexer (scripts/tools/index_figures.py) and the portal all agree.

    parse("maps/gulf_of_mexico/temperature_0m/2026/10/"
          "gulf-of-mexico_2026-10-12T000000Z_temperature-0m_rtofs-vs-espc.png")
    -> FigureMeta(family="map", product="model_comparison",
                  region="gulf_of_mexico", variable="temperature", depth=0,
                  models=("rtofs", "espc"), ...)

Paths are relative to the published plots root (configs.path_plots on the
server, i.e. what Apache serves under /hurricane/model_comparisons/).
`build(meta)` is the inverse and is round-trip tested against real paths in
tests/fixtures/sample_paths.txt — if you change how plotting.py names files,
add the new pattern here and a sample path there.

Pure standard library on purpose: it is vendored into the web app.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from urllib.parse import unquote

TS_FMT = "%Y-%m-%dT%H%M%SZ"

# Filename tokens of known models, longest first so "rtofs-p" wins over
# "rtofs" when splitting hyphen-joined OHC names like "rtofs-p-espc".
# registry.py can extend this from models.yml (see register_model_tokens).
_MODEL_TOKENS = [
    "rtofs-p", "rtofs", "espc", "cmems", "copernicus", "eccofs", "nesdis",
    "amseas", "cnaps", "doppio", "gofs", "lusitania",
]


def register_model_tokens(tokens):
    """Add filename tokens (e.g. from models.yml) so OHC names split correctly."""
    for t in tokens:
        t = str(t).lower()
        if t and t not in _MODEL_TOKENS:
            _MODEL_TOKENS.append(t)
    _MODEL_TOKENS.sort(key=len, reverse=True)


_MODEL_TOKENS.sort(key=len, reverse=True)


@dataclass
class FigureMeta:
    family: str                      # map | sst | ohc | profile | adaptive
    product: str                     # model_comparison | single | streamplot | sst | ohc | ohc_single
                                     # | ohc_combined | glider_profile | argo_profile | fvon_profile | asg
    region: str | None               # folder name on disk (filesystem truth)
    variable: str | None
    depth: int | None
    valid_time: _dt.datetime
    models: tuple = ()
    platform_type: str | None = None
    platform_id: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def comparison(self) -> str | None:
        return "-vs-".join(self.models) if self.models else None

    @property
    def var_depth(self) -> str | None:
        """The portal's `variable_depth` key: temperature_0m, currents_150m,
        ocean_heat_content, ..."""
        if self.family == "ohc":
            return "ocean_heat_content"
        if self.variable and self.depth is not None and self.family in ("map", "sst"):
            return f"{self.variable}_{self.depth}m"
        return None


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------
_TS = r"\d{4}-\d{2}-\d{2}T\d{6}Z"
_YM = r"(?P<y>\d{4})/(?P<mo>\d{2})/"
_TOK = r"[A-Za-z0-9-]+"

_RE_MAP = re.compile(
    rf"^maps/(?P<folder>[^/]+)/(?P<var>temperature|salinity)_(?P<d>\d+)m/{_YM}"
    rf"(?P<slug>[^/]+)_(?P<ts>{_TS})_(?P=var)-(?P=d)m_(?P<m1>{_TOK}?)-vs-(?P<m2>{_TOK})\.png$")
_RE_MAP_SINGLE = re.compile(
    rf"^maps/(?P<folder>[^/]+)/(?P<var>temperature|salinity)_(?P<d>\d+)m/{_YM}"
    rf"(?P<slug>[^/]+)_(?P<ts>{_TS})_(?P=var)-(?P=d)m_(?P<m1>[A-Za-z0-9]+(?:-p)?)\.png$")
_RE_CURRENTS = re.compile(
    rf"^maps/(?P<folder>[^/]+)/currents_(?P<d>\d+)m/(?:{_YM})?"
    rf"(?P<slug>[^/]+)_(?P<ts>{_TS})_currents-(?P=d)m_(?P<m1>{_TOK}?)-vs-(?P<m2>{_TOK})\.png$")
_RE_OHC = re.compile(
    rf"^maps/(?P<folder>[^/]+)/ocean_heat_content/{_YM}"
    rf"(?P<slug>[^/]+)_(?P<ts>{_TS})_heat_content_(?P<models>[a-z0-9-]+)\.png$")
_RE_OHC_COMBINED = re.compile(
    rf"^maps/(?P<folder>[^/]+)/ocean_heat_content/{_YM}"
    rf"(?P<slug>[^/]+)_(?P<ts>{_TS})_ohc_(?P<models>[a-z0-9-]+)\.png$")
_RE_GLIDER = re.compile(
    r"^profiles/(?P<group>gliders|gulf_gliders)/(?P<y>\d{4})/(?P<md>\d{2}-\d{2})/"
    r"(?P<id>[^/]+?)_(?P<d0>\d{8})_to_(?P<d1>\d{8})_(?P<depth>\d+)m\.png$")
_RE_PROFILE = re.compile(
    r"^profiles/(?P<ptype>argo|fvon)/(?P<region>[^/]+)/(?P<y>\d{4})/(?P<mo>\d{2})/(?P<dd>\d{2})/"
    r"(?P<id>[^/]+?)-profile-(?P<ts>\d{4}-\d{2}-\d{2}T\d{4}(?:\d{2})?Z)\.png$")
_RE_ASG = re.compile(
    rf"^adaptive_sampling_guidance/maps/(?P<region>[^/]+)/(?P<varf>[^/]+)/(?P<panel>[^/]+)/"
    rf"(?P<model>[^/]+)/{_YM}(?P<name>[^/]+)_(?P<ts>{_TS})_(?P<code>[a-z]+)_rtofs-(?P<cmp>[a-z]+)-diff\.png$")

_SAT = re.compile(r"^GOES(\d{2})$", re.IGNORECASE)


def _ts(s: str) -> _dt.datetime:
    return _dt.datetime.strptime(s, TS_FMT)


def split_models(joined: str) -> tuple | None:
    """Split a hyphen-joined model list ('rtofs-p-espc') using known tokens."""
    out, rest = [], joined.lower()
    while rest:
        for tok in _MODEL_TOKENS:
            if rest == tok or rest.startswith(tok + "-"):
                out.append(tok)
                rest = rest[len(tok) + 1:]
                break
        else:
            return None
    return tuple(out)


def _norm_model(tok: str) -> str:
    return tok.lower()


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------
def parse(rel_path) -> FigureMeta | None:
    """Return FigureMeta for a published plot path, or None if unrecognized.

    Never raises. Paths through last_14_days/ (symlink mirrors) return None.
    """
    try:
        return _parse(unquote(str(rel_path)).replace("\\", "/").lstrip("/"))
    except Exception:
        return None


def _parse(p: str) -> FigureMeta | None:
    if "/last_14_days/" in p or not p.endswith(".png"):
        return None

    if p.startswith("maps/"):
        m = _RE_MAP.match(p)
        if m:
            m1, m2 = m["m1"], m["m2"]
            sat = _SAT.match(m2)
            fam, prod = ("sst", "sst") if sat else ("map", "model_comparison")
            return FigureMeta(
                family=fam, product=prod, region=m["folder"], variable=m["var"],
                depth=int(m["d"]), valid_time=_ts(m["ts"]),
                models=(_norm_model(m1), _norm_model(m2)),
                extra={"slug": m["slug"], "tokens": [m1, m2]})
        m = _RE_CURRENTS.match(p)
        if m:
            return FigureMeta(
                family="map", product="streamplot", region=m["folder"], variable="currents",
                depth=int(m["d"]), valid_time=_ts(m["ts"]),
                models=(_norm_model(m["m1"]), _norm_model(m["m2"])),
                extra={"slug": m["slug"], "tokens": [m["m1"], m["m2"]], "dated_dirs": bool(m["y"])})
        m = _RE_MAP_SINGLE.match(p)
        if m:
            return FigureMeta(
                family="map", product="single", region=m["folder"], variable=m["var"],
                depth=int(m["d"]), valid_time=_ts(m["ts"]), models=(_norm_model(m["m1"]),),
                extra={"slug": m["slug"], "tokens": [m["m1"]]})
        for rx, prod in ((_RE_OHC, "ohc"), (_RE_OHC_COMBINED, "ohc_combined")):
            m = rx.match(p)
            if m:
                models = split_models(m["models"])
                if not models:
                    return None
                if prod == "ohc" and len(models) == 1:
                    prod = "ohc_single"
                return FigureMeta(
                    family="ohc", product=prod, region=m["folder"], variable="ocean_heat_content",
                    depth=None, valid_time=_ts(m["ts"]), models=models,
                    extra={"slug": m["slug"]})
        return None

    if p.startswith("profiles/"):
        m = _RE_GLIDER.match(p)
        if m:
            d0 = _dt.datetime.strptime(m["d0"], "%Y%m%d")
            return FigureMeta(
                family="profile", product="glider_profile", region=None, variable=None,
                depth=int(m["depth"]), valid_time=d0, platform_type="glider",
                platform_id=m["id"],
                extra={"group": m["group"], "dir_year": m["y"], "d1": m["d1"]})
        m = _RE_PROFILE.match(p)
        if m:
            ts = m["ts"]
            fmt = TS_FMT if len(ts) == len("2026-01-01T000000Z") else "%Y-%m-%dT%H%MZ"
            return FigureMeta(
                family="profile", product=f"{m['ptype']}_profile", region=m["region"],
                variable=None, depth=None, valid_time=_dt.datetime.strptime(ts, fmt),
                platform_type=m["ptype"], platform_id=m["id"], extra={"ts_fmt": fmt})
        return None

    if p.startswith("adaptive_sampling_guidance/"):
        m = _RE_ASG.match(p)
        if m:
            return FigureMeta(
                family="adaptive", product="asg", region=m["region"], variable=m["varf"],
                depth=None, valid_time=_ts(m["ts"]), models=("rtofs", m["cmp"]),
                extra={"panel": m["panel"], "model_folder": m["model"], "name": m["name"],
                       "code": m["code"]})
        return None

    return None


# --------------------------------------------------------------------------
# build (inverse of parse)
# --------------------------------------------------------------------------
def region_slug(folder: str, variable: str) -> str:
    """File-name prefix plotting.py uses: hyphenated folder, except currents."""
    return folder if variable == "currents" else folder.replace("_", "-")


def build(meta: FigureMeta) -> str:
    """Relative path for meta. build(parse(p)) == p for every supported p."""
    t = meta.valid_time
    ym = t.strftime("%Y/%m/")
    ts = t.strftime(TS_FMT)
    x = meta.extra or {}
    if meta.product in ("model_comparison", "sst", "single"):
        slug = x.get("slug") or region_slug(meta.region, meta.variable)
        toks = x.get("tokens") or list(meta.models)
        tail = "-vs-".join(toks)
        return (f"maps/{meta.region}/{meta.variable}_{meta.depth}m/{ym}"
                f"{slug}_{ts}_{meta.variable}-{meta.depth}m_{tail}.png")
    if meta.product == "streamplot":
        slug = x.get("slug") or region_slug(meta.region, "currents")
        toks = x.get("tokens") or list(meta.models)
        dirs = ym if x.get("dated_dirs", True) else ""
        return (f"maps/{meta.region}/currents_{meta.depth}m/{dirs}"
                f"{slug}_{ts}_currents-{meta.depth}m_{'-vs-'.join(toks)}.png")
    if meta.product in ("ohc", "ohc_single"):
        slug = x.get("slug") or meta.region
        return (f"maps/{meta.region}/ocean_heat_content/{ym}"
                f"{slug}_{ts}_heat_content_{'-'.join(meta.models)}.png")
    if meta.product == "ohc_combined":
        return (f"maps/{meta.region}/ocean_heat_content/{ym}"
                f"{x['slug']}_{ts}_ohc_{'-'.join(meta.models)}.png")
    if meta.product == "glider_profile":
        d1 = x.get("d1") or (t + _dt.timedelta(days=1)).strftime("%Y%m%d")
        year = x.get("dir_year") or t.strftime("%Y")
        return (f"profiles/{x.get('group', 'gliders')}/{year}/{t:%m-%d}/"
                f"{meta.platform_id}_{t:%Y%m%d}_to_{d1}_{meta.depth}m.png")
    if meta.product in ("argo_profile", "fvon_profile"):
        fmt = x.get("ts_fmt", TS_FMT)
        return (f"profiles/{meta.platform_type}/{meta.region}/{t:%Y/%m/%d}/"
                f"{meta.platform_id}-profile-{t.strftime(fmt)}.png")
    if meta.product == "asg":
        return (f"adaptive_sampling_guidance/maps/{meta.region}/{meta.variable}/{x['panel']}/"
                f"{x['model_folder']}/{ym}{x['name']}_{ts}_{x['code']}_rtofs-{meta.models[1]}-diff.png")
    raise ValueError(f"cannot build a path for product {meta.product!r}")


def to_doc(meta: FigureMeta) -> dict:
    """Catalog-document fields derived from the path (see catalog.py)."""
    doc = {
        "family": meta.family,
        "product": meta.product,
        "region": meta.region,
        "variable": meta.variable,
        "depth": meta.depth,
        "var_depth": meta.var_depth,
        "valid_time": meta.valid_time,
        "models": list(meta.models),
        "comparison": meta.comparison,
    }
    if meta.platform_type:
        doc["platform"] = {"type": meta.platform_type, "id": meta.platform_id}
    if meta.extra.get("group"):
        doc["group"] = meta.extra["group"]
    return doc
