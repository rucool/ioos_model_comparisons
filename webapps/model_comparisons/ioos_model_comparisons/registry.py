"""
registry.py — load and validate the YAML configuration in config/.

    from ioos_model_comparisons.registry import load_registry
    reg = load_registry()
    reg.region_defaults("gom")                   # what region_config(["gom"]) returns
    reg.map_panels("guam", "temperature_0m")      # portal panels for that slot
    reg.rtofs_version_at("parallel", when)        # version tag for catalog entries

Files (ioos_model_comparisons/config/, or $IMC_CONFIG_DIR):
    site.yml          plot roots, published URL base, catalog settings
    models.yml        models/products and their file-name tokens
    comparisons.yml   comparisons, portal panels and their region/var rules
    rtofs.yml         RTOFS streams, URLs, harvester settings, version history
    portal.yml        portal lists (map regions, ASG, profile region boxes)
    regions/<key>.yml one region's defaults (regions.region_config output)

Everything is validated at load, including cross-references (a panel naming a
comparison that doesn't exist, a rule naming an unknown region folder, a typo'd
key) — load_registry() raises RegistryError listing every problem, so a bad
edit fails tests/CI instead of quietly misbehaving in production.

Vendored into the web app; needs only PyYAML and pydantic.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CONFIG_DIR = Path(__file__).resolve().parent / "config"


class RegistryError(ValueError):
    pass


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogCfg(_Strict):
    database: str = "hurricanes"
    collection: str = "figures"
    stale_after_hours: float = 12


class SiteCfg(_Strict):
    plots_root: dict[str, str]
    publish_root: str
    public_base_url: str
    catalog: CatalogCfg = CatalogCfg()


class ModelSpec(_Strict):
    label: str
    filename_token: str
    kind: Literal["model", "satellite", "product"] = "model"
    rtofs_stream: Optional[str] = None
    loader: Optional[str] = None


class Rule(_Strict):
    include: Optional[list[str]] = None
    exclude: Optional[list[str]] = None

    def allows(self, value) -> bool:
        if self.include is not None and value not in self.include:
            return False
        if self.exclude is not None and value in self.exclude:
            return False
        return True


class ComparisonSpec(_Strict):
    models: list[str] = Field(min_length=2)
    regions: Optional[Rule] = None
    var_depths: Optional[Rule] = None


class ExcludeCombo(_Strict):
    region: Optional[str] = None
    var_depth: Optional[str] = None

    def matches(self, region, var_depth) -> bool:
        return ((self.region is None or self.region == region) and
                (self.var_depth is None or self.var_depth == var_depth))


class PanelSpec(_Strict):
    key: str
    label: str
    short: str
    comparisons: list[str] = Field(min_length=1)
    regions: Optional[Rule] = None
    var_depths: Optional[Rule] = None
    exclude: list[ExcludeCombo] = []


class ComparisonsCfg(_Strict):
    comparisons: dict[str, ComparisonSpec]
    map_panels: list[PanelSpec]
    overview_order: list[str]


class RtofsStream(_Strict):
    model_attr: str
    thredds: dict[str, str]
    aws_url: str
    nomads_url: Optional[str] = None
    data_dirs: dict[str, str] = {}


class RtofsHistory(_Strict):
    stream: str
    version: Optional[str] = None
    from_: Optional[_dt.date] = Field(default=None, alias="from")


class RtofsCfg(_Strict):
    streams: dict[str, RtofsStream]
    history: list[RtofsHistory] = []
    files: list[str] = []
    rename: dict[str, str] = {}


class BoxSpec(_Strict):
    label: Optional[str] = None
    extent: Optional[list[float]] = None
    extent_from: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self):
        if (self.extent is None) == (self.extent_from is None):
            raise ValueError("give exactly one of extent / extent_from")
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError("extent must be [lonmin, lonmax, latmin, latmax]")
        return self


class AsgCfg(_Strict):
    regions: dict[str, list[str]]
    variables: dict[str, list[str]]
    models: dict[str, str]


class PortalCfg(_Strict):
    map_regions: list[str]
    adaptive_sampling: AsgCfg
    argo_regions: dict[str, BoxSpec]
    glider_regions: dict[str, BoxSpec]
    fvon_regions: list[str]
    fvon_folder_map: dict[str, str] = {}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class Registry:
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        problems: list[str] = []

        def _load(name, schema):
            path = self.config_dir / name
            try:
                raw = yaml.safe_load(path.read_text()) or {}
                return schema.model_validate(raw), raw
            except FileNotFoundError:
                problems.append(f"{name}: missing")
            except yaml.YAMLError as e:
                problems.append(f"{name}: YAML syntax error: {e}")
            except ValidationError as e:
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    problems.append(f"{name}: {loc}: {err['msg']}")
            return None, None

        self.site, _ = _load("site.yml", SiteCfg)
        models, _ = _load("models.yml", _ModelsFile)
        self.models: dict[str, ModelSpec] = models.models if models else {}
        cmp_cfg, _ = _load("comparisons.yml", ComparisonsCfg)
        self.comparisons: dict[str, ComparisonSpec] = cmp_cfg.comparisons if cmp_cfg else {}
        self.panels: list[PanelSpec] = cmp_cfg.map_panels if cmp_cfg else []
        self.overview_order: list[str] = cmp_cfg.overview_order if cmp_cfg else []
        self.rtofs, _ = _load("rtofs.yml", RtofsCfg)
        self.portal, _ = _load("portal.yml", PortalCfg)
        self._regions = self._load_regions(problems)

        if not problems:
            self._cross_check(problems)
        if problems:
            raise RegistryError("invalid configuration in %s:\n  - %s"
                                % (self.config_dir, "\n  - ".join(problems)))

        try:  # keep figure_names' OHC model splitting in sync with models.yml
            from ioos_model_comparisons import figure_names
            figure_names.register_model_tokens(m.filename_token for m in self.models.values())
        except Exception:
            pass

    # -- loading ----------------------------------------------------------
    def _load_regions(self, problems) -> dict:
        from ioos_model_comparisons.regions_store import validate

        regions = {}
        rdir = self.config_dir / "regions"
        files = sorted(rdir.glob("*.yml")) if rdir.is_dir() else []
        if not files:
            problems.append("regions/: no region files")
        for path in files:
            key = path.stem
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError as e:
                problems.append(f"regions/{path.name}: YAML syntax error: {e}")
                continue
            if not isinstance(doc, dict):
                problems.append(f"regions/{path.name}: must be a mapping")
                continue
            order = doc.pop("order", None)
            if not isinstance(order, (int, float)):
                problems.append(f"regions/{path.name}: order: a number is required")
                order = 1e9
            for k in ("name", "folder", "extent"):
                if k not in doc:
                    problems.append(f"regions/{path.name}: {k}: required")
            errors, _warnings = validate({"region": key, **doc})
            problems.extend(f"regions/{path.name}: {e}" for e in errors)
            regions[key] = (order, doc)
        ordered = sorted(regions.items(), key=lambda kv: (kv[1][0], kv[0]))
        return {k: doc for k, (_o, doc) in ordered}

    def _cross_check(self, problems):
        folders = {d["folder"] for d in self._regions.values()}
        tokens = {}
        for mid, m in self.models.items():
            if m.filename_token.lower() in tokens:
                problems.append(f"models.yml: {mid}: filename_token {m.filename_token!r} "
                                f"already used by {tokens[m.filename_token.lower()]}")
            tokens[m.filename_token.lower()] = mid
            if m.rtofs_stream and m.rtofs_stream not in self.rtofs.streams:
                problems.append(f"models.yml: {mid}: unknown rtofs_stream {m.rtofs_stream!r}")

        def _check_rule(where, rule):
            if rule is None:
                return
            for f in (rule.include or []) + (rule.exclude or []):
                if f not in folders:
                    problems.append(f"{where}: unknown region folder {f!r}")

        for cid, c in self.comparisons.items():
            for m in c.models:
                if m not in self.models:
                    problems.append(f"comparisons.yml: {cid}: unknown model {m!r}")
            _check_rule(f"comparisons.yml: {cid}.regions", c.regions)

        keys = [p.key for p in self.panels]
        if len(set(keys)) != len(keys):
            problems.append("comparisons.yml: map_panels: duplicate key")
        for p in self.panels:
            for c in p.comparisons:
                if c not in self.comparisons:
                    problems.append(f"comparisons.yml: panel {p.key}: unknown comparison {c!r}")
            _check_rule(f"comparisons.yml: panel {p.key}.regions", p.regions)
            for ex in p.exclude:
                if ex.region and ex.region not in folders:
                    problems.append(f"comparisons.yml: panel {p.key}.exclude: unknown region folder {ex.region!r}")
        if sorted(self.overview_order) != sorted(keys):
            problems.append("comparisons.yml: overview_order must list every map_panels key exactly once")

        for h in self.rtofs.history:
            if h.stream not in self.rtofs.streams:
                problems.append(f"rtofs.yml: history: unknown stream {h.stream!r}")

        for k in self.portal.map_regions:
            if k not in self._regions:
                problems.append(f"portal.yml: map_regions: unknown region {k!r}")
        for group in ("argo_regions", "glider_regions"):
            for k, box in getattr(self.portal, group).items():
                if box.extent_from and box.extent_from not in self._regions:
                    problems.append(f"portal.yml: {group}.{k}: unknown region {box.extent_from!r}")

    # -- regions ----------------------------------------------------------
    def region_keys(self) -> list[str]:
        """Region keys in legacy regions.py order."""
        return list(self._regions)

    def region_defaults(self, key) -> dict:
        """A deep copy of the region's YAML defaults (no MongoDB overrides)."""
        return copy.deepcopy(self._regions[key])

    def folder_to_key(self, folder):
        for k, d in self._regions.items():
            if d["folder"] == folder:
                return k
        return None

    def name_to_key(self, name):
        for k, d in self._regions.items():
            if d["name"] == name:
                return k
        return None

    # -- comparisons and panels ---------------------------------------------
    def comparison_applies(self, comp_id, region_folder, var_depth=None) -> bool:
        c = self.comparisons.get(comp_id)
        if c is None:
            return False
        if c.regions and not c.regions.allows(region_folder):
            return False
        if var_depth is not None and c.var_depths and not c.var_depths.allows(var_depth):
            return False
        return True

    def panel_applies(self, panel: PanelSpec, region_folder, var_depth) -> bool:
        if panel.regions and not panel.regions.allows(region_folder):
            return False
        if panel.var_depths and not panel.var_depths.allows(var_depth):
            return False
        if any(ex.matches(region_folder, var_depth) for ex in panel.exclude):
            return False
        return any(self.comparison_applies(c, region_folder, var_depth) for c in panel.comparisons)

    def map_panels(self, region_folder, var_depth) -> list[PanelSpec]:
        """Panels shown for one region/var_depth, in Maps display order."""
        return [p for p in self.panels if self.panel_applies(p, region_folder, var_depth)]

    def panel_defs(self) -> list[dict]:
        return [{"key": p.key, "label": p.label, "short": p.short} for p in self.panels]

    def comparisons_for_model_ids(self, model_ids) -> Optional[str]:
        """Comparison id whose model list equals model_ids, if any."""
        model_ids = list(model_ids)
        for cid, c in self.comparisons.items():
            if c.models == model_ids:
                return cid
        return None

    def predict_rel(self, comp_id, region_folder, region_name, var_depth, valid_time):
        """Relative path the plotting scripts write for this comparison/slot,
        built from models.yml filename tokens and figure_names. Used by the
        portal when it isn't reading the catalog (legacy discovery)."""
        import re
        from ioos_model_comparisons import figure_names as fn

        comp = self.comparisons[comp_id]
        tokens = [self.models[m].filename_token for m in comp.models]
        if var_depth == "ocean_heat_content":
            combined = len(tokens) > 2
            meta = fn.FigureMeta(
                family="ohc", product="ohc_combined" if combined else "ohc",
                region=region_folder, variable="ocean_heat_content", depth=None,
                valid_time=valid_time, models=tuple(t.lower() for t in tokens),
                extra={"slug": region_name if combined else region_folder})
            return fn.build(meta)
        m = re.fullmatch(r"(?P<var>[a-z_]+)_(?P<d>\d+)m", var_depth or "")
        if not m:
            return None
        var, depth = m["var"], int(m["d"])
        product = "streamplot" if var == "currents" else (
            "sst" if any(self.models[x].kind == "satellite" for x in comp.models) else "model_comparison")
        meta = fn.FigureMeta(
            family="map", product=product, region=region_folder, variable=var, depth=depth,
            valid_time=valid_time, models=tuple(t.lower() for t in tokens),
            extra={"slug": fn.region_slug(region_folder, var), "tokens": tokens})
        return fn.build(meta)

    def model_for_token(self, token):
        for mid, m in self.models.items():
            if m.filename_token.lower() == str(token).lower():
                return mid
        return None

    # -- RTOFS ------------------------------------------------------------
    def rtofs_stream(self, name) -> RtofsStream:
        return self.rtofs.streams[name]

    def rtofs_version_at(self, stream, when=None) -> Optional[str]:
        """Version tag for `stream` at `when` (date/datetime; default today)."""
        when = when or _dt.date.today()
        if isinstance(when, _dt.datetime):
            when = when.date()
        best = None
        for h in self.rtofs.history:
            if h.stream != stream:
                continue
            if h.from_ is None or h.from_ <= when:
                if best is None or (h.from_ or _dt.date.min) >= (best.from_ or _dt.date.min):
                    best = h
        return best.version if best else None

    def rtofs_tag_for_models(self, models, when=None) -> Optional[dict]:
        """{stream: version} for every RTOFS model in a figure's model list."""
        out = {}
        for tok in models:
            mid = self.model_for_token(tok) or tok
            spec = self.models.get(mid)
            if spec and spec.rtofs_stream:
                out[spec.rtofs_stream] = self.rtofs_version_at(spec.rtofs_stream, when)
        return out or None

    # -- portal -----------------------------------------------------------
    def _boxes(self, group) -> list[dict]:
        out = []
        for k, box in getattr(self.portal, group).items():
            src = self._regions.get(box.extent_from) if box.extent_from else None
            extent = list(src["extent"]) if src else list(box.extent)
            label = box.label or (src["name"] if src else k)
            out.append({"key": k, "label": label, "extent": extent})
        return out

    def argo_boxes(self) -> list[dict]:
        return self._boxes("argo_regions")

    def glider_boxes(self) -> list[dict]:
        return self._boxes("glider_regions")

    # -- site -------------------------------------------------------------
    def plots_root(self) -> Path:
        plat = "darwin" if sys.platform == "darwin" else "linux"
        return Path(os.path.expanduser(self.site.plots_root[plat]))

    def publish_root(self) -> Path:
        return Path(os.path.expanduser(os.getenv("IMC_PLOTS_ROOT") or self.site.publish_root))

    def public_url(self, rel_path) -> str:
        from urllib.parse import quote
        return self.site.public_base_url.rstrip("/") + "/" + quote(str(rel_path).lstrip("/"))

    def config_hash(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.config_dir.rglob("*.yml")):
            h.update(p.relative_to(self.config_dir).as_posix().encode())
            h.update(p.read_bytes())
        return h.hexdigest()[:12]


class _ModelsFile(_Strict):
    models: dict[str, ModelSpec]


@lru_cache(maxsize=4)
def _load_cached(config_dir: str) -> Registry:
    return Registry(Path(config_dir))


def load_registry(config_dir=None) -> Registry:
    """The validated configuration (cached per config directory)."""
    return _load_cached(str(config_dir or os.getenv("IMC_CONFIG_DIR") or CONFIG_DIR))


def reload_registry():
    """Drop the cache (tests, or after editing YAML in a long-lived process)."""
    _load_cached.cache_clear()
