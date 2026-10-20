"""
Generate a PowerPoint presentation of the most recent figures shown in the
IOOS Model Comparisons Archive Explorer 'Overview' tab.

One slide per region/variable/depth combination, showing every RTOFS-involving
comparison available (RTOFS vs CMEMS, RTOFS vs ESPC, RTOFS vs GOES for SST).
ESPC-vs-CMEMS is intentionally never shown — only comparisons against RTOFS
are of interest here. "Latest" is anchored on the newest cycle that actually
has an RTOFS image (require="rtofs" in get_latest_map_info /
get_available_hour_for_date), not just the newest file of any kind — some
regions/cycles only publish ESPC-vs-CMEMS (e.g. Guam's RTOFS comparisons
currently only run at the 00Z cycle, not all four), which would otherwise
look like "the latest" despite having no RTOFS data at all.
Ocean Heat Content additionally falls back to the older single combined
RTOFS/ESPC/CMEMS panel format when the standard two-file format isn't
published for that cycle (see build_ohc_legacy_url) — Fiji still only ever
produces that legacy format; Guam produces both during its migration.

Usage:
    python overview_powerpoint.py
    python overview_powerpoint.py --output my_overview.pptx
    python overview_powerpoint.py --regions Caribbean "Gulf of Mexico"

    # Storm-focused briefing deck for a single region. --nhc-search scrapes
    # nhc.noaa.gov itself to find the matching active storm and pull its
    # advisory graphics (forecast cone, wind history, key messages, current
    # wind field) at full resolution — no URLs to find or paste:
    python overview_powerpoint.py \\
        --regions "Hawaii" \\
        --storm "Hurricane Fausto — Approaching Hawaii" \\
        --nhc-search "Fausto" \\
        --output fausto_hawaii.pptx

    # See what NHC currently considers active (name + storm ID):
    python overview_powerpoint.py --list-storms

    # Multiple storms in one deck: repeat --storm-spec "NAME:Region1,Region2"
    # — each gets its own NHC advisory section, region maps, Argo profiles,
    # and OHC trend, all in a single output file:
    python overview_powerpoint.py \\
        --storm-spec "Fausto:Hawaii" \\
        --storm-spec "Seven-E:Eastern Pacific - Mexico" \\
        --output active_pacific_storms.pptx

    # --nhc-images still works for pasting specific graphic URLs by hand
    # instead of auto-discovery (e.g. a storm NHC has already archived).

Dependencies (install if missing):
    pip install python-pptx requests

    # Optional, only needed for --argo-track-filter (on by default whenever a
    # storm is matched): tropycal (storm track) and this repo's own
    # ioos_model_comparisons package (live Argo positions via ERDDAP).
"""
import argparse
import datetime
import io
import math
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
except ImportError:
    sys.exit("python-pptx is required:  pip install python-pptx")

# ── Metadata ───────────────────────────────────────────────────────────────

# Default region order for the presentation (products are discovered dynamically).
OVERVIEW_REGIONS = [
    "Caribbean",
    "Gulf of Mexico",
    "South Atlantic Bight",
    "Mid Atlantic Bight",
    "West Florida Shelf",
    "Tropical Western Atlantic",
    "Eastern Pacific - Mexico",
    "Hawaii",
    "Guam",
    "Fiji",
]

# Short codes matching ioos_model_comparisons/regions.py's internal
# region_config() keys, mapped to the canonical display name this script
# (and the server's folder-naming convention) expects for --regions.
REGION_ALIASES = {
    "gom":                        "Gulf of Mexico",
    "sab":                        "South Atlantic Bight",
    "mab":                        "Mid Atlantic Bight",
    "caribbean":                  "Caribbean",
    "west_florida_shelf":         "West Florida Shelf",
    "tropical_western_atlantic":  "Tropical Western Atlantic",
    "mexico_pacific":             "Eastern Pacific - Mexico",
    "hawaii":                     "Hawaii",
    "guam":                       "Guam",
    "fiji":                       "Fiji",
}

# Reverse of REGION_ALIASES: canonical display name -> regions.py short code.
# Used to look up a region's geographic extent (for the Argo track-distance
# filter) via ioos_model_comparisons.regions.region_config().
REGION_CODE_BY_NAME = {name: code for code, name in REGION_ALIASES.items()}


def resolve_region(name: str) -> str:
    """
    Resolve a --regions value to the canonical display name this script
    expects. Accepts the display name itself (case-insensitive, e.g. 'guam',
    'GUAM', 'Guam') or a regions.py-style short code (e.g. 'mab' ->
    'Mid Atlantic Bight'). Unrecognized input passes through unchanged so
    discover_products()'s own "not found" reporting can surface it.
    """
    stripped = name.strip()
    for canonical in OVERVIEW_REGIONS:
        if stripped.lower() == canonical.lower():
            return canonical
    return REGION_ALIASES.get(stripped.lower(), name)

# Variable display order; depths within each variable are sorted numerically.
VARIABLE_ORDER = ["temperature", "salinity", "ocean_heat_content", "currents"]

# No per-region "ESPC vs CMEMS only" allowlist is kept anymore:
# build_region_slides()/add_ohc_trend_slides() just try every comparison type
# (including, for OHC, the older combined-panel format via
# build_ohc_legacy_url) and skip whichever ones don't exist for that
# region/cycle — more robust than a hand-maintained list, since coverage
# varies by region and can even vary by 6-hour cycle within the same region
# (e.g. Guam's RTOFS comparisons currently only publish at 00Z, and its OHC
# pipeline is mid-migration, producing both the old and new formats).

REGION_FOLDER_MAP = {
    "Eastern Pacific - Mexico": "mexico_pacific",
}

BASE_URL      = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/maps"
BASE_PROFILES = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles"

# Known nhc.noaa.gov storm_graphics filename stems -> a readable slide label.
# Anything not in this table falls back to a title-cased version of the stem.
NHC_LABELS = {
    "5day_cone": "Forecast Cone (5-Day)",
    "3day_cone": "Forecast Cone (3-Day)",
    "5day_expCone": "Forecast Cone — Experimental (5-Day)",
    "wind_history": "Wind Speed History / Swath",
    "current_wind": "Current Wind Field",
    "key_messages": "Key Messages",
    "spanish_key_messages": "Key Messages (Spanish)",
    "peak_surge": "Peak Storm Surge Risk",
}

NHC_HOME = "https://www.nhc.noaa.gov/"

# query-param keys accepted by NHC's /refresh/graphics_{slot}+shtml/{advisory_id}.shtml
# detail sub-pages; these are the ones auto-discovery fetches by default.
DEFAULT_NHC_GRAPHIC_KEYS = ["cone", "swath", "key_messages", "radii"]

# Slide colours
NAVY  = RGBColor(0x0d, 0x2b, 0x5e)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY  = RGBColor(0x55, 0x55, 0x55)


# ── Server helpers ─────────────────────────────────────────────────────────

def _region_url_key(region: str) -> str:
    return REGION_FOLDER_MAP.get(region, region.replace(" ", "_").lower())


def _product_sort_key(product: str) -> tuple[int, int]:
    """Sort by VARIABLE_ORDER index, then by depth numerically ascending."""
    for i, var in enumerate(VARIABLE_ORDER):
        if product == var or product.startswith(var + "_"):
            m = re.search(r'(\d+)m$', product)
            return (i, int(m.group(1)) if m else 0)
    return (len(VARIABLE_ORDER), 0)


def _product_label(product: str) -> str:
    """Human-readable label: 'temperature_200m' → 'Temperature 200m'."""
    parts = product.split("_")
    if re.match(r'^\d+m$', parts[-1]):
        # Keep depth token lowercase (e.g. '200m', not '200M')
        return " ".join(p.title() for p in parts[:-1]) + " " + parts[-1]
    return " ".join(p.title() for p in parts)


def discover_products(region: str) -> list[str]:
    """
    Scrape the region's maps directory on the server and return every
    variable_depth folder that exists, sorted by variable then depth.

    This means the presentation always reflects what is actually published —
    no hardcoded depth lists required.
    """
    key  = _region_url_key(region)
    base = f"{BASE_URL}/{key}/"
    try:
        r = requests.get(base, timeout=10)
        if r.status_code != 200:
            return []
        folders = re.findall(r'href="([a-z][a-z_0-9]*)/"', r.text)
        products = [
            f for f in folders
            if any(f == v or f.startswith(v + "_") for v in VARIABLE_ORDER)
        ]
        return sorted(set(products), key=_product_sort_key)
    except Exception as exc:
        print(f"  Warning: could not discover products for {region}: {exc}")
        return []


def _dated_pngs(html: str, require: str | None = None) -> list[tuple[str, str]]:
    """Extract (date_str, time_code) pairs from every '...YYYY-MM-DDTHHMMSSZ...png'
    href in a directory listing, optionally keeping only filenames containing
    the `require` substring (e.g. 'rtofs' to ignore ESPC-vs-CMEMS-only cycles)."""
    out = []
    for fname in re.findall(r'href="([^"]+\.png)"', html):
        if require and require not in fname:
            continue
        m = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{6})Z', fname)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def get_latest_map_info(region: str, variable_depth: str,
                        date_filter: str | None = None,
                        require: str | None = None):
    """
    Scrape the server directory listing to find the most recent date and
    6-hourly time step for the given region / variable-depth product.

    If date_filter is "YYYY-MM-DD", returns the latest time available for
    that specific date rather than the overall most recent date.

    If `require` is given, only filenames containing that substring count
    (e.g. require="rtofs" to find the latest cycle with an actual RTOFS
    comparison — some regions/cycles only publish ESPC-vs-CMEMS, which would
    otherwise look like "the latest" even though it has no RTOFS image).

    Returns (date_str "YYYY-MM-DD", time_str "HHZ") or None.
    """
    key  = _region_url_key(region)
    base = f"{BASE_URL}/{key}/{variable_depth}/"
    try:
        if date_filter:
            year  = date_filter[:4]
            month = date_filter[5:7]
            r = requests.get(f"{base}{year}/{month}/", timeout=10)
            matches = [(d, t) for d, t in _dated_pngs(r.text, require) if d == date_filter]
        else:
            today_str = datetime.date.today().isoformat()
            r = requests.get(base, timeout=10)
            years = sorted(re.findall(r'href="(\d{4})/"', r.text), reverse=True)
            years = [y for y in years if y <= today_str[:4]]
            if not years:
                return None
            year = years[0]
            r = requests.get(f"{base}{year}/", timeout=10)
            months = sorted(re.findall(r'href="(\d{2})/"', r.text), reverse=True)
            months = [m for m in months if f"{year}-{m}" <= today_str[:7]]
            if not months:
                return None
            month = months[0]
            r = requests.get(f"{base}{year}/{month}/", timeout=10)
            # Guard against stray/mis-dated folders (e.g. a future month placed
            # under the current year) so "latest" never resolves ahead of today.
            matches = [(d, t) for d, t in _dated_pngs(r.text, require) if d <= today_str]

        if not matches:
            return None
        date_str, time_code = sorted(set(matches), reverse=True)[0]
        return date_str, time_code[:2] + "Z"
    except Exception as exc:
        print(f"  Warning: could not scrape {region}/{variable_depth}: {exc}")
        return None


def build_map_urls(region: str, variable_depth: str,
                   date_obj: datetime.date, time_str: str):
    """
    Return (url_copernicus, url_espc, url_espc_cmems, url_goes).
    Mirrors build_map_urls() in webapps/app.py exactly.
    url_goes is None for OHC and ESPC-only regions.
    """
    vd_mod  = variable_depth.replace("_", "-")
    year    = date_obj.year
    month   = f"{date_obj.month:02d}"
    day     = f"{date_obj.day:02d}"
    hour    = time_str.replace("Z", "").zfill(2)
    tcode   = f"{hour}0000"
    key     = _region_url_key(region)
    slug    = key if variable_depth.startswith("currents") else key.replace("_", "-")
    folder  = f"{BASE_URL}/{key}/{variable_depth}/{year}/{month}/"

    if variable_depth == "ocean_heat_content":
        stem = f"{key}_{year}-{month}-{day}T{tcode}Z"
        return (
            f"{folder}{stem}_heat_content_rtofs-cmems.png",
            f"{folder}{stem}_heat_content_rtofs-espc.png",
            f"{folder}{stem}_heat_content_espc-cmems.png",
            None,
        )
    else:
        stem = f"{slug}_{year}-{month}-{day}T{tcode}Z_{vd_mod}"
        return (
            f"{folder}{stem}_rtofs-vs-cmems.png",
            f"{folder}{stem}_rtofs-vs-espc.png",
            f"{folder}{stem}_espc-vs-cmems.png",
            f"{folder}{stem}_rtofs-vs-GOES.png",
        )


def build_ohc_legacy_url(region: str, date_obj: datetime.date, time_str: str) -> str:
    """
    The older single-panel [RTOFS | ESPC | CMEMS] Ocean Heat Content filename
    format. Fiji's pipeline still only produces this; Guam's produces it
    alongside the newer two-file format during its migration (see the caller
    in build_region_slides for how the two are reconciled). Try this only
    as a fallback when the standard build_map_urls() OHC files are missing —
    don't assume by region, since coverage can vary by cycle.
    """
    year, month, day = date_obj.year, f"{date_obj.month:02d}", f"{date_obj.day:02d}"
    hour  = time_str.replace("Z", "").zfill(2)
    tcode = f"{hour}0000"
    key    = _region_url_key(region)
    folder = f"{BASE_URL}/{key}/ocean_heat_content/{year}/{month}/"
    return f"{folder}{region}_{year}-{month}-{day}T{tcode}Z_ohc_rtofs-espc-cmems.png"


def nhc_image_label(url: str) -> str:
    """Derive a readable slide label from an nhc.noaa.gov storm_graphics URL.

    The descriptive name reliably lives in the folder segment
    ("{STORMID}_{descriptor}[_sm]+png"), not the filename — full-resolution
    URLs often reduce the filename to just the advisory id (e.g. '241434.png'),
    so parse the folder rather than the filename:
      '.../EP062026_5day_cone+png/241434.png'                  -> 'Forecast Cone (5-Day)'
      '.../AL022026_wind_history_sm+png/201146_wind_history_sm.png' -> 'Wind Speed History / Swath'
    """
    parts  = [p for p in url.rstrip("/").split("/") if p]
    folder = next((p for p in reversed(parts[:-1]) if re.search(r"\+(png|gif)$", p, re.IGNORECASE)), None)
    if folder:
        stem = re.sub(r"\+(png|gif)$", "", folder, flags=re.IGNORECASE)
    else:
        stem = re.sub(r"\.(png|gif)$", "", parts[-1], flags=re.IGNORECASE)
        stem = re.sub(r"^\d+_", "", stem)  # drop leading advisory-id prefix
    stem = re.sub(r"^[A-Za-z]{2}\d{6}_", "", stem)  # drop e.g. "EP062026_" / "AL022026_"
    stem = re.sub(r"_sm\d*$", "", stem)             # drop trailing _sm / _sm1 thumbnail suffix
    return NHC_LABELS.get(stem, stem.replace("_", " ").title())


def discover_active_storms() -> list[dict]:
    """
    Scrape the NHC homepage for currently active storms.

    NHC embeds an HTML comment near the top of each storm's table listing
    its official ID and name, e.g. "<!--storm identification: EP062026
    Hurricane Fausto-->", followed (within that same table) by
    "/refresh/graphics_{slot}+shtml/{advisory_id}.shtml?{key}" links.
    `slot` here is a basin+sequence token (e.g. 'ep1') NOT the storm's own
    number — it just indicates the storm's current on-page ordering, so it
    must be scraped rather than derived from the storm ID.

    Returns a list of dicts: {storm_id, name, basin, slot, advisory_id}.
    """
    try:
        r = requests.get(NHC_HOME, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        print(f"  Warning: could not fetch NHC homepage: {exc}")
        return []

    html = r.text
    id_matches = list(re.finditer(
        r'<!--storm identification: ([A-Z]{2})(\d{2})(\d{4}) (.+?)-->', html
    ))
    storms = []
    for i, m in enumerate(id_matches):
        basin, number, year, name = m.groups()
        start = m.start()
        end   = id_matches[i + 1].start() if i + 1 < len(id_matches) else len(html)
        section = html[start:end]
        g = re.search(r'/refresh/graphics_([a-z]{2}\d+)\+shtml/(\d+)\.shtml\?', section)
        if not g:
            continue
        slot, advisory_id = g.groups()
        storms.append({
            "storm_id":    f"{basin}{number}{year}",
            "name":        name.strip(),
            "basin":       basin,
            "slot":        slot,
            "advisory_id": advisory_id,
        })
    return storms


def match_storm(storms: list[dict], query: str) -> dict | None:
    """Find the active storm whose name best matches a free-text query
    (e.g. 'Fausto' or 'Hurricane Fausto — Approaching Hawaii')."""
    query_l = query.lower()
    for s in storms:
        name_l = s["name"].lower()               # e.g. "hurricane fausto"
        proper = name_l.split()[-1]               # e.g. "fausto" / "seven-e"
        if name_l in query_l or proper in query_l:
            return s
    return None


def get_storm_graphic_urls(slot: str, advisory_id: str,
                           keys: list[str] = None) -> list[str]:
    """
    For a storm's graphics slot + advisory id, visit each detail sub-page
    (e.g. .../241434.shtml?cone) and return the full-resolution image URL
    it links to. Keys with no matching graphic for this storm (e.g. 'surge'
    on a storm with no coastal surge risk) are silently skipped.
    """
    keys = keys or DEFAULT_NHC_GRAPHIC_KEYS
    urls: list[str] = []
    for key in keys:
        page_url = f"https://www.nhc.noaa.gov/refresh/graphics_{slot}+shtml/{advisory_id}.shtml?{key}#contents"
        try:
            r = requests.get(page_url, timeout=10)
            if r.status_code != 200:
                continue
            hrefs = re.findall(r'href="([^"]*storm_graphics[^"]*\.(?:png|gif))"', r.text)
        except Exception as exc:
            print(f"  Warning: could not fetch NHC graphic '{key}': {exc}")
            continue
        for href in hrefs:
            full = f"https://www.nhc.noaa.gov{href}" if href.startswith("/") else href
            if full not in urls:
                urls.append(full)
    return urls


def fetch_image(url: str | None) -> io.BytesIO | None:
    """Download an image and return a BytesIO buffer, or None if unavailable."""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=20)
        ct = r.headers.get("Content-Type", "")
        if r.status_code == 200 and ct.startswith("image"):
            return io.BytesIO(r.content)
    except Exception:
        pass
    return None


# ── Slide builders ─────────────────────────────────────────────────────────

def _text_box(slide, left, top, width, height, text, font_size, bold=False,
              color=None, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = False
    p  = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    run = p.runs[0]
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    return tb


def add_title_slide(prs: Presentation, subtitle: str, storm: str | None = None):
    layout = prs.slide_layouts[6]  # blank
    slide  = prs.slides.add_slide(layout)
    W, H   = prs.slide_width, prs.slide_height

    # Navy background
    bg = slide.shapes.add_shape(
        1, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    _text_box(slide, Inches(0.5), Inches(2.8), W - Inches(1), Inches(1.2),
              "IOOS Ocean Data-Model Comparisons", 40, bold=True, color=WHITE,
              align=PP_ALIGN.CENTER)
    _text_box(slide, Inches(0.5), Inches(4.1), W - Inches(1), Inches(0.7),
              storm or "Overview — Most Recent Figures", 24, color=RGBColor(0xAA, 0xCC, 0xFF),
              align=PP_ALIGN.CENTER)
    _text_box(slide, Inches(0.5), Inches(5.0), W - Inches(1), Inches(0.5),
              subtitle, 16, color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.CENTER)


def add_section_slide(prs: Presentation, region: str):
    layout = prs.slide_layouts[6]  # blank
    slide  = prs.slides.add_slide(layout)
    W, H   = prs.slide_width, prs.slide_height

    # Dark accent bar on left
    bar = slide.shapes.add_shape(1, 0, 0, Inches(0.4), H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    _text_box(slide, Inches(0.7), Inches(3.5), W - Inches(1), Inches(1.2),
              region, 40, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
    _text_box(slide, Inches(0.7), Inches(4.7), W - Inches(1), Inches(0.5),
              "Model Comparison Figures  ·  Most Recent Available", 14,
              color=GREY, align=PP_ALIGN.LEFT)


def add_image_slide(prs: Presentation, region: str, product_label: str,
                    images: list[tuple[io.BytesIO | None, str]],
                    date_str: str, time_str: str):
    """
    Add a content slide.

    images: list of (BytesIO_or_None, label_string)
    """
    layout = prs.slide_layouts[6]  # blank
    slide  = prs.slides.add_slide(layout)
    W, H   = prs.slide_width, prs.slide_height

    # Header bar
    hdr_h = Inches(0.55)
    hdr = slide.shapes.add_shape(1, 0, 0, W, hdr_h)
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = NAVY
    hdr.line.fill.background()

    _text_box(slide, Inches(0.15), 0, W - Inches(3), hdr_h,
              f"{region}  ·  {product_label}", 13, bold=True, color=WHITE)
    _text_box(slide, W - Inches(2.8), 0, Inches(2.7), hdr_h,
              f"{date_str}  {time_str}", 12, color=RGBColor(0xAA, 0xCC, 0xFF),
              align=PP_ALIGN.RIGHT)

    # Image area
    margin   = Inches(0.12)
    lbl_h    = Inches(0.28)
    top      = hdr_h + margin
    avail_h  = H - top - margin
    avail_w  = W - 2 * margin

    valid = [(buf, lbl) for buf, lbl in images if buf is not None]
    n = len(valid)
    if n == 0:
        return

    img_w = (avail_w - (n - 1) * margin) / n
    img_h = avail_h - lbl_h

    for i, (buf, label) in enumerate(valid):
        left = margin + i * (img_w + margin)
        # Label
        lbl_box = slide.shapes.add_textbox(left, top, img_w, lbl_h)
        lbl_tf  = lbl_box.text_frame
        lbl_p   = lbl_tf.paragraphs[0]
        lbl_p.text = label
        lbl_p.alignment = PP_ALIGN.CENTER
        lbl_run = lbl_p.runs[0]
        lbl_run.font.size = Pt(10)
        lbl_run.font.bold = True
        lbl_run.font.color.rgb = GREY
        # Image — scale to fit slot without stretching, then centre
        try:
            pic = slide.shapes.add_picture(buf, 0, 0)  # natural size first
            scale  = min(img_w / pic.width, img_h / pic.height)
            fit_w  = int(pic.width  * scale)
            fit_h  = int(pic.height * scale)
            pic.left   = int(left + (img_w - fit_w) / 2)
            pic.top    = int(top + lbl_h + (img_h - fit_h) / 2)
            pic.width  = fit_w
            pic.height = fit_h
        except Exception as exc:
            print(f"  Warning: could not embed image '{label}': {exc}")


# ── Glider profile helpers ─────────────────────────────────────────────────

def get_latest_glider_date() -> datetime.date | None:
    """Scrape the glider profiles directory for the most recent dated folder."""
    base = f"{BASE_PROFILES}/gliders/"
    try:
        today_str = datetime.date.today().isoformat()
        r = requests.get(base, timeout=10)
        years = sorted(re.findall(r'href="(\d{4})/"', r.text), reverse=True)
        years = [y for y in years if y <= today_str[:4]]
        if not years:
            return None
        year = years[0]
        r = requests.get(f"{base}{year}/", timeout=10)
        month_days = sorted(re.findall(r'href="(\d{2}-\d{2})/"', r.text), reverse=True)
        # Guard against stray/mis-dated folders (e.g. a future month placed
        # under the current year) so "latest" never resolves ahead of today.
        month_days = [md for md in month_days if f"{year}-{md}" <= today_str]
        if not month_days:
            return None
        month_str, day_str = month_days[0].split("-")
        return datetime.date(int(year), int(month_str), int(day_str))
    except Exception as exc:
        print(f"  Warning: could not find latest glider date: {exc}")
        return None


def get_glider_files(date_obj: datetime.date) -> dict[str, str]:
    """
    Scrape the glider profiles directory for date_obj and return
    {glider_id: full_url} for every 400m PNG found.

    Using the actual filenames from the server listing avoids any
    mismatch between a reconstructed URL and what was actually saved.
    """
    year      = date_obj.year
    month_day = f"{date_obj.month:02d}-{date_obj.day:02d}"
    base_url  = f"{BASE_PROFILES}/gliders/{year}/{month_day}/"
    try:
        r = requests.get(base_url, timeout=8)
        if r.status_code != 200:
            return {}
        result: dict[str, str] = {}
        for m in re.finditer(r'href="([^"]+_400m\.png)"', r.text):
            fname = m.group(1)
            gid   = fname.split("_20")[0]
            result[gid] = f"{base_url}{fname}"
        return result
    except Exception as exc:
        print(f"  Warning: could not get glider files for {date_obj}: {exc}")
        return {}


def get_glider_ids_in_bbox(bbox: list[float], t0: datetime.date, t1: datetime.date,
                           timeout: int = 8) -> set[str] | None:
    """
    Return the set of IOOS Glider DAC dataset IDs (e.g. 'ng344-20260713T0000')
    whose declared spatial coverage overlaps `bbox` and whose time coverage
    overlaps [t0, t1], via the NGDAC ERDDAP's own geospatial search — the
    same query get_active_gliders() in ioos_model_comparisons.platforms uses,
    but without downloading each dataset's full profile data.

    get_glider_files() scrapes a single global (not per-region) directory,
    so unlike get_argo_files()/filter_argo_by_track(), nothing upstream ever
    narrows gliders down to the region(s) actually being built — every
    active glider in the whole network shows up in every deck's "Glider
    Profiles" section regardless of --regions. This is the region-side
    filter that was missing; None means the ERDDAP search itself failed, so
    callers should fail open (keep everything) rather than treat "found no
    matches" and "couldn't check" the same way.
    """
    try:
        import pandas as pd
        from erddapy import ERDDAP
    except ImportError:
        print("    Warning: erddapy/pandas not importable — skipping glider region filter")
        return None

    e = ERDDAP(server="NGDAC")
    e.requests_kwargs["timeout"] = timeout
    kw = dict(
        min_time=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_time=(t1 + datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        min_lon=bbox[0], max_lon=bbox[1],
        min_lat=bbox[2], max_lat=bbox[3],
    )
    try:
        search = pd.read_csv(e.get_search_url(search_for=None, response="csv", **kw))
    except Exception as exc:
        print(f"    Warning: glider region search failed ({exc}) — skipping glider region filter")
        return None
    return set(search["Dataset ID"].tolist())


def union_extent(regions: list[str]) -> list[float] | None:
    """Bounding box covering every region in `regions`, via each region's
    regions.py extent [lonmin, lonmax, latmin, latmax]. None if none resolve."""
    try:
        from ioos_model_comparisons.regions import region_config
    except ImportError:
        return None
    boxes = []
    for region in regions:
        code = REGION_CODE_BY_NAME.get(region)
        if not code:
            continue
        try:
            boxes.append(region_config(regions=[code])["extent"])
        except Exception:
            continue
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes), max(b[1] for b in boxes),
        min(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


# ── Argo profile helpers ───────────────────────────────────────────────────
# Argo profiles are scraped per-region (unlike gliders, which are global),
# mirroring get_argo_ids()/get_argo_profile_url() in webapps/model_comparisons/app.py.

def get_argo_files(region: str, ref_date: datetime.date,
                   lookback_days: int = 7) -> dict[str, tuple[str, str]]:
    """
    Scan the region's dated argo profile folders over the last `lookback_days`
    days (inclusive of ref_date) and return {wmo_id: (url, date_str)} — the
    most recent profile PNG per float within that window.

    Argo floats only surface intermittently (often every ~5-10 days), so
    restricting to a single day misses most of them; a multi-day window is
    the normal way to see "recent" floats for a region.
    """
    key  = _region_url_key(region)
    best: dict[str, tuple[datetime.date, str]] = {}  # wmo -> (date, url)
    for days_back in range(lookback_days + 1):
        d = ref_date - datetime.timedelta(days=days_back)
        year, month, day = d.year, f"{d.month:02d}", f"{d.day:02d}"
        base_url = f"{BASE_PROFILES}/argo/{key}/{year}/{month}/{day}/"
        try:
            r = requests.get(base_url, timeout=8)
            if r.status_code != 200:
                continue
        except Exception as exc:
            print(f"  Warning: could not list argo files for {region}/{d}: {exc}")
            continue
        for m in re.finditer(r'href="([^"]+\.png)"', r.text):
            fname = m.group(1)
            wmo   = fname.split("-")[0]
            if wmo not in best or d > best[wmo][0]:
                best[wmo] = (d, f"{base_url}{fname}")
    return {wmo: (url, d.strftime("%Y-%m-%d")) for wmo, (d, url) in best.items()}


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in nautical miles."""
    EARTH_RADIUS_NM = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(a))


def get_storm_track_points(storm_id: str) -> list[tuple[float, float]] | None:
    """
    Fetch a storm's best-track fixes plus its latest NHC forecast cone as a
    list of (lat, lon) points, via tropycal's realtime feed — the same
    approach used by scripts/maps/models/synchronous/rtofs_binary_ohc_comparisons.py.
    Used by filter_argo_by_track() to keep only Argo floats actually near the
    storm's path instead of every float in the region.

    Returns None if tropycal isn't installed, or the storm can't be found in
    the realtime feed (e.g. it has already dissipated by the time this runs)
    — callers should treat that as "can't filter" and fall back to showing
    every float, not "no floats are near the storm".
    """
    try:
        from tropycal import realtime as trealtime
    except ImportError:
        print("    Warning: tropycal not installed — skipping Argo track-distance filter (pip install tropycal)")
        return None

    try:
        rt = trealtime.Realtime()
        storm = rt.get_storm(storm_id)
    except Exception as exc:
        print(f"    Warning: could not fetch tropycal track for storm '{storm_id}': {exc}")
        return None

    points = list(zip(storm.lat, storm.lon))
    try:
        ofcl = storm.get_operational_forecasts().get("OFCL", {})
        if ofcl:
            latest_cycle = max(ofcl.items(), key=lambda kv: kv[0])[1]
            points.extend(zip(latest_cycle.get("lat", []), latest_cycle.get("lon", [])))
    except Exception as exc:
        print(f"    Warning: could not fetch forecast cone for '{storm_id}': {exc}")

    return points or None


def filter_argo_by_track(argo_files: dict[str, tuple[str, str]], region: str,
                         ref_date: datetime.date, lookback_days: int,
                         track_points: list[tuple[float, float]],
                         max_distance_nm: float) -> dict[str, tuple[str, str]]:
    """
    Keep only the floats in `argo_files` (as returned by get_argo_files,
    which has no coordinates of its own — just filenames scraped off the
    profile image directory) whose most recent known position is within
    `max_distance_nm` of any point on the storm's track.

    Positions come from a live query against the IFREMER Argo ERDDAP via
    ioos_model_comparisons.platforms.get_argo_floats_by_time(). Any failure
    here (missing dependency, network error, no ERDDAP coverage) fails open
    — returns argo_files unfiltered — so a live-data hiccup can't silently
    drop floats that the static profile PNGs confirm do exist.
    """
    if not argo_files:
        return argo_files

    try:
        from ioos_model_comparisons.platforms import get_argo_floats_by_time
        from ioos_model_comparisons.regions import region_config
    except ImportError:
        print("    Warning: ioos_model_comparisons not importable — skipping Argo track-distance filter")
        return argo_files

    code = REGION_CODE_BY_NAME.get(region)
    if not code:
        print(f"    Warning: no regions.py extent for '{region}' — skipping Argo track-distance filter")
        return argo_files

    try:
        extent = region_config(regions=[code])["extent"]
        df = get_argo_floats_by_time(
            bbox=extent,
            time_start=ref_date - datetime.timedelta(days=lookback_days),
            time_end=ref_date + datetime.timedelta(days=1),
        )
    except Exception as exc:
        print(f"    Warning: could not fetch Argo positions for track filter: {exc}")
        return argo_files

    if df.empty:
        print("    Warning: Argo ERDDAP query returned no positions — skipping track filter")
        return argo_files

    # ERDDAP returns platform_number as int64; argo_files' WMO keys (from
    # scraped filenames) are strings — normalize so the .loc[[wmo]] lookup
    # below actually matches instead of silently KeyError-ing on every float.
    df = df.reset_index()
    df["argo"] = df["argo"].astype(str)
    df = df.set_index(["argo", "time"]).sort_index()

    kept = {}
    for wmo, (url, date_str) in argo_files.items():
        try:
            rows = df.loc[[wmo]]
        except KeyError:
            continue
        last = rows.sort_index(level="time").iloc[-1]
        dist = min(_haversine_nm(last["lat"], last["lon"], tlat, tlon) for tlat, tlon in track_points)
        if dist <= max_distance_nm:
            kept[wmo] = (url, date_str)
    return kept


# ── Ocean Heat Content trend ────────────────────────────────────────────────

def get_available_hour_for_date(region: str, variable_depth: str, date_obj: datetime.date,
                                preferred_hour: str = "12Z", require: str | None = None) -> str | None:
    """
    Return the best available 'HHZ' time string published for a specific date
    of a product, preferring `preferred_hour` but falling back to whatever
    hour is actually published that day. Some products run every 6 hours
    (00/06/12/18Z); others — e.g. ocean_heat_content in some regions — only
    run once daily, so a fixed hour can silently miss data that does exist.

    If `require` is given, only filenames containing that substring count
    (e.g. require="rtofs" to skip cycles that only published ESPC-vs-CMEMS).
    """
    key  = _region_url_key(region)
    year, month = date_obj.year, f"{date_obj.month:02d}"
    base = f"{BASE_URL}/{key}/{variable_depth}/{year}/{month}/"
    try:
        r = requests.get(base, timeout=10)
        if r.status_code != 200:
            return None
        date_str = date_obj.isoformat()
        times = sorted({t[:2] for d, t in _dated_pngs(r.text, require) if d == date_str})
        if not times:
            return None
        preferred = preferred_hour.replace("Z", "").zfill(2)
        return f"{preferred}Z" if preferred in times else f"{times[0]}Z"
    except Exception as exc:
        print(f"  Warning: could not check available times for {region}/{variable_depth}/{date_obj}: {exc}")
        return None


def add_ohc_trend_slides(prs: Presentation, region: str, ref_date: datetime.date,
                         total_slides: int, days: int = 3, hour: str = "12Z") -> int:
    """
    Add an Ocean Heat Content time series: one slide-set per day for the last
    `days` days up to and including ref_date, preferring `hour` each day but
    falling back to whatever hour is actually published (see
    get_available_hour_for_date). Meant to show how much heat content was
    available for a storm to draw on as it approached.
    """
    for offset in range(days - 1, -1, -1):  # oldest -> newest
        date_obj = ref_date - datetime.timedelta(days=offset)
        date_str = date_obj.strftime("%Y-%m-%d")

        actual_hour = get_available_hour_for_date(
            region, "ocean_heat_content", date_obj, preferred_hour=hour, require="rtofs"
        )
        if not actual_hour:
            print(f"    {date_str}  not found")
            continue
        note = "" if actual_hour == hour else f"  (using {actual_hour}, {hour} not published)"

        url_cop, url_espc, _url_espc_cmems, _ = build_map_urls(region, "ocean_heat_content", date_obj, actual_hour)

        buf_cop  = fetch_image(url_cop)
        buf_espc = fetch_image(url_espc)
        pairs = [
            (buf_cop,  "RTOFS vs CMEMS (Copernicus)"),
            (buf_espc, "RTOFS vs ESPC"),
        ]
        if not (buf_cop or buf_espc):
            legacy_buf = fetch_image(build_ohc_legacy_url(region, date_obj, actual_hour))
            if legacy_buf:
                pairs = [(legacy_buf, "Combined RTOFS/ESPC/CMEMS Panel")]

        n_ok = sum(1 for buf, _ in pairs if buf is not None)
        print(f"    {date_str} {actual_hour}{note}  {'ok' if n_ok else 'unavailable'}  ({n_ok} image(s))")
        for buf, img_label in pairs:
            if buf is None:
                continue
            add_image_slide(
                prs, region, f"OHC Trend  ·  {date_str} {actual_hour}  ·  {img_label}",
                [(buf, img_label)], date_str, actual_hour,
            )
            total_slides += 1
    return total_slides


def build_storm_advisory_slides(prs: Presentation, query: str, args: argparse.Namespace,
                                today: str, total_slides: int) -> tuple[int, bool, str | None, str | None]:
    """
    Search NHC for an active storm matching `query`, add its advisory
    graphics as slides, and report whether a match was found (used to gate
    the OHC trend and the Argo track-distance filter). Returns
    (total_slides, storm_found, matched_storm_name, matched_storm_id).
    """
    print(f"\n{'='*60}\n  Searching NHC for a storm matching '{query}'")
    storms  = discover_active_storms()
    matched = match_storm(storms, query)
    if not matched:
        active = ", ".join(s["name"] for s in storms) or "none"
        print(f"  No active storm matched '{query}'. Currently active: {active}")
        return total_slides, False, None, None

    print(f"  Matched: {matched['name']} ({matched['storm_id']})")
    nhc_urls = get_storm_graphic_urls(matched["slot"], matched["advisory_id"], args.nhc_keys)
    if not nhc_urls:
        print("  No graphics found for this storm.")
        return total_slides, True, matched["name"], matched["storm_id"]

    print(f"\n{'='*60}\n  NHC Advisory Graphics — {matched['name']}")
    add_section_slide(prs, matched["name"])
    for url in nhc_urls:
        label = nhc_image_label(url)
        buf = fetch_image(url)
        status = "ok" if buf else "unavailable"
        print(f"  {label:<40} {status}")
        if buf:
            add_image_slide(prs, "NHC Advisory", label, [(buf, label)], today, "")
            total_slides += 1
    return total_slides, True, matched["name"], matched["storm_id"]


def build_region_slides(prs: Presentation, region: str, args: argparse.Namespace,
                        ref_date: datetime.date, storm_found: bool, total_slides: int,
                        track_points: list[tuple[float, float]] | None = None) -> int:
    """Add the model-comparison, Argo profile, and (if a storm was found) OHC
    trend slides for a single region. Returns the updated total_slides.

    track_points, when given, is the matched storm's track (see
    get_storm_track_points) — Argo floats are filtered down to ones within
    args.argo_max_distance_nm of it (see filter_argo_by_track)."""
    print(f"\n{'='*60}\n  Region: {region}")

    products = discover_products(region)
    if not products:
        print(
            f"  !! No products found on server for region '{region}' — SKIPPING IT. "
            f"Known region names: {', '.join(OVERVIEW_REGIONS)} "
            f"(short codes like 'gom'/'mab' are also accepted)."
        )
        return total_slides

    add_section_slide(prs, region)

    for product in products:
        label = _product_label(product)
        print(f"  {product:<35}", end="", flush=True)

        # require="rtofs": the plain "latest" timestamp can land on a cycle
        # that only published ESPC-vs-CMEMS (no RTOFS at all) — e.g. Guam's
        # RTOFS comparisons currently only publish at the 00Z cycle, not
        # every 6 hours — so anchor on the latest cycle that actually has
        # an RTOFS image instead.
        latest = get_latest_map_info(region, product, date_filter=args.date, require="rtofs")
        if not latest:
            print("not found")
            continue

        date_str, time_str = latest
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        if (ref_date - date_obj).days > 1:
            print(f"stale ({date_str})")
            continue
        url_cop, url_espc, _url_espc_cmems, url_goes = build_map_urls(
            region, product, date_obj, time_str
        )

        buf_cop  = fetch_image(url_cop)
        buf_espc = fetch_image(url_espc)
        pairs = [
            (buf_cop,               "RTOFS vs CMEMS (Copernicus)"),
            (buf_espc,              "RTOFS vs ESPC"),
            (fetch_image(url_goes), "RTOFS vs GOES SST"),
        ]
        if product == "ocean_heat_content" and not (buf_cop or buf_espc):
            # Standard two-file OHC format is missing this cycle — fall back
            # to the older single combined RTOFS/ESPC/CMEMS panel some
            # regions still (sometimes only intermittently) produce.
            legacy_buf = fetch_image(build_ohc_legacy_url(region, date_obj, time_str))
            if legacy_buf:
                pairs = [(legacy_buf, "Combined RTOFS/ESPC/CMEMS Panel")]

        n_ok = sum(1 for buf, _ in pairs if buf is not None)
        if n_ok == 0:
            print("images unavailable")
            continue

        for buf, img_label in pairs:
            if buf is None:
                continue
            add_image_slide(prs, region, f"{label}  ·  {img_label}", [(buf, img_label)], date_str, time_str)
            total_slides += 1
        print(f"ok  ({date_str} {time_str}, {n_ok} image(s))")

    # ── Argo profiles for this region ──────────────────────────────
    print(f"  Argo Profiles ({region}, last {args.argo_lookback_days}d)")
    argo_files = get_argo_files(region, ref_date, lookback_days=args.argo_lookback_days)
    argo_total = len(argo_files)
    if argo_files and track_points and args.argo_track_filter:
        argo_files = filter_argo_by_track(
            argo_files, region, ref_date, args.argo_lookback_days,
            track_points, args.argo_max_distance_nm,
        )
        print(f"    Track filter: kept {len(argo_files)}/{argo_total} float(s) "
              f"within {args.argo_max_distance_nm:.0f} nm of the storm track")

    if argo_files:
        for wmo, (url, date_str) in sorted(argo_files.items()):
            buf = fetch_image(url)
            status = "ok" if buf else "unavailable"
            print(f"    Argo {wmo:<12} {status}  ({date_str})")
            if buf:
                add_image_slide(prs, region, f"Argo Float {wmo}", [(buf, f"Argo {wmo}")], date_str, "")
                total_slides += 1
    elif argo_total:
        print(f"    All {argo_total} float(s) filtered out — none within "
              f"{args.argo_max_distance_nm:.0f} nm of the storm track")
    else:
        print(f"    No argo profiles found in the last {args.argo_lookback_days} days")

    # ── Ocean Heat Content trend — only when a storm was found ──────
    if storm_found or args.ohc_trend:
        print(f"  Ocean Heat Content Trend ({region}, last {args.ohc_trend_days}d @ {args.ohc_trend_hour})")
        total_slides = add_ohc_trend_slides(
            prs, region, ref_date, total_slides,
            days=args.ohc_trend_days, hour=args.ohc_trend_hour,
        )

    return total_slides


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a PowerPoint of IOOS Model Comparisons overview figures."
    )
    parser.add_argument(
        "--output", "-o",
        default="ioos_model_comparisons_overview.pptx",
        help="Output .pptx filename (default: ioos_model_comparisons_overview.pptx)",
    )
    parser.add_argument(
        "--regions", nargs="+", metavar="REGION", default=None,
        help="Limit to specific region(s) (default: all overview regions)",
    )
    parser.add_argument(
        "--date", "-d", metavar="YYYY-MM-DD", default=None,
        help="Use figures from a specific date instead of the most recent available",
    )
    parser.add_argument(
        "--storm", metavar="NAME", default=None,
        help="Storm name to show on the title slide (e.g. 'Tropical Depression Two — Gulf of Mexico')",
    )
    parser.add_argument(
        "--nhc-images", nargs="+", metavar="URL", default=None,
        help=(
            "Explicit NHC advisory graphic URL(s) to insert as intro slides. "
            "Prefer --nhc-search, which finds these automatically."
        ),
    )
    parser.add_argument(
        "--nhc-search", metavar="TEXT", default=None,
        help=(
            "Auto-discover NHC advisory graphics for the currently active storm "
            "matching this name (e.g. 'Fausto') by scraping nhc.noaa.gov — no "
            "URLs needed. If omitted but --storm is given, --storm is used as "
            "the search text."
        ),
    )
    parser.add_argument(
        "--nhc-keys", nargs="+", metavar="KEY", default=None,
        help=(
            "Which NHC graphic types to auto-fetch with --nhc-search, e.g. "
            "cone swath key_messages radii surge tswind120 "
            f"(default: {' '.join(DEFAULT_NHC_GRAPHIC_KEYS)})"
        ),
    )
    parser.add_argument(
        "--list-storms", action="store_true",
        help="Print all currently active NHC storms (name, ID) and exit",
    )
    parser.add_argument(
        "--argo-lookback-days", type=int, metavar="N", default=7,
        help=(
            "How many days back to look for Argo float profiles per region "
            "(default: 7). Floats only surface intermittently, so a single "
            "day usually misses most of them."
        ),
    )
    parser.add_argument(
        "--ohc-trend-days", type=int, metavar="N", default=3,
        help=(
            "How many days of Ocean Heat Content (at --ohc-trend-hour) to include "
            "per region when a storm is found via --nhc-search/--storm (default: 3)"
        ),
    )
    parser.add_argument(
        "--ohc-trend-hour", metavar="HHZ", default="12Z",
        help="Fixed synoptic hour used for the OHC trend slides (default: 12Z)",
    )
    parser.add_argument(
        "--ohc-trend", action="store_true",
        help=(
            "Force-include the OHC trend slides even when no storm was auto-matched "
            "(e.g. when using --nhc-images instead of --nhc-search)"
        ),
    )
    parser.add_argument(
        "--argo-track-filter", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "When a storm is matched via --nhc-search/--storm, only show Argo "
            "floats near its track (best track + NHC forecast cone, via tropycal). "
            "On by default; use --no-argo-track-filter to show every float in "
            "the region regardless of the storm. Has no effect without a matched storm."
        ),
    )
    parser.add_argument(
        "--argo-max-distance-nm", type=float, metavar="NM", default=200.0,
        help="Max distance (nautical miles) from the storm track for --argo-track-filter (default: 200)",
    )
    parser.add_argument(
        "--storm-spec", action="append", metavar="NAME:REGION[,REGION...]",
        help=(
            "Build one deck covering multiple storms: repeat this flag once per "
            "storm, pairing an NHC search name with its own region(s), e.g. "
            "--storm-spec \"Fausto:Hawaii\" "
            "--storm-spec \"Seven-E:Eastern Pacific - Mexico\". "
            "Each gets its own NHC advisory section, region maps, Argo profiles, "
            "and OHC trend. Overrides --regions/--nhc-search/--storm/--nhc-images."
        ),
    )
    args = parser.parse_args()

    if args.list_storms:
        storms = discover_active_storms()
        if not storms:
            print("No active storms found on nhc.noaa.gov.")
        else:
            print("Active NHC storms:")
            for s in storms:
                print(f"  {s['storm_id']:<10} {s['name']}")
        return

    if args.date:
        try:
            datetime.datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            sys.exit(f"Invalid --date value '{args.date}' — expected YYYY-MM-DD")

    ref_date = (
        datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.date.today()
    )

    # Resolve storm/region groups. --storm-spec builds one deck covering
    # several storms, each paired with its own region(s); otherwise there's
    # a single implicit group using the existing --regions/--nhc-search/
    # --storm/--nhc-images flags, preserving prior single-storm behaviour.
    groups = []
    if args.storm_spec:
        for spec in args.storm_spec:
            if ":" not in spec:
                sys.exit(f"--storm-spec must be 'NAME:Region1,Region2' — got '{spec}'")
            name, region_str = spec.split(":", 1)
            regs = [resolve_region(r) for r in region_str.split(",") if r.strip()]
            if not regs:
                sys.exit(f"--storm-spec '{spec}' has no region(s) after ':'")
            groups.append({"search": name.strip(), "regions": regs, "nhc_images": None})
    else:
        groups.append({
            "search":     args.nhc_search or args.storm,
            "regions":    [resolve_region(r) for r in args.regions] if args.regions else OVERVIEW_REGIONS,
            "nhc_images": args.nhc_images,
        })

    prs = Presentation()
    prs.slide_width  = Inches(16)
    prs.slide_height = Inches(9)

    today = datetime.date.today().strftime("%Y-%m-%d")
    subtitle = f"Figures for {args.date}  ·  Generated {today}" if args.date else f"Generated {today}"
    storm_title = args.storm or (
        ", ".join(spec.split(":", 1)[0].strip() for spec in args.storm_spec)
        if args.storm_spec else None
    )
    add_title_slide(prs, subtitle, storm=storm_title)

    total_slides = 0
    all_regions: list[str] = []

    for group in groups:
        query, regions, nhc_images = group["search"], group["regions"], group["nhc_images"]
        all_regions.extend(regions)
        storm_found  = False
        track_points = None

        if nhc_images:
            print(f"\n{'='*60}\n  NHC Advisory Graphics")
            add_section_slide(prs, args.storm or "Storm Advisory")
            for url in nhc_images:
                label = nhc_image_label(url)
                buf = fetch_image(url)
                status = "ok" if buf else "unavailable"
                print(f"  {label:<40} {status}")
                if buf:
                    add_image_slide(prs, "NHC Advisory", label, [(buf, label)], today, "")
                    total_slides += 1
        elif query:
            total_slides, storm_found, _matched_name, storm_id = build_storm_advisory_slides(
                prs, query, args, today, total_slides
            )
            if storm_found and storm_id and args.argo_track_filter:
                track_points = get_storm_track_points(storm_id)

        for region in regions:
            total_slides = build_region_slides(
                prs, region, args, ref_date, storm_found, total_slides, track_points
            )

    # ── Glider profiles ───────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Glider Profiles")
    if args.date:
        glider_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        glider_date = get_latest_glider_date()

    if not glider_date:
        print("  Could not determine latest glider date")
    elif (ref_date - glider_date).days > 1:
        print(f"  Glider data is stale ({glider_date}, {(ref_date - glider_date).days} days old) — skipping")
    else:
        print(f"  Date: {glider_date}")
        glider_files = get_glider_files(glider_date)
        print(f"  {len(glider_files)} glider(s) published network-wide for {glider_date}")

        bbox = union_extent(all_regions)
        if bbox and glider_files:
            active_ids = get_glider_ids_in_bbox(bbox, glider_date, glider_date, timeout=15)
            if active_ids is None:
                print("  Could not verify glider region — keeping all (fail open)")
            else:
                dropped = sorted(gid for gid in glider_files if gid not in active_ids)
                glider_files = {gid: url for gid, url in glider_files.items() if gid in active_ids}
                if dropped:
                    preview = ", ".join(dropped[:6]) + ("..." if len(dropped) > 6 else "")
                    print(f"  Filtered to {', '.join(all_regions)} bbox {bbox}: "
                          f"kept {len(glider_files)}, dropped {len(dropped)} outside it ({preview})")
        if glider_files:
            add_section_slide(prs, "Glider Profiles")
            date_str = glider_date.strftime("%Y-%m-%d")
            for gid, url in glider_files.items():
                buf = fetch_image(url)
                status = "ok" if buf else "unavailable"
                print(f"  {gid:<40} {status}")
                if buf:
                    add_image_slide(prs, "Glider Profiles", gid, [(buf, gid)], date_str, "")
                    total_slides += 1
        else:
            print(f"  No glider profiles found for {glider_date}")

    output = Path(args.output)
    prs.save(output)
    print(f"\n{'='*60}")
    print(f"Saved {total_slides} content slides → {output.resolve()}")


if __name__ == "__main__":
    main()
