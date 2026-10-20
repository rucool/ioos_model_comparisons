"""
Focused briefing deck for Breanna: why RTOFS's temperature structure is
diverging from CMEMS/the glider near sg625 (the center of the three CICESE
gliders — sg623/sg625/sg650 — currently deployed off western Mexico), and
why that makes RTOFS's Ocean Heat Content unrealistically low on some days.

Per the boss's review of the first draft, this tells one story instead of
covering everything:
  1. A map showing Hurricane Simon's and Hurricane Trudy's tracks passing
     close to the CICESE gliders (the existing Ocean Heat Content product
     already overlays both, so no new plot was needed for this).
  2. Daily sg625-vs-RTOFS/CMEMS profile comparison, Oct 10 -> Oct 14, so the
     evolution is visible — notably Oct 12 (RTOFS Ocean Heat Content collapses
     to ~2.6 kJ/cm^2 against the glider's ~87.6) and Oct 13 (~12.7 vs ~70.1).
  3. The 50 m RTOFS-vs-CMEMS temperature map for the same five days, with
     sg625's position circled (see the new `highlight` parameter on
     ioos_model_comparisons.plotting.plot_model_region_comparison).

Like storm_focus_deck.py this reuses overview_powerpoint.py's slide-building
helpers, but — same as the first draft of this deck — embeds our own
model-comparison PNGs straight from the local plots directory
(ioos_model_comparisons.configs.path_plots) rather than scraping the
published site, since the 50 m highlight and the profile comparisons aren't
deployed there.

Usage:
    python storm_focus_deck_simon_trudy.py
    python storm_focus_deck_simon_trudy.py --output my_deck.pptx
"""
import argparse
import datetime as dt
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import overview_powerpoint as o  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

import ioos_model_comparisons.configs as conf  # noqa: E402
from ioos_model_comparisons.figure_names import FigureMeta, build as build_fig_path  # noqa: E402

REGION_FOLDER  = "mexico_pacific"
REGION_DISPLAY = "Eastern Pacific - Mexico"
GLIDER_ID      = "sg625-20260630T0000"   # center of the 3 CICESE gliders

# One map per day, Oct 10 -> Oct 14, all at the 12Z cycle.
HISTORY_DAYS = [dt.datetime(2026, 10, d, 12) for d in range(10, 15)]

# RTOFS / glider Ocean Heat Content (kJ/cm^2) read off each day's profile
# comparison, for the slide captions — see the figure itself for the full
# Glider/RTOFS/CMEMS/NESDIS line.
OHC_NOTE = {
    dt.date(2026, 10, 10): "Glider 120.5, RTOFS 87.9 — close",
    dt.date(2026, 10, 11): "Glider 121.0, RTOFS 24.9 — RTOFS pulling away",
    dt.date(2026, 10, 12): "Glider 87.6, RTOFS 2.6 — RTOFS collapses",
    dt.date(2026, 10, 13): "Glider 70.1, RTOFS 12.7 — still far off",
    dt.date(2026, 10, 14): "Glider 72.0, RTOFS 43.4 — partial recovery",
}


def section(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    W, H = prs.slide_width, prs.slide_height
    bar = slide.shapes.add_shape(1, 0, 0, Inches(0.4), H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = o.NAVY
    bar.line.fill.background()
    o._text_box(slide, Inches(0.7), Inches(3.2), W - Inches(1), Inches(1.2),
                title, 40, bold=True, color=o.NAVY)
    o._text_box(slide, Inches(0.7), Inches(4.5), W - Inches(1), Inches(0.5),
                subtitle, 16, color=o.GREY)


def local_file(rel_path, label):
    """(BytesIO_or_None, label) for a figure already on this machine."""
    path = conf.path_plots / rel_path
    if path.is_file():
        return io.BytesIO(path.read_bytes()), label
    return None, label


def pad_to_common_size(bufs):
    """Pad a list of PNG BytesIO buffers (may be None) to a shared canvas —
    the max width/height across the group, image anchored top-left.

    plot_model_region_comparison(fixed_layout=True) pins the map panels to
    identical pixel positions regardless of that day's Argo/glider legend
    content, but the legend itself can still overflow its fixed box on a
    content-heavy day, and savefig's bbox_inches='tight' then crops to that
    taller/wider extent — so output canvas size can still differ by a few
    pixels. Anchoring top-left (rather than centering) when padding to a
    common canvas keeps the map position — which is already identical
    top-left-relative — from shifting; any extra canvas only grows
    rightward/downward into the legend's own margin.
    """
    images = [Image.open(b) for b in bufs if b is not None]
    if not images:
        return bufs
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)

    out = []
    i = 0
    for b in bufs:
        if b is None:
            out.append(None)
            continue
        im = images[i]
        i += 1
        canvas = Image.new("RGBA", (max_w, max_h), (255, 255, 255, 255))
        canvas.paste(im, (0, 0))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        buf.seek(0)
        out.append(buf)
    return out


def add_annotated_slide(prs, region, product_label, images, date_str, time_str, caption):
    """Same layout as overview_powerpoint.add_image_slide(), with an added
    caption band between the header and the image(s) explaining what the
    slide shows — written for someone (Breanna) who wasn't in the room for
    the analysis, not just a figure label."""
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)
    W, H   = prs.slide_width, prs.slide_height

    hdr_h = Inches(0.55)
    hdr = slide.shapes.add_shape(1, 0, 0, W, hdr_h)
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = o.NAVY
    hdr.line.fill.background()

    o._text_box(slide, Inches(0.15), 0, W - Inches(3), hdr_h,
                f"{region}  ·  {product_label}", 13, bold=True, color=o.WHITE)
    o._text_box(slide, W - Inches(2.8), 0, Inches(2.7), hdr_h,
                f"{date_str}  {time_str}", 12, color=o.RGBColor(0xAA, 0xCC, 0xFF),
                align=PP_ALIGN.RIGHT)

    # Caption band: light background strip with wrapped explanatory text.
    cap_h = Inches(0.85)
    cap_bg = slide.shapes.add_shape(1, 0, hdr_h, W, cap_h)
    cap_bg.fill.solid()
    cap_bg.fill.fore_color.rgb = o.RGBColor(0xEE, 0xF2, 0xF8)
    cap_bg.line.fill.background()
    cap_tb = slide.shapes.add_textbox(Inches(0.25), hdr_h + Inches(0.08), W - Inches(0.5), cap_h - Inches(0.16))
    cap_tf = cap_tb.text_frame
    cap_tf.word_wrap = True
    cap_p = cap_tf.paragraphs[0]
    cap_p.text = caption
    cap_run = cap_p.runs[0]
    cap_run.font.size = Pt(12)
    cap_run.font.color.rgb = o.NAVY

    margin  = Inches(0.12)
    lbl_h   = Inches(0.28)
    top     = hdr_h + cap_h + margin
    avail_h = H - top - margin
    avail_w = W - 2 * margin

    valid = [(buf, lbl) for buf, lbl in images if buf is not None]
    n = len(valid)
    if n == 0:
        return

    img_w = (avail_w - (n - 1) * margin) / n
    img_h = avail_h - lbl_h

    for i, (buf, label) in enumerate(valid):
        left = margin + i * (img_w + margin)
        lbl_box = slide.shapes.add_textbox(left, top, img_w, lbl_h)
        lbl_p = lbl_box.text_frame.paragraphs[0]
        lbl_p.text = label
        lbl_p.alignment = PP_ALIGN.CENTER
        lbl_run = lbl_p.runs[0]
        lbl_run.font.size = Pt(10)
        lbl_run.font.bold = True
        lbl_run.font.color.rgb = o.GREY
        try:
            pic = slide.shapes.add_picture(buf, 0, 0)
            scale = min(img_w / pic.width, img_h / pic.height)
            fit_w = int(pic.width * scale)
            fit_h = int(pic.height * scale)
            pic.left = int(left + (img_w - fit_w) / 2)
            pic.top = int(top + lbl_h + (img_h - fit_h) / 2)
            pic.width = fit_w
            pic.height = fit_h
        except Exception as exc:
            print(f"  Warning: could not embed image '{label}': {exc}")


OHC_CAPTION = (
    "Trudy's forecast track (colored dots, intensifying toward Cat 4) passes almost directly over "
    "sg625 (red triangle). Simon's track (yellow dots, curving northwest) stays further north, near "
    "sg623/sg650. This same Ocean Heat Content map is the data source behind the next section's numbers."
)
PROFILE_CAPTION = (
    "Each line is a profile at sg625 that day: the glider (blue) vs. RTOFS (red) and CMEMS (magenta) "
    "sampled at the same time/place. Where RTOFS's red line sits left of the glider's blue line, RTOFS "
    "is too cold there — that gap below ~20m drags RTOFS's Ocean Heat Content (bottom text) far below "
    "the glider's own measurement."
)
TEMPERATURE_50M_CAPTION = (
    "Yellow circle marks sg625's position. Compare water color inside the circle between RTOFS (left) "
    "and CMEMS (right) — a visibly darker/cooler patch in RTOFS here is the same cold bias the profile "
    "plots quantify at this location."
)


def ohc_map_slide(prs, valid_time, models=("rtofs", "cmems")):
    """The existing Ocean Heat Content product for this time — already shows
    storm tracks (via tropycal) and glider markers together."""
    meta = FigureMeta(family="ohc", product="ohc", region=REGION_FOLDER,
                       variable=None, depth=None, valid_time=valid_time, models=models)
    label = " vs ".join(m.upper() for m in models)
    buf, label = local_file(build_fig_path(meta), label)
    if buf:
        add_annotated_slide(prs, REGION_DISPLAY, f"Storm tracks near the CICESE gliders  ·  {label}",
                            [(buf, label)], valid_time.date().isoformat(), f"{valid_time:%H}Z",
                            OHC_CAPTION)
        return 1
    print(f"    OHC {label}: not found")
    return 0


def profile_slide(prs, day):
    """sg625's daily profile-comparison composite (temperature/salinity/
    density vs RTOFS/CMEMS, OHC numbers, mini track map)."""
    d0 = day.date()
    d1 = d0 + dt.timedelta(days=1)
    rel = f"profiles/gliders/{d0.year}/{d0:%m-%d}/{GLIDER_ID}_{d0:%Y%m%d}_to_{d1:%Y%m%d}_150m.png"
    buf, label = local_file(rel, "sg625 vs RTOFS/CMEMS")
    if buf:
        note = OHC_NOTE.get(d0, "")
        add_annotated_slide(prs, REGION_DISPLAY, f"sg625 daily profile  ·  {note}",
                            [(buf, label)], d0.isoformat(), "", PROFILE_CAPTION)
        return 1
    print(f"    Profile {d0}: not found")
    return 0


def temperature_50m_slides(prs, valid_times, models=("rtofs", "cmems")):
    """One slide per valid_time, all padded to identical pixel dimensions
    first (see pad_to_common_size) so the map renders at the same size on
    every slide."""
    label = " vs ".join(m.upper() for m in models)
    bufs = []
    for valid_time in valid_times:
        meta = FigureMeta(family="map", product="model_comparison", region=REGION_FOLDER,
                           variable="temperature", depth=50, valid_time=valid_time, models=models)
        buf, _ = local_file(build_fig_path(meta), label)
        bufs.append(buf)

    bufs = pad_to_common_size(bufs)

    n = 0
    for valid_time, buf in zip(valid_times, bufs):
        if buf is None:
            print(f"    Temperature 50m {label} {valid_time.date()}: not found")
            continue
        add_annotated_slide(prs, REGION_DISPLAY, f"Temperature 50m  ·  sg625 circled  ·  {label}",
                            [(buf, label)], valid_time.date().isoformat(), f"{valid_time:%H}Z",
                            TEMPERATURE_50M_CAPTION)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o",
                    default=str(Path.home() / f"Documents/{dt.date.today().isoformat()}_sg625_rtofs_cold_bias.pptx"))
    args = ap.parse_args()

    today = dt.date.today()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    o.add_title_slide(prs, f"Prepared {today.isoformat()}",
                      "sg625 (CICESE) vs RTOFS  ·  Why 50 m, and why RTOFS's heat content is unrealistically low")

    # ── 1. Storm tracks near the CICESE gliders ─────────────────────────
    print("\nStorm tracks near the gliders")
    section(prs, "Simon & Trudy vs. the CICESE Gliders",
            "Both storms' tracks pass close to sg623 / sg625 / sg650")
    ohc_map_slide(prs, dt.datetime(2026, 10, 13, 12))

    # ── 2. Daily sg625 profile evolution ────────────────────────────────
    print("\nsg625 daily profile evolution")
    section(prs, "sg625 Daily Profile Evolution",
            "Oct 10 → Oct 14  ·  Temperature / Salinity / Density vs RTOFS & CMEMS, with Ocean Heat Content")
    for day in HISTORY_DAYS:
        profile_slide(prs, day)

    # ── 3. 50m temperature maps, sg625 circled ──────────────────────────
    print("\n50m temperature — RTOFS vs CMEMS")
    section(prs, "Temperature at 50 m",
            "RTOFS vs CMEMS, Oct 10 → Oct 14  ·  sg625's position circled")
    temperature_50m_slides(prs, HISTORY_DAYS)

    prs.save(args.output)
    print(f"\nSaved {len(prs.slides)} slides -> {args.output}")


if __name__ == "__main__":
    main()
