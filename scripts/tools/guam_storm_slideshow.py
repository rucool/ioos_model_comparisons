"""
Small briefing deck for Champi (ex-"Twenty-Nine"), the system that formed
east of Guam: where the storm track actually is, today's Ocean Heat Content
around Guam, and the two currently-deployed gliders (ng252, ng738) vs.
RTOFS/ESPC/CMEMS.

Sources — all generated locally today via the ~/boardwalk mount of the
production server's /home/hurricaneadm/data (Guam's RTOFS comes from a
pre-processed binary archive, not the OPeNDAP feed the rest of the project
uses, so it isn't reachable without that mount):
  - Storm-track map: scripts/tools/guam_storm_track_map.py (bathymetry +
    JTWC track + gliders, no gridded model data needed).
  - OHC maps: scripts/maps/models/synchronous/rtofs_binary_ohc_comparisons.py
    --data-dir ~/boardwalk/hurricaneadm/data/rtofs_archv --regions guam
    --start/--end pinned to today's 12Z.
  - Glider profiles: scripts/profiles/gliders/synchronous/
    daily_glider_profiles_comparisons_400m_guam.py, which now prefers
    ~/boardwalk/hurricaneadm/data/rtofs_archv automatically when present.

Falls back to scraping the live published site for anything not found
locally (e.g. if the mount isn't up on a later run).

Usage:
    python guam_storm_slideshow.py
"""
import argparse
import datetime as dt
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import overview_powerpoint as o  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

import ioos_model_comparisons.configs as conf  # noqa: E402

REGION_DISPLAY = "Guam"
TODAY = dt.date.today()


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


def local_file(path):
    path = Path(path)
    if path.is_file():
        return io.BytesIO(path.read_bytes())
    return None


def glider_profile(gid, deployment="20260713T0000", max_lookback_days=4):
    """Local file first (today, generated against the boardwalk mount);
    falls back to the live published server, walking back a few days, if
    the local file isn't there. Returns (BytesIO_or_None, date_used, "local"|"server")."""
    rel = f"profiles/gliders/{TODAY.year}/{TODAY:%m-%d}/{gid}-{deployment}_{TODAY:%Y%m%d}_to_{(TODAY+dt.timedelta(days=1)):%Y%m%d}_400m.png"
    buf = local_file(conf.path_plots / rel)
    if buf:
        return buf, TODAY, "local"

    for back in range(max_lookback_days + 1):
        d0 = TODAY - dt.timedelta(days=back)
        d1 = d0 + dt.timedelta(days=1)
        url = (f"{o.BASE_PROFILES}/gliders/{d0.year}/{d0:%m-%d}/"
               f"{gid}-{deployment}_{d0:%Y%m%d}_to_{d1:%Y%m%d}_400m.png")
        buf = o.fetch_image(url)
        if buf:
            return buf, d0, "server"
    return None, None, None


def add_annotated_slide(prs, region, product_label, images, date_str, time_str, caption):
    """Same as overview_powerpoint.add_image_slide(), plus a caption band
    under the header explaining what the slide shows."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o",
                    default=str(Path.home() / f"Documents/{TODAY.isoformat()}_guam_storm_watch.pptx"))
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    o.add_title_slide(prs, f"Prepared {TODAY.isoformat()}",
                      "Storm Watch near Guam  ·  Ocean Heat Content  ·  ng252 & ng738")

    # ── 1. Storm tracks ──────────────────────────────────────────────────
    print("\nStorm tracks")
    section(prs, "Where the Storm Is",
            "Champi (JTWC), tracking toward the Marianas, plus both active gliders")
    buf = local_file(conf.path_plots / "maps" / "guam" / "storm_tracks" / f"guam_storm_tracks_{TODAY.isoformat()}.png")
    if buf:
        add_annotated_slide(
            prs, REGION_DISPLAY, "Storm tracks & active gliders", [(buf, "JTWC track")],
            TODAY.isoformat(), "",
            "Champi (JTWC) is now a named tropical storm — intensified from 30kt to 60kt in the last "
            "24 hours — tracking west from ~160°E toward the Marianas/Guam longitude. The weaker "
            "system to the northwest ('Twenty-Eight') has dropped off the active list. No forecast cone "
            "yet (JTWC hasn't published model guidance for Champi), so this is best-track position only. "
            "ng252 and ng738 (triangles) are Guam's two active gliders."
        )
        print("  OK")
    else:
        print("  MISSING — run scripts/tools/guam_storm_track_map.py first")

    # ── 2. Ocean Heat Content ────────────────────────────────────────────
    print("\nOcean Heat Content")
    section(prs, "Ocean Heat Content around Guam",
            f"RTOFS vs CMEMS / ESPC, {TODAY.isoformat()} 12Z  ·  generated locally from the boardwalk mount")
    for pair, label in [("rtofs-cmems", "RTOFS vs CMEMS"), ("rtofs-espc", "RTOFS vs ESPC")]:
        rel = f"maps/guam/ocean_heat_content/{TODAY.year}/{TODAY.month:02d}/guam_{TODAY.isoformat()}T120000Z_heat_content_{pair}.png"
        buf = local_file(conf.path_plots / rel)
        if buf:
            add_annotated_slide(
                prs, REGION_DISPLAY, f"Ocean Heat Content  ·  {label}", [(buf, label)],
                TODAY.isoformat(), "12Z",
                "Heat content Champi has to draw on if it continues tracking this way — mostly "
                "100-150+ kJ/cm² (orange/yellow) along its current path, plenty of fuel for further "
                "intensification. Both gliders (triangles) sit right next to Guam (red diamond) for context."
            )
            print(f"  {label} OK")
        else:
            print(f"  {label} MISSING")

    # ── 3. Glider profiles ───────────────────────────────────────────────
    print("\nGlider profiles")
    section(prs, "ng252 & ng738 vs. RTOFS/ESPC/CMEMS",
            f"{TODAY.isoformat()}  ·  generated locally from the boardwalk mount")
    for gid in ("ng252", "ng738"):
        buf, used_date, source = glider_profile(gid)
        if buf:
            note = "" if source == "local" and used_date == TODAY else f"  (from {source}, {used_date.isoformat()})"
            add_annotated_slide(
                prs, REGION_DISPLAY, f"{gid} daily profile", [(buf, f"{gid} vs RTOFS/ESPC/CMEMS")],
                used_date.isoformat(), "",
                f"{gid}'s measured profile (blue) vs. RTOFS (red), ESPC (green) and CMEMS (magenta).{note} "
                "Generated locally today now that Guam's RTOFS binary archive is reachable via the "
                "boardwalk mount."
            )
            print(f"  {gid} OK ({source}, {used_date})")
        else:
            print(f"  {gid} MISSING everywhere")

    prs.save(args.output)
    print(f"\nSaved {len(prs.slides)} slides -> {args.output}")


if __name__ == "__main__":
    main()
