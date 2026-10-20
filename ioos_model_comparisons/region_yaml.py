"""
region_yaml.py — targeted, comment-preserving edits to config/regions/*.yml.

Used by scripts/tools/colorbar_tuner.py ("Save region YAML") to write tuned
colorbar limits back into the region defaults. Instead of load-and-dump
(which would drop every comment), it locates the exact YAML node with the
parser and splices new text in at that character range.

    update_limits({("gom", "temperature", 0): [26, 32, 0.5],
                   ("gom", "currents", 150): [0, 1.2, 0.1]})
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ioos_model_comparisons.registry import CONFIG_DIR, reload_registry


def _fmt(limits):
    return "[" + ", ".join(repr(float(v)) if not float(v).is_integer() else str(int(v))
                           for v in limits) + "]"


def _get(mapping_node, key):
    for k, v in mapping_node.value:
        if k.value == str(key):
            return k, v
    return None, None


def _find_depth_item(seq_node, depth):
    for item in seq_node.value:
        _k, d = _get(item, "depth")
        if d is not None and float(d.value) == float(depth):
            return item
    return None


def update_limits(edits, config_dir=None):
    """Apply {(region, var, depth): [vmin, vmax, stride]}. var is a key under
    `variables` (temperature, salinity), "sea_surface_height", or "currents"
    (writes currents.limits_by_depth[depth]). Returns a list of messages."""
    rdir = Path(config_dir or CONFIG_DIR) / "regions"
    by_region = {}
    for (region, var, depth), limits in edits.items():
        by_region.setdefault(region, []).append((var, depth, limits))

    msgs = []
    for region, items in by_region.items():
        path = rdir / f"{region}.yml"
        if not path.is_file():
            msgs.append(f"{region}: no {path.name}")
            continue
        for var, depth, limits in items:
            text = path.read_text()
            root = yaml.compose(text)
            new_text, msg = _apply(text, root, var, depth, limits)
            msgs.append(f"{region}.{var}[{depth}]: {msg}")
            if new_text is not None:
                yaml.safe_load(new_text)  # never write a file we can't read back
                path.write_text(new_text)
    reload_registry()
    return msgs


def _splice(text, node, replacement):
    return text[:node.start_mark.index] + replacement + text[node.end_mark.index:]


def _apply(text, root, var, depth, limits):
    new = _fmt(limits)
    if var == "currents":
        _k, cur = _get(root, "currents")
        if cur is None:
            return None, "region has no currents block — skipped"
        _k, lbd = _get(cur, "limits_by_depth")
        dkey = str(int(depth)) if float(depth).is_integer() else str(depth)
        if lbd is not None:
            _kk, val = _get(lbd, dkey)
            if val is not None:
                return _splice(text, val, new), "updated"
            if lbd.flow_style:
                inner = text[lbd.start_mark.index + 1:lbd.end_mark.index - 1].strip()
                body = f"{inner}, {dkey}: {new}" if inner else f"{dkey}: {new}"
                return _splice(text, lbd, "{" + body + "}"), "added"
            indent = " " * lbd.value[0][0].start_mark.column
            # end_mark sits on the next key's line; insert at that line's start.
            insert_at = lbd.end_mark.index - lbd.end_mark.column
            return text[:insert_at] + f"{indent}{dkey}: {new}\n" + text[insert_at:], "added"
        # No limits_by_depth yet: add it as the first child of currents.
        first_key = cur.value[0][0]
        indent = " " * first_key.start_mark.column
        at = first_key.start_mark.index - first_key.start_mark.column
        return text[:at] + f"{indent}limits_by_depth: {{{dkey}: {new}}}\n" + text[at:], "added"

    if var == "sea_surface_height":
        _k, seq = _get(root, "sea_surface_height")
    else:
        _k, variables = _get(root, "variables")
        seq = _get(variables, var)[1] if variables is not None else None
    if seq is None or not hasattr(seq, "value") or not isinstance(seq.value, list):
        return None, "no such variable list — skipped"
    item = _find_depth_item(seq, depth)
    if item is None:
        return None, "depth not configured — skipped (add it to the YAML first)"
    _k, lim = _get(item, "limits")
    if lim is None:
        return None, "depth has no limits entry — skipped"
    return _splice(text, lim, new), "updated"
