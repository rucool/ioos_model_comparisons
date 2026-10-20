# RTOFS upgrades

Everything that depends on the RTOFS version lives in
`ioos_model_comparisons/config/rtofs.yml`. An upgrade is an edit to that
file plus the THREDDS aggregation on `tds.marine.rutgers.edu`, which is
outside this repo.

What reads `rtofs.yml`:

| Consumer | Uses |
|---|---|
| `models.rtofs(source=...)` and `models.RTOFS` | `streams.<stream>.thredds`, `model_attr`, `rename` |
| `scripts/harvest/grab_rtofs_aws.py` | `aws_url`, `nomads_url`, `data_dirs`, `files` |
| Figure catalog and portal | `history`: each figure is tagged `rtofs: {stream: version}` |

`model_attr` (`RTOFS`, `RTOFS-P`) becomes part of every plot filename (as
`rtofs` and `rtofs-p`). Never change it for an existing stream, or the
portal loses the link to every existing figure.

## A new parallel version (e.g. v2.5)

1. In `rtofs.yml`, edit `streams.parallel`:
   - set `aws_url` to `https://noaa-nws-rtofs-pds.s3.amazonaws.com/rtofs.parallel.v2.5`;
   - set `data_dirs` to the new download directories (keep east and west in
     separate trees; the comment in the harvester explains why);
   - update `thredds.east` if the aggregation is renamed.
2. Add a `history` row: `- {stream: parallel, version: v2.5, from: 2026-11-01}`.
3. Point the THREDDS aggregation at the new data directory.
4. Run `make check`, commit, and `git pull` on the plot host. Harvest, plots
   and catalog tags all follow.

## Parallel becomes operational

1. Add `- {stream: operational, version: v2.5, from: <cut-over date>}` to
   `history`.
2. If NOAA keeps publishing the old paths, nothing else changes. If not,
   update `streams.operational` the same way as above.
3. Retire or re-point `streams.parallel`. If no parallel run exists, delete
   the stream; the harvester then skips it.

## Fixing version tags after the fact

Version tags are applied when a figure is cataloged. After correcting
`history` dates, re-tag the affected window:

```bash
python scripts/tools/index_figures.py --retag-rtofs 2026-06-01 2026-11-01
```

## Not yet config-driven

The binary pipeline still defines its grid in code:
`hycom/rtofs_binary.py` (`RTOFS_STANDARD_Z_LEVELS`, `idm=4500, jdm=3298`)
and `scripts/harvest/grab_rtofs_archv_aws.py`. A grid change in an upgrade
needs edits there.
