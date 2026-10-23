# Hurricane-season glider and Argo vs model comparison

Compares glider or Argo profiles with CMEMS and RTOFS over a hurricane season, for the Caribbean, Hawaii, Guam and
the Mexican Pacific. For each pair it computes ocean heat content (OHC), the 26 °C isotherm depth, the mixed
layer depth and the salinity maximum, and plots each variable as a hexbin.

Data, logs and plots go to `~/Documents/glider_ohc_<year>/` (Argo: its `argo/` subfolder), outside git.
Choose the platform with `SEASON_PLATFORM=glider|argo` (default glider), and the location or season with
`GLIDER_OHC_ROOT`, `SEASON_YEAR` and `SEASON_START` (see `season_config.py`). Everything runs in the
`model_comps` env.

```bash
caffeinate -i ./run_all.sh            # steps 0-3, all regions except Guam RTOFS (several hours, mostly RTOFS)
./run_guam_rtofs.sh                   # Guam RTOFS, extracted on baffin via boardwalk
./rerun_metrics_plots.sh              # steps 2-3 only, after changing a calculation or plot
SEASON_PLATFORM=argo caffeinate -i ./run_all.sh   # the same for Argo (then SEASON_PLATFORM=argo ./run_guam_rtofs.sh)
```

| Step | Script | Output |
|---|---|---|
| 0 | `step0_download_gliders.py <region>` | `<year>_<region>_gliders.csv`: DAC profiles, clipped to the region and cleaned (Mexico Pacific is Pacific only, south of 32.5N; duplicate datasets dropped) |
| 0 | `step0_download_argos.py <region>` | `<year>_<region>_argos.csv`: IFREMER ArgoFloats, good time/position/pres/temp/psal flags, depth from pressure. Float id goes in the `glider` column |
| 1 | `step1_pair_model_profiles.py <region> <model>` | `<region>/<glider>/<glider>_<time>[_<model>].csv`: at each 6-hourly model time, the closest glider profile within ±3 h and the nearest model column. Resumable. |
| 2 | `step2_calculate_metrics.py <region> <model>` | `<region>/<glider>/calculated/`: per-profile metrics |
| 3 | `step3_plot_hexbin.py <title> <model> <t0> <t1> <region>` | `plots/*.png`, `plots/summary.csv` |

Method notes:
- **OHC** (`ohc_with_qc` in step 2) is computed the same way for glider and model. The profile is put on a
  1 m grid from the surface (the top value is held constant above the first sample) and integrated to the
  interpolated 26 °C crossing. A profile is excluded if it starts deeper than 10 m or never gets below 26 °C.
  Pairs where both glider and model OHC are 0 are left out of the plot. This differs from
  `calc.ocean_heat_content`, which starts at the first sample and stops at the last sample of 26 °C or warmer.
- **Fit line** is an orthogonal (total least squares) fit. Both axes are noisy, and an ordinary least
  squares fit of model on glider comes out flatter than the scatter by a factor of about r.
- **Hexagon colours** are profile counts.
- **Model access:**
  - CMEMS is read from the time-series-chunked ARCO store, grouped by its chunks (32 × 32 cells × 526
    six-hourly steps), so each chunk is read once for all the profiles in it. The map-chunked store that
    `models.CMEMS` uses takes about a minute per profile.
  - Guam RTOFS is read on baffin from `/home/hurricaneadm/data/rtofs_archv`, then from
    `~/rtofs_archv_backfill` for days missing there (made with
    `grab_rtofs_archv_aws.py --date D --regions guam --no-global --output-dir ~/rtofs_archv_backfill`).
  - RTOFS comes from the Rutgers THREDDS. That is a production server, so the run uses only 2 workers.
- **CMEMS caveat:** each week of CMEMS is re-run with that week's observations, which may include these
  gliders, so it is not a fully independent test.
- **Argo caveat:** both models assimilate Argo, so the Argo comparison measures how well each model holds
  onto data it was given, not independent skill.
