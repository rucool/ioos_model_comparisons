# Configuration: regions, models, comparisons

All of this lives in `ioos_model_comparisons/config/`. `registry.py` loads
and validates it. A typo, an unknown model, a reversed colorbar range or a
dangling reference makes `load_registry()` raise with a list of every
problem, so `make check` catches bad edits before they ship.

After any edit:

```bash
make check     # validates config, runs tests, re-checks the web app's vendored copy
make sync-vendor   # if check-vendor says the web app's copy is stale
```

Then commit. Plot scripts pick the change up on the next run after a
`git pull` on the plot host. The portal picks it up on the next deploy.

## Add a region

1. Create `config/regions/<key>.yml`; copying a similar region is easiest.
   - Required: `order` (position in the list), `name`, `folder` (the
     directory plots go in), `extent`.
   - Then: `figure`, `sea_surface_height`, `eez`, `variables`
     (per-depth colorbar `limits: [min, max, step]`).
   - Optionally: `currents`, `salinity_max`, `ocean_heat_content`.
2. To show it on the Maps Archive / Overview tabs, add the key to
   `portal.yml` → `map_regions`.
3. To limit which comparisons it gets, add its folder to a comparison's
   `regions.include/exclude` in `comparisons.yml`. For example, ECCOFS only
   covers the Atlantic.
4. Add the key to the plot script or cron `--regions` list that should draw it.

`region_config("<key>")` now returns it, and `region_catalog.ALL_REGIONS`,
the `/regions` editor and the portal's region list include it automatically.

### Region values vs MongoDB overrides

The YAML holds the defaults. MongoDB `hurricanes.region_configs` holds runtime
overrides from the `/regions` editor and the colorbar tuners, and
`db.apply_colorbar_overrides()` applies them on top. MongoDB stores only keys
that differ from the YAML, so YAML edits take effect for everything nobody has
deliberately overridden. The editor shows which keys are overridden.
"Drop override" goes back to the YAML.

`colorbar_tuner.py`'s "Save region YAML" writes limits straight into these
files (`region_yaml.update_limits`), keeping their comments.

Optional, and only needed if you plan to edit region YAML: older seeding
copied every default into MongoDB, so those stored keys hide YAML edits. If
you make all region changes in the `/regions` editor, skip this.
`python scripts/tools/unseed_region_overrides.py` reports what it would remove.
`--apply` backs up the collection, then removes those copies.

## Add a model or comparison

1. In `models.yml`, add the model with its `label` and `filename_token`. The
   token is exactly how it appears in plot filenames, i.e.
   `ds.attrs['model'].lower()` from its loader.
2. In `comparisons.yml` → `comparisons`, add the pair, e.g.
   `rtofs-vs-newmodel: {models: [rtofs, newmodel]}`, with optional
   `regions` / `var_depths` rules.
3. To show it in the portal, add a `map_panels` entry and its key to
   `overview_order`. The panel key becomes the `/api/maps` JSON key and the
   Overview radio button id.

The portal then shows the panel wherever the rules allow, finds the images
through the catalog (or predicts their URLs in legacy mode), and needs no
code change.

Producing the plots is separate. Until the Phase 2 config-driven runner
exists, `rtofs-gofs-cmems-amseas.py` still needs a loader and a plot block
for a new model.

## Portal lists

`portal.yml` holds what used to be hand-typed in `app.py`:
- the map regions and their order;
- the adaptive-sampling regions, variables and models;
- the Argo/glider boxes on the Profiles map (`extent_from: <region key>`
  reuses a region's extent);
- the FVON folders.

## Plot paths and the catalog

`site.yml` sets where plots are written per platform and the published root
and URL. Only figures under `publish_root` are cataloged. Set
`IMC_PLOTS_ROOT` to point a local catalog at a local plot tree. Filename
conventions live in `figure_names.py`. A new plot product needs a parser
there and a sample line in `tests/fixtures/sample_paths.txt` before it
appears in the catalog.
