/* Region config CRUD.
 *
 * These documents drive every production plot, so the emphasis is on making a
 * wrong value hard to enter and easy to undo:
 *   - the extent is drawn on a map, because a lon/lat swap is valid-looking
 *     as numbers but obvious as a box in the wrong hemisphere
 *   - "Check" dry-runs validation and shows the diff before anything is written
 *   - every save is a version, so a mistake is one click to revert
 *
 * Form and JSON tabs edit the SAME object; switching tabs syncs.
 */

const $ = (id) => document.getElementById(id);
let state = { region: null, doc: null, dirty: false, map: null, box: null };

const api = (p) => `${API_BASE}/regions/api/${p}`;
const post = (p, body) => fetch(api(p), {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': CSRF_TOKEN },
  body: JSON.stringify(body),
}).then(async r => ({ ok: r.ok, status: r.status, data: await r.json() }));

/* ------------------------------------------------------------ region list */
async function loadRegions() {
  const { regions } = await (await fetch(api('regions'))).json();
  const q = ($('filter').value || '').toLowerCase();
  const el = $('regionList'); el.innerHTML = '';
  regions.filter(r => !q || (r.region || '').toLowerCase().includes(q)
                        || (r.name || '').toLowerCase().includes(q))
    .forEach(r => {
      const a = document.createElement('a');
      a.className = 'list-group-item list-group-item-action'
                  + (r.region === state.region ? ' active' : '');
      a.innerHTML = `<div class="rg-key">${r.region}</div>`
                  + `<div class="small text-muted">${r.name || ''}`
                  + (r.n_versions ? ` · v${r.n_versions}` : '') + `</div>`;
      a.onclick = () => selectRegion(r.region);
      el.appendChild(a);
    });
}

async function selectRegion(region) {
  if (state.dirty && !confirm('Discard unsaved changes?')) return;
  const r = await fetch(api(`region?region=${encodeURIComponent(region)}`));
  if (!r.ok) { showIssues([`could not load ${region}`], []); return; }
  const j = await r.json();
  state = { ...state, region, doc: j.doc, dirty: false };
  $('regionTitle').textContent = `${j.doc.name || region}`;
  $('versionBadge').style.display = '';
  $('versionBadge').textContent = `${(j.versions || []).length} version(s)`;
  ['validateBtn', 'saveBtn', 'deleteBtn'].forEach(b => $(b).disabled = false);
  renderVersions(j.versions);
  renderForm();
  syncJsonFromDoc();
  showIssues(j.errors, j.warnings);
  loadRegions();
}

/* ------------------------------------------------------------------- form */
function field(label, value, onChange, {type = 'text', step, cls = ''} = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'col-auto';
  wrap.innerHTML = `<label class="form-label small mb-0">${label}</label>`;
  const inp = document.createElement('input');
  inp.className = `form-control form-control-sm ${cls}`;
  inp.type = type; if (step) inp.step = step;
  inp.value = value ?? '';
  inp.oninput = () => { onChange(inp.value); state.dirty = true; };
  wrap.appendChild(inp);
  return wrap;
}

function renderForm() {
  const d = state.doc || {}, body = $('formBody');
  body.innerHTML = '';

  const row = (title) => {
    const h = document.createElement('div');
    h.className = 'col-12 mt-2';
    h.innerHTML = `<div class="fw-semibold small text-uppercase text-muted">${title}</div>`;
    body.appendChild(h);
  };

  row('identity');
  body.appendChild(field('region key', d.region, v => d.region = v.trim()));
  body.appendChild(field('name', d.name, v => d.name = v));
  body.appendChild(field('folder', d.folder, v => d.folder = v));
  const eez = document.createElement('div');
  eez.className = 'col-auto';
  eez.innerHTML = `<label class="form-label small mb-0 d-block">eez</label>
    <div class="form-check form-switch"><input class="form-check-input" type="checkbox"
      ${d.eez ? 'checked' : ''} id="eezChk"></div>`;
  body.appendChild(eez);
  setTimeout(() => { const c = $('eezChk'); if (c) c.onchange = () => {
    d.eez = c.checked; state.dirty = true; }; }, 0);

  row('extent  [lon_min, lon_max, lat_min, lat_max]');
  d.extent = d.extent || [0, 0, 0, 0];
  ['lon min', 'lon max', 'lat min', 'lat max'].forEach((lab, i) => {
    body.appendChild(field(lab, d.extent[i], v => {
      d.extent[i] = v === '' ? null : parseFloat(v); drawExtent();
    }, { type: 'number', step: 'any' }));
  });
  const mapCol = document.createElement('div');
  mapCol.className = 'col-12';
  mapCol.innerHTML = `<div id="extentMap"></div>
    <div class="form-text">The box is drawn from the four numbers above. A
      lon/lat swap is valid as numbers but lands in the wrong hemisphere here.</div>`;
  body.appendChild(mapCol);
  setTimeout(initMap, 0);

  row('colorbar limits  [vmin, vmax, stride]');
  const vars = d.variables = d.variables || {};
  Object.keys(vars).forEach(name => limitList(body, `variables.${name}`, vars[name] || [],
                                              arr => vars[name] = arr));
  limitList(body, 'sea_surface_height', d.sea_surface_height || [],
            arr => d.sea_surface_height = arr);

  if (d.currents && typeof d.currents === 'object') {
    row('currents');
    const c = d.currents;
    const en = document.createElement('div');
    en.className = 'col-auto';
    en.innerHTML = `<label class="form-label small mb-0 d-block">enabled</label>
      <div class="form-check form-switch"><input class="form-check-input" type="checkbox"
        ${c.bool ? 'checked' : ''} id="curChk"></div>`;
    body.appendChild(en);
    setTimeout(() => { const k = $('curChk'); if (k) k.onchange = () => {
      c.bool = k.checked; state.dirty = true; }; }, 0);
    const lbd = c.limits_by_depth || {};
    Object.keys(lbd).forEach(depth => {
      const g = document.createElement('div');
      g.className = 'col-12 limits-row d-flex align-items-end gap-2';
      g.innerHTML = `<div class="small text-muted" style="width:9rem">${depth} m</div>`;
      ['vmin', 'vmax', 'stride'].forEach((lab, i) => {
        g.appendChild(field(lab, lbd[depth][i], v => {
          lbd[depth][i] = parseFloat(v); }, { type: 'number', step: 'any' }));
      });
      body.appendChild(g);
    });
  }
}

function limitList(body, label, items, setter) {
  const head = document.createElement('div');
  head.className = 'col-12 d-flex align-items-center gap-2';
  head.innerHTML = `<span class="small fw-semibold">${label}</span>`;
  const add = document.createElement('button');
  add.className = 'btn btn-sm btn-outline-secondary py-0';
  add.textContent = '+';
  add.onclick = () => { items.push({ depth: 0, limits: [0, 1, 0.1] });
                        setter(items); state.dirty = true; renderForm(); };
  head.appendChild(add);
  body.appendChild(head);

  (items || []).forEach((it, idx) => {
    const g = document.createElement('div');
    g.className = 'col-12 limits-row d-flex align-items-end gap-2';
    g.appendChild(field('depth', it.depth, v => it.depth = parseFloat(v),
                        { type: 'number', step: 'any' }));
    it.limits = it.limits || [0, 1, 0.1];
    ['vmin', 'vmax', 'stride'].forEach((lab, i) =>
      g.appendChild(field(lab, it.limits[i], v => it.limits[i] = parseFloat(v),
                          { type: 'number', step: 'any' })));
    const del = document.createElement('button');
    del.className = 'btn btn-sm btn-outline-danger py-0 ms-2';
    del.textContent = '×';
    del.onclick = () => { items.splice(idx, 1); setter(items);
                          state.dirty = true; renderForm(); };
    g.appendChild(del);
    body.appendChild(g);
  });
}

/* -------------------------------------------------------------- extent map */
function initMap() {
  if (!$('extentMap')) return;
  if (!state.map) {
    state.map = L.map('extentMap', { attributionControl: false })
                 .setView([25, -60], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
                { maxZoom: 8 }).addTo(state.map);
  }
  drawExtent();
  setTimeout(() => state.map.invalidateSize(), 50);
}

function drawExtent() {
  const e = (state.doc || {}).extent;
  if (!state.map || !e || e.some(v => v === null || isNaN(v))) return;
  let [lon0, lon1, lat0, lat1] = e;
  // 0-360 configs exist; Leaflet is happier with -180..180
  const wrap = (x) => x > 180 ? x - 360 : x;
  if (state.box) state.map.removeLayer(state.box);
  state.box = L.rectangle([[lat0, wrap(lon0)], [lat1, wrap(lon1)]],
                          { color: '#0d6efd', weight: 2, fillOpacity: 0.08 })
               .addTo(state.map);
  try { state.map.fitBounds(state.box.getBounds(), { padding: [20, 20], maxZoom: 6 }); }
  catch (_) {}
}

/* ------------------------------------------------------------- json + sync */
function syncJsonFromDoc() { $('jsonBox').value = JSON.stringify(state.doc, null, 2); }
function syncDocFromJson() {
  try { state.doc = JSON.parse($('jsonBox').value); return true; }
  catch (exc) { showIssues([`JSON is not parseable: ${exc.message}`], []); return false; }
}
document.addEventListener('shown.bs.tab', (e) => {
  const t = e.target.getAttribute('data-bs-target');
  if (t === '#jsonPane') syncJsonFromDoc();
  if (t === '#formPane' && syncDocFromJson()) renderForm();
});

/* ------------------------------------------------------------------ issues */
function showIssues(errors, warnings, diff) {
  const el = $('issues'); el.innerHTML = '';
  (errors || []).forEach(m => el.insertAdjacentHTML('beforeend',
    `<div class="small issue-err">✕ ${m}</div>`));
  (warnings || []).forEach(m => el.insertAdjacentHTML('beforeend',
    `<div class="small issue-warn">⚠ ${m}</div>`));
  if (diff && Object.keys(diff).length) {
    el.insertAdjacentHTML('beforeend',
      `<div class="small mt-2 fw-semibold">changes vs saved:</div>`);
    Object.entries(diff).forEach(([path, [a, b]]) =>
      el.insertAdjacentHTML('beforeend',
        `<div class="small font-monospace">${path}: `
        + `<span class="diff-del">${JSON.stringify(a)}</span> → `
        + `<span class="diff-add">${JSON.stringify(b)}</span></div>`));
  }
  if (!el.innerHTML) el.innerHTML = '<div class="small text-success">✓ no problems</div>';
}

/* ---------------------------------------------------------------- versions */
function renderVersions(versions) {
  const sel = $('versionSelect'); sel.innerHTML = '<option value="">current</option>';
  const hist = $('histBody'); hist.innerHTML = '';
  (versions || []).forEach(v => {
    sel.insertAdjacentHTML('beforeend',
      `<option value="${v.version}">v${v.version} · ${v.origin}</option>`);
    const changed = v.changed && Object.keys(v.changed).length
      ? Object.keys(v.changed).slice(0, 6).join(', ') : '—';
    hist.insertAdjacentHTML('beforeend',
      `<div class="hist-row d-flex gap-2 align-items-baseline">
         <b>v${v.version}</b>
         <span class="text-muted">${v.origin}</span>
         <span class="text-muted">${v.edited_by || ''}</span>
         <span class="text-muted small">${(v.created_at || '').slice(0, 16).replace('T', ' ')}</span>
         <span class="font-monospace small flex-grow-1">${changed}</span>
         <button class="btn btn-sm btn-outline-secondary py-0"
                 onclick="revertTo(${v.version})">revert</button>
       </div>`);
  });
  sel.onchange = async () => {
    if (!sel.value) return selectRegion(state.region);
    const v = await (await fetch(api(
      `version?region=${encodeURIComponent(state.region)}&version=${sel.value}`))).json();
    state.doc = v.doc; state.dirty = true;
    renderForm(); syncJsonFromDoc();
    showIssues([], [`viewing v${sel.value} — Save writes it as a new version`]);
  };
}

async function revertTo(version) {
  if (!confirm(`Revert ${state.region} to v${version}?\n\n`
             + `This appends a new version; nothing in the history is removed.`)) return;
  const r = await post('revert', { region: state.region, version });
  if (!r.ok) return showIssues([r.data.error || 'revert failed'], []);
  selectRegion(state.region);
}
window.revertTo = revertTo;

/* ------------------------------------------------------------------ actions */
$('validateBtn').onclick = async () => {
  if (!syncDocFromJson()) return;
  const r = await post('validate', { region: state.region, doc: state.doc });
  showIssues(r.data.errors, r.data.warnings, r.data.diff);
};

$('saveBtn').onclick = async () => {
  if (!syncDocFromJson()) return;
  const chk = await post('validate', { region: state.region, doc: state.doc });
  showIssues(chk.data.errors, chk.data.warnings, chk.data.diff);
  if ((chk.data.errors || []).length) return;
  const n = Object.keys(chk.data.diff || {}).length;
  if (!confirm(`Save ${state.region}? ${n} field(s) change.\n\n`
             + `This takes effect on the next plotting run.`)) return;
  const r = await post('save', { region: state.region, doc: state.doc });
  if (!r.ok) return showIssues([r.data.error || 'save failed'], []);
  state.dirty = false;
  selectRegion(state.region);
};

$('deleteBtn').onclick = async () => {
  if (!confirm(`Drop the MongoDB override for ${state.region}?\n\n`
             + `regions.py defaults will apply again. History is kept.`)) return;
  const r = await post('delete', { region: state.region });
  if (!r.ok) return showIssues([r.data.error || 'delete failed'], []);
  state.region = null; state.doc = null;
  $('regionTitle').textContent = 'Select a region';
  $('formBody').innerHTML = ''; $('jsonBox').value = '';
  loadRegions();
};

$('newBtn').onclick = () => {
  const key = (prompt('New region key (e.g. gulf_stream):') || '').trim();
  if (!key) return;
  state = { ...state, region: key, dirty: true, doc: {
    region: key, name: key, folder: key, extent: [-80, -60, 30, 45], eez: false,
    variables: { temperature: [{ depth: 0, limits: [10, 30, 0.5] }] },
    sea_surface_height: [], figure: { figsize: [14, 8] } } };
  $('regionTitle').textContent = key;
  ['validateBtn', 'saveBtn', 'deleteBtn'].forEach(b => $(b).disabled = false);
  renderForm(); syncJsonFromDoc();
  showIssues([], ['new region — not saved yet. It will only appear in scripts '
                + 'that read region_configs, not in regions.py.']);
};

$('filter').oninput = loadRegions;
window.addEventListener('beforeunload', (e) => {
  if (state.dirty) { e.preventDefault(); e.returnValue = ''; }
});
loadRegions();
