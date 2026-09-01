/* Gulf Stream front editor (served by webapps/model_comparisons).
 *
 * Data comes from MongoDB via /fronts/api/*, not from files. Saving
 * appends a VERSION rather than overwriting, so the automatic
 * full-resolution wall survives every edit.
 *
 * Loads a day's wall + rings from the Flask API, lets Leaflet-Geoman edit
 * them, and posts the result back.
 *
 * Geometry note: the overlay PNGs are rendered in Web Mercator by
 * ioos_model_comparisons/fronts/webmap.py, which is why a plain
 * L.imageOverlay registers correctly here. If you ever swap in an
 * equirectangular image it will sit ~35 km off in latitude at mid-domain.
 */

const map = L.map('map', { preferCanvas: false }).setView([38.5, -63.5], 5);

L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap, &copy; CARTO', maxZoom: 12,
}).addTo(map);

const wallLayer = L.featureGroup().addTo(map);
const ringLayer = L.featureGroup().addTo(map);

let overlay = null;        // current L.imageOverlay
let state = null;          // { stamp, simplify, wall_properties, overlay }
let undoStack = [];
let dirtyRings = false;
let saveModal = null;

const $ = (id) => document.getElementById(id);
const msg = (t, cls = '') => { const e = $('statMsg'); e.textContent = t; e.className = 'ms-auto ' + cls; };

const WALL_STYLE = { color: '#111', weight: 3, opacity: 1 };
const RING_STYLE = {
  warm: { color: '#ff3fd8', weight: 2, fill: false },
  cold: { color: '#3fe0ff', weight: 2, fill: false },
};

/* ---------------------------------------------------------------- undo */
function snapshot() {
  if (!CAN_EDIT) return;
  undoStack.push(JSON.stringify({
    wall: wallLayer.toGeoJSON(),
    rings: ringLayer.toGeoJSON(),
  }));
  if (undoStack.length > 40) undoStack.shift();
  $('undoBtn').disabled = undoStack.length === 0;
}

function undo() {
  if (!CAN_EDIT) return;
  if (!undoStack.length) return;
  const prev = JSON.parse(undoStack.pop());
  drawWall(prev.wall);
  drawRings(prev.rings);
  $('undoBtn').disabled = undoStack.length === 0;
  updateStats();
}

/* -------------------------------------------------------------- drawing */
function drawWall(fc) {
  wallLayer.clearLayers();
  L.geoJSON(fc, {
    style: WALL_STYLE,
    onEachFeature: (f, layer) => {
      layer.options.className = 'wall-halo';
      layer.feature = f;
      wallLayer.addLayer(layer);
      if (CAN_EDIT) {
        layer.on('pm:edit pm:dragend pm:vertexadded pm:vertexremoved', () => {
          snapshot(); updateStats();
        });
      }
    },
  });
}

function drawRings(fc) {
  ringLayer.clearLayers();
  L.geoJSON(fc, {
    style: (f) => RING_STYLE[f.properties.ring_kind] || RING_STYLE.warm,
    onEachFeature: (f, layer) => {
      layer.feature = f;
      const p = f.properties || {};
      const days = p.days_tracked == null ? '?' : p.days_tracked;
      layer.bindTooltip(
        `${p.ring_kind || '?'} ring · r=${p.radius_km ?? '?'} km · tracked ${days} d`,
        { sticky: true });
      ringLayer.addLayer(layer);
      if (CAN_EDIT) {
        layer.on('pm:edit pm:dragend pm:vertexadded pm:vertexremoved', () => {
          dirtyRings = true;
          if (layer.feature) layer.feature.properties.dirty = true;
          snapshot(); updateStats();
        });
      }
    },
  });
  ringLayer.setStyle && ringLayer.eachLayer(l => l.setStyle(
    RING_STYLE[(l.feature.properties || {}).ring_kind] || RING_STYLE.warm));
}

/* ------------------------------------------------------------ OISST layer */
/* The GOES/SLA overlays are pre-rendered PNGs stored in Mongo by the nightly
 * digitizer. OISST instead renders per request, so date and colour scale can
 * be changed live. The slice is cached server-side, so only the first request
 * for a date hits the network — changing cmap/limits is a ~40 ms re-render. */
const oisst = { meta: null, ready: false };

function oisstParams() {
  const p = new URLSearchParams({ date: $('oDate').value, cmap: $('oCmap').value });
  if (state && state.stamp) p.set('stamp', state.stamp);
  ['Vmin', 'Vmax', 'Stride'].forEach(k => {
    const v = $('o' + k).value;
    if (v !== '') p.set(k.toLowerCase(), v);
  });
  return p;
}

function drawColorbar() {
  const c = $('oBar'); if (!c) return;
  const ctx = c.getContext('2d');
  const vmin = parseFloat($('oVmin').value), vmax = parseFloat($('oVmax').value);
  ctx.clearRect(0, 0, c.width, c.height);
  if (!isFinite(vmin) || !isFinite(vmax) || vmax <= vmin) return;
  // The bar is a scaled-down copy of the same image endpoint, so what you see
  // in the legend is literally the colormap being drawn on the map.
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, c.width, 18);
    ctx.fillStyle = '#333'; ctx.font = '10px system-ui';
    ctx.fillText(vmin.toFixed(1), 0, 30);
    ctx.fillText(vmax.toFixed(1), c.width - 24, 30);
  };
  const p = oisstParams(); p.set('bar', '1');
  img.src = `${API_BASE}/fronts/api/oisst?${p.toString()}`;
}

async function loadOisstMeta() {
  if (oisst.ready) return;
  const p = new URLSearchParams();
  if (state && state.stamp) p.set('stamp', state.stamp);
  if ($('oDate').value) p.set('date', $('oDate').value);
  const r = await fetch(`${API_BASE}/fronts/api/oisst/meta?${p}`);
  if (!r.ok) { msg('OISST metadata unavailable', 'stat-warn'); return; }
  oisst.meta = await r.json();
  const sel = $('oCmap');
  if (!sel.options.length) {
    (oisst.meta.colormaps || []).forEach(c => {
      const o = document.createElement('option');
      o.value = c.id; o.textContent = c.label;
      sel.appendChild(o);
    });
    sel.value = 'cmo.thermal';
  }
  const av = oisst.meta.available;
  if (av) { $('oDate').min = av.first; $('oDate').max = av.last;
            if (!$('oDate').value) $('oDate').value = av.last; }
  if (oisst.meta.stats && $('oVmin').value === '') {
    $('oVmin').value = oisst.meta.stats.suggest_vmin;
    $('oVmax').value = oisst.meta.stats.suggest_vmax;
    $('oStride').value = 0.5;
  }
  oisst.ready = true;
}

async function autoLimits() {
  const p = new URLSearchParams({ date: $('oDate').value });
  if (state && state.stamp) p.set('stamp', state.stamp);
  const r = await fetch(`${API_BASE}/fronts/api/oisst/meta?${p}`);
  if (!r.ok) return;
  const j = await r.json();
  if (j.stats) {
    $('oVmin').value = j.stats.suggest_vmin;
    $('oVmax').value = j.stats.suggest_vmax;
    setOverlay('oisst');
  }
}

function stepDay(n) {
  const d = new Date($('oDate').value + 'T00:00:00Z');
  if (isNaN(d)) return;
  d.setUTCDate(d.getUTCDate() + n);
  const iso = d.toISOString().slice(0, 10);
  const av = oisst.meta && oisst.meta.available;
  if (av && (iso < av.first || iso > av.last)) return;
  $('oDate').value = iso;
  setOverlay('oisst');
}

/* ---------------------------------------------------------------- overlay */
async function setOverlay(field) {
  const bar = $('oisstBar');
  if (bar) bar.style.setProperty('display', field === 'oisst' ? 'flex' : 'none', 'important');
  if (overlay) { map.removeLayer(overlay); overlay = null; }
  if (!state || field === 'none') return;

  let url, bounds;
  if (field === 'oisst') {
    await loadOisstMeta();
    if (!$('oDate').value) return;
    const ext = (oisst.meta && oisst.meta.extent)
             || (state.overlay && state.overlay.extent);
    if (!ext) { msg('no extent for the OISST layer', 'stat-warn'); return; }
    bounds = [[ext[2], ext[0]], [ext[3], ext[1]]];
    url = `${API_BASE}/fronts/api/oisst?${oisstParams().toString()}`;
    msg('loading OISST…');
  } else {
    const ext = state.overlay && state.overlay.extent;
    if (!ext) { msg('no overlay for this day (re-run the digitizer)', 'stat-warn'); return; }
    bounds = [[ext[2], ext[0]], [ext[3], ext[1]]];
    url = `${API_BASE}/fronts/api/overlay?stamp=${state.stamp}&field=${field}`;
  }

  overlay = L.imageOverlay(url, bounds, {
    opacity: parseFloat($('opacity').value), interactive: false,
  }).addTo(map);
  overlay.on('load', () => { if (field === 'oisst') { msg(''); drawColorbar(); } });
  overlay.on('error', () => msg(
    field === 'oisst' ? 'OISST unavailable for that date' : 'overlay failed', 'stat-bad'));
  overlay.bringToFront();
  wallLayer.bringToFront();
  ringLayer.bringToFront();
}

/* ---------------------------------------------------------------- stats */
function countVertices(fc) {
  return (fc.features || []).reduce((n, f) => {
    const g = f.geometry || {};
    if (g.type === 'LineString') return n + g.coordinates.length;
    if (g.type === 'Polygon') return n + g.coordinates[0].length;
    return n;
  }, 0);
}

function updateStats() {
  if (!state) return;
  const wf = wallLayer.toGeoJSON();
  const n = countVertices(wf);
  const orig = state.simplify.original_n_vertices;
  $('statVertices').textContent =
    `${wf.features.length} piece(s) · ${n} vertices (auto had ${orig})`;

  const p = state.wall_properties || {};
  const qc = p.qc_pass === false ? 'QC FAIL' : (p.qc_pass === true ? 'QC pass' : 'QC ?');
  $('statQc').innerHTML = `<span class="${p.qc_pass === false ? 'stat-bad' : 'stat-good'}">${qc}</span>` +
    (p.support_frac != null ? ` · support ${(p.support_frac * 100).toFixed(0)}%` : '') +
    (p.qc_stale ? ` · <span class="stat-warn">qc_stale</span>` : '');

  $('statEdited').innerHTML = p.edited_by_hand
    ? `<span class="stat-warn">hand-edited ${p.edited_at || ''}</span>`
    : 'auto';
}

/* ----------------------------------------------------------------- load */
async function loadDays() {
  const r = await fetch(`${API_BASE}/fronts/api/days`);
  const { days } = await r.json();
  const sel = $('daySelect');
  sel.innerHTML = '';
  days.forEach(d => {
    const o = document.createElement('option');
    o.value = d.stamp;
    o.textContent = `${d.stamp}${d.edited_by_hand ? '  ✎' : ''}${d.qc_pass === false ? '  ⚠' : ''}`;
    sel.appendChild(o);
  });
  if (days.length) await loadDay(days[0].stamp);
  else msg('no digitized days found in outputs/gulf_stream_fronts/', 'stat-bad');
}

async function loadDay(stamp) {
  msg('loading…');
  const r = await fetch(`${API_BASE}/fronts/api/features?stamp=${encodeURIComponent(stamp)}`);
  if (!r.ok) { msg(`load failed: ${(await r.json()).error}`, 'stat-bad'); return; }
  state = await r.json();
  undoStack = []; dirtyRings = false;
  $('undoBtn').disabled = true;

  drawWall(state.wall);
  drawRings(state.rings);
  setOverlay(document.querySelector('input[name=field]:checked').value);
  ringLayer.eachLayer(l => l.setStyle(RING_STYLE[(l.feature.properties||{}).ring_kind] || RING_STYLE.warm));
  if (!$('showRings').checked) map.removeLayer(ringLayer);

  const b = wallLayer.getBounds();
  if (b.isValid()) map.fitBounds(b, { padding: [30, 30] });
  updateStats();
  msg('');
}

/* ----------------------------------------------------------------- save */
function openSaveDialog() {
  if (!CAN_EDIT) return;
  if (!state) return;
  const wf = wallLayer.toGeoJSON();
  const n = countVertices(wf);
  const orig = state.simplify.original_n_vertices;
  $('saveSummary').textContent =
    `Saving ${wf.features.length} wall piece(s), ${n} vertices, for ${state.stamp}.` +
    (dirtyRings ? ' Ring geometry was changed.' : '');
  $('simplifyWarning').innerHTML =
    `This wall was simplified to <b>${state.simplify.edit_n_vertices}</b> vertices for editing, ` +
    `down from <b>${orig}</b>. Saving writes the simplified geometry — the ` +
    `full-resolution original is kept only in <code>auto_backup/</code>.` +
    (dirtyRings
      ? `<br><br><b>Rings were edited.</b> Tomorrow's run matches rings against this file, ` +
        `so <code>days_tracked</code> persistence counts will reset or shift for changed rings.`
      : '');
  saveModal.show();
}

async function doSave() {
  if (!CAN_EDIT) return;
  saveModal.hide();
  msg('saving…');
  const payload = {
    stamp: state.stamp,
    wall: wallLayer.toGeoJSON(),
    rings: dirtyRings ? ringLayer.toGeoJSON() : null,
    original_n_vertices: state.simplify.original_n_vertices,
    tolerance_deg: state.simplify.tolerance_deg,
  };
  const r = await fetch(`${API_BASE}/fronts/api/save`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': CSRF_TOKEN },
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  if (!r.ok) { msg(`save failed: ${j.error}`, 'stat-bad'); return; }
  msg(`saved ${j.n_pieces} piece(s), ${j.n_vertices} vertices`, 'stat-good');
  await loadDay(state.stamp);
  await refreshDayLabels();
}

async function refreshDayLabels() {
  const cur = $('daySelect').value;
  await loadDays();
  $('daySelect').value = cur;
}

async function doRevert() {
  if (!CAN_EDIT) return;
  if (!state) return;
  if (!confirm(`Revert ${state.stamp} to the automatic wall?\n\n`
    + `This appends a new version — your hand edit stays in the version `
    + `history and can be viewed again, it just stops being current.`)) return;
  const r = await fetch(`${API_BASE}/fronts/api/revert`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': CSRF_TOKEN },
    body: JSON.stringify({ stamp: state.stamp }),
  });
  const j = await r.json();
  if (!r.ok) { msg(`revert failed: ${j.error}`, 'stat-bad'); return; }
  msg(`reverted to automatic v${j.restored_from} (now v${j.version})`, 'stat-good');
  await loadDay(state.stamp);
}

/* ------------------------------------------------------------ geoman UI */
if (CAN_EDIT) {
  map.pm.addControls({
    position: 'topleft',
    drawMarker: false, drawCircle: false, drawCircleMarker: false,
    drawRectangle: false, drawText: false, drawPolygon: true,
    drawPolyline: true, editMode: true, dragMode: true,
    cutPolygon: false, removalMode: true, rotateMode: false,
  });
  map.pm.setGlobalOptions({ snappable: true, snapDistance: 12, allowSelfIntersection: true });

  // a newly drawn line joins the wall; a newly drawn polygon joins the rings
  map.on('pm:create', (e) => {
    snapshot();
    const gj = e.layer.toGeoJSON();
    map.removeLayer(e.layer);
    if (gj.geometry.type === 'LineString') {
      gj.properties = { kind: 'wall', piece: wallLayer.getLayers().length };
      const merged = wallLayer.toGeoJSON(); merged.features.push(gj);
      drawWall(merged);
    } else if (gj.geometry.type === 'Polygon') {
      gj.properties = { kind: 'ring', ring_kind: 'warm', dirty: true };
      const merged = ringLayer.toGeoJSON(); merged.features.push(gj);
      dirtyRings = true;
      drawRings(merged);
    }
    updateStats();
  });

  map.on('pm:remove', (e) => {
    // Geoman already detached the layer; record the pre-removal state so undo
    // can bring it back, and note whether a ring changed
    if (e.layer && e.layer.feature && e.layer.feature.properties
        && e.layer.feature.properties.kind === 'ring') dirtyRings = true;
    updateStats();
  });
  map.on('pm:drawstart pm:removelayerstart', snapshot);
}

/* ------------------------------------------------------------- bindings */
$('daySelect').addEventListener('change', (e) => loadDay(e.target.value));
document.querySelectorAll('input[name=field]').forEach(el =>
  el.addEventListener('change', () => setOverlay(el.value)));
['oCmap', 'oVmin', 'oVmax', 'oStride'].forEach(id => {
  const el = $(id); if (!el) return;
  el.addEventListener('change', () => setOverlay('oisst'));
});
if ($('oDate'))   $('oDate').addEventListener('change', () => setOverlay('oisst'));
if ($('oPrev'))   $('oPrev').onclick = () => stepDay(-1);
if ($('oNext'))   $('oNext').onclick = () => stepDay(1);
if ($('oAuto'))   $('oAuto').onclick = autoLimits;
$('opacity').addEventListener('input', (e) => { if (overlay) overlay.setOpacity(parseFloat(e.target.value)); });
$('showRings').addEventListener('change', (e) =>
  e.target.checked ? map.addLayer(ringLayer) : map.removeLayer(ringLayer));
$('undoBtn').addEventListener('click', undo);
$('saveBtn').addEventListener('click', openSaveDialog);
$('saveConfirm').addEventListener('click', doSave);
$('revertBtn').addEventListener('click', doRevert);
if (CAN_EDIT) {
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo(); }
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); openSaveDialog(); }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  if (CAN_EDIT) saveModal = new bootstrap.Modal($('saveModal'));
  loadDays();
});
