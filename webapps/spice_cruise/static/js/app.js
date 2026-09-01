/* ══════════════════════════════════════════════════════════
   SPICE Cruise Data Support Portal – Client-side JS
   Scoped entirely to the Tropical Western Atlantic (REGION).
   All API calls go to Flask endpoints; zero page reloads.
   ══════════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  // Data Archive
  mapSource: 'satellite',  // 'satellite' | 'glider' | 'model'
  mapDate: document.getElementById('mapDate').value,
  mapTime: '00Z',
  mapVarDepth: '',

  // Satellite / Glider (within the Data Archive tab)
  satDate: document.getElementById('satDate').value,
  satVariable: '',
  satCatalog: {},   // {variable: [ {filename, time, url}, ... ]} for satDate
  satIdx: 0,        // index into satCatalog[satVariable]

  // RT Glider Plots
  rtGliders: [],       // [glider_id, ...]
  rtGliderId: '',
  rtCatalog: {},        // {category: {period: [ {variable, filename, url}, ... ]}}
  rtCategory: 'profiles',  // 'profiles' | 'xsection' | 'TS'
  rtPeriod: 'synoptic',    // 'synoptic' | 'last_24h' | 'last_48h'
  rtVarIdx: 0,
};

// ─── Utilities ───────────────────────────────────────────────────────────────

function addDays(dateStr, days) {
  const d = new Date(dateStr + 'T12:00:00');
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

// 6-hourly time steps matching server cadence
const TIME_STEPS = ['00Z', '06Z', '12Z', '18Z'];

function stepTimeHelper(direction, dateStr, timeStr) {
  const idx = TIME_STEPS.indexOf(timeStr);
  if (direction === 'forward') {
    if (idx < TIME_STEPS.length - 1) {
      return { date: dateStr, time: TIME_STEPS[idx + 1] };
    } else {
      return { date: addDays(dateStr, 1), time: TIME_STEPS[0] };
    }
  } else {
    if (idx > 0) {
      return { date: dateStr, time: TIME_STEPS[idx - 1] };
    } else {
      return { date: addDays(dateStr, -1), time: TIME_STEPS[TIME_STEPS.length - 1] };
    }
  }
}

function stepTime(direction) {
  const next = stepTimeHelper(direction, state.mapDate, state.mapTime);
  state.mapTime = next.time;
  state.mapDate = next.date;

  document.getElementById('mapDate').value = state.mapDate;
  document.getElementById('mapTimeDisplay').textContent = state.mapTime;
}

function setLoading(containerId, text = 'Loading…') {
  document.getElementById(containerId).innerHTML = `
    <div class="loading-state">
      <div class="spinner-border text-primary" style="width:2rem;height:2rem;" role="status">
        <span class="visually-hidden">Loading</span>
      </div>
      <span class="fw-500">${text}</span>
    </div>`;
}

function setUnavailable(containerId, label) {
  document.getElementById(containerId).innerHTML = `
    <div class="unavailable-state">
      <i class="bi bi-exclamation-circle" style="font-size:2.5rem;color:#adb5bd;"></i>
      <p class="mb-0 mt-1 text-muted">${label}: image not available for this time</p>
    </div>`;
}

// Date-specific version for the satellite/glider catalog browser, where the
// variable dropdown always lists the full product set — this is what tells a
// user "this product just has nothing on this particular date" as they step
// back through dates, rather than the variable silently vanishing.
function setSatelliteUnavailable(containerId, variable, dateStr) {
  document.getElementById(containerId).innerHTML = `
    <div class="unavailable-state">
      <i class="bi bi-exclamation-circle" style="font-size:2.5rem;color:#adb5bd;"></i>
      <p class="mb-0 mt-1 text-muted">No "${variable}" images available for ${dateStr}</p>
    </div>`;
}

function setImage(containerId, url) {
  document.getElementById(containerId).innerHTML = `
    <img src="${url}" class="plot-image" alt="Model comparison plot"
         onerror="setUnavailable('${containerId}', 'Image')" />`;
}

async function downloadPlot(containerId, imageId) {
  let img;
  if (imageId) {
    img = document.getElementById(imageId);
  } else {
    const container = document.getElementById(containerId);
    if (container) {
      img = container.querySelector('img');
    }
  }

  if (img && img.src && !img.src.includes('undefined') && !img.classList.contains('d-none')) {
    const downloadUrl = `${API_BASE}/api/download?url=` + encodeURIComponent(img.src);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = downloadUrl;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } else {
    showToast("No image available to download.");
  }
}

// ─── Image Lightbox ────────────────────────────────────────────────────────
// Delegated to the document so it works for every plot image on the page,
// including ones re-created via setImage()'s innerHTML replacement.
const lightbox = document.getElementById('imageLightbox');
const lightboxImage = document.getElementById('lightboxImage');

function openLightbox(src, alt) {
  if (!src) return;
  lightboxImage.src = src;
  lightboxImage.alt = alt || '';
  lightbox.classList.add('show');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  lightbox.classList.remove('show');
  lightboxImage.src = '';
  document.body.style.overflow = '';
}

document.addEventListener('click', (e) => {
  const img = e.target.closest('.plot-image');
  if (img && !img.classList.contains('d-none') && img.src) {
    openLightbox(img.src, img.alt);
    return;
  }
  if (e.target === lightbox) {
    closeLightbox();
  }
});

document.getElementById('lightboxClose').addEventListener('click', closeLightbox);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && lightbox.classList.contains('show')) {
    closeLightbox();
  }
  if (e.key === 'Escape' && textLightbox.classList.contains('show')) {
    closeTextLightbox();
  }
});

// ─── Text Lightbox ─────────────────────────────────────────────────────────
// Some products (e.g. eddy_trajectory) publish a daily plain-text summary
// alongside their plots — shown here instead of trying to shoehorn it into
// the image viewer.
const textLightbox = document.getElementById('textLightbox');
const textLightboxTitle = document.getElementById('textLightboxTitle');
const textLightboxContent = document.getElementById('textLightboxContent');
const textLightboxDownload = document.getElementById('textLightboxDownload');
let currentForecastTextUrl = null;

function openTextLightbox(title, content, downloadUrl) {
  textLightboxTitle.textContent = title;
  textLightboxContent.textContent = content;
  currentForecastTextUrl = downloadUrl || null;
  textLightbox.classList.add('show');
  document.body.style.overflow = 'hidden';
}

function closeTextLightbox() {
  textLightbox.classList.remove('show');
  document.body.style.overflow = '';
}

async function openForecastText(dateStr, variable) {
  if (!variable) return;
  const url = `${API_BASE}/api/satellite-text?date=${encodeURIComponent(dateStr)}&variable=${encodeURIComponent(variable)}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    if (!data.available) {
      showToast(`No forecast data available for ${variable} on ${dateStr}.`);
      return;
    }
    openTextLightbox(`${variable} — ${dateStr}`, data.content, data.url);
  } catch (err) {
    console.error(err);
    showToast('Network error while fetching forecast data.');
  }
}

document.getElementById('textLightboxClose').addEventListener('click', closeTextLightbox);

textLightboxDownload.addEventListener('click', () => {
  if (!currentForecastTextUrl) return;
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = `${API_BASE}/api/download?url=` + encodeURIComponent(currentForecastTextUrl);
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

document.addEventListener('click', (e) => {
  if (e.target === textLightbox) {
    closeTextLightbox();
  }
});

function showToast(msg) {
  document.getElementById('alertToastMsg').textContent = msg;
  const el = document.getElementById('alertToast');
  const toast = bootstrap.Toast.getOrCreateInstance(el, { delay: 4000 });
  toast.show();
}

// ─── Variable/Depth dropdown population (Model Comparisons) ──────────────────

function populateVarDepth() {
  const info = REGION_INFO[REGION];
  if (!info) return;
  const sel = document.getElementById('mapVarDepth');
  sel.innerHTML = '';
  for (const v of info.variables) {
    if (v === 'ocean_heat_content') {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = "Ocean Heat Content";
      sel.appendChild(opt);
    } else {
      for (const d of info.depths) {
        const opt = document.createElement('option');
        opt.value = `${v}_${d}`;
        opt.textContent = `${v} @ ${d}`;
        sel.appendChild(opt);
      }
    }
  }
  state.mapVarDepth = sel.value;
}

// ─── Maps – Model Comparisons Toggle State ───────────────────────────────────

const mapModels = [
  { key: 'copernicus', label: 'RTOFS vs. Copernicus (CMEMS)' },
  { key: 'espc',       label: 'RTOFS vs. ESPC' },
];
let mapCurrentModelIdx = 0;
let mapCachedUrls = { copernicus: null, espc: null, espc_cmems: null };

function renderMapModel() {
  const model = mapModels[mapCurrentModelIdx];
  const url = mapCachedUrls[model.key];

  // Update tab active states
  mapModels.forEach((m, i) => {
    const tab = document.getElementById(`mapTab-${m.key}`);
    if (tab) tab.classList.toggle('active', i === mapCurrentModelIdx);
  });

  if (url) {
    setImage('mapImageContainer', url);
  } else {
    setUnavailable('mapImageContainer', model.label);
  }
}

function stepMapModel(direction) {
  mapCurrentModelIdx = (mapCurrentModelIdx + direction + mapModels.length) % mapModels.length;
  renderMapModel();
}

// Wire up the click handlers for the (static) model tabs
mapModels.forEach((m, i) => {
  const btn = document.getElementById(`mapTab-${m.key}`);
  if (btn) {
    btn.addEventListener('click', () => {
      mapCurrentModelIdx = i;
      renderMapModel();
    });
  }
});

const apiCache = {};

function preloadImage(url) {
  if (!url) return;
  const img = new Image();
  img.src = url;
}

function preloadAdjacentMaps() {
  const directions = ['back', 'forward'];
  directions.forEach(async (dir) => {
    const next = stepTimeHelper(dir, state.mapDate, state.mapTime);
    const params = new URLSearchParams({
      variable_depth: state.mapVarDepth,
      date: next.date,
      time: next.time,
    });
    const url = `${API_BASE}/api/maps?${params.toString()}`;
    if (!apiCache[url]) {
      try {
        const res = await fetch(url);
        const data = await res.json();
        apiCache[url] = data;
        if (data.copernicus && data.copernicus.available) preloadImage(data.copernicus.url);
        if (data.espc && data.espc.available) preloadImage(data.espc.url);
        if (data.espc_cmems && data.espc_cmems.available) preloadImage(data.espc_cmems.url);
      } catch (e) {
        // silently fail on preload errors
      }
    } else {
      const data = apiCache[url];
      if (data.copernicus && data.copernicus.available) preloadImage(data.copernicus.url);
      if (data.espc && data.espc.available) preloadImage(data.espc.url);
      if (data.espc_cmems && data.espc_cmems.available) preloadImage(data.espc_cmems.url);
    }
  });
}

async function loadMaps() {
  state.mapDate = document.getElementById('mapDate').value;
  state.mapVarDepth = document.getElementById('mapVarDepth').value;

  const btn = document.getElementById('mapUpdateBtn');
  btn.disabled = true;

  const params = new URLSearchParams({
    variable_depth: state.mapVarDepth,
    date: state.mapDate,
    time: state.mapTime,
  });

  const cacheKey = `${API_BASE}/api/maps?${params.toString()}`;

  const renderData = (data) => {
    mapCachedUrls.copernicus = data.copernicus && data.copernicus.available ? data.copernicus.url : null;
    mapCachedUrls.espc       = data.espc && data.espc.available       ? data.espc.url       : null;
    mapCachedUrls.espc_cmems = data.espc_cmems && data.espc_cmems.available ? data.espc_cmems.url : null;

    renderMapModel();

    if (!mapCachedUrls.copernicus && !mapCachedUrls.espc && !mapCachedUrls.espc_cmems) {
      showToast('No images available for the selected date/time.');
    }
  };

  if (apiCache[cacheKey]) {
    renderData(apiCache[cacheKey]);
    btn.disabled = false;
    preloadAdjacentMaps();
    return;
  }

  setLoading('mapImageContainer', 'Fetching maps…');

  try {
    const res = await fetch(cacheKey);
    const data = await res.json();
    apiCache[cacheKey] = data;
    renderData(data);
    preloadAdjacentMaps();
  } catch (err) {
    console.error(err);
    mapCachedUrls = { copernicus: null, espc: null, espc_cmems: null };
    setUnavailable('mapImageContainer', 'Maps');
    showToast('Network error while fetching images.');
  } finally {
    btn.disabled = false;
  }
}


// ─── Satellite vs. Glider variable classification ────────────────────────────
// The feed's variable folders mix true satellite/ocean products with
// per-glider diagnostic plots named "{glider_id}_{diagnostic}" (e.g.
// ru29_turner, ru29_CT). A satellite-name allowlist used to gate this, but
// that broke the moment a new satellite product (eddy_trajectory, nfai)
// showed up and silently fell through to "glider" by default. Classification
// now goes the other way: match against the currently-known glider IDs
// (sourced from the same RT-glider listing used elsewhere in the app — see
// loadKnownGliderIds) plus known staircase-only platform prefixes, and default
// anything else to satellite. New satellite products need no changes at all
// since they're the default bucket now.
let knownGliderIds = [];

// Some staircase_analyses products are generated for in-situ platforms that
// do not have matching RT glider deployment folders, or whose staircase folder
// names differ from the RT deployment ID. Keep those aliases here so they stay
// in the Glider source instead of falling through to Satellite.
const STAIRCASE_GLIDER_MATCHERS = [
  /^falkor(?:_|$)/i,
  /(?:^|_)sea044(?:_|$)/i,
];

// RT deployment IDs are "{short_name}<sep>{deployment date}" (currently
// "ru29-20260623T2102", hyphen-separated, but nothing guarantees future
// deployments won't use an underscore instead — e.g. "ru20_20260812"), while
// the staircase_analyses diagnostics use just the short platform name as
// their prefix ("ru29_turner"). Strip whatever separator precedes an 8-digit
// deployment date, rather than assuming one specific separator.
function gliderShortName(fullId) {
  const m = fullId.match(/^(.+?)[-_]\d{8}/);
  return m ? m[1] : fullId;
}

function isGliderVariable(name) {
  const normalizedName = name.toLowerCase();
  return STAIRCASE_GLIDER_MATCHERS.some(re => re.test(name)) ||
    knownGliderIds.some(id => {
      const normalizedId = id.toLowerCase();
      return normalizedName === normalizedId || normalizedName.startsWith(`${normalizedId}_`);
    });
}

function filterVariablesForSource(variableNames, source) {
  return variableNames
    .filter(v => source === 'glider' ? isGliderVariable(v) : !isGliderVariable(v))
    .sort();
}

function sourceLabel(source) {
  return source === 'glider' ? 'Glider' : 'Satellite';
}

// ─── Master variable list ─────────────────────────────────────────────────────
// The variable dropdowns default to this stable, full product list (every
// variable ever seen, from the server) rather than just what today's/this
// date's catalog happens to contain — so stepping back through dates keeps
// the same product selected and shows "no images for this date" instead of
// the option disappearing. Falls back to the current catalog's own keys until
// this loads (fire-and-forget from init, typically resolves in a second or two).
let masterSatVariables = [];

function variableUniverse(catalog) {
  return masterSatVariables.length ? masterSatVariables : Object.keys(catalog);
}

async function loadMasterSatVariables() {
  try {
    const res = await fetch(`${API_BASE}/api/satellite-master-variables`);
    const data = await res.json();
    masterSatVariables = data.variables || [];
  } catch (err) {
    console.error(err);
    return;
  }

  // Upgrade the Data Archive dropdown if it's currently populated now that
  // the full list is available.
  if (state.mapSource !== 'model') {
    populateSatVariableSelect(state.satVariable);
    renderSatelliteImage();
  }
}

// Fetches the same glider-ID list the RT Glider Plots tab uses, purely to
// drive isGliderVariable() above. Fire-and-forget from init like
// loadMasterSatVariables() — until it resolves, everything is classified as
// satellite (the default bucket), then the Data Archive dropdown gets
// reclassified in place.
async function loadKnownGliderIds() {
  try {
    const res = await fetch(`${API_BASE}/api/gliders-rt`);
    const data = await res.json();
    knownGliderIds = (data.gliders || []).map(gliderShortName);
  } catch (err) {
    console.error(err);
    return;
  }

  if (state.mapSource !== 'model') {
    populateSatVariableSelect(state.satVariable);
    renderSatelliteImage();
  }
}


// ─── Data Archive – Satellite / Glider ────────────────────────────────────────
// Timestamps here are irregular (not a fixed 6-hour grid) and the set of
// variables changes day to day, so this browses a per-date catalog fetched
// from the server rather than constructing predictable URLs client-side.

function renderSatelliteImage() {
  const images = state.satCatalog[state.satVariable] || [];
  const img = images[state.satIdx];

  document.getElementById('satDateLabel').textContent = state.satDate;
  document.getElementById('satVariableLabel').textContent = state.satVariable || '—';
  document.getElementById('mapForecastBtn').classList.toggle('d-none', state.satVariable !== 'eddy_trajectory');

  if (!img) {
    document.getElementById('satTimeDisplay').textContent = '--:--';
    setSatelliteUnavailable('mapImageContainer', state.satVariable || sourceLabel(state.mapSource), state.satDate);
    return;
  }
  document.getElementById('satTimeDisplay').textContent = img.time + ' UTC';
  setImage('mapImageContainer', img.url);
}

// The dropdown always lists the full master product set (see loadMasterSatVariables)
// so a chosen variable stays selected even on dates where it has no images —
// renderSatelliteImage() is what shows "no images for this date" in that case.
function populateSatVariableSelect(preferredVariable) {
  const sel = document.getElementById('satVariable');
  const noMsg = document.getElementById('satNoDataMsg');
  const variables = filterVariablesForSource(variableUniverse(state.satCatalog), state.mapSource);
  sel.innerHTML = '';

  if (variables.length === 0) {
    noMsg.classList.remove('d-none');
    state.satVariable = '';
    document.getElementById('satTimeDisplay').textContent = '--:--';
    setSatelliteUnavailable('mapImageContainer', sourceLabel(state.mapSource), state.satDate);
    return;
  }
  noMsg.classList.add('d-none');

  for (const v of variables) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
  state.satVariable = variables.includes(preferredVariable) ? preferredVariable : variables[0];
  sel.value = state.satVariable;
  state.satIdx = 0;
}

async function loadSatelliteCatalog(preferredVariable) {
  state.satDate = document.getElementById('satDate').value;
  const cacheKey = `${API_BASE}/api/satellite-catalog?date=${state.satDate}`;

  const applyData = (data) => {
    state.satCatalog = data || {};
    populateSatVariableSelect(preferredVariable || state.satVariable);
    renderSatelliteImage();
  };

  if (apiCache[cacheKey]) {
    applyData(apiCache[cacheKey]);
    return;
  }

  setLoading('mapImageContainer', 'Fetching catalog…');
  try {
    const res = await fetch(cacheKey);
    const data = await res.json();
    apiCache[cacheKey] = data;
    applyData(data);
  } catch (err) {
    console.error(err);
    setUnavailable('mapImageContainer', sourceLabel(state.mapSource));
    showToast('Network error while fetching catalog.');
  }
}

// Step to the next/previous available image for the current variable, crossing
// into the adjacent day's catalog when the current day's list is exhausted.
// The variable itself never changes here — only the date does — so repeatedly
// stepping stays on the same product across however many empty dates it takes
// to find (or not find) the next image.
async function stepSatelliteTime(direction) {
  const images = state.satCatalog[state.satVariable] || [];
  const nextIdx = state.satIdx + direction;

  if (nextIdx >= 0 && nextIdx < images.length) {
    state.satIdx = nextIdx;
    renderSatelliteImage();
    return;
  }

  state.satDate = addDays(state.satDate, direction > 0 ? 1 : -1);
  document.getElementById('satDate').value = state.satDate;

  const cacheKey = `${API_BASE}/api/satellite-catalog?date=${state.satDate}`;
  let data = apiCache[cacheKey];
  if (!data) {
    setLoading('mapImageContainer', 'Fetching catalog…');
    try {
      const res = await fetch(cacheKey);
      data = await res.json();
      apiCache[cacheKey] = data;
    } catch (err) {
      console.error(err);
      setUnavailable('mapImageContainer', sourceLabel(state.mapSource));
      return;
    }
  }

  state.satCatalog = data || {};
  const images2 = state.satCatalog[state.satVariable] || [];
  state.satIdx = direction > 0 ? 0 : Math.max(images2.length - 1, 0);
  populateSatVariableSelect(state.satVariable);
  renderSatelliteImage();
}

function setMapSource(source) {
  state.mapSource = source;
  const isModel = source === 'model';

  document.getElementById('mapModelControls').classList.toggle('d-none', !isModel);
  document.getElementById('mapSatControls').classList.toggle('d-none', isModel);
  document.getElementById('mapModelHeader').classList.toggle('d-none', !isModel);
  document.getElementById('mapSatHeader').classList.toggle('d-none', isModel);
  document.getElementById('mapLegendCard').classList.toggle('d-none', !isModel);
  document.getElementById('mapSatAboutCard').classList.toggle('d-none', isModel);

  if (isModel) {
    const hasData = Object.values(mapCachedUrls).some(url => url);
    if (hasData) {
      renderMapModel();
    } else {
      loadMaps();
    }
    return;
  }

  if (Object.keys(state.satCatalog).length === 0) {
    loadSatelliteCatalog();
  } else {
    populateSatVariableSelect(state.satVariable);
    renderSatelliteImage();
  }
}

document.querySelectorAll('input[name="mapSource"]').forEach(el => {
  el.addEventListener('change', (e) => setMapSource(e.target.value));
});

document.getElementById('satDate').addEventListener('change', () => {
  loadSatelliteCatalog();
});

document.getElementById('satVariable').addEventListener('change', (e) => {
  state.satVariable = e.target.value;
  state.satIdx = 0;
  renderSatelliteImage();
});

document.getElementById('satTimeBack').addEventListener('click', () => stepSatelliteTime(-1));
document.getElementById('satTimeForward').addEventListener('click', () => stepSatelliteTime(1));

// "Latest" button next to the date picker — products can lag days behind
// each other, so this jumps to the newest date with data for whichever
// variable is currently selected, rather than just resetting to today.
async function jumpToLatestSatelliteDate() {
  if (!state.satVariable) return;
  const btn = document.getElementById('satLatestBtn');
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/satellite-latest-date?variable=${encodeURIComponent(state.satVariable)}`);
    const data = await res.json();
    if (!data.available) {
      showToast(`No images found for "${state.satVariable}" in the archive.`);
      return;
    }
    document.getElementById('satDate').value = data.date;
    loadSatelliteCatalog(state.satVariable);
  } catch (err) {
    console.error(err);
    showToast('Network error while finding the latest date.');
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('satLatestBtn').addEventListener('click', jumpToLatestSatelliteDate);


// ─── Event listeners ─────────────────────────────────────────────────────────

// Navbar timestamp
function updateClock() {
  const el = document.getElementById('currentDateTime');
  if (!el) return;
  el.textContent = new Date().toUTCString().replace('GMT', 'UTC');
}
updateClock();
setInterval(updateClock, 1000);

// Map: auto-update helper
function autoUpdateMaps() {
  loadMaps();
}

// Map: time step buttons
document.getElementById('mapTimeBack').addEventListener('click', () => {
  stepTime('back');
  autoUpdateMaps();
});
document.getElementById('mapTimeForward').addEventListener('click', () => {
  stepTime('forward');
  autoUpdateMaps();
});

// Map: date input change
document.getElementById('mapDate').addEventListener('change', (e) => {
  state.mapDate = e.target.value;
  autoUpdateMaps();
});

// Map: variable/depth change
document.getElementById('mapVarDepth').addEventListener('change', () => {
  autoUpdateMaps();
});


// ══════════════════════════════════════════════════════════
// ─── RT GLIDER PLOTS TAB ───
// Plots are overwritten in place by a ~30 min cron (filenames never change),
// so the backend appends a cache-busting token each time the catalog is
// fetched — no client-side polling needed, just refetch on tab interaction.
// ══════════════════════════════════════════════════════════

function currentRtImages() {
  const periods = state.rtCatalog[state.rtCategory] || {};
  return periods[state.rtPeriod] || [];
}

function renderRtGliderImage() {
  const images = currentRtImages();
  const img = images[state.rtVarIdx];

  document.getElementById('rtGliderBadge').textContent = state.rtGliderId || '—';
  document.getElementById('rt_placeholder').classList.add('d-none');
  document.getElementById('rt_loading').classList.add('d-none');

  const isMultiVar = state.rtCategory !== 'TS';
  document.getElementById('rt_prev_var').classList.toggle('d-none', !isMultiVar);
  document.getElementById('rt_next_var').classList.toggle('d-none', !isMultiVar);

  if (!img) {
    document.getElementById('rt_image').classList.add('d-none');
    document.getElementById('rt_unavailable').classList.remove('d-none');
    document.getElementById('rtVariableBadge').textContent = '—';
    return;
  }

  document.getElementById('rt_unavailable').classList.add('d-none');
  document.getElementById('rtVariableBadge').textContent = img.variable || 'T-S Diagram';

  const el = document.getElementById('rt_image');
  el.onerror = () => {
    el.classList.add('d-none');
    document.getElementById('rt_unavailable').classList.remove('d-none');
  };
  el.src = img.url;
  el.classList.remove('d-none');
}

function populateRtVariableSelect() {
  const group = document.getElementById('rtVariableGroup');
  const sel = document.getElementById('rtVariableSelect');
  const noMsg = document.getElementById('rtNoDataMsg');

  if (state.rtCategory === 'TS') {
    group.classList.add('d-none');
    state.rtVarIdx = 0;
    renderRtGliderImage();
    return;
  }
  group.classList.remove('d-none');

  const images = currentRtImages();
  sel.innerHTML = '';

  if (images.length === 0) {
    noMsg.classList.remove('d-none');
    state.rtVarIdx = 0;
    renderRtGliderImage();
    return;
  }
  noMsg.classList.add('d-none');

  images.forEach((img, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = img.variable;
    sel.appendChild(opt);
  });

  if (state.rtVarIdx >= images.length) state.rtVarIdx = 0;
  sel.value = state.rtVarIdx;
  renderRtGliderImage();
}

async function loadRtGliderCatalog() {
  if (!state.rtGliderId) return;
  const url = `${API_BASE}/api/gliders-rt/${encodeURIComponent(state.rtGliderId)}/catalog`;

  document.getElementById('rt_placeholder').classList.add('d-none');
  document.getElementById('rt_loading').classList.remove('d-none');

  try {
    const res = await fetch(url);
    const data = await res.json();
    state.rtCatalog = (data && data.available) ? data.catalog : {};
    state.rtVarIdx = 0;
    populateRtVariableSelect();
  } catch (err) {
    console.error(err);
    document.getElementById('rt_loading').classList.add('d-none');
    document.getElementById('rt_unavailable').classList.remove('d-none');
    showToast('Network error while fetching glider plots.');
  }
}

async function loadRtGliderList() {
  try {
    const res = await fetch(`${API_BASE}/api/gliders-rt`);
    const data = await res.json();
    state.rtGliders = (data && data.gliders) || [];

    const sel = document.getElementById('rtGliderSelect');
    sel.innerHTML = '';
    state.rtGliders.forEach(g => {
      const opt = document.createElement('option');
      opt.value = g;
      opt.textContent = g;
      sel.appendChild(opt);
    });

    if (state.rtGliders.length === 0) {
      document.getElementById('rt_placeholder').classList.add('d-none');
      document.getElementById('rt_unavailable').classList.remove('d-none');
      return;
    }

    state.rtGliderId = state.rtGliders[0];
    sel.value = state.rtGliderId;
    loadRtGliderCatalog();
  } catch (err) {
    console.error(err);
    document.getElementById('rt_placeholder').classList.add('d-none');
    document.getElementById('rt_unavailable').classList.remove('d-none');
    showToast('Network error while fetching glider list.');
  }
}

function stepRtVariable(direction) {
  const images = currentRtImages();
  if (images.length === 0) return;
  state.rtVarIdx = (state.rtVarIdx + direction + images.length) % images.length;
  document.getElementById('rtVariableSelect').value = state.rtVarIdx;
  renderRtGliderImage();
}

document.getElementById('rtGliderSelect').addEventListener('change', (e) => {
  state.rtGliderId = e.target.value;
  loadRtGliderCatalog();
});

document.querySelectorAll('input[name="rt_category"]').forEach(el => {
  el.addEventListener('change', (e) => {
    state.rtCategory = e.target.value;
    state.rtVarIdx = 0;
    populateRtVariableSelect();
  });
});

document.querySelectorAll('input[name="rt_period"]').forEach(el => {
  el.addEventListener('change', (e) => {
    state.rtPeriod = e.target.value;
    state.rtVarIdx = 0;
    populateRtVariableSelect();
  });
});

document.getElementById('rtVariableSelect').addEventListener('change', (e) => {
  state.rtVarIdx = parseInt(e.target.value, 10);
  renderRtGliderImage();
});

document.getElementById('rt_prev_var').addEventListener('click', () => stepRtVariable(-1));
document.getElementById('rt_next_var').addEventListener('click', () => stepRtVariable(1));


// Keyboard Navigation (Data Archive + RT Glider Plots tabs)
document.addEventListener('keydown', (e) => {
  // Skip if user is typing in a form field, or the lightbox is open (its own
  // keydown listener handles Escape; arrow keys shouldn't change the image
  // behind it while it's showing a snapshot of a specific image).
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (lightbox.classList.contains('show')) return;

  const mapsPane      = document.getElementById('maps-pane');
  const rtGlidersPane = document.getElementById('rtgliders-pane');

  const inMaps      = mapsPane      && mapsPane.classList.contains('active');
  const inRtGliders = rtGlidersPane && rtGlidersPane.classList.contains('active');

  if (inMaps) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      state.mapSource === 'model' ? stepMapModel(-1) : stepSatelliteTime(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      state.mapSource === 'model' ? stepMapModel(1) : stepSatelliteTime(1);
    }
  } else if (inRtGliders) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      stepRtVariable(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      stepRtVariable(1);
    }
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
(function init() {
  // Populate variable/depth for the Model Comparisons controls (cheap, DOM-only)
  populateVarDepth();

  // Load whichever source is active by default in each tab
  setMapSource(state.mapSource);
  loadRtGliderList();

  // Fire-and-forget: upgrades the satellite/glider dropdowns to the full
  // master product list once it resolves (see loadMasterSatVariables), and
  // to the current glider-ID list for satellite/glider classification (see
  // loadKnownGliderIds).
  loadMasterSatVariables();
  loadKnownGliderIds();
})();
