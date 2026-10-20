/* ══════════════════════════════════════════════════════════
   IOOS Model Comparisons – Client-side JS
   All API calls go to Flask endpoints; zero page reloads.
   ══════════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  // Overview
  ovRegions: ["Caribbean", "Gulf of Mexico", "South Atlantic Bight", "Mid Atlantic Bight", "West Florida Shelf", "Tropical Western Atlantic", "Eastern Pacific - Mexico", "Hawaii", "Guam", "Fiji"],
  ovCurrentRegionIndex: 0,

  // Maps
  mapDate: document.getElementById('mapDate').value,
  mapTime: '00Z',
  mapRegion: document.getElementById('mapRegion').value,
  mapVarDepth: '',

  // Profiles
  profileDate: document.getElementById('profileDate').value,
  profileType: 'Glider',
  gliderIds: [],
  argoIds: {},  // {wmo: {filename, region}} — merged across every Argo region folder, see app.py's ARGO_REGIONS
  fvonIds: {},  // {id: {filename, region}} — merged across every FVON region folder, see app.py's FVON_REGIONS
  stickyGliderId: null,  // set on day nav to restore same glider on new date
  stickyArgoId: null,    // set on day nav to restore same Argo float on new date
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

// Back to the neutral "nothing selected yet" state — used whenever day-nav,
// a date change, or switching profile type would otherwise leave the user
// looking at a stale previous platform's plot (or worse, silently auto-load
// whichever platform happened to default to the top of the freshly-
// repopulated dropdown). The user picks; nothing gets picked for them.
function resetProfilePlaceholder() {
  document.getElementById('profileImageContainer').innerHTML = `
    <div class="placeholder-state">
      <i class="bi bi-graph-up text-muted" style="font-size:3rem;"></i>
      <p class="text-muted mt-2">Select a platform to view its profile</p>
    </div>`;
  document.getElementById('profileLabel').textContent = '';
}

// Single source of truth for "nothing to show for this date" — previously
// the sidebar's #noPlatformMsg and the main panel could both be silent at
// once (dropdown says "No gliders available", main panel just shows
// whatever was there before), or occasionally say conflicting things.
// Replaces that sidebar message entirely.
function showNoPlatformsMessage(label) {
  document.getElementById('profileImageContainer').innerHTML = `
    <div class="unavailable-state">
      <i class="bi bi-exclamation-circle" style="font-size:2.5rem;color:#adb5bd;"></i>
      <p class="mb-0 mt-1 text-muted">No ${label} available for this date</p>
    </div>`;
  document.getElementById('profileLabel').textContent = '';
}

function setUnavailable(containerId, label, action) {
  const container = document.getElementById(containerId);
  container.innerHTML = `
    <div class="unavailable-state">
      <i class="bi bi-exclamation-circle" style="font-size:2.5rem;color:#adb5bd;"></i>
      <p class="mb-0 mt-1 text-muted">${label}: image not available for this time</p>
      ${action ? `<button type="button" class="btn btn-sm btn-outline-secondary mt-2" id="unavailableActionBtn">${action.label}</button>` : ''}
    </div>`;
  if (action) {
    document.getElementById('unavailableActionBtn').addEventListener('click', action.onClick);
  }
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
    showToast("No image available to download.", 'secondary');
  }
}

// ─── Image Lightbox ────────────────────────────────────────────────────────
// Delegated to the document so it works for every plot image on the page,
// including ones re-created via setImage()'s innerHTML replacement.
const lightbox = document.getElementById('imageLightbox');
const lightboxImage = document.getElementById('lightboxImage');

// Keyboard/screen-reader support: the element that had focus before opening
// (so closing returns focus there instead of leaving it on a now-hidden
// background element), plus a focus trap so Tab/Shift+Tab stay inside the
// lightbox while it's open rather than cycling through the page behind it.
let lightboxTriggerEl = null;

function lightboxFocusableEls() {
  return Array.from(lightbox.querySelectorAll(
    'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
  )).filter(el => el.offsetParent !== null);  // skip anything hidden (e.g. lightboxImage when profile mode is showing lightboxBody instead)
}

function openLightboxCommon() {
  lightboxTriggerEl = document.activeElement;
  lightbox.classList.add('show');
  document.body.style.overflow = 'hidden';
  // Deferred one tick so profile mode's just-moved controls are already
  // visible/focusable by the time we look for them.
  setTimeout(() => {
    const focusable = lightboxFocusableEls();
    (focusable[0] || lightbox).focus();
  }, 0);
}

function openLightbox(src, alt) {
  if (!src) return;
  lightboxImage.src = src;
  lightboxImage.alt = alt || '';
  lightboxImage.classList.remove('d-none');
  openLightboxCommon();
}

// Profile mode: instead of handing the lightbox a static image src, move the
// real controls (Date/Day-nav/Type/Platform) and the real
// #profileImageContainer into it — not clones, the same elements, so their
// existing event listeners and IDs keep working and whatever the panel is
// currently showing (image, loading, unavailable) carries over with no
// extra syncing. closeLightbox() puts everything back where it came from.
let lightboxMovedNodes = [];

function openProfileLightbox() {
  lightboxMovedNodes = [];
  const moveInto = (targetEl, sourceId) => {
    const node = document.getElementById(sourceId);
    if (!node) return;
    lightboxMovedNodes.push({ node, parent: node.parentNode, nextSibling: node.nextSibling });
    targetEl.appendChild(node);
  };

  const toolbar = document.getElementById('lightboxToolbar');
  ['profileDateGroup', 'profileDayNavGroup', 'profileTypeGroupWrap', 'platformIdGroup']
    .forEach(id => moveInto(toolbar, id));
  moveInto(document.getElementById('lightboxBody'), 'profileImageContainer');

  lightboxImage.classList.add('d-none');
  openLightboxCommon();
}

function closeLightbox() {
  lightbox.classList.remove('show');
  lightboxImage.src = '';
  lightboxImage.classList.remove('d-none');
  document.body.style.overflow = '';

  if (lightboxMovedNodes.length) {
    // Restore in REVERSE of the order they were moved in. Each node's
    // nextSibling was captured relative to whatever was still in place at
    // move time — e.g. profileDateGroup's nextSibling is profileDayNavGroup,
    // which hadn't moved yet when that snapshot was taken. Restoring in the
    // same (forward) order means insertBefore's reference node (DayNavGroup)
    // is still sitting in the lightbox toolbar, not yet back in the sidebar
    // — insertBefore throws NotFoundError in that case, aborting the loop
    // and leaving every remaining control stranded in the (now-hidden)
    // lightbox instead of back in the sidebar. Reverse order restores each
    // reference node before anything needs to point at it.
    for (const { node, parent, nextSibling } of lightboxMovedNodes.slice().reverse()) {
      parent.insertBefore(node, nextSibling);
    }
    lightboxMovedNodes = [];
    // The map's corner inset may have resized/reflowed while hidden behind
    // the modal — Leaflet needs a nudge to recompute its tile layout.
    if (profileMap) setTimeout(() => profileMap.invalidateSize(), 50);
  }

  // Return focus to whatever opened the lightbox (a marker, the maximize
  // button, etc.) — otherwise a keyboard user's focus silently lands back
  // at the top of the document instead of where they were.
  if (lightboxTriggerEl && typeof lightboxTriggerEl.focus === 'function') {
    lightboxTriggerEl.focus();
  }
  lightboxTriggerEl = null;
}

document.addEventListener('click', (e) => {
  const img = e.target.closest('.plot-image');
  if (img && !img.classList.contains('d-none') && img.src) {
    if (img.closest('#profileImageContainer')) {
      openProfileLightbox();
    } else {
      openLightbox(img.src, img.alt);
    }
    return;
  }
  if (e.target === lightbox) {
    closeLightbox();
  }
});

document.getElementById('lightboxClose').addEventListener('click', closeLightbox);

document.addEventListener('keydown', (e) => {
  if (!lightbox.classList.contains('show')) return;

  if (e.key === 'Escape') {
    closeLightbox();
    return;
  }

  // Focus trap: keep Tab/Shift+Tab cycling inside the lightbox instead of
  // continuing into the (visually hidden, but still in the DOM) page
  // behind it. Queried fresh each press since profile mode's focusable set
  // changes (day-nav/platform controls live here while open).
  if (e.key === 'Tab') {
    const focusable = lightboxFocusableEls();
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
});

// severity: 'danger' (default) for real failures (network errors etc.),
// 'secondary' for benign "nothing to show" cases (valid selection, just no
// data for it) — previously every toast looked like an alarming error
// regardless of which kind of message it was.
function showToast(msg, severity = 'danger') {
  document.getElementById('alertToastMsg').textContent = msg;
  const el = document.getElementById('alertToast');
  el.classList.remove('text-bg-danger', 'text-bg-secondary');
  el.classList.add(`text-bg-${severity}`);
  const toast = bootstrap.Toast.getOrCreateInstance(el, { delay: 4000 });
  toast.show();
}

// ─── Variable/Depth dropdown population ──────────────────────────────────────

function populateVarDepth(region) {
  const info = REGION_INFO[region];
  if (!info) return;
  const sel = document.getElementById('mapVarDepth');
  const previous = state.mapVarDepth;  // try to carry this over to the new region
  sel.innerHTML = '';
  const sortedVariables = [...info.variables].sort((a, b) => a.localeCompare(b));
  for (const v of sortedVariables) {
    if (v === 'ocean_heat_content') {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = "Ocean Heat Content";
      sel.appendChild(opt);
    } else {
      const depths = (info.depths && info.depths[v]) || [];
      const label = v.charAt(0).toUpperCase() + v.slice(1);
      for (const d of depths) {
        const opt = document.createElement('option');
        opt.value = `${v}_${d}`;
        opt.textContent = `${label} @ ${d}`;
        sel.appendChild(opt);
      }
    }
  }
  // Keep the same variable/depth selected across a region switch when the
  // new region also has it (e.g. Ocean Heat Content in Caribbean -> Gulf),
  // instead of always resetting to the alphabetically-first option
  // (previously always "Currents @ 0m", regardless of what you'd been
  // looking at).
  if (previous && Array.from(sel.options).some(o => o.value === previous)) {
    sel.value = previous;
  }
  state.mapVarDepth = sel.value;
}

// ─── Overview: dynamic depth buttons ─────────────────────────────────────────

function updateOverviewDepths(region, variable) {
  const info = REGION_INFO[region];
  const container = document.getElementById('ov_depth_btns');
  if (!info || !info.depths || !container) return;

  const depths = info.depths[variable] || [];
  container.innerHTML = '';
  depths.forEach((depth, i) => {
    const safeId = 'ov_dep_' + depth.replace(/[^a-z0-9]/gi, '_');

    const input = document.createElement('input');
    input.type = 'radio';
    input.className = 'btn-check';
    input.name = 'ov_depth';
    input.id = safeId;
    input.value = depth;
    input.autocomplete = 'off';
    if (i === 0) input.checked = true;
    input.addEventListener('change', () => {
      updateOverviewGoesVisibility();
      loadOverview();
    });

    const label = document.createElement('label');
    label.className = 'btn btn-outline-grey btn-sm';
    label.setAttribute('for', safeId);
    label.textContent = depth;

    container.appendChild(input);
    container.appendChild(label);
  });
}

// Which comparison panels exist for a region / variable_depth comes from
// comparisons.yml via MAP_PANELS (injected by index()); the old per-model
// rules (GOES only at 0 m, ECCOFS exclusions, NESDIS only for OHC, Fiji/Guam
// ESPC-vs-CMEMS) all live there now.
function panelKeysFor(region, varDepth) {
  const byRegion = MAP_PANELS[region] || {};
  return byRegion[varDepth] || PANEL_DEFS.slice(0, 2).map(p => p.key);
}

function overviewVarDepth() {
  const varInput = document.querySelector('input[name="ov_variable"]:checked');
  const depInput = document.querySelector('input[name="ov_depth"]:checked');
  if (!varInput) return null;
  if (varInput.value === 'ocean_heat_content') return 'ocean_heat_content';
  return depInput ? `${varInput.value}_${depInput.value}` : null;
}

function updateOverviewPanels() {
  const regionInput = document.querySelector('input[name="ov_region"]:checked');
  if (!regionInput) return;
  const vd = overviewVarDepth();
  let visible = vd ? panelKeysFor(regionInput.value, vd) : null;
  if (!visible) {
    // No depth chosen yet: show anything this region offers for the variable.
    const varInput = document.querySelector('input[name="ov_variable"]:checked');
    const byRegion = MAP_PANELS[regionInput.value] || {};
    visible = [...new Set(Object.entries(byRegion)
      .filter(([k]) => varInput && k.startsWith(varInput.value))
      .flatMap(([, keys]) => keys))];
  }
  const radios = Array.from(document.querySelectorAll('input[name="ov_model"]'));
  radios.forEach(btn => {
    const show = visible.includes(btn.value);
    btn.style.display = show ? '' : 'none';
    btn.nextElementSibling.style.display = show ? '' : 'none';
  });
  const checked = document.querySelector('input[name="ov_model"]:checked');
  if (!checked || !visible.includes(checked.value)) {
    const first = radios.find(btn => visible.includes(btn.value));
    if (first) first.checked = true;
  }
}

// Kept as names because several event handlers call them.
function updateOverviewGoesVisibility() { updateOverviewPanels(); }

// ─── Maps – Toggle State ──────────────────────────────────────────────────────

let mapModels = PANEL_DEFS.slice(0, 2);
let mapCurrentModelIdx = 0;
function preloadPanels(data) {
  if (!data) return;
  PANEL_DEFS.forEach(p => { if (data[p.key] && data[p.key].available) preloadImage(data[p.key].url); });
}
const emptyMapUrls = () => Object.fromEntries(PANEL_DEFS.map(p => [p.key, null]));
let mapCachedUrls = emptyMapUrls();

function updateMapModelsConfig(region, varDepth) {
  mapModels = panelKeysFor(region, varDepth)
    .map(key => PANEL_DEFS.find(p => p.key === key))
    .filter(Boolean)
    .map(p => ({ key: p.key, label: p.label }));

  if (mapCurrentModelIdx >= mapModels.length) {
    mapCurrentModelIdx = 0;
  }

  const tabsContainer = document.getElementById('mapModelTabs');
  tabsContainer.innerHTML = '';

  mapModels.forEach((m, i) => {
    const li = document.createElement('li');
    li.className = 'nav-item';
    li.setAttribute('role', 'presentation');

    const btn = document.createElement('button');
    btn.className = 'nav-link fw-medium px-4 py-2';
    if (i === mapCurrentModelIdx) btn.classList.add('active');
    btn.id = `mapTab-${m.key}`;
    btn.dataset.model = m.key;
    btn.setAttribute('role', 'tab');
    btn.style.cssText = 'border-radius: var(--radius-md) var(--radius-md) 0 0; font-size:.85rem;';
    btn.textContent = m.label;

    btn.addEventListener('click', () => {
      mapCurrentModelIdx = i;
      renderMapModel();
    });

    li.appendChild(btn);
    tabsContainer.appendChild(li);
  });
}

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

// Initial tabs are set up by updateMapModelsConfig()

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
      region: state.mapRegion,
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
        preloadPanels(data);
      } catch (e) {
        // silently fail on preload errors
      }
    } else {
      const data = apiCache[url];
      preloadPanels(data);
    }
  });
}

async function loadMaps() {
  state.mapDate = document.getElementById('mapDate').value;
  state.mapRegion = document.getElementById('mapRegion').value;
  state.mapVarDepth = document.getElementById('mapVarDepth').value;

  const params = new URLSearchParams({
    region: state.mapRegion,
    variable_depth: state.mapVarDepth,
    date: state.mapDate,
    time: state.mapTime,
  });
  
  const cacheKey = `${API_BASE}/api/maps?${params.toString()}`;
  
  const renderData = (data) => {
    mapCachedUrls = Object.fromEntries(PANEL_DEFS.map(p =>
      [p.key, data[p.key] && data[p.key].available ? data[p.key].url : null]));

    renderMapModel();

    if (!Object.values(mapCachedUrls).some(Boolean)) {
      showToast('No images available for the selected date/time.', 'secondary');
    }
  };

  if (apiCache[cacheKey]) {
    renderData(apiCache[cacheKey]);
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
    mapCachedUrls = emptyMapUrls();
    setUnavailable('mapImageContainer', 'Maps');
    showToast('Network error while fetching images.');
  }
}


// ─── Profiles – Glider ───────────────────────────────────────────────────────

async function fetchGliderIds() {
  const sel = document.getElementById('platformId');
  sel.innerHTML = '<option>Loading…</option>';

  const params = new URLSearchParams({ date: state.profileDate });
  try {
    const res = await fetch(`${API_BASE}/api/glider-ids?${params}`);
    const data = await res.json();
    state.gliderIds = data.ids || [];
    sel.innerHTML = '';
    if (state.gliderIds.length === 0) {
      sel.innerHTML = '<option value="">No gliders available</option>';
    } else {
      // Placeholder first — a freshly-populated <select> defaults to its
      // first <option> "selected" with no user action, which used to mean
      // whichever glider sorted first silently became "the" selection
      // (and could even auto-load, depending on the caller). This makes
      // "nothing chosen yet" the explicit, visible default instead.
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = 'Select a glider…';
      sel.appendChild(placeholder);
      for (const id of state.gliderIds) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
    }

    // Single source of truth for platform availability is the main panel
    // (see showNoPlatformsMessage) — covers every combination below,
    // including the one the old sidebar-only message missed: zero gliders
    // and nothing sticky to restore.
    if (state.stickyGliderId) {
      if (state.gliderIds.includes(state.stickyGliderId)) {
        // Glider has a plot on this date — select and load it
        sel.value = state.stickyGliderId;
        loadProfile();
      } else {
        // Glider has no plot on this date — show unavailable, keep sticky for further nav
        setUnavailable('profileImageContainer', 'Glider Profile');
      }
    } else if (state.gliderIds.length > 0) {
      sel.value = '';
      resetProfilePlaceholder();
    } else {
      showNoPlatformsMessage('gliders');
    }
  } catch {
    sel.innerHTML = '<option value="">Error loading IDs</option>';
  }
}

// ─── Profiles – Argo ─────────────────────────────────────────────────────────

async function fetchArgoIds() {
  const sel = document.getElementById('platformId');
  sel.innerHTML = '<option>Loading…</option>';

  // Merged across every Argo region folder server-side (see ARGO_REGIONS in
  // app.py) — no region picker here, since most of regions.py's finer
  // subregions were never their own Argo folder (e.g. floats plotted under
  // "Caribbean" never show up if you'd picked "Windward Islands").
  const params = new URLSearchParams({ date: state.profileDate });
  try {
    const res = await fetch(`${API_BASE}/api/argo-ids?${params}`);
    const data = await res.json();
    state.argoIds = data.ids || {};
    const keys = Object.keys(state.argoIds).sort();
    sel.innerHTML = '';
    if (keys.length === 0) {
      sel.innerHTML = '<option value="">No Argo floats available</option>';
    } else {
      // Placeholder first — a freshly-populated <select> defaults to its
      // first <option> "selected" with no user action, which used to mean
      // whichever float sorted first silently became "the" selection (and
      // could even auto-load, depending on the caller). This makes "nothing
      // chosen yet" the explicit, visible default instead.
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = 'Select a float…';
      sel.appendChild(placeholder);
      for (const id of keys) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
    }

    // Single source of truth for platform availability is the main panel
    // (see showNoPlatformsMessage) — covers every combination below,
    // including the one the old sidebar-only message missed: zero floats
    // and nothing sticky to restore.
    if (state.stickyArgoId) {
      if (keys.includes(state.stickyArgoId)) {
        // Float has a listing on this date — select and load it
        sel.value = state.stickyArgoId;
        loadProfile();
      } else {
        // Float has no listing on this date (floats don't profile daily) —
        // keep it sticky and offer a way back to its last real profile.
        setUnavailable('profileImageContainer', 'Argo Profile', {
          label: `Jump to ${state.stickyArgoId}'s last profile`,
          onClick: () => jumpToLastArgoProfile(state.stickyArgoId),
        });
      }
    } else if (keys.length > 0) {
      sel.value = '';
      resetProfilePlaceholder();
    } else {
      showNoPlatformsMessage('Argo floats');
    }
  } catch {
    sel.innerHTML = '<option value="">Error loading IDs</option>';
  }
}

const ARGO_LOOKBACK_DAYS = 60;  // floats typically profile every ~10 days

// Walk backward from the current date until argoKey shows up in the ids
// list again, then jump the whole Profiles view (date, dropdown, image,
// map) to that date.
async function jumpToLastArgoProfile(argoKey) {
  setLoading('profileImageContainer', `Searching for ${argoKey}'s last profile…`);

  for (let i = 1; i <= ARGO_LOOKBACK_DAYS; i++) {
    const d = addDays(state.profileDate, -i);
    let ids;
    try {
      const res = await fetch(`${API_BASE}/api/argo-ids?` +
        new URLSearchParams({ date: d }));
      ids = (await res.json()).ids || {};
    } catch (err) {
      console.error('Error searching for last Argo profile:', err);
      break;
    }

    if (ids[argoKey]) {
      state.profileDate = d;
      document.getElementById('profileDate').value = d;
      state.argoIds = ids;

      const sel = document.getElementById('platformId');
      sel.innerHTML = '';
      for (const id of Object.keys(ids).sort()) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
      sel.value = argoKey;
      state.stickyArgoId = argoKey;

      loadProfile();
      fetchProfileLocations();
      return;
    }
  }

  document.getElementById('profileImageContainer').innerHTML = `
    <div class="unavailable-state">
      <i class="bi bi-exclamation-circle" style="font-size:2.5rem;color:#adb5bd;"></i>
      <p class="mb-0 mt-1 text-muted">No profile found for ${argoKey} in the last ${ARGO_LOOKBACK_DAYS} days</p>
    </div>`;
}

// ─── Profiles – FVON ─────────────────────────────────────────────────────────

async function fetchFvonIds() {
  const sel = document.getElementById('platformId');
  sel.innerHTML = '<option>Loading…</option>';

  // Merged across every FVON region folder server-side (see FVON_REGIONS in
  // app.py) — no region picker; Bahamas and Fiji are both queried and
  // whichever has data for this date shows up.
  const params = new URLSearchParams({ date: state.profileDate });
  try {
    const res = await fetch(`${API_BASE}/api/fvon-ids?${params}`);
    const data = await res.json();
    state.fvonIds = data.ids || {};
    const keys = Object.keys(state.fvonIds).sort();
    sel.innerHTML = '';
    if (keys.length === 0) {
      sel.innerHTML = '<option value="">No FVON profiles available</option>';
      // Single source of truth for platform availability is the main panel
      // (see showNoPlatformsMessage), not a separate sidebar message.
      showNoPlatformsMessage('FVON profiles');
    } else {
      // Placeholder first — a freshly-populated <select> defaults to its
      // first <option> "selected" with no user action, which used to mean
      // whichever profile sorted first silently became "the" selection (and
      // could even auto-load, depending on the caller). This makes "nothing
      // chosen yet" the explicit, visible default instead. FVON has no
      // sticky-selection concept (unlike Argo/Glider), so this always
      // applies rather than only when nothing was previously selected.
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = 'Select a profile…';
      sel.appendChild(placeholder);
      for (const id of keys) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
      sel.value = '';
      resetProfilePlaceholder();
    }
  } catch {
    sel.innerHTML = '<option value="">Error loading IDs</option>';
  }
}

// ─── Profile Map ─────────────────────────────────────────────────────────────
const PLATFORM_STYLES = {
  glider: { fillColor: '#0077cc', label: 'Glider', iconUrl: `${API_BASE}/static/img/slocum_glider.png` },
  argo:   { fillColor: '#e07b00', label: 'Argo' },
  fvon:   { fillColor: '#17a85c', label: 'FVON' },
};

function argoColor(id) {
  let hash = 0;
  const str = String(id);
  for (let i = 0; i < str.length; i++) hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  return `hsl(${Math.round((hash * 137.508) % 360)}, 75%, 50%)`;
}

const PROFILE_CLICK_ZOOM = 7;
let profileMap = null;
// Whether the map has already been auto-fit to the current profile type's
// markers. Only the first render fits bounds — after that, a manual zoom
// (e.g. into the Gulf) persists across day-nav instead of being undone by
// every subsequent date change re-fitting to the whole merged dataset. Reset
// when the profile type changes (Argo/Glider/FVON are different datasets
// worth re-fitting to once).
let profileMapAutoFitDone = false;
let profileMapTracks = null;
let profileMapMarkers = null;
let profileMapRegionBoxes = null;
let profileMapRegionLegend = null;
let profileMapStorms = null;
let profileMapStormLegend = null;
let profileMapStormLegendDiv = null;

// idLabel -> [lat, lon] for whatever's currently on the map, rebuilt on every
// renderProfileMarkers() call. Lets the platform dropdown's own change
// listener fly to a platform the same way clicking its marker already does,
// without re-deriving location from the API response itself.
let profilePlatformLocations = {};

// Of a point's lon, lon-360, lon+360 (the three wrapped copies markers/boxes
// are drawn at — see renderProfileMarkers), pick whichever is closest to the
// map's current center so flyTo/setView doesn't pan the long way around the
// world to reach a platform that's actually already near the visible copy.
function nearestWrappedLon(lon, centerLon) {
  return [lon - 360, lon, lon + 360]
    .reduce((a, b) => Math.abs(b - centerLon) < Math.abs(a - centerLon) ? b : a);
}

// Box color per platform type — also used by the legend swatch below, so
// the two stay visually tied together.
const REGION_BOX_COLORS = { Argo: '#6c8ebf', Glider: '#d97706' };

// Region box fill opacity is user-adjustable (slider in #regionBoxOpacityModal)
// and remembered across sessions, same pattern as basemaps.js's basemap choice.
const REGION_BOX_OPACITY_KEY = 'ioosRegionBoxFillOpacity';
const REGION_BOX_OPACITY_DEFAULT = 0.015;

function loadRegionBoxOpacity() {
  try {
    const v = parseFloat(localStorage.getItem(REGION_BOX_OPACITY_KEY));
    return Number.isFinite(v) ? v : REGION_BOX_OPACITY_DEFAULT;
  } catch (e) { return REGION_BOX_OPACITY_DEFAULT; }
}
function saveRegionBoxOpacity(v) {
  try { localStorage.setItem(REGION_BOX_OPACITY_KEY, String(v)); } catch (e) { /* private mode */ }
}

let regionBoxFillOpacity = loadRegionBoxOpacity();

// Which box set belongs to the currently-selected profile type. FVON has no
// defined search boxes, so it's absent here on purpose.
function activeRegionBoxSet() {
  if (state.profileType === 'Argo')   return { boxes: ARGO_REGION_BOXES,   color: REGION_BOX_COLORS.Argo };
  if (state.profileType === 'Glider') return { boxes: GLIDER_REGION_BOXES, color: REGION_BOX_COLORS.Glider };
  return null;
}

let profileMapRegionLegendDiv = null;

// A small on-map key explaining the dashed boxes, so it's clear what they
// mean without having to hover one first. Shown/hidden together with the
// boxes themselves (toggleProfileRegionBoxes below); content tracks whichever
// profile type is selected (updateRegionBoxLegend below).
function createRegionBoxLegend() {
  const legend = L.control({ position: 'bottomleft' });
  legend.onAdd = function () {
    profileMapRegionLegendDiv = L.DomUtil.create('div', 'map-region-legend');
    updateRegionBoxLegend();
    L.DomEvent.disableClickPropagation(profileMapRegionLegendDiv);
    return profileMapRegionLegendDiv;
  };
  return legend;
}

function updateRegionBoxLegend() {
  if (!profileMapRegionLegendDiv) return;
  const active = activeRegionBoxSet();
  if (!active) {
    profileMapRegionLegendDiv.innerHTML =
      '<div class="map-region-legend-note">FVON has no defined search regions</div>';
    return;
  }
  profileMapRegionLegendDiv.innerHTML =
    '<div class="map-region-legend-row">' +
    `<span class="map-region-legend-swatch" style="border-color:${active.color}"></span>` +
    `<span>${state.profileType} search region</span></div>` +
    '<div class="map-region-legend-note">Platforms outside a box aren\'t found or listed</div>';
}

// Draws a dashed box (plus ±360° wrapped copies, same trick as the markers)
// for every region the CURRENTLY SELECTED profile type's box set lists — see
// activeRegionBoxSet(). Only one type's boxes show at a time, so picking Argo
// doesn't leave Glider's boxes (or vice versa) sitting on the map. Re-run on
// every profileType change (see that radio's listener); doesn't need
// re-running on date/platform changes.
function renderRegionBoxes() {
  if (!profileMapRegionBoxes) return;
  profileMapRegionBoxes.clearLayers();
  updateRegionBoxLegend();

  const active = activeRegionBoxSet();
  if (!active || !Array.isArray(active.boxes)) return;

  for (const { label, extent } of active.boxes) {
    const [lonmin, lonmax, latmin, latmax] = extent;
    for (const dLon of [-360, 0, 360]) {
      L.rectangle(
        [[latmin, lonmin + dLon], [latmax, lonmax + dLon]],
        {
          color: active.color,
          weight: 1,
          opacity: 0.45,
          dashArray: '4 4',
          fill: true,
          fillOpacity: regionBoxFillOpacity,
          interactive: true,
        }
      ).bindTooltip(`${state.profileType}: ${label}`, { sticky: true }).addTo(profileMapRegionBoxes);
    }
  }
}

// Wires the opacity modal's slider: live-updates the already-drawn boxes via
// setStyle (cheap — a handful of rectangles) rather than a full re-render on
// every drag tick, and persists the choice for next time.
function initRegionBoxOpacityControl() {
  const slider = document.getElementById('regionBoxOpacitySlider');
  const valueLabel = document.getElementById('regionBoxOpacityValue');
  if (!slider || !valueLabel) return;

  const pct = Math.round(regionBoxFillOpacity * 100);
  slider.value = String(pct);
  valueLabel.textContent = `${pct}%`;

  slider.addEventListener('input', () => {
    const p = Number(slider.value);
    valueLabel.textContent = `${p}%`;
    regionBoxFillOpacity = p / 100;
    saveRegionBoxOpacity(regionBoxFillOpacity);
    if (profileMapRegionBoxes) {
      profileMapRegionBoxes.eachLayer(l => l.setStyle({ fillOpacity: regionBoxFillOpacity }));
    }
  });
}

function toggleProfileRegionBoxes(visible) {
  if (!profileMap || !profileMapRegionBoxes) return;
  if (visible) {
    profileMap.addLayer(profileMapRegionBoxes);
    if (profileMapRegionLegend) profileMapRegionLegend.addTo(profileMap);
  } else {
    profileMap.removeLayer(profileMapRegionBoxes);
    if (profileMapRegionLegend) profileMapRegionLegend.remove();
  }
}

// ─── Storm tracks (Profiles map) ───────────────────────────────────────────
// One tropycal-backed source (see storm_track.py) covers both historical and
// recent dates, so unlike region boxes this isn't gated by profileType —
// whatever storm was active near state.profileDate shows regardless of
// whether Glider/Argo/FVON is selected. Atlantic + East Pacific only (v1);
// see storm_track.py for why Guam/Fiji show nothing for now.

function createStormLegend() {
  const legend = L.control({ position: 'bottomleft' });
  legend.onAdd = function () {
    profileMapStormLegendDiv = L.DomUtil.create('div', 'map-region-legend');
    updateStormLegend({});
    L.DomEvent.disableClickPropagation(profileMapStormLegendDiv);
    return profileMapStormLegendDiv;
  };
  return legend;
}

function updateStormLegend(storms) {
  if (!profileMapStormLegendDiv) return;
  const entries = Object.values(storms);
  if (!entries.length) {
    profileMapStormLegendDiv.innerHTML =
      '<div class="map-region-legend-note">No active storms near this date (Atlantic/East Pacific)</div>';
    return;
  }
  // One row per storm rather than per category — there's usually only a
  // handful active at once, and the name matters more here than it does
  // for the region-box legend's fixed Argo/Glider categories.
  profileMapStormLegendDiv.innerHTML = entries.map(s =>
    '<div class="map-region-legend-row">' +
    `<span class="map-region-legend-swatch" style="border-color:${s.color};background:${s.color}"></span>` +
    `<span>${s.name} — ${s.category_label}</span></div>`
  ).join('') + '<div class="map-region-legend-note">Track shown through the selected date; shaded area is the NHC forecast cone, where one exists</div>';
}

// Drawn at the same ±360° wrapped copies as every other layer on this map
// (region boxes, markers) so Pacific storms show up regardless of which
// copy of the world is currently in view.
function renderStormTracks(data) {
  if (!profileMapStorms) return;
  profileMapStorms.clearLayers();
  const storms = (data && data.storms) || {};
  updateStormLegend(storms);

  for (const storm of Object.values(storms)) {
    const pts = (storm.track || []).filter(p => p.lat != null && p.lon != null);
    if (!pts.length) continue;
    const last = pts[pts.length - 1];

    const popupHtml =
      '<div style="font-family:Inter,sans-serif;font-size:13px;min-width:160px;">' +
      `<b style="color:${storm.color}">${storm.name}</b><br>` +
      `<span style="color:#555">${storm.category_label}</span>` +
      (last.vmax != null ? ` · ${last.vmax} kt` : '') + '<br>' +
      `<span style="color:#888;font-size:11px;">as of ${last.time || ''}</span>` +
      '</div>';

    for (const dLon of [-360, 0, 360]) {
      if (pts.length > 1) {
        L.polyline(pts.map(p => [p.lat, p.lon + dLon]), {
          color: storm.color,
          weight: 3,
          opacity: 0.85,
        }).bindPopup(popupHtml).addTo(profileMapStorms);
      }

      if (storm.cone && storm.cone.length > 2) {
        L.polygon(storm.cone.map(([lat, lon]) => [lat, lon + dLon]), {
          color: storm.color,
          weight: 1,
          dashArray: '4 4',
          fill: true,
          fillColor: storm.color,
          fillOpacity: 0.12,
        }).bindTooltip(`${storm.name}: NHC forecast cone`, { sticky: true }).addTo(profileMapStorms);
      }

      // Each earlier observation as its own small point — hover/click for
      // that point's own time/intensity, not just the latest position.
      for (const p of pts.slice(0, -1)) {
        L.circleMarker([p.lat, p.lon + dLon], {
          radius: 4,
          color: 'white',
          weight: 1,
          fillColor: storm.color,
          fillOpacity: 0.85,
        }).bindTooltip(
          `${storm.name} — ${p.type || ''}` + (p.vmax != null ? ` ${p.vmax} kt` : '') + `<br>${p.time || ''}`
        ).addTo(profileMapStorms);
      }

      L.circleMarker([last.lat, last.lon + dLon], {
        radius: 7,
        color: 'white',
        weight: 2,
        fillColor: storm.color,
        fillOpacity: 0.95,
      }).bindPopup(popupHtml).addTo(profileMapStorms);

      // NHC forecast track — dashed to read as "predicted," picking up
      // from the current position through each advisory point (fhr 0, 3,
      // 12, 24, ...). Only present while the storm was active (empty most
      // of a historical storm's lifetime — see storm_track.py).
      const fpts = (storm.forecast || []).filter(p => p.lat != null && p.lon != null);
      if (fpts.length) {
        L.polyline([[last.lat, last.lon + dLon], ...fpts.map(p => [p.lat, p.lon + dLon])], {
          color: storm.color,
          weight: 2,
          opacity: 0.7,
          dashArray: '6 6',
        }).addTo(profileMapStorms);

        for (const p of fpts) {
          L.circleMarker([p.lat, p.lon + dLon], {
            radius: 5,
            color: storm.color,
            weight: 2,
            fillColor: '#fff',
            fillOpacity: 0.9,
          }).bindTooltip(
            `${storm.name} — forecast +${p.fhr}h` + (p.vmax != null ? ` · ${p.vmax} kt` : '') + `<br>${p.time || ''}`
          ).addTo(profileMapStorms);
        }
      }
    }
  }
}

async function fetchStormTracks() {
  if (!profileMap || !profileMapStorms) return;
  const cacheKey = `storm-track|${state.profileDate}`;
  if (apiCache[cacheKey]) { renderStormTracks(apiCache[cacheKey]); return; }

  try {
    const res = await fetch(`${API_BASE}/api/storm-track?date=${state.profileDate}`);
    const data = await res.json();
    apiCache[cacheKey] = data;
    renderStormTracks(data);
  } catch (err) {
    console.error('Error fetching storm tracks:', err);
  }
}

function toggleStormLayer(visible) {
  if (!profileMap || !profileMapStorms) return;
  if (visible) {
    profileMap.addLayer(profileMapStorms);
    if (profileMapStormLegend) profileMapStormLegend.addTo(profileMap);
  } else {
    profileMap.removeLayer(profileMapStorms);
    if (profileMapStormLegend) profileMapStormLegend.remove();
  }
}

function initProfileMap() {
  setTimeout(() => {
    const container = document.getElementById('profileMapContainer');
    if (!container) return;

    if (profileMap) {
      profileMap.invalidateSize();
      fetchProfileLocations();
      fetchStormTracks();
      return;
    }

    profileMap = L.map('profileMapContainer', {
      zoomControl: true,
      scrollWheelZoom: true,
      // Merged Argo/FVON positions now span both the Atlantic and Pacific
      // (Guam, Fiji) in one view — worldCopyJump lets panning past ±180°
      // wrap seamlessly into the next copy of the world instead of hitting
      // a hard edge.
      worldCopyJump: true,
    }).setView([30.0, -75.0], 4);

    addBasemaps(profileMap, { maxZoom: 13 });

    profileMapRegionBoxes = L.layerGroup().addTo(profileMap);
    profileMapTracks      = L.layerGroup().addTo(profileMap);
    profileMapMarkers     = L.layerGroup().addTo(profileMap);
    profileMapStorms      = L.layerGroup().addTo(profileMap);
    renderRegionBoxes();
    profileMapRegionLegend = createRegionBoxLegend();
    profileMapStormLegend = createStormLegend();

    // Boxes/storms should always start on. Force both the checkbox and the
    // layer in sync explicitly (rather than relying on the checkbox's static
    // HTML `checked` attribute) — some browsers restore a checkbox's
    // previous checked state across a page refresh regardless of that
    // attribute.
    const boxToggle = document.getElementById('profileRegionBoxToggle');
    if (boxToggle) boxToggle.checked = true;
    toggleProfileRegionBoxes(true);

    const stormToggle = document.getElementById('profileStormToggle');
    if (stormToggle) stormToggle.checked = true;
    toggleStormLayer(true);

    // Watch for resizes to reflow Leaflet map. Coalesced onto a single
    // rAF tick — calling invalidateSize() synchronously inside the observer
    // callback risks a ResizeObserver feedback loop (invalidateSize changes
    // layout -> fires the observer again) under rapid-fire notifications.
    const mapPanel = document.getElementById('profileMapPanel');
    if (mapPanel && window.ResizeObserver) {
      let resizeRaf = null;
      new ResizeObserver(() => {
        if (resizeRaf) return;
        resizeRaf = requestAnimationFrame(() => {
          resizeRaf = null;
          if (profileMap) profileMap.invalidateSize();
        });
      }).observe(mapPanel);
    }

    fetchProfileLocations();
    fetchStormTracks();
  }, 350);
}

function renderProfileMarkers(allData) {
  profileMapMarkers.clearLayers();
  const bounds = [];
  profilePlatformLocations = {};

  const activeType = state.profileType.toLowerCase();

  // Which platform (if any) is currently selected, so day-nav clicks can hold
  // the view on it instead of re-fitting to the whole-region overview.
  // For gliders and Argo, the dropdown races with fetchGliderIds()/fetchArgoIds()
  // on nav clicks (it briefly reads "Loading…"), so the sticky id is the
  // source of truth there.
  const platformSel = document.getElementById('platformId');
  const liveSel = platformSel && platformSel.value !== 'Loading…' ? platformSel.value : null;
  const selectedId = state.profileType === 'Glider' ? (state.stickyGliderId || liveSel)
    : state.profileType === 'Argo' ? (state.stickyArgoId || liveSel)
    : liveSel;
  let selectedLatLng = null;

  for (const [ptype, locations] of Object.entries(allData)) {
    if (ptype !== activeType) continue;

    const style = PLATFORM_STYLES[ptype] || PLATFORM_STYLES.argo;

    for (const [filename, loc] of Object.entries(locations)) {
      if (loc.lat == null || loc.lon == null) continue;

      const idLabel = loc.glider_id || loc.wmo || loc.wigos || filename.split('-')[0];
      const dotColor = ptype === 'argo' ? argoColor(idLabel) : style.fillColor;
      if (selectedId && idLabel === selectedId) selectedLatLng = [loc.lat, loc.lon];
      profilePlatformLocations[idLabel] = [loc.lat, loc.lon];

      const popupHtml =
        '<div style="font-family:Inter,sans-serif;font-size:13px;min-width:160px;">' +
        '<b style="color:' + dotColor + '">' + style.label + '</b><br>' +
        '<span style="color:#555">ID:</span> <b>' + idLabel + '</b><br>' +
        '<span style="color:#888;font-size:11px;">' + (loc.time || state.profileDate) + '</span><br>' +
        '<small style="color:#aaa">Click to load profile →</small>' +
        '</div>';

      // Draw a copy at lon and at lon±360 so platforms near the antimeridian
      // (Guam, Fiji) show up on whichever copy of the world is currently in
      // view, instead of requiring a long scroll from an Atlantic-centered map.
      for (const dLon of [-360, 0, 360]) {
        const latlng = [loc.lat, loc.lon + dLon];
        let marker;
        if (style.iconUrl) {
          const icon = L.icon({
            iconUrl: style.iconUrl,
            iconSize: [32, 32],
            iconAnchor: [16, 16],
            popupAnchor: [0, -18],
          });
          marker = L.marker(latlng, { icon }).addTo(profileMapMarkers);
        } else {
          marker = L.circleMarker(latlng, {
            radius: 9,
            color: 'white',
            weight: 2,
            fillColor: dotColor,
            fillOpacity: 0.9
          }).addTo(profileMapMarkers);
        }

        marker.bindPopup(popupHtml);

        marker.on('click', () => {
          marker.openPopup();
          // Zoom in a bit on the clicked platform, but never zoom back out
          profileMap.flyTo(latlng, Math.max(profileMap.getZoom(), PROFILE_CLICK_ZOOM), { duration: 0.6 });
          if (loc.url) {
            setImage('profileImageContainer', loc.url);
            document.getElementById('profileLabel').textContent =
              style.label + ' – ' + idLabel + ' – ' + (loc.time || state.profileDate);
            // Sync dropdown so back/forward navigation follows this platform
            const sel = document.getElementById('platformId');
            if (sel && Array.from(sel.options).some(o => o.value === idLabel)) {
              sel.value = idLabel;
            }
          }
        });
      }

      bounds.push([loc.lat, loc.lon]);
    }
  }

  if (profileMap) {
    profileMap.invalidateSize();
    if (selectedLatLng) {
      // A platform is selected (e.g. via day-nav) — hold the view on it
      // instead of re-fitting to the whole-region overview. Pick whichever
      // wrapped copy (lon, lon±360) is closest to the current view so this
      // doesn't pan all the way around the world to reach it.
      const center = profileMap.getCenter();
      const [slat, slon] = selectedLatLng;
      profileMap.setView([slat, nearestWrappedLon(slon, center.lng)], Math.max(profileMap.getZoom(), PROFILE_CLICK_ZOOM));
    } else if (bounds.length > 0 && !profileMapAutoFitDone) {
      // Only the first render for this profile type auto-fits — otherwise a
      // manual zoom (e.g. into the Gulf) would get silently undone back to
      // the whole merged region on every single day-nav click.
      bounds.length === 1
        ? profileMap.setView(bounds[0], 6)
        : profileMap.fitBounds(bounds, { padding: [40, 40], maxZoom: 8 });
      profileMapAutoFitDone = true;
    }
  }
}

async function fetchProfileLocations() {
  if (!profileMap) return;

  const cacheKey = `profile-pos|${state.profileDate}`;
  if (apiCache[cacheKey]) {
    renderProfileMarkers(apiCache[cacheKey]);
    fetchGliderTracks();
    return;
  }

  // Merged across every Argo/FVON region server-side — see ARGO_REGIONS/
  // FVON_REGIONS in app.py.
  const params = new URLSearchParams({ date: state.profileDate });

  try {
    const res = await fetch(`${API_BASE}/api/profile-positions?` + params);
    const allData = await res.json();
    apiCache[cacheKey] = allData;
    renderProfileMarkers(allData);
  } catch (err) {
    console.error('Error fetching profile locations:', err);
  }

  fetchGliderTracks();
}

async function fetchGliderTracks() {
  if (!profileMap || !profileMapTracks) return;
  profileMapTracks.clearLayers();
  // Tracks are glider-only — without this, switching to Argo/FVON left
  // whatever glider tracks were last drawn sitting on the map indefinitely,
  // since this was always fetching/drawing them regardless of profileType.
  if (state.profileType !== 'Glider') return;

  const params = new URLSearchParams({ date: state.profileDate });
  try {
    const res = await fetch(`${API_BASE}/api/glider-tracks?` + params);
    const tracks = await res.json();

    for (const [, points] of Object.entries(tracks)) {
      const valid = points.filter(p => p.lat != null && p.lon != null);
      if (valid.length < 2) continue;
      L.polyline(valid.map(p => [p.lat, p.lon]), {
        color: 'white',
        weight: 2.5,
        opacity: 0.75,
      }).addTo(profileMapTracks);
    }
  } catch (err) {
    console.error('Error fetching glider tracks:', err);
  }
}

// ─── Load profile image ────────────────────────────────────────────────────

async function loadProfile() {
  state.profileDate = document.getElementById('profileDate').value;
  const platformSel = document.getElementById('platformId');

  setLoading('profileImageContainer', 'Loading profile…');

  try {
    let res, data;
    if (state.profileType === 'Glider') {
      const glider_id = platformSel.value;
      if (!glider_id) {
        setUnavailable('profileImageContainer', 'Glider Profile');
        return;
      }
      const params = new URLSearchParams({ glider_id, date: state.profileDate });
      res = await fetch(`${API_BASE}/api/glider-profile?${params}`);
      data = await res.json();
    } else if (state.profileType === 'Argo') {
      const argo_key = platformSel.value;
      const entry = state.argoIds[argo_key];
      if (!entry) {
        setUnavailable('profileImageContainer', 'Argo Profile', argo_key ? {
          label: `Jump to ${argo_key}'s last profile`,
          onClick: () => jumpToLastArgoProfile(argo_key),
        } : undefined);
        return;
      }
      const params = new URLSearchParams({
        region: entry.region,
        filename: entry.filename,
        date: state.profileDate,
      });
      res = await fetch(`${API_BASE}/api/argo-profile?${params}`);
      data = await res.json();
    } else if (state.profileType === 'FVON') {
      const fvon_key = platformSel.value;
      const entry = state.fvonIds[fvon_key];
      if (!entry) {
        setUnavailable('profileImageContainer', 'FVON Profile');
        return;
      }
      const params = new URLSearchParams({
        region: entry.region,
        filename: entry.filename,
        date: state.profileDate,
      });
      res = await fetch(`${API_BASE}/api/fvon-profile?${params}`);
      data = await res.json();
    }

    if (data && data.available) {
      setImage('profileImageContainer', data.url);
      document.getElementById('profileLabel').textContent =
        `${state.profileType} – ${platformSel.value} – ${state.profileDate}`;
    } else if (state.profileType === 'Argo' && platformSel.value) {
      const argo_key = platformSel.value;
      setUnavailable('profileImageContainer', 'Argo Profile', {
        label: `Jump to ${argo_key}'s last profile`,
        onClick: () => jumpToLastArgoProfile(argo_key),
      });
    } else {
      setUnavailable('profileImageContainer', `${state.profileType} Profile`);
    }
  } catch (err) {
    console.error(err);
    setUnavailable('profileImageContainer', 'Profile');
    showToast('Network error while fetching profile.');
  }
}

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

// Map: region change → repopulate variable/depth
document.getElementById('mapRegion').addEventListener('change', (e) => {
  state.mapRegion = e.target.value;
  populateVarDepth(state.mapRegion);
  updateMapModelsConfig(state.mapRegion, state.mapVarDepth);
  autoUpdateMaps();
});

// Map: variable/depth change
document.getElementById('mapVarDepth').addEventListener('change', (e) => {
  state.mapVarDepth = e.target.value;
  updateMapModelsConfig(state.mapRegion, state.mapVarDepth);
  autoUpdateMaps();
});


// Profile: day navigation
// No unconditional autoUpdateProfile() here — if a platform was explicitly
// selected it stays sticky and refreshProfileIds()'s fetchGliderIds/
// fetchArgoIds restores it (or shows "unavailable") on their own; if
// nothing was selected, resetProfilePlaceholder() leaves it that way rather
// than silently loading whatever the repopulated dropdown defaults to.
document.getElementById('profileDayBack').addEventListener('click', () => {
  const cur = document.getElementById('platformId').value;
  if (cur && cur !== 'Loading…') {
    if (state.profileType === 'Glider') state.stickyGliderId = cur;
    if (state.profileType === 'Argo') state.stickyArgoId = cur;
  }
  state.profileDate = addDays(state.profileDate, -1);
  document.getElementById('profileDate').value = state.profileDate;
  if (!state.stickyGliderId && !state.stickyArgoId) resetProfilePlaceholder();
  refreshProfileIds();
});
document.getElementById('profileDayForward').addEventListener('click', () => {
  const cur = document.getElementById('platformId').value;
  if (cur && cur !== 'Loading…') {
    if (state.profileType === 'Glider') state.stickyGliderId = cur;
    if (state.profileType === 'Argo') state.stickyArgoId = cur;
  }
  state.profileDate = addDays(state.profileDate, 1);
  document.getElementById('profileDate').value = state.profileDate;
  if (!state.stickyGliderId && !state.stickyArgoId) resetProfilePlaceholder();
  refreshProfileIds();
});

// Profile: date input change — a calendar jump always resets sticky (it's
// not "the next/previous day for the same platform" the way day-nav is), so
// this always lands on the placeholder rather than auto-loading anything.
document.getElementById('profileDate').addEventListener('change', (e) => {
  state.stickyGliderId = null;
  state.stickyArgoId = null;
  state.profileDate = e.target.value;
  resetProfilePlaceholder();
  refreshProfileIds();
});

// Profile: type radio
document.querySelectorAll('input[name="profileType"]').forEach(radio => {
  radio.addEventListener('change', async (e) => {
    state.profileType = e.target.value;
    resetProfilePlaceholder();

    // No region picker for Argo/FVON — they're merged across every region
    // their pipelines actually publish to server-side (see ARGO_REGIONS/
    // FVON_REGIONS in app.py). *-latest-date already aggregates the same
    // way, so this is just "ask for the latest date for this type."
    const endpoint = state.profileType === 'FVON'   ? `${API_BASE}/api/fvon-latest-date`
                    : state.profileType === 'Argo'  ? `${API_BASE}/api/argo-latest-date`
                    : `${API_BASE}/api/glider-latest-date`;
    // Gliders' most recent per-day file is frequently still sparse (most of
    // the fleet hasn't been processed into it yet) — the same "today's file
    // often isn't complete" issue Argo/FVON already route around via their
    // own *-latest-date. Without this, opening the Profiles tab (Glider is
    // the default type) shows only a handful of platforms instead of the
    // fleet, and looks like gliders are broken rather than a data lag.
    try {
      const res = await fetch(endpoint);
      const data = await res.json();
      if (data.date) {
        state.profileDate = data.date;
        document.getElementById('profileDate').value = state.profileDate;
      }
    } catch (err) {
      console.error(err);
    }

    // Swap which type's search-region boxes are shown (Argo/Glider only —
    // see activeRegionBoxSet()).
    renderRegionBoxes();

    // Argo/Glider/FVON are different datasets with different extents — fit
    // once to the new type's markers, then leave the view alone across
    // day-nav within that type (see profileMapAutoFitDone).
    profileMapAutoFitDone = false;

    // No unconditional autoUpdateProfile() here — fetchGliderIds/
    // fetchArgoIds restore a sticky selection for this type on their own if
    // one exists; otherwise the resetProfilePlaceholder() above stands and
    // the user picks a platform themselves.
    refreshProfileIds();
  });
});

// Profile: platform ID change
document.getElementById('platformId').addEventListener('change', (e) => {
  state.stickyGliderId = null;  // manual selection clears sticky
  state.stickyArgoId = null;
  autoUpdateProfile();

  // Zoom to this platform the same way clicking its marker already does —
  // picking it from the dropdown wasn't moving the map at all before.
  const loc = profilePlatformLocations[e.target.value];
  if (loc && profileMap) {
    const [lat, lon] = loc;
    const center = profileMap.getCenter();
    profileMap.flyTo([lat, nearestWrappedLon(lon, center.lng)], Math.max(profileMap.getZoom(), PROFILE_CLICK_ZOOM), { duration: 0.6 });
  }
});

function autoUpdateProfile() {
  loadProfile();
}

function refreshProfileIds() {
  if (state.profileType === 'Glider') {
    fetchGliderIds();
  } else if (state.profileType === 'Argo') {
    fetchArgoIds();
  } else if (state.profileType === 'FVON') {
    fetchFvonIds();
  }
  fetchProfileLocations();
  fetchStormTracks();
}

// ─── Bootstrap tab switch → reload IDs ───────────────────────────────────────
document.getElementById('profiles-tab').addEventListener('shown.bs.tab', () => {
  initProfileMap();
  refreshProfileIds();
});

// Maps Archive: load when the tab is shown rather than eagerly at page
// load, matching Profiles/Adaptive Sampling — apiCache already makes a
// redundant call here cheap if nothing's changed since last time.
document.getElementById('maps-tab').addEventListener('shown.bs.tab', () => {
  loadMaps();
});


// ─── Adaptive Sampling state ──────────────────────────────────────────────────
const asgState = {
  date: document.getElementById('asgDate').value,   // populated from server's asg_latest
  time: '00Z',
  region:   document.getElementById('asgRegion').value,
  variable: document.getElementById('asgVariable').value,
  model:    document.getElementById('asgModel').value,
};

function stepAsgTime(direction) {
  const next = stepTimeHelper(direction, asgState.date, asgState.time);
  asgState.time = next.time;
  asgState.date = next.date;

  document.getElementById('asgDate').value = asgState.date;
  document.getElementById('asgTimeDisplay').textContent = asgState.time;
}

function preloadAdjacentASG() {
  const directions = ['back', 'forward'];
  directions.forEach(async (dir) => {
    const next = stepTimeHelper(dir, asgState.date, asgState.time);
    const params = new URLSearchParams({
      region:   asgState.region,
      variable: asgState.variable,
      model:    asgState.model,
      date:     next.date,
      time:     next.time,
    });
    const url = `${API_BASE}/api/adaptive-sampling?${params.toString()}`;
    if (!apiCache[url]) {
      try {
        const res = await fetch(url);
        const data = await res.json();
        apiCache[url] = data;
        if (data.available) preloadImage(data.url);
      } catch (e) {
        // silently fail on preload errors
      }
    } else {
      const data = apiCache[url];
      if (data.available) preloadImage(data.url);
    }
  });
}

async function loadAdaptiveSampling() {
  asgState.date     = document.getElementById('asgDate').value;
  asgState.region   = document.getElementById('asgRegion').value;
  asgState.variable = document.getElementById('asgVariable').value;
  asgState.model    = document.getElementById('asgModel').value;

  const params = new URLSearchParams({
    region:   asgState.region,
    variable: asgState.variable,
    model:    asgState.model,
    date:     asgState.date,
    time:     asgState.time,
  });

  const cacheKey = `${API_BASE}/api/adaptive-sampling?${params.toString()}`;

  const renderData = (data) => {
    if (data.available) {
      setImage('asgImageContainer', data.url);
      document.getElementById('asgLabel').textContent =
        `${asgState.region} · ${asgState.variable} · ${asgState.model} · ${asgState.date} ${asgState.time}`;
    } else {
      setUnavailable('asgImageContainer', 'Adaptive Sampling Guidance');
    }
  };

  if (apiCache[cacheKey]) {
    renderData(apiCache[cacheKey]);
    preloadAdjacentASG();
    return;
  }

  setLoading('asgImageContainer', 'Loading adaptive sampling figure…');

  try {
    const res  = await fetch(cacheKey);
    const data = await res.json();
    apiCache[cacheKey] = data;
    renderData(data);
    preloadAdjacentASG();
  } catch (err) {
    console.error(err);
    setUnavailable('asgImageContainer', 'Adaptive Sampling Guidance');
    showToast('Network error while fetching adaptive sampling figure.');
  }
}

// Adaptive Sampling: auto-update helper
function autoUpdateASG() {
  loadAdaptiveSampling();
}

// Adaptive Sampling: time step buttons
document.getElementById('asgTimeBack').addEventListener('click', () => {
  stepAsgTime('back');
  autoUpdateASG();
});
document.getElementById('asgTimeForward').addEventListener('click', () => {
  stepAsgTime('forward');
  autoUpdateASG();
});

// Adaptive Sampling: date change
document.getElementById('asgDate').addEventListener('change', (e) => {
  asgState.date = e.target.value;
  autoUpdateASG();
});

// Adaptive Sampling: region, variable, model dropdowns
['asgRegion', 'asgVariable', 'asgModel'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    autoUpdateASG();
  });
});

// Adaptive Sampling: load when the tab is shown, same as Maps/Profiles —
// apiCache already makes a redundant call here cheap if nothing's changed.
document.getElementById('adaptive-tab').addEventListener('shown.bs.tab', () => {
  loadAdaptiveSampling();
});

function showDashboardTabFromHash() {
  const targetByHash = {
    '#overview': 'overview-tab',
    '#maps': 'maps-tab',
    '#profiles': 'profiles-tab',
    '#adaptive': 'adaptive-tab',
  };
  const tabId = targetByHash[window.location.hash];
  if (!tabId) return;
  const tab = document.getElementById(tabId);
  if (tab) bootstrap.Tab.getOrCreateInstance(tab).show();
}

window.addEventListener('DOMContentLoaded', showDashboardTabFromHash);
window.addEventListener('hashchange', showDashboardTabFromHash);


// ══════════════════════════════════════════════════════════
// ─── OVERVIEW TAB ───
// ══════════════════════════════════════════════════════════

function preloadOverviewImages(data) {
  if (!data) return;
  preloadPanels(data);
}

function preloadOverviewAdjacentProducts(region, currentVar, currentDep) {
  const ovVariables = ["temperature", "salinity", "currents", "ocean_heat_content"];
  const idx = ovVariables.indexOf(currentVar);
  if (idx === -1) return;
  
  const prevIdx = (idx - 1 + ovVariables.length) % ovVariables.length;
  const nextIdx = (idx + 1) % ovVariables.length;
  
  const varsToPreload = [ovVariables[prevIdx], ovVariables[nextIdx]];
  
  varsToPreload.forEach(async (v) => {
    const prod = v === 'ocean_heat_content' ? v : v + '_' + currentDep;
    const url = `${API_BASE}/api/overview-latest?region=${encodeURIComponent(region)}&variable_depth=${encodeURIComponent(prod)}`;
    if (!apiCache[url]) {
      try {
        const res = await fetch(url);
        const data = await res.json();
        apiCache[url] = data;
        preloadOverviewImages(data);
      } catch (e) {
        // silently fail on preload errors
      }
    } else {
      preloadOverviewImages(apiCache[url]);
    }
  });
}

function loadOverview(allowFallback = true) {
  const regionInput = document.querySelector('input[name="ov_region"]:checked');
  const region = regionInput ? regionInput.value : state.ovRegions[state.ovCurrentRegionIndex];
  const ovVar = document.querySelector('input[name="ov_variable"]:checked').value;
  const depthInput = document.querySelector('input[name="ov_depth"]:checked');
  const ovDep = depthInput ? depthInput.value : '0m';
  const product = ovVar === 'ocean_heat_content' ? ovVar : ovVar + '_' + ovDep;
  const model = document.querySelector('input[name="ov_model"]:checked').value;

  const url = `${API_BASE}/api/overview-latest?region=${encodeURIComponent(region)}&variable_depth=${encodeURIComponent(product)}`;
  
  // UI updates
  document.getElementById('ov_region_label').textContent = region;
  document.getElementById('ov_placeholder').classList.add('d-none');
  document.getElementById('ov_unavailable').classList.add('d-none');
  document.getElementById('ov_image').classList.add('d-none');
  document.getElementById('ov_fallback_note').classList.add('d-none');

  const renderData = (data) => {
    document.getElementById('ov_loading').classList.add('d-none');
    if (!data.available) {
      document.getElementById('ov_unavailable').classList.remove('d-none');
      document.getElementById('ov_date_label').textContent = "YYYY-MM-DD";
      document.getElementById('ov_time_badge').textContent = "--Z";
      return;
    }

    // Each model comparison finishes rendering on its own schedule, so the
    // selected model may not be ready yet even though others are. On automatic
    // loads (region/variable/depth change, initial load) fall back to the
    // first visible model that is ready. A model the user explicitly picked
    // (clicking it, or the prev/next arrows) is always respected as-is —
    // if it's not ready, say so instead of silently swapping it out.
    let chosenModel = model;
    let usedFallback = false;
    if (allowFallback && !(data[chosenModel] && data[chosenModel].available)) {
      const fallback = getVisibleOverviewModels().find(m => data[m] && data[m].available);
      if (fallback) {
        chosenModel = fallback;
        usedFallback = true;
        const radio = document.querySelector(`input[name="ov_model"][value="${chosenModel}"]`);
        if (radio) radio.checked = true;
      }
    }

    // Make a silent model substitution visible — otherwise a user can be
    // looking at a different comparison than the one still checked in their
    // own sidebar without any indication it happened.
    const fallbackNote = document.getElementById('ov_fallback_note');
    if (usedFallback) {
      const label = document.querySelector(`label[for="ov_mod_${chosenModel}"]`);
      fallbackNote.textContent = `Showing ${label ? label.textContent.trim() : chosenModel} — your selected model isn't ready yet`;
      fallbackNote.classList.remove('d-none');
    } else {
      fallbackNote.classList.add('d-none');
    }

    const imgData = data[chosenModel];
    if (imgData && imgData.available) {
      const img = document.getElementById('ov_image');
      img.onerror = () => {
        img.classList.add('d-none');
        document.getElementById('ov_unavailable').classList.remove('d-none');
      };
      img.src = imgData.url;
      img.classList.remove('d-none');

      document.getElementById('ov_date_label').textContent = data.date;
      document.getElementById('ov_time_badge').textContent = data.time;
    } else {
      document.getElementById('ov_unavailable').classList.remove('d-none');
    }
  };

  if (apiCache[url]) {
    renderData(apiCache[url]);
    preloadOverviewImages(apiCache[url]);
    preloadOverviewAdjacentProducts(region, ovVar, ovDep);
    return;
  }

  document.getElementById('ov_loading').classList.remove('d-none');

  fetch(url)
    .then(r => r.json())
    .then(data => {
      apiCache[url] = data;
      renderData(data);
      preloadOverviewImages(data);
      preloadOverviewAdjacentProducts(region, ovVar, ovDep);
    })
    .catch(err => {
      console.error(err);
      document.getElementById('ov_loading').classList.add('d-none');
      document.getElementById('ov_unavailable').classList.remove('d-none');
    });
}

// Overview Event Listeners

// Variable: show/hide depth group (OHC has no depth dimension), rebuild depth
// buttons (each variable has its own depth set), reload
document.querySelectorAll('input[name="ov_variable"]').forEach(el => {
  el.addEventListener('change', (e) => {
    const region = document.querySelector('input[name="ov_region"]:checked').value;
    document.getElementById('ov_depth_group').style.display =
      e.target.value === 'ocean_heat_content' ? 'none' : '';
    updateOverviewModelVisibility(region, e.target.value);
    updateOverviewDepths(region, e.target.value);
    updateOverviewGoesVisibility();
    loadOverview();
  });
});

// Model: reload on change. A model the user explicitly clicked should be
// respected as-is (no silent fallback) — show "not available" if it isn't ready.
document.querySelectorAll('input[name="ov_model"]').forEach(el => {
  el.addEventListener('change', () => loadOverview(false));
});

// Region: update model labels/visibility, rebuild depth buttons, reload
document.querySelectorAll('input[name="ov_region"]').forEach(el => {
  el.addEventListener('change', (e) => {
    const region = e.target.value;
    const variable = document.querySelector('input[name="ov_variable"]:checked').value;
    state.ovCurrentRegionIndex = state.ovRegions.indexOf(region);
    updateOverviewModelVisibility(region, variable);
    updateOverviewDepths(region, variable);
    updateOverviewGoesVisibility();
    loadOverview();
  });
});

function updateOverviewModelVisibility(region, variable) { updateOverviewPanels(); }

function getVisibleOverviewModels() {
  return Array.from(document.querySelectorAll('input[name="ov_model"]'))
    .filter(el => el.style.display !== 'none')
    .map(el => el.value);
}

function toggleOverviewModel(direction = 1) {
  const models = getVisibleOverviewModels();
  const currentModel = document.querySelector('input[name="ov_model"]:checked').value;
  const idx = models.indexOf(currentModel);
  const nextIdx = (idx + direction + models.length) % models.length;
  document.querySelector(`input[name="ov_model"][value="${models[nextIdx]}"]`).checked = true;
  loadOverview(false);
}

document.getElementById('ov_prev_model').addEventListener('click', () => toggleOverviewModel(-1));
document.getElementById('ov_next_model').addEventListener('click', () => toggleOverviewModel(1));


// Keyboard Navigation (Overview + Maps tabs)
document.addEventListener('keydown', (e) => {
  // Skip if user is typing in a form field, or the lightbox is open (its own
  // keydown listener handles Escape; arrow keys shouldn't change the image
  // behind it while it's showing a snapshot of a specific image).
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (lightbox.classList.contains('show')) return;

  const overviewPane = document.getElementById('overview-pane');
  const mapsPane     = document.getElementById('maps-pane');

  const inOverview = overviewPane && overviewPane.classList.contains('active');
  const inMaps     = mapsPane     && mapsPane.classList.contains('active');

  if (inOverview) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      toggleOverviewModel(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      toggleOverviewModel(1);
    }
  } else if (inMaps) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      stepMapModel(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      stepMapModel(1);
    }
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
// Collapses each tab's controls sidebar by default on mobile only (desktop
// always shows it — the header is only a toggle below the 900px breakpoint
// that also stacks the sidebar above the content, see app.css).
function initCollapsibleControls() {
  const mq = window.matchMedia('(max-width: 900px)');
  const headers = document.querySelectorAll('.controls-header[data-collapse-target]');

  const applyState = () => {
    headers.forEach(header => {
      const body = document.getElementById(header.dataset.collapseTarget);
      if (!body) return;
      const collapsed = mq.matches;
      body.classList.toggle('is-collapsed', collapsed);
      header.setAttribute('aria-expanded', String(!collapsed));
    });
  };
  applyState();
  mq.addEventListener('change', applyState);

  headers.forEach(header => {
    const body = document.getElementById(header.dataset.collapseTarget);
    if (!body) return;
    const toggle = () => {
      if (!mq.matches) return; // desktop: not a toggle, always expanded
      const collapsed = body.classList.toggle('is-collapsed');
      header.setAttribute('aria-expanded', String(!collapsed));
    };
    header.addEventListener('click', toggle);
    header.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
  });
}

(function init() {
  initCollapsibleControls();
  initRegionBoxOpacityControl();

  // Bootstrap only auto-wires data-bs-toggle for modal/collapse/etc. —
  // popovers need an explicit `new bootstrap.Popover(el)` per element.
  document.querySelectorAll('[data-bs-toggle="popover"]').forEach(el => new bootstrap.Popover(el));

  // Populate variable/depth for default region
  populateVarDepth(state.mapRegion);

  // Populate map tabs based on default region + variable/depth
  updateMapModelsConfig(state.mapRegion, state.mapVarDepth);

  // Initialize overview depth buttons for the default region + variable
  const defaultOvRegion = document.querySelector('input[name="ov_region"]:checked');
  const defaultOvVariable = document.querySelector('input[name="ov_variable"]:checked');
  if (defaultOvRegion && defaultOvVariable) {
    updateOverviewModelVisibility(defaultOvRegion.value, defaultOvVariable.value);
    updateOverviewDepths(defaultOvRegion.value, defaultOvVariable.value);
  }
  updateOverviewGoesVisibility();

  // Overview is the default active tab, so it loads immediately — the same
  // "load when shown" rule the other tabs follow via their own
  // shown.bs.tab listeners below, since Overview's "shown" moment IS page
  // load. Maps Archive used to also eager-load here regardless of whether
  // the user ever opened that tab; it now follows the same lazy pattern as
  // Profiles/Adaptive Sampling (see the maps-tab listener below).
  loadOverview();

  // Trigger profile type setup
  const checkedRadio = document.querySelector('input[name="profileType"]:checked');
  if (checkedRadio) {
    checkedRadio.dispatchEvent(new Event('change'));
  }
})();
