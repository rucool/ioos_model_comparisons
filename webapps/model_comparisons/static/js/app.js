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
  argoRegion: document.getElementById('argoRegion').value,
  gliderIds: [],
  argoIds: {},
  stickyGliderId: null,  // set on day nav to restore same glider on new date
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

function showToast(msg) {
  document.getElementById('alertToastMsg').textContent = msg;
  const el = document.getElementById('alertToast');
  const toast = bootstrap.Toast.getOrCreateInstance(el, { delay: 4000 });
  toast.show();
}

// ─── Variable/Depth dropdown population ──────────────────────────────────────

function populateVarDepth(region) {
  const info = REGION_INFO[region];
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

// ─── Overview: dynamic depth buttons ─────────────────────────────────────────

function updateOverviewDepths(region) {
  const info = REGION_INFO[region];
  const container = document.getElementById('ov_depth_btns');
  if (!info || !info.depths || !container) return;

  container.innerHTML = '';
  info.depths.forEach((depth, i) => {
    const safeId = 'ov_dep_' + depth.replace(/[^a-z0-9]/gi, '_');

    const input = document.createElement('input');
    input.type = 'radio';
    input.className = 'btn-check';
    input.name = 'ov_depth';
    input.id = safeId;
    input.value = depth;
    input.autocomplete = 'off';
    if (i === 0) input.checked = true;
    input.addEventListener('change', loadOverview);

    const label = document.createElement('label');
    label.className = 'btn btn-outline-ocean btn-sm';
    label.setAttribute('for', safeId);
    label.textContent = depth;

    container.appendChild(input);
    container.appendChild(label);
  });
}

// ─── Maps – Toggle State ──────────────────────────────────────────────────────

let mapModels = [
  { key: 'copernicus', label: 'RTOFS vs. Copernicus (CMEMS)' },
  { key: 'espc',       label: 'RTOFS vs. ESPC' },
  { key: 'goes',       label: 'RTOFS vs. GOES' },
];
let mapCurrentModelIdx = 0;
let mapCachedUrls = { copernicus: null, espc: null, espc_cmems: null, goes: null };

function updateMapModelsConfig(region) {
  if (region === 'Guam' || region === 'Fiji') {
    mapModels = [
      { key: 'espc_cmems', label: 'ESPC vs. CMEMS' }
    ];
  } else {
    mapModels = [
      { key: 'copernicus', label: 'RTOFS vs. Copernicus (CMEMS)' },
      { key: 'espc',       label: 'RTOFS vs. ESPC' },
      { key: 'goes',       label: 'RTOFS vs. GOES' },
    ];
  }

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
        if (data.copernicus && data.copernicus.available) preloadImage(data.copernicus.url);
        if (data.espc && data.espc.available) preloadImage(data.espc.url);
        if (data.espc_cmems && data.espc_cmems.available) preloadImage(data.espc_cmems.url);
        if (data.goes && data.goes.available) preloadImage(data.goes.url);
      } catch (e) {
        // silently fail on preload errors
      }
    } else {
      const data = apiCache[url];
      if (data.copernicus && data.copernicus.available) preloadImage(data.copernicus.url);
      if (data.espc && data.espc.available) preloadImage(data.espc.url);
      if (data.espc_cmems && data.espc_cmems.available) preloadImage(data.espc_cmems.url);
      if (data.goes && data.goes.available) preloadImage(data.goes.url);
    }
  });
}

async function loadMaps() {
  state.mapDate = document.getElementById('mapDate').value;
  state.mapRegion = document.getElementById('mapRegion').value;
  state.mapVarDepth = document.getElementById('mapVarDepth').value;

  const btn = document.getElementById('mapUpdateBtn');
  btn.disabled = true;

  const params = new URLSearchParams({
    region: state.mapRegion,
    variable_depth: state.mapVarDepth,
    date: state.mapDate,
    time: state.mapTime,
  });
  
  const cacheKey = `${API_BASE}/api/maps?${params.toString()}`;
  
  const renderData = (data) => {
    mapCachedUrls.copernicus = data.copernicus && data.copernicus.available ? data.copernicus.url : null;
    mapCachedUrls.espc       = data.espc && data.espc.available       ? data.espc.url       : null;
    mapCachedUrls.espc_cmems = data.espc_cmems && data.espc_cmems.available ? data.espc_cmems.url : null;
    mapCachedUrls.goes       = data.goes && data.goes.available       ? data.goes.url       : null;

    renderMapModel();

    if (!mapCachedUrls.copernicus && !mapCachedUrls.espc && !mapCachedUrls.espc_cmems && !mapCachedUrls.goes) {
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
    mapCachedUrls = { copernicus: null, espc: null, espc_cmems: null, goes: null };
    setUnavailable('mapImageContainer', 'Maps');
    showToast('Network error while fetching images.');
  } finally {
    btn.disabled = false;
  }
}


// ─── Profiles – Glider ───────────────────────────────────────────────────────

async function fetchGliderIds() {
  const sel = document.getElementById('platformId');
  const noMsg = document.getElementById('noPlatformMsg');
  sel.innerHTML = '<option>Loading…</option>';
  noMsg.classList.add('d-none');

  const params = new URLSearchParams({ date: state.profileDate });
  try {
    const res = await fetch(`${API_BASE}/api/glider-ids?${params}`);
    const data = await res.json();
    state.gliderIds = data.ids || [];
    sel.innerHTML = '';
    if (state.gliderIds.length === 0) {
      sel.innerHTML = '<option value="">No gliders available</option>';
      noMsg.classList.remove('d-none');
    } else {
      for (const id of state.gliderIds) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
      if (state.stickyGliderId) {
        if (state.gliderIds.includes(state.stickyGliderId)) {
          // Glider has a plot on this date — select and load it
          sel.value = state.stickyGliderId;
          loadProfile();
        } else {
          // Glider has no plot on this date — show unavailable, keep sticky for further nav
          setUnavailable('profileImageContainer', 'Glider Profile');
        }
      }
    }
  } catch {
    sel.innerHTML = '<option value="">Error loading IDs</option>';
  }
}

// ─── Profiles – Argo ─────────────────────────────────────────────────────────

async function fetchArgoIds() {
  const sel = document.getElementById('platformId');
  const noMsg = document.getElementById('noPlatformMsg');
  sel.innerHTML = '<option>Loading…</option>';
  noMsg.classList.add('d-none');
  state.argoRegion = document.getElementById('argoRegion').value;

  const params = new URLSearchParams({
    region: state.argoRegion,
    date: state.profileDate,
  });
  try {
    const res = await fetch(`${API_BASE}/api/argo-ids?${params}`);
    const data = await res.json();
    state.argoIds = data.ids || {};
    const keys = Object.keys(state.argoIds).sort();
    sel.innerHTML = '';
    if (keys.length === 0) {
      sel.innerHTML = '<option value="">No Argo floats available</option>';
      noMsg.classList.remove('d-none');
    } else {
      for (const id of keys) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
    }
  } catch {
    sel.innerHTML = '<option value="">Error loading IDs</option>';
  }
}

// ─── Profiles – FVON ─────────────────────────────────────────────────────────

async function fetchFvonIds() {
  const sel = document.getElementById('platformId');
  const noMsg = document.getElementById('noPlatformMsg');
  sel.innerHTML = '<option>Loading…</option>';
  noMsg.classList.add('d-none');
  state.argoRegion = document.getElementById('argoRegion').value;

  const params = new URLSearchParams({
    region: state.argoRegion,
    date: state.profileDate,
  });
  try {
    const res = await fetch(`${API_BASE}/api/fvon-ids?${params}`);
    const data = await res.json();
    state.fvonIds = data.ids || {};
    const keys = Object.keys(state.fvonIds).sort();
    sel.innerHTML = '';
    if (keys.length === 0) {
      sel.innerHTML = '<option value="">No FVON profiles available</option>';
      noMsg.classList.remove('d-none');
    } else {
      for (const id of keys) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
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

let profileMap = null;
let profileMapTracks = null;
let profileMapMarkers = null;

function initProfileMap() {
  setTimeout(() => {
    const container = document.getElementById('profileMapContainer');
    if (!container) return;

    if (profileMap) {
      profileMap.invalidateSize();
      fetchProfileLocations();
      return;
    }

    profileMap = L.map('profileMapContainer', {
      zoomControl: true,
      scrollWheelZoom: true
    }).setView([30.0, -75.0], 4);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }).addTo(profileMap);

    profileMapTracks  = L.layerGroup().addTo(profileMap);
    profileMapMarkers = L.layerGroup().addTo(profileMap);

    // Watch for resizes to reflow Leaflet map
    const mapPanel = document.getElementById('profileMapPanel');
    if (mapPanel && window.ResizeObserver) {
      new ResizeObserver(() => {
        if (profileMap) profileMap.invalidateSize();
      }).observe(mapPanel);
    }

    fetchProfileLocations();
  }, 350);
}

function renderProfileMarkers(allData) {
  profileMapMarkers.clearLayers();
  const bounds = [];

  const activeType = state.profileType.toLowerCase();

  for (const [ptype, locations] of Object.entries(allData)) {
    if (ptype !== activeType) continue;

    const style = PLATFORM_STYLES[ptype] || PLATFORM_STYLES.argo;

    for (const [filename, loc] of Object.entries(locations)) {
      if (loc.lat == null || loc.lon == null) continue;

      const idLabel = loc.glider_id || loc.wmo || loc.wigos || filename.split('-')[0];
      const dotColor = ptype === 'argo' ? argoColor(idLabel) : style.fillColor;

      let marker;
      if (style.iconUrl) {
        const icon = L.icon({
          iconUrl: style.iconUrl,
          iconSize: [32, 32],
          iconAnchor: [16, 16],
          popupAnchor: [0, -18],
        });
        marker = L.marker([loc.lat, loc.lon], { icon }).addTo(profileMapMarkers);
      } else {
        marker = L.circleMarker([loc.lat, loc.lon], {
          radius: 9,
          color: 'white',
          weight: 2,
          fillColor: dotColor,
          fillOpacity: 0.9
        }).addTo(profileMapMarkers);
      }

      marker.bindPopup(
        '<div style="font-family:Inter,sans-serif;font-size:13px;min-width:160px;">' +
        '<b style="color:' + dotColor + '">' + style.label + '</b><br>' +
        '<span style="color:#555">ID:</span> <b>' + idLabel + '</b><br>' +
        '<span style="color:#888;font-size:11px;">' + (loc.time || state.profileDate) + '</span><br>' +
        '<small style="color:#aaa">Click to load profile →</small>' +
        '</div>'
      );

      marker.on('click', () => {
        marker.openPopup();
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

      bounds.push([loc.lat, loc.lon]);
    }
  }

  if (profileMap) {
    profileMap.invalidateSize();
    if (bounds.length > 0) {
      bounds.length === 1
        ? profileMap.setView(bounds[0], 6)
        : profileMap.fitBounds(bounds, { padding: [40, 40], maxZoom: 8 });
    }
  }
}

async function fetchProfileLocations() {
  if (!profileMap) return;

  const cacheKey = `profile-pos|${state.argoRegion}|${state.profileDate}`;
  if (apiCache[cacheKey]) {
    renderProfileMarkers(apiCache[cacheKey]);
    fetchGliderTracks();
    return;
  }

  const params = new URLSearchParams({
    region: state.argoRegion,
    date: state.profileDate
  });

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
  const btn = document.getElementById('profileUpdateBtn');
  btn.disabled = true;
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
      const filename = state.argoIds[argo_key];
      if (!filename) {
        setUnavailable('profileImageContainer', 'Argo Profile');
        return;
      }
      const params = new URLSearchParams({
        region: state.argoRegion,
        filename,
        date: state.profileDate,
      });
      res = await fetch(`${API_BASE}/api/argo-profile?${params}`);
      data = await res.json();
    } else if (state.profileType === 'FVON') {
      const fvon_key = platformSel.value;
      const filename = state.fvonIds[fvon_key];
      if (!filename) {
        setUnavailable('profileImageContainer', 'FVON Profile');
        return;
      }
      const params = new URLSearchParams({
        region: state.argoRegion,
        filename,
        date: state.profileDate,
      });
      res = await fetch(`${API_BASE}/api/fvon-profile?${params}`);
      data = await res.json();
    }

    if (data && data.available) {
      setImage('profileImageContainer', data.url);
      document.getElementById('profileLabel').textContent =
        `${state.profileType} – ${platformSel.value} – ${state.profileDate}`;
    } else {
      setUnavailable('profileImageContainer', `${state.profileType} Profile`);
    }
  } catch (err) {
    console.error(err);
    setUnavailable('profileImageContainer', 'Profile');
    showToast('Network error while fetching profile.');
  } finally {
    btn.disabled = false;
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
  if (document.getElementById('mapAutoUpdate').checked) loadMaps();
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
  updateMapModelsConfig(state.mapRegion);
  populateVarDepth(state.mapRegion);
  autoUpdateMaps();
});

// Map: variable/depth change
document.getElementById('mapVarDepth').addEventListener('change', () => {
  autoUpdateMaps();
});


// Profile: day navigation
document.getElementById('profileDayBack').addEventListener('click', () => {
  if (state.profileType === 'Glider') {
    const cur = document.getElementById('platformId').value;
    if (cur && cur !== 'Loading…') state.stickyGliderId = cur;
  }
  state.profileDate = addDays(state.profileDate, -1);
  document.getElementById('profileDate').value = state.profileDate;
  refreshProfileIds();
  if (!state.stickyGliderId) autoUpdateProfile();
});
document.getElementById('profileDayForward').addEventListener('click', () => {
  if (state.profileType === 'Glider') {
    const cur = document.getElementById('platformId').value;
    if (cur && cur !== 'Loading…') state.stickyGliderId = cur;
  }
  state.profileDate = addDays(state.profileDate, 1);
  document.getElementById('profileDate').value = state.profileDate;
  refreshProfileIds();
  if (!state.stickyGliderId) autoUpdateProfile();
});

// Profile: date input change
document.getElementById('profileDate').addEventListener('change', (e) => {
  state.stickyGliderId = null;  // calendar jump resets sticky
  state.profileDate = e.target.value;
  refreshProfileIds();
  autoUpdateProfile();
});

// Profile: type radio
document.querySelectorAll('input[name="profileType"]').forEach(radio => {
  radio.addEventListener('change', async (e) => {
    state.profileType = e.target.value;
    const argoGroup = document.getElementById('argoRegionGroup');
    const argoRegionSelect = document.getElementById('argoRegion');

    if (state.profileType === 'Argo' || state.profileType === 'FVON') {
      argoGroup.style.setProperty('display', 'block', 'important');

      // Filter options based on profile type
      Array.from(argoRegionSelect.options).forEach(opt => {
        if (state.profileType === 'FVON') {
          if (opt.value === 'Fiji' || opt.value === 'Bahamas') {
            opt.style.display = '';
          } else {
            opt.style.display = 'none';
          }
        } else {
          // Argo uses MAP_REGIONS
          if (MAP_REGIONS.includes(opt.value)) {
            opt.style.display = '';
          } else {
            opt.style.display = 'none';
          }
        }
      });

      // If the currently selected option is hidden, switch to the first visible one
      const selectedOption = argoRegionSelect.options[argoRegionSelect.selectedIndex];
      if (selectedOption && selectedOption.style.display === 'none') {
        const firstVisible = Array.from(argoRegionSelect.options).find(o => o.style.display !== 'none');
        if (firstVisible) {
          argoRegionSelect.value = firstVisible.value;
          state.argoRegion = firstVisible.value;
        }
      }

      if (state.profileType === 'FVON' || state.profileType === 'Argo') {
        const endpoint = state.profileType === 'FVON' ? `${API_BASE}/api/fvon-latest-date` : `${API_BASE}/api/argo-latest-date`;
        const params = new URLSearchParams({ region: state.argoRegion });
        try {
          const res = await fetch(`${endpoint}?${params}`);
          const data = await res.json();
          if (data.date) {
            state.profileDate = data.date;
            document.getElementById('profileDate').value = state.profileDate;
          }
        } catch (err) {
          console.error(err);
        }
      }

    } else {
      argoGroup.style.setProperty('display', 'none', 'important');
    }
    refreshProfileIds();
    autoUpdateProfile();
  });
});

// Profile: argo region change
document.getElementById('argoRegion').addEventListener('change', async (e) => {
  state.argoRegion = e.target.value;
  if (state.profileType === 'FVON' || state.profileType === 'Argo') {
    const endpoint = state.profileType === 'FVON' ? '${API_BASE}/api/fvon-latest-date' : '${API_BASE}/api/argo-latest-date';
    const params = new URLSearchParams({ region: state.argoRegion });
    try {
      const res = await fetch(`${endpoint}?${params}`);
      const data = await res.json();
      if (data.date) {
        state.profileDate = data.date;
        document.getElementById('profileDate').value = state.profileDate;
      }
    } catch (err) {
      console.error(err);
    }
  }
  refreshProfileIds();
  autoUpdateProfile();
});

// Profile: platform ID change
document.getElementById('platformId').addEventListener('change', () => {
  state.stickyGliderId = null;  // manual selection clears sticky
  autoUpdateProfile();
});

function autoUpdateProfile() {
  if (document.getElementById('profileAutoUpdate').checked) loadProfile();
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
}

// ─── Bootstrap tab switch → reload IDs ───────────────────────────────────────
document.getElementById('profiles-tab').addEventListener('shown.bs.tab', () => {
  initProfileMap();
  refreshProfileIds();
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

  const btn = document.getElementById('asgUpdateBtn');
  btn.disabled = true;

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
    btn.disabled = false;
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
  } finally {
    btn.disabled = false;
  }
}

// Adaptive Sampling: auto-update helper
function autoUpdateASG() {
  if (document.getElementById('asgAutoUpdate').checked) loadAdaptiveSampling();
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

// Adaptive Sampling: auto-load when tab is first shown
document.getElementById('adaptive-tab').addEventListener('shown.bs.tab', () => {
  const c = document.getElementById('asgImageContainer');
  if (c && c.querySelector('.placeholder-state')) {
    loadAdaptiveSampling();
  }
});


// ══════════════════════════════════════════════════════════
// ─── OVERVIEW TAB ───
// ══════════════════════════════════════════════════════════

function preloadOverviewImages(data) {
  if (!data) return;
  if (data.copernicus && data.copernicus.url) preloadImage(data.copernicus.url);
  if (data.espc && data.espc.url) preloadImage(data.espc.url);
  if (data.espc_cmems && data.espc_cmems.url) preloadImage(data.espc_cmems.url);
  if (data.goes && data.goes.url) preloadImage(data.goes.url);
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

function loadOverview() {
  const regionInput = document.querySelector('input[name="ov_region"]:checked');
  const region = regionInput ? regionInput.value : state.ovRegions[state.ovCurrentRegionIndex];
  const ovVar = document.querySelector('input[name="ov_variable"]:checked').value;
  const depthInput = document.querySelector('input[name="ov_depth"]:checked');
  const ovDep = depthInput ? depthInput.value : '0m';
  const product = ovVar === 'ocean_heat_content' ? ovVar : ovVar + '_' + ovDep;
  let model = document.querySelector('input[name="ov_model"]:checked').value;

  if (region === 'Guam' || region === 'Fiji') {
    model = 'espc_cmems';
  }

  const url = `${API_BASE}/api/overview-latest?region=${encodeURIComponent(region)}&variable_depth=${encodeURIComponent(product)}`;
  
  // UI updates
  document.getElementById('ov_region_label').textContent = region;
  document.getElementById('ov_placeholder').classList.add('d-none');
  document.getElementById('ov_unavailable').classList.add('d-none');
  document.getElementById('ov_image').classList.add('d-none');

  const renderData = (data) => {
    document.getElementById('ov_loading').classList.add('d-none');
    if (!data.available) {
      document.getElementById('ov_unavailable').classList.remove('d-none');
      document.getElementById('ov_date_label').textContent = "YYYY-MM-DD";
      document.getElementById('ov_time_badge').textContent = "--Z";
      return;
    }
    
    const imgData = data[model];
    if (imgData && imgData.url) {
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

// Variable: show/hide depth group (OHC has no depth dimension)
document.querySelectorAll('input[name="ov_variable"]').forEach(el => {
  el.addEventListener('change', (e) => {
    document.getElementById('ov_depth_group').style.display =
      e.target.value === 'ocean_heat_content' ? 'none' : '';
    loadOverview();
  });
});

// Model: reload on change
document.querySelectorAll('input[name="ov_model"]').forEach(el => {
  el.addEventListener('change', loadOverview);
});

// Region: update model labels/visibility, rebuild depth buttons, reload
document.querySelectorAll('input[name="ov_region"]').forEach(el => {
  el.addEventListener('change', (e) => {
    const region = e.target.value;
    state.ovCurrentRegionIndex = state.ovRegions.indexOf(region);
    const isSpecial = region === 'Guam' || region === 'Fiji';
    const copEl  = document.getElementById('ov_mod_copernicus');
    const goesEl = document.getElementById('ov_mod_goes');
    const espcEl = document.getElementById('ov_mod_espc');
    [copEl, goesEl].forEach(btn => {
      btn.style.display = isSpecial ? 'none' : '';
      btn.nextElementSibling.style.display = isSpecial ? 'none' : '';
    });
    espcEl.nextElementSibling.textContent = isSpecial ? 'ESPC vs CMEMS' : 'ESPC';
    if (isSpecial) espcEl.checked = true;
    updateOverviewDepths(region);
    loadOverview();
  });
});

function toggleOverviewModel(direction = 1) {
  const region = document.querySelector('input[name="ov_region"]:checked').value;
  const isSpecialRegion = region === 'Guam' || region === 'Fiji';
  if (isSpecialRegion) return;

  const models = ['espc', 'copernicus', 'goes'];
  const currentModel = document.querySelector('input[name="ov_model"]:checked').value;
  const idx = models.indexOf(currentModel);
  const nextIdx = (idx + direction + models.length) % models.length;
  document.querySelector(`input[name="ov_model"][value="${models[nextIdx]}"]`).checked = true;
  loadOverview();
}

document.getElementById('ov_prev_model').addEventListener('click', () => toggleOverviewModel(-1));
document.getElementById('ov_next_model').addEventListener('click', () => toggleOverviewModel(1));


// Keyboard Navigation (Overview + Maps tabs)
document.addEventListener('keydown', (e) => {
  // Skip if user is typing in a form field
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

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
(function init() {
  // Populate map tabs based on default region
  updateMapModelsConfig(state.mapRegion);

  // Populate variable/depth for default region
  populateVarDepth(state.mapRegion);

  // Initialize overview depth buttons for the default region
  const defaultOvRegion = document.querySelector('input[name="ov_region"]:checked');
  if (defaultOvRegion) updateOverviewDepths(defaultOvRegion.value);

  // Pre-load maps on page load
  loadOverview();
  loadMaps();

  // Trigger profile type setup
  const checkedRadio = document.querySelector('input[name="profileType"]:checked');
  if (checkedRadio) {
    checkedRadio.dispatchEvent(new Event('change'));
  }
})();
