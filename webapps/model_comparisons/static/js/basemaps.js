/**
 * Shared Leaflet basemap switcher (no API keys required).
 *
 *   addBasemaps(map, { maxZoom: 12 });
 *
 * Adds a layer control to `map` and remembers the viewer's choice across the
 * Profiles, Regions and Fronts maps via localStorage.
 */
(function () {
  const STORE_KEY = 'ioosBasemap';
  const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services';
  const esri = (svc) => `${ESRI}/${svc}/MapServer/tile/{z}/{y}/{x}`;
  const ESRI_ATTR = 'Tiles &copy; Esri';

  function defs(maxZoom) {
    const z = (native) => ({ maxNativeZoom: native, maxZoom: Math.max(maxZoom, native) });
    const t = (url, opts) => L.tileLayer(url, Object.assign({}, opts));
    return {
      'Ocean': L.layerGroup([
        t(esri('Ocean/World_Ocean_Base'), Object.assign({
          attribution: ESRI_ATTR + ' &mdash; GEBCO, NOAA, CHS, OSU, and others' }, z(13))),
        t(esri('Ocean/World_Ocean_Reference'), z(13)),
      ]),
      'Satellite': L.layerGroup([
        t(esri('World_Imagery'), Object.assign({
          attribution: ESRI_ATTR + ' &mdash; Maxar, Earthstar Geographics, and others' }, z(17))),
        t(esri('Reference/World_Boundaries_and_Places'), z(13)),
      ]),
      'Light': L.layerGroup([
        t(esri('Canvas/World_Light_Gray_Base'), Object.assign({ attribution: ESRI_ATTR }, z(16))),
        t(esri('Canvas/World_Light_Gray_Reference'), z(16)),
      ]),
      'Dark': L.layerGroup([
        t(esri('Canvas/World_Dark_Gray_Base'), Object.assign({ attribution: ESRI_ATTR }, z(16))),
        t(esri('Canvas/World_Dark_Gray_Reference'), z(16)),
      ]),
      'Topographic': t(esri('World_Topo_Map'), Object.assign({ attribution: ESRI_ATTR }, z(19))),
      'Streets': t('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxNativeZoom: 19, maxZoom: Math.max(maxZoom, 19) }),
    };
  }

  function load() { try { return localStorage.getItem(STORE_KEY); } catch (e) { return null; } }
  function save(v) { try { localStorage.setItem(STORE_KEY, v); } catch (e) { /* private mode */ } }

  window.addBasemaps = function (map, opts) {
    opts = opts || {};
    const layers = defs(opts.maxZoom || 12);
    const saved = load();
    const initial = layers[saved] ? saved : (opts.default || 'Ocean');
    layers[initial].addTo(map);
    L.control.layers(layers, null, { position: 'topright', collapsed: true }).addTo(map);
    map.on('baselayerchange', (e) => save(e.name));
    return layers;
  };
})();
