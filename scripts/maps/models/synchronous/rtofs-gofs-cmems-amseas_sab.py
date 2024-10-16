import datetime as dt
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import copy
import logging

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import rtofs, amseas, CMEMS, ESPC, cnaps
from ioos_model_comparisons.platforms import (
    get_active_gliders,
    get_argo_floats_by_time, get_goes
    )
from ioos_model_comparisons.plotting import (
    plot_model_region_comparison,
    plot_model_region_comparison_streamplot,
    plot_sst
)
from ioos_model_comparisons.regions import region_config
from cool_maps.plot import get_bathymetry

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

matplotlib.use('agg')  # Set matplotlib to use non-interactive backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Start time to measure script execution duration
start_time = time.time()

# Save path for plots
path_save = conf.path_plots / "maps"

# Model selection flags
plot_rtofs = True
plot_para = True
plot_espc = True
plot_cmems = True
plot_amseas = False
plot_cnaps = False

# Keyword arguments for map plots
kwargs = {
    'transform': conf.projection,
    'dpi': conf.dpi,
    'overwrite': False,
    'colorbar': True,
    'legend': True,
}

# Configuration for regions and days to process
conf.days = 1
conf.regions = ['sab']

# Date range for processing
today = dt.date.today()
date_start = today - dt.timedelta(days=conf.days)
date_end = today + dt.timedelta(days=1)
freq = '6H'
date_list = pd.date_range(date_start, date_end, freq=freq)
date_list_2 = pd.date_range(date_start - dt.timedelta(days=1), date_end, freq=freq)

# Set up global extent based on regions
search_start = date_list[0] - dt.timedelta(hours=conf.search_hours)
extent_list = [region_config(region)["extent"] for region in conf.regions]
extent_df = pd.DataFrame(extent_list, columns=['lonmin', 'lonmax', 'latmin', 'latmax'])
global_extent = [extent_df.lonmin.min(), extent_df.lonmax.max(), extent_df.latmin.min(), extent_df.latmax.max()]

# Retrieve platform data with error handling
if conf.argo:
    try:
        argo_data = get_argo_floats_by_time(global_extent, search_start, date_end) if conf.argo else pd.DataFrame()
        logger.info(f"Argo data {'loaded' if not argo_data.empty else 'not available'}.")
    except Exception as e:
        logger.error(f"Failed to load Argo data: {e}")
        argo_data = pd.DataFrame()

if conf.gliders:
    try:
        glider_data = get_active_gliders(global_extent, search_start, date_end, parallel=False, timeout=60) if conf.gliders else pd.DataFrame()
        logger.info(f"Glider data {'loaded' if not glider_data.empty else 'not available'}.")
    except Exception as e:
        logger.error(f"Failed to load Glider data: {e}")
        glider_data = pd.DataFrame()

try:
    bathy_data = get_bathymetry(global_extent) if conf.bathy else None
    logger.info("Bathymetry data loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Bathymetry data: {e}")
    bathy_data = None

try:
    sst_sorted = get_goes()
    logger.info('GOES SST data loaded successfully.')
except Exception as e:
    logger.error(f"Failed to load GOES SST data: {e}")
    sst_sorted = None

# Updated load_model function to handle the RTOFS parallel model
def load_model(model_func, model_name, source=None, rename=True):
    try:
        logger.info(f'Loading {model_name} model data.')
        if source:
            model_data = model_func(rename=rename, source=source)
        else:
            model_data = model_func(rename=rename)
        logger.info(f'{model_name} model data loaded successfully.')
        return model_data
    except Exception as e:
        logger.error(f"Failed to load {model_name} model data: {e}")
        return None

# Load selected models with error handling
rds = load_model(rtofs, 'RTOFS')
rtofs_para = load_model(rtofs, 'RTOFS Parallel', source='parallel') if plot_para else None
# gds = load_model(espc, 'ESPC') if plot_espc else None
gds_instance = ESPC() if plot_espc else None
cmems_instance = CMEMS() if plot_cmems else None
am = load_model(amseas, 'AMSEAS') if plot_amseas else None
cn = load_model(cnaps, 'CNAPS') if plot_cnaps else None

# Set up grid values if RTOFS model loaded
grid_lons, grid_lats, grid_x, grid_y = None, None, None, None
if rds is not None:
    grid_lons, grid_lats = rds.lon.values[0, :], rds.lat.values[:, 0]
    grid_x, grid_y = rds.x.values, rds.y.values

def main():
    for ctime in date_list:
        logger.info(f"Starting processing for time: {ctime}")

        # Attempt to load data for each model
        rdt_flag, rdt = attempt_data_load(rds, ctime, "RTOFS")
        rdtp_flag, rdtp = attempt_data_load(rtofs_para, ctime, "RTOFS Parallel") if plot_para else (False, None)
        amt_flag, amt = attempt_data_load(am, ctime, "AMSEAS") if plot_amseas else (False, None)
        cnt_flag, cnt = attempt_data_load(cn, ctime, "CNAPS") if plot_cnaps else (False, None)

        # Process each region
        for item in conf.regions:
            region = region_config(item)

            gdt_flag, gdt = attempt_cmems_data_load(gds_instance, ctime, region['extent']) if plot_espc else (False, None)
            cdt_flag, cdt = attempt_cmems_data_load(cmems_instance, ctime, region['extent']) if plot_cmems else (False, None)
            logger.info(f"Processing region: {region['name']} at time: {ctime}")
            process_region(ctime, rdt_flag, rdt, rdtp_flag, rdtp, gdt_flag, gdt, cdt_flag, cdt,
                           amt_flag, amt, cnt_flag, cnt, region)

    logger.info(f'All processing complete. Total execution time: {time.time() - start_time} seconds.')

def attempt_data_load(model, ctime, model_name):
    """Attempt to load data for a given model and time."""
    try:
        if model is None:
            raise ValueError(f"{model_name} model data is not available.")
        data = model.sel(time=ctime)
        # if data is None: # or data.isnull().all():
            # raise ValueError(f"No valid data found for {model_name} at {ctime}.")
        logger.info(f"{model_name}: Data successfully loaded for time {ctime}.")
        return True, data
    except (KeyError, ValueError) as e:
        logger.warning(f"{model_name}: Data not available for time {ctime} - {e}")
        return False, None
    
def attempt_cmems_data_load(cmems_instance, ctime, extent):
    """Attempt to load CMEMS data for a given time and region extent."""
    try:
        if cmems_instance is None:
            raise ValueError("CMEMS instance is not initialized.")
        
        lon_extent = extent[:2]  # Longitude range
        lat_extent = extent[2:]  # Latitude range

        # Lazy-load the combined subset of CMEMS data (temperature, salinity, u, v) for the time and region
        data = cmems_instance.get_combined_subset(lon_extent, lat_extent, time=ctime)
        
        if data is None:# or data.isnull().all():
            raise ValueError(f"No valid CMEMS data found for time {ctime}.")

        logger.info(f"CMEMS: Data successfully loaded for time {ctime}.")
        return True, data

    except (KeyError, ValueError, Exception) as e:
        logger.warning(f"CMEMS: Data not available for time {ctime} - {e}")
        return False, None

def process_region(ctime, rdt_flag, rdt, rdtp_flag, rdtp, gdt_flag, gdt, cdt_flag, cdt,
                   amt_flag, amt, cnt_flag, cnt, region):
    """Process a specific region for the given time."""
    extent = region['extent']
    logger.info(f"Subsetting data for region: {region['name']} with extent {extent} at time {ctime}")
    kwargs['path_save'] = path_save / region['folder']

    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr)

    if 'eez' in region:
        kwargs["eez"] = region["eez"]

    if region['currents']['bool']:
        kwargs['currents'] = region['currents']

    if 'figure' in region:
        if 'legend' in region['figure']:
            kwargs['cols'] = region['figure']['legend']['columns']
        if 'figsize' in region['figure']:
            kwargs['figsize'] = region['figure']['figsize']

    extended = np.add(extent, [-1, 1, -1, 1]).tolist()
    lon360 = lon180to360(extended[:2])  # Convert from -180, 180 to 0, 360 longitude

    try:
        # Subset data based on the region extent
        rds_sub = subset_data(rdt, extended, grid_lons, grid_lats, grid_x, grid_y) if rdt_flag else None
        rdtp_sub = subset_data(rdtp, extended, grid_lons, grid_lats, grid_x, grid_y) if rdtp_flag else None
        # gds_sub = subset_data_lonlat(gdt, lon360, extended) if gdt_flag else None
        gds_sub = gdt
        # cds_sub = cmems_instance.get_combined_subset(extended[:2], extended[2:], ctime) if cdt_flag else None
        cds_sub = cdt
        am_sub = subset_data_lonlat(amt, lon360, extended) if amt_flag else None

        # Subset downloaded Argo data to this region and time
        if not argo_data.empty:
            lon = argo_data['lon']
            lat = argo_data['lat']

            # Mask out anything beyond the extent
            mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
            argo_region = argo_data[mask]
            argo_region.sort_index(inplace=True)

            # Mask out any argo floats beyond the time window
            idx = pd.IndexSlice
            kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

        # Subset downloaded glider data to this region and time
        if not glider_data.empty:
            lon = glider_data['lon']
            lat = glider_data['lat']
            
            # Mask out anything beyond the extent
            mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
            glider_region = glider_data[mask]

            # Mask out any gliders beyond the time window
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                &
                (glider_region.index.get_level_values('time') < search_window_t1)
                ]
            kwargs['gliders'] = glider_region

        # Process SST data
        if sst_sorted is not None:
            sst = process_sst_data(sst_sorted, extent, ctime)
        else:
            sst = None
            logger.warning(f"SST data unavailable for region {region['name']} at time {ctime}")

        # Plot data
        try:
            if rdt_flag and gdt_flag:
                plot_model_region_comparison(rds_sub, gds_sub, region, **kwargs)
                plot_model_region_comparison_streamplot(rds_sub, gds_sub, region, **kwargs)
                logger.info(f"Successfully plotted RTOFS vs ESPC for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs ESPC at {ctime} for region {region['name']}: {e}")

        try:
            if rdt_flag and rdtp_flag:
                plot_model_region_comparison(rds_sub, rdtp_sub, region, **kwargs)
                plot_model_region_comparison_streamplot(rds_sub, rdtp_sub, region, **kwargs)
                logger.info(f"Successfully plotted RTOFS vs Parallel for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs Parallel at {ctime} for region {region['name']}: {e}")

        try:
            if rdt_flag and cdt_flag:
                plot_model_region_comparison(rds_sub, cds_sub, region, **kwargs)
                plot_model_region_comparison_streamplot(rds_sub, cds_sub, region, **kwargs)
                logger.info(f"Successfully plotted RTOFS vs CMEMS for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs CMEMS at {ctime} for region {region['name']}: {e}")

        try:
            if rdt_flag and amt_flag:
                plot_model_region_comparison(rds_sub, am_sub, region, **kwargs)
                plot_model_region_comparison_streamplot(rds_sub, am_sub, region, **kwargs)
                logger.info(f"Successfully plotted RTOFS vs AMSEAS for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs AMSEAS at {ctime} for region {region['name']}: {e}")

        if sst is not None:
            plot_sst(rds_sub, sst, region, **remove_kwargs(['eez', 'currents']))
            logger.info(f"Successfully plotted SST for region {region['name']} at time {ctime}")

    except Exception as e:
        logger.error(f"Failed to process region {region['name']} at time {ctime}: {e}")

def subset_data(data, extent, grid_lons, grid_lats, grid_x, grid_y):
    """Subset data based on the region extent."""
    try:
        lons_ind = np.interp(extent[:2], grid_lons, grid_x)
        lats_ind = np.interp(extent[2:], grid_lats, grid_y)
        extent_ind = [int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1])),
                      int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1]))]
        logger.debug(f"Subsetting data for extent indices: {extent_ind}")
        return data.isel(x=slice(extent_ind[0], extent_ind[1]),
                         y=slice(extent_ind[2], extent_ind[3])).set_coords(['u', 'v'])
    except Exception as e:
        logger.error(f"Error during data subsetting: {e}")
        return None

def subset_data_lonlat(data, lon_extent, lat_extent):
    """Subset data using longitude and latitude extents."""
    try:
        logger.debug(f"Subsetting {data.attrs['model']} data for lon extent: {lon_extent} and lat extent: {lat_extent}")
        return data.sel(lon=slice(lon_extent[0], lon_extent[1]),
                        lat=slice(lat_extent[2], lat_extent[3])).set_coords(['u', 'v'])
    except Exception as e:
        logger.error(f"Error during lon/lat data subsetting: {e}")
        return None

def process_sst_data(sst_data: xr.DataArray, extent: list, ctime: dt.datetime) -> xr.DataArray:
    """Process SST data to convert from Kelvin to Celsius and subset."""
    try:
        logger.debug(f"Processing SST data for extent: {extent} at time {ctime}")
        sst = sst_data.sel(lon=slice(extent[0], extent[1]),
                           lat=slice(extent[2], extent[3])).sel(time=str(ctime), method='nearest')
        sst['SST_C'] = (('lat', 'lon'), kelvin_to_celsius(sst['SST'].values))
        return sst
    except Exception as e:
        logger.error(f"Error during SST data processing: {e}")
        return None

def kelvin_to_celsius(kelvin_temps: np.ndarray) -> np.ndarray:
    """Convert temperature from Kelvin to Celsius."""
    try:
        return kelvin_temps - 273.15
    except Exception as e:
        logger.error(f"Error converting temperature from Kelvin to Celsius: {e}")
        return kelvin_temps  # Return original temps if conversion fails

def remove_kwargs(keys: list) -> dict:
    """Return a copy of kwargs with specified keys removed."""
    try:
        new_kwargs = copy.deepcopy(kwargs)
        for key in keys:
            new_kwargs.pop(key, None)
        return new_kwargs
    except Exception as e:
        logger.error(f"Error removing kwargs: {e}")
        return kwargs

if __name__ == "__main__":
    main()
