import datetime as dt
from collections import namedtuple
from pathlib import Path

import gsw
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seawater
import xarray as xr
from erddapy import ERDDAP
from gsw import CT_from_t, SA_from_SP, p_from_z, rho
from requests.exceptions import HTTPError as rHTTPError

from ioos_model_comparisons.plotting import export_fig


################################################################################
def ohc_from_profile(depth, temp, dens):
    """
    This function Calculates the ocean heat content from a temperature and
    density profile (Leipper, Dale F., and Douglas Volgenau. "Hurricane heat
    potential of the Gulf of Mexico". Journal of Physical Oceanography 2.3
    (1972): 218-224).
    
    Q = ρ Cp ΔT ΔZ 
    
    Q: Hurricane Heat Potential
    ρ: density (gm cm-3)
    Cp: specific heat at constant pressure (cal cm-3) (C)
    ΔT: Average temperature difference above 26C for a given depth increment
    ΔZ: Depth increment (taken as 500cm)

    Args:
        depth (numpy.ndarray or pandas.Series or xarray.Series): depth (m)
        temp (numpy.ndarray or pandas.Series or xarray.Series): temperature (c)
        dens (numpy.ndarray or pandas.Series or xarray.Series): density (gm/cm^3)

    Returns:
        numpy.ndarray: Ocean heat content of the water column in kJ/cm^2
    """

    cp = 3985 #Heat capacity in J/(kg K)
    ok26 = temp >= 26 
    depth = np.abs(depth) #absolute value of depth

    if len(depth[ok26]) != 0:
        if np.nanmin(depth[ok26])>10:
            OHC = np.nan
        else:
            rho0 = np.nanmean(dens[ok26]) # don't include nans
            OHC = np.abs(cp * rho0 * np.trapz(temp[ok26]-26,depth[ok26]))
            OHC = OHC * 10**(-7) # in kJ/cm^2
    else:
        OHC = np.nan
    return OHC


def mld_temp_crit(depth, temp, ref_depth=10, thres_depth=10, dtemp=.2):
    """
    This function calculates the mixed layer depth and temperature
    based on a temperature criteria: T - T_at_ref_depth <= dtemp

    Args:
        depth (_type_): Profile Depth
        temp (_type_): Profile temperature
        ref_depth (int, optional): Reference depth from the mixed layer depth definition used. 
            Defaults to 10.
        thres_depth (int, optional): Depth threshold.
            Defaults to 10.
        dtemp (float, optional): Delta temperature from the mixed layer depth definition used. 
            Defaults to .2.

    Returns:
        _type_: mixed_layer_depth, mixed_layer_temperature
    """
    if ref_depth > thres_depth:
        print("Threshold depth must be greater (deeper) than the reference depth")
        return

    # MLD and MLT: mixed layer depth and Mixed layer temperature
    # find where profile depths are greater than the reference depth
    ok_ref_depth = np.where(depth >= ref_depth)[0]

    # Get the first depth that is greater than the reference depth
    top_ref_depth = ok_ref_depth[0]
    top_profile_depth = depth[top_ref_depth]

    # Check if the shallowest depth of the profile is deeper than the reference depth
    tol = np.abs(thres_depth - ref_depth)/2
    if np.isclose(top_profile_depth, ref_depth+tol, atol=tol):
        # Get the temperature at the first depth that is greater than the reference depth
        temp_ref_depth = temp[top_ref_depth]

        # subtract every single temperature from the reference depth
        delta_T = temp_ref_depth - temp 
        ok_mld_temp = np.where(delta_T <= dtemp)[0]

        if ok_mld_temp.size == 0:
            MLD = np.nan
            MLT = np.nan
        else:
            MLD = depth[ok_mld_temp[-1]]
            MLT = np.nanmean(temp[ok_mld_temp])
        # print(f"Reference Depth {ref_depth} (m) is deeper than shallowest Profile Depth {top_profile_depth} (m).")
        # print(f"MLD: {MLD}, MLT: {MLT}")
        return MLD, MLT, top_profile_depth, ref_depth
    else:
        print(f"Reference Depth {ref_depth} (m) is shallower than shallowest Profile Depth {top_profile_depth} (m). Skipping this profile.")
        return np.nan, np.nan, top_profile_depth, ref_depth
        # return

################################################################################
def mld_dens_crit(drho, ref_depth, depth, temp, dens):
    """
    This function calculates the mixed layer depth andtemperature
    based on a density criteria: rho_at_ref_depth - rho <= drho

    Args:
        drho (array): delta density from the mixed layer depth definition used
        ref_depth (array): Reference depth from the mixed layer depth definition used
        depth (_type_): Profile Depth
        temp (_type_): Profile temperature
        dens (array): Profile density

    Returns:
        np.ndarray: mixed layer depth, Mixed layer temperature
    """
    ok_ref_depth = np.where(depth >= ref_depth)[0][0]
    rho_ref_depth = dens[ok_ref_depth]
    delta_rho = -(rho_ref_depth - dens)
    ok_mld_rho = np.where(delta_rho <= drho)[0]

    if ok_mld_rho.size == 0:
        MLD = np.nan
        MLT = np.nan
    else:
        MLD = depth[ok_mld_rho[-1]]
        MLT = np.nanmean(temp[ok_mld_rho])

    return MLD, MLT

################################################################################
def temp_average_depth(depth, temp, depth_range=[0, 100]):
    """
    This function calculates the depth average temperature between a minimum
    and maximum depth. The default is set to calculate the average Temperature
    in the top 100 m.

    Args:
        depth (numpy.ndarray or pandas.Series or xarray.Series): depth
        temp (numpy.ndarray or pandas.Series or xarray.Series): temperature

    Returns:
        np.ndarray: depth average temperature in the top 100 meters
    """
    # Get index where depths lie between depth_range
    ind = (depth_range[0] <= np.abs(depth)) & (np.abs(depth) <= depth_range[1])
    
    if len( np.where( np.isnan(temp[ind]) )[0] ) > 10:
        averaged = np.nan
    else:
        averaged = np.nanmean(temp[ind])
    return averaged

################################################################################
def potential_energy_anomaly100(depth, dens):
    """
    This function calculates the potential energy anomaly
    (Simpson J, Brown J, Matthews J, Allen G (1990) Tidal straining, density
    currents and stirring in the control of estuarine stratification.
    Estuaries 13(2):125-132), in the top 100 meters

    Args:
        depth (numpy.ndarray or pandas.Series or xarray.Series): depth
        dens (numpy.ndarray or pandas.Series or xarray.Series): density

    Returns:
        np.ndarray: potential energy anomaly in J/m^3
    """

    g = 9.8 #m/s
    dindex = np.fliplr(np.where(np.asarray(np.abs(depth)) <= 100))[0]
    if len(dindex) == 0:
        PEA = np.nan
    else:
        zz = np.asarray(np.abs(depth[dindex]))
        denss = np.asarray(dens[dindex])
        ok = np.isfinite(denss)
        z = zz[ok]
        densi = denss[ok]
        if len(z)==0 or len(densi)==0 or np.min(zz) > 10 or np.max(zz) < 30:
            PEA = np.nan
        else:
            if z[-1] - z[0] > 0:
                # So PEA is < 0
                # sign = -1
                # Adding 0 to sigma integral is normalized
                z = np.append(0,z)
            else:
                # So PEA is < 0
                # sign = 1
                # Adding 0 to sigma integral is normalized
                z = np.flipud(z)
                z = np.append(0,z)
                densit = np.flipud(densi)

            # adding density at depth = 0
            densitt = np.interp(z,z[1:],densit)
            density = np.flipud(densitt)

            # defining sigma
            max_depth = np.nanmax(zz[ok])
            sigma = -1*z/max_depth
            sigma = np.flipud(sigma)

            rhomean = np.trapz(density,sigma,axis=0)
            drho = rhomean - density
            torque = drho * sigma
            PEA = g * max_depth * np.trapz(torque,sigma,axis=0)

    return PEA


def calculate_density(temperature, salinity, depth, lat):
    pressure = seawater.eos80.pres(depth, lat)
    density = seawater.eos80.dens(salinity, temperature, pressure)
    return density


def calculate_upper_ocean_metrics(time, temp, salinity, depth, density=None):
    """_summary_

    Args:
        time (_type_): time
        temp (_type_): temperature (c)
        salinity (_type_): salinity (1)
        depth (_type_): depth (m)
        density (_type_, optional): _description_. Defaults to None.
        lat (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # if not isinstance(density, pd.Series):
        # density = pd.Series(dtype=object)
        
    # Reference variables
    dtemp = 0.2
    reference_depth = 10
    threshold_depth = 11
    drho = 0.125

    # Init empty dict 
    ddict = {}
     
    # Calculate density of gofs profile. Converting from depth (m) to pressure 
    # internally.
    
    # if not density.any():
        # density = calculate_density(temp, salinity, depth, lat)
    
    # Ocean heat content
    # Depth, temperature, density
    ohc = ohc_from_profile(depth, temp, density)
    ddict['ocean_heat_content'] = ohc

    # Mixed Layer Depth calculated from temperature
    # dtemp, reference depth, depth, and temperature
    depth_mld_t, temp_mld_t, tdepth, rdepth= mld_temp_crit(
        depth, 
        temp, 
        reference_depth, 
        threshold_depth, 
        dtemp)
    ddict['mixed_layer_depth_from_temp'] = depth_mld_t
    ddict['mixed_layer_temp_from_temp'] = temp_mld_t
    ddict['tdepth'] = tdepth
    ddict['rdepth'] = rdepth
    ddict['sdepth'] = depth.min()

    # Mixed Layer Depth calculated from density
    # ddensity, reference depth, depth, temperature, and density
    depth_mld_d, temp_mld_d = mld_dens_crit(drho, reference_depth, depth, temp,  density)
    ddict['mixed_layer_depth_from_density'] = depth_mld_d
    ddict['mixed_layer_temp_from_density'] = temp_mld_d

    # Average temperature in the top 100m 
    # depth, temperature
    ddict['average_temp_mldt_to_100m'] = temp_average_depth(depth, temp, depth_range=[depth_mld_t,100])
    ddict['average_temp_mlds_to_100m'] = temp_average_depth(depth, temp, depth_range=[depth_mld_d,100])
    ddict['average_temp_000m_to_100m'] = temp_average_depth(depth, temp, depth_range=[0,100])
    ddict['average_temp_100m_to_200m'] = temp_average_depth(depth, temp, depth_range=[100,200])
    ddict['average_temp_200m_to_300m'] = temp_average_depth(depth, temp, depth_range=[200,300])
    ddict['average_temp_300m_to_400m'] = temp_average_depth(depth, temp, depth_range=[300,400])
    ddict['average_temp_400m_to_500m'] = temp_average_depth(depth, temp, depth_range=[400,500])
    ddict['average_temp_500m_to_600m'] = temp_average_depth(depth, temp, depth_range=[500,600])   
    ddict['average_temp_600m_to_700m'] = temp_average_depth(depth, temp, depth_range=[600,700])
    ddict['average_temp_700m_to_800m'] = temp_average_depth(depth, temp, depth_range=[700,800])
    ddict['average_temp_800m_to_900m'] = temp_average_depth(depth, temp, depth_range=[800,900])
    ddict['average_temp_900m_to_1000m'] = temp_average_depth(depth, temp, depth_range=[900,1000])


    # Potential Energy Anomaly in the top 100 meters
    # depth, density
    pea = potential_energy_anomaly100(depth, density)
    ddict['potential_energy_anomaly_100m'] = pea

    # Salinity at surface
    # Should this be an average or the very first reading?
    sal_surf_idx = np.nanargmin(depth)
    ddict['salinity_surface'] = salinity[sal_surf_idx]
    # ddict['salinity_surface_depth'] = salinity[sal_surf_idx]

    # Salinity maximum
    try:
        sal_max_idx = np.nanargmax(salinity)
        ddict['salinity_max'] = salinity[sal_max_idx]
        ddict['salinity_max_depth'] = depth[sal_max_idx]
    except ValueError:
        ddict['salinity_max'] = np.nan
        ddict['salinity_max_depth'] = np.nan

    df = pd.DataFrame(data=ddict, index=[pd.to_datetime(time)])
    return df


def amseas(rename=False):
    url = "https://www.ncei.noaa.gov/thredds-coastal/dodsC/ncom_amseas_agg/AmSeas_Dec_17_2020_to_Current_best.ncd"
    ds = xr.open_dataset(url)
    ds.attrs['model'] = 'AMSEAS'
    if rename:
        ds = ds.rename(
            {
                "surf_el": "sea_surface_height",
                "water_temp": "temperature", 
                "water_u": "u",
                "water_v": "v"
                }
            )
    return ds


def rtofs():
    url = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped"
    ds = xr.open_dataset(url).set_coords(['lon', 'lat'])
    ds.attrs['model'] = 'RTOFS'
    return ds


def gofs(rename=False):
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
    ds = xr.open_dataset(url, drop_variables="tau")
    ds.attrs['model'] = 'GOFS'
    if rename:
        ds = ds.rename(
            {
                "surf_el": "sea_surface_height",
                "water_temp": "temperature",
                "water_u": "u",
                "water_v": "v"
                }
            )
    return ds


def cmems(rename=False):
    username = 'maristizabalvar'
    password = 'MariaCMEMS2018'
    dataset = 'global-analysis-forecast-phy-001-024'
    from pydap.cas.get_cookies import setup_session
    from pydap.client import open_url
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits

    # Downloading and reading Copernicus grid
    ds = xr.open_dataset(data_store, drop_variables='tau')
    ds.attrs['model'] = 'CMEMS'

    if rename:
        ds = ds.rename(
            {
                'thetao': 'temperature', 
                'so': 'salinity',
                'latitude': 'lat',
                'longitude': 'lon',
                'uo': 'u',
                'vo': 'v'
                }
            )
    return ds

def density(temperature, depth, salinity, latitude, longitude):
    """
    Calculates density given practical salinity, depth, latitude,
    and longitude using Gibbs gsw SA_from_SP and rho functions.

    Args:
        temperature (_type_): temperature (C)
        depth (_type_): depth, positive up (m)
        salinity (array): salinity
        latitude (array): latitude (decimal degrees)
        longitude (array): longitude (decimal degrees)

    Returns:
        density: Density calculated using the Gibbs GSW
    """

    # Calculates sea pressure from height using computationally-efficient 
    # 75-term expression for density, in terms of SA, CT and p 
    # (Roquet et al., 2015). 
    pressure = p_from_z(
        depth,
        latitude,
    )

    # Calculates Absolute Salinity from Practical Salinity. 
    # Since SP is non-negative by definition,
    # this function changes any negative input values of SP to be zero.
    absolute_salinity = SA_from_SP(
        salinity,
        pressure,
        longitude,
        latitude
    )

    # Calculates Conservative Temperature of seawater from in-situ temperature.
    conservative_temperature = CT_from_t(
        absolute_salinity,
        temperature,
        pressure
    )

    # Calculates in-situ density from Absolute Salinity and
    # Conservative Temperature, using the computationally-efficient expression 
    # for specific volume in terms of SA, CT and p (Roquet et al., 2015).
    density = rho(
        absolute_salinity,
        conservative_temperature,
        pressure
    )

    return density

def lon180to360(array):
    array = np.array(array)
    return np.mod(array, 360)

def lon360to180(array):
    array = np.array(array)
    return np.mod(array+180, 360)-180

def get_argo_floats_by_time(bbox=(-100, -45, 5, 46),
                            time_start=None, time_end=dt.date.today(),
                            wmo_id=None, variables=None):
    """_summary_

    Args:
        bbox (_type_, optional): _description_. Defaults to None.
        time_start (_type_, optional): Start time. Defaults to None.
        time_end (_type_, optional): End time. Defaults to dt.date.today().
        floats (_type_, optional): _description_. Defaults to None.
        add_vars (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Accept both tuples and lists, but turn tuple into a list if passed a tuple
    if isinstance(variables, tuple):
        variables = list(variables)
        
    time_start = time_start or (time_end - dt.timedelta(days=1))
    
    default_variables = ['platform_number', 'time', 'longitude', 'latitude']
    
    constraints = {
        'time>=': str(time_start),
        'time<=': str(time_end),
    }

    if bbox:
        constraints['longitude>='] = bbox[0]
        constraints['longitude<='] = bbox[1]
        constraints['latitude>='] = bbox[2]
        constraints['latitude<='] = bbox[3]

    if wmo_id:
        if isinstance(wmo_id, int) or isinstance(wmo_id, float):
            wmo_id = str(wmo_id)
            
        constraints['platform_number='] = wmo_id

    if variables:
        default_variables = default_variables + variables
        default_variables = list(set(default_variables)) # remove duplicates
        
    e = ERDDAP(
        server='IFREMER',
        protocol='tabledap',
        response='csv'
    )

    e.dataset_id = 'ArgoFloats'
    e.constraints = constraints
    e.variables = default_variables

    try:
        df = e.to_pandas(
            index_col="time (UTC)",
            parse_dates=True,
        ).dropna().tz_localize(None)
        df = df.reset_index().rename(rename_argo, axis=1)
        df = df.set_index(["argo", "time"]).sort_index()
    except rHTTPError:
        df = pd.DataFrame()
    return df


def transect2rtofs(pts, grid_lons, grid_lats, grid_x, grid_y):
    # if not grid_x:
    #     grid_x = np.arange(0, len(grid_lons))
    # if not grid_y:
    #     grid_y = np.arange(0, len(grid_lats))
    
    # Convert points to x and y index for rtofs
    # Use piecewise linear interpolation (np.interp) to find the partial index (float instead of an int) of the points (lon,lat) we are calculating would lie
    # on the x,y grid for RTOFS
    # We pass the following:
    # np.interp(x, grid_lon, grid_x)
    # np.interp(y, grid_lat, grid_y)
    lonidx = np.interp(pts[:,0], grid_lons, grid_x)
    latidx = np.interp(pts[:,1], grid_lats, grid_y)
    return lonidx, latidx









# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# # Part 1 - Leave Part 2 uncommented.
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# path_impact_calculated = Path('/Users/mikesmith/Documents/')
# float_id = '4903248'
# t0 = dt.datetime(2021, 8, 1, 0, 0, 0)
# t1 = dt.datetime(2021, 10, 31, 23, 59, 59)
# variables = ['pres', 'temp', 'psal']


# Argo = namedtuple('Argo', ['name', 'lon', 'lat'])
# Glider = namedtuple('Glider', ['name', 'lon', 'lat'])
# time_formatter = '%Y-%m-%dT%H:%M:%SZ'

# rename_gliders = {}
# # rename_gliders["time (UTC)"] = "time"
# rename_gliders["longitude"] = "lon"
# rename_gliders["latitude"] = "lat"

# rename_argo = {}
# rename_argo["platform_number"] = "argo"
# rename_argo["time (UTC)"] = "time"
# rename_argo["longitude (degrees_east)"] = "lon"
# rename_argo["latitude (degrees_north)"] = "lat"

# # Calculate Argo metrics
# df = get_argo_floats_by_time(time_start=t0,
#                              time_end=t1, 
#                              wmo_id=float_id, 
#                              variables=variables)
# rename = {'psal (PSU)': 'salinity',
#           'temp (degree_Celsius)': 'temperature',
#           'pres (decibar)': 'pressure'
#           }
# df = df.rename(columns=rename)

# # Calculate depth from pressure
# df['depth'] = gsw.z_from_p(df['pressure'].values, df['lat'].values)

# # Calculate density (gsw-python)
# df['density'] = density(df['temperature'].values, df['depth'].values, df['salinity'].values, df['lat'].values, df['lon'].values)

# calc = []
# glon = []
# glat = []
# for time, group in df.groupby(level=1):
#     print(time)
#     calc.append(
#         calculate_upper_ocean_metrics(
#             time,
#             group['temperature'].values,
#             group['salinity'].values,
#             -group['depth'].values,
#             group['density'].values
#         )
#     )         
#     glon.append(group['lon'].iloc[0])
#     glat.append(group['lat'].iloc[0])   
# argo_metrics = pd.concat(calc)
# argo_metrics.to_pickle(path_impact_calculated / f"{float_id}_calculated_argo_data.pkl")
# gtime = argo_metrics.index.to_numpy()

# # Calculate RTOFS metrics
# rds = rtofs()

# # Convert to the x, y indexes so we can work with the RTOFS model
# grid_lons = rds.lon.values[0,:]
# grid_lats = rds.lat.values[:,0]
# grid_x = rds.x.values
# grid_y = rds.y.values
# lonidx, latidx = transect2rtofs(np.column_stack((glon, glat)), grid_lons, grid_lats, grid_x=grid_x, grid_y=grid_y)

# # Interpolate RTOFS Transect
# rds = rds.sel(
#     time=xr.DataArray(gtime, dims='point'),
#     x=xr.DataArray(lonidx, dims="point"),
#     y=xr.DataArray(latidx, dims="point"),
#     method='nearest'
# )

# # Calculate upper ocean metrics from gofs and add to DataFrame
# rtofs_df = pd.DataFrame()

# for t in rds.point:
#     temp = rds.sel(point=t)
#     temp.load()
#     tdensity = density(temp.temperature.values, -temp.depth.values, temp.salinity.values, temp.lat.values, temp.lon.values)

#     # try:
#     gdf = uom_functions.calculate_upper_ocean_metrics(
#         pd.to_datetime(temp.time.values), 
#         temp.temperature.values, 
#         temp.salinity.values,
#         temp.depth.values, 
#         tdensity
#         )
#     rtofs_df = pd.concat([rtofs_df, gdf]) 
# rtofs_df.to_pickle(path_impact_calculated / f"{float_id}_calculated_rtofs_data.pkl")

# # Calculate GOFS metrics
# gds = gofs(rename=True)

# gds = gds.sel(
#     time=xr.DataArray(gtime, dims='point'),
#     lon=xr.DataArray(lon180to360(glon), dims='point'),
#     lat=xr.DataArray(glat, dims='point'),
#     method='nearest'
# )

# # Calculate upper ocean metrics from gofs and add to DataFrame
# gofs_df = pd.DataFrame()

# for t in gds.point:
#     temp = gds.sel(point=t)
#     temp.load()
#     tdensity = density(temp.temperature.values, -temp.depth.values, temp.salinity.values, temp.lat.values, temp.lon.values)

#     # try:
#     gdf = calculate_upper_ocean_metrics(
#         pd.to_datetime(temp.time.values), 
#         temp.temperature.values, 
#         temp.salinity.values,
#         temp.depth.values, 
#         tdensity
#         )
#     gofs_df = pd.concat([gofs_df, gdf]) 
#     # gofs_df.to_csv(path_impact_calculated / f"{glider}_calculated_gofs_data.csv")
# gofs_df.to_pickle(path_impact_calculated / f"{float_id}_calculated_gofs_data.pkl")

# # Calculate Copernicus metrics
# cds = cmems(rename=True)

# cds = cds.sel(
#     time=xr.DataArray(gtime, dims='point'),
#     lon=xr.DataArray(glon, dims='point'),
#     lat=xr.DataArray(glat, dims='point'),
#     method='nearest'
# )

# cdf = pd.DataFrame()
# for t in cds.point:
#     temp = cds.sel(point=t)
#     temp.load()
    
#     tdensity = density(temp.temperature.values, -temp.depth.values, temp.salinity.values, temp.lat.values, temp.lon.values)
    
#     try:
#         gdf = calculate_upper_ocean_metrics(
#             pd.to_datetime(temp.time.values), 
#             temp.temperature.values, 
#             temp.salinity.values, 
#             temp.depth.values,
#             tdensity
#             )
#         cdf = pd.concat([cdf, gdf])
#     except ValueError:
#         continue
# cdf.to_pickle(path_impact_calculated / f"{float_id}_calculated_cmems_data.pkl")

# root_dir = Path.home() / "Documents"
# # data_dir = root_dir / "data"
# # impact_calc = data_dir / "impact_metrics" / "calculated"

# adata = "4903248_calculated_argo_data.pkl"
# gdata = "4903248_calculated_gofs_data.pkl"
# rdata = "4903248_calculated_rtofs_data.pkl"
# cdata =  "4903248_calculated_cmems_data.pkl"
# sname = "4903248_merged_data.pkl"

# argo = pd.read_pickle(root_dir / adata)
# gofs = pd.read_pickle(root_dir / gdata)
# rtofs = pd.read_pickle(root_dir / rdata)
# cmems = pd.read_pickle(root_dir / cdata)

# argo["source"] = float_id
# gofs["source"] = "gofs"
# rtofs["source"] = "rtofs"
# cmems['source'] = 'cmems'

# df = pd.concat([argo, gofs, rtofs, cmems])
# df.to_pickle(root_dir / sname)










# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Part 2 of script - Uncomment after commenting lines 20 through 191
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to save plots and figures
path_plot = (root_dir / "plots" / script_name.replace(".py", "")) # create plot path

# User defined variables
glid = "4903248"
combined = f"{glid}_merged_data.pkl"
freq = None # interval of time for each plot. None for entire record.


# # Read glider data pickle file
# glider_erddap = pd.read_pickle(path_gliders / f"{glid}_data.pkl")

# # Create lon and lat variables for the entire glider track
# glider_lon = glider_erddap['longitude (degrees_east)']
# glider_lat = glider_erddap['latitude (degrees_north)']

# # Create date range from start date to end date of glider data
# start_date = glider_erddap.index[0].floor(freq="1D")
# end_date = glider_erddap.index[-1].ceil(freq='1D')

# if freq:
#     ranges = pd.date_range(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), freq=freq)
# else:
ranges = [t0, t1]

# Read a precreated pickle file of gofs, rtofs, copernicus, and the glider data
df = pd.read_pickle(path_impact_calculated / combined)
# df = pd.read_pickle("/Users/mikesmith/Documents/offset.pkl")

# Create dictionary of plotting inputs for each variable
vars = {}

# Heat content
vars['ocean_heat_content'] = dict(name='Ocean Heat Content', units = 'kJ cm-2', flip_y=False, ylim=[0, 130])
vars['potential_energy_anomaly_100m'] = dict(name='Potential Energy Anomaly (100m)', units='J m-3', flip_y=False, ylim=[100, 650])

# Mixed Layer Depth
vars['mixed_layer_depth_from_temp'] = dict(name='MLD_temp', units='m', flip_y=True, ylim=[0, 55])
vars['mixed_layer_temp_from_temp'] = dict(name='MLT_temp', units='dec C', flip_y=False, ylim=[25, 32])

# Salinity
vars['salinity_surface'] = dict(name='Surface Salinity', units=' ', flip_y=False, ylim=[28, 38])
vars['salinity_max'] = dict(name='Max Salinity ', units=' ', flip_y=False, ylim=[36, 37.2])
vars['salinity_max_depth'] = dict(name='Max Salinity - Depth', units='m', flip_y=True, ylim=[0, 205])

# Average temperatures
vars['average_temp_mldt_to_100m'] = dict(name='Mean Temp. (MLDt to 100m)', units='deg C', flip_y=False, ylim=[20, 32])
vars['average_temp_000m_to_100m'] = dict(name='Mean Temp. (0 to 100m)', units='deg C', flip_y=False, ylim=[20, 32])
vars['average_temp_100m_to_200m'] = dict(name='Mean Temp. (100 to 200m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_200m_to_300m'] = dict(name='Mean Temp. (200 to 300m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_300m_to_400m'] = dict(name='Mean Temp. (300 to 400m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_400m_to_500m'] = dict(name='Mean Temp. (400 to 500m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_500m_to_600m'] = dict(name='Mean Temp. (500 to 600m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_600m_to_700m'] = dict(name='Mean Temp. (600 to 700m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_700m_to_800m'] = dict(name='Mean Temp. (700 to 800m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_800m_to_900m'] = dict(name='Mean Temp. (800 to 900m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_900m_to_1000m'] = dict(name='Mean Temp. (900 to 1000m)', units='deg C', flip_y=False, ylim=[4, 25])


for index, value in enumerate(ranges[1:], start=1):    
    # Save time range as variables
    t0 = ranges[index-1].strftime("%Y-%m-%d")
    t1 = ranges[index].strftime("%Y-%m-%d")
    t1s = ranges[index].strftime("%Y-%m-%d")
    print(f"{index-1} to {index}, {t0} to {t1}")
    datefmt = f"{t0}_to_{t1s}" # String for date range in filename

    # Split out data sources into separate dataframes
    rtofs = df[df.source == 'rtofs']
    gofs = df[df.source == 'gofs']
    copernicus = df[df.source == 'cmems']
    glider = df[df.source == '4903248']

    for k, v in vars.items():
        var_title = v['name']
        if not v['units'] == ' ':
            var_title += f" ({v['units']})"

        # Create figure 
        fig, ax = plt.subplots(figsize=(13.333, 7.5))

        # Plot each model 
        h1 = ax.plot(glider.index, glider[k], 'b-o', markersize=2, label='4903248')
        h2 = ax.plot(gofs.index, gofs[k], 'g-', markersize=2, label='GOFS')
        h3 = ax.plot(rtofs.index, rtofs[k], 'r-o', markersize=2, label='RTOFS')
        # h3 = ax.plot(rtofs.index.shift(-offset, freq='D'), rtofs[k], 'r-o', markersize=2, label='RTOFS')
        h4 = ax.plot(copernicus.index, copernicus[k], 'm-', markersize=2, label='Copernicus')

        # Add grid
        ax.grid(True)

        # # Add minor grid
        ax.grid(True, which="major")
        # ax.grid(True, which="minor", linestyle="-.", linewidth=0.25, alpha=.5)
        # plt.minorticks_on()

        # Make the plot have minor ticks
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        # Set axis limits 
        ax.set_ylim(v['ylim'])

        # Invert axis if flip_y is True
        if v['flip_y']:
            ax.invert_yaxis()

        # Adjust axes labels
        ax.set_xlabel('Datetime (GMT)', fontsize=14, fontweight='bold')
        ax.set_ylabel(var_title, fontsize=14, fontweight='bold')

        # Create and set title
        gname = glid.split('-')[0]
        title = f"{gname} - {t0} to {t1} - {v['name']}"
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Add legend
        plt.legend()

        # Save figure 
        savedir = path_plot / f"{glid}"
        export_fig(savedir, f'argo_{k}_{datefmt}.png', dpi=200)
        plt.close()
