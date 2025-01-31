import sys
from pyproj import Geod
import numpy as np
import pandas as pd
from gsw import SA_from_SP, CT_from_t, rho, p_from_z
import datetime as dt
from pyproj import Geod
g = Geod(ellps="WGS84")

def create_datetime_list(
    ctime=dt.datetime.utcnow(), days_before=2, days_after=1, freq="6H", time_offset=0
):
    """_summary_

    Args:
        ctime (datetime, optional): Center Time. Defaults to dt.datetime.utcnow().
        days_before (int, optional): Days before center time. Defaults to 2.
        days_after (int, optional): Days after center time. Defaults to 1.
        freq (str, optional): Pandas frequency string. Defaults to '6H'.
        time_offset (int, optional): _description_. Defaults to 0.

    Returns:
        list: list of datetimes
    """

    # Convert
    ctime = dt.datetime(*ctime.timetuple()[:3])

    # If you want to start the date list at a specific day
    if time_offset:
        ctime = ctime + dt.timedelta(hours=time_offset)

    # Calculate window around center time
    date_start = ctime - dt.timedelta(days=days_before)
    date_end = ctime + dt.timedelta(days=days_after)

    # Create dates that we want to plot
    date_list = pd.date_range(date_start, date_end, freq=freq)

    return date_list


def ocean_heat_content(depth, temp, density):
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

    cp = 3985  # Heat capacity in J/(kg K)
    temp_mask = temp >= 26  # only want data greater than or equal to 26C
    depth = np.abs(depth)  # absolute value of depth

    # Mask the variables based off the temperature mask
    depth_m = depth[temp_mask]
    density_m = density[temp_mask]
    temp_m = temp[temp_mask]

    # If the number of depths do not equal 0
    if len(depth_m) != 0:
        # If the minimum depth is shallower than 10m
        if np.nanmin(depth_m) > 10:
            OHC = np.nan
        # If the minimum depth is deeper than 10m
        else:
            rho0 = np.nanmean(density_m)  # don't include nans
            OHC = np.abs(cp * rho0 * np.trapz(temp_m - 26, depth_m))
            OHC = OHC * 10 ** (-7)  # in kJ/cm^2
    # If the number of depths do equal 0
    else:
        OHC = np.nan
    return OHC


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
    absolute_salinity = SA_from_SP(salinity, pressure, longitude, latitude)

    # Calculates Conservative Temperature of seawater from in-situ temperature.
    conservative_temperature = CT_from_t(
        absolute_salinity, temperature, pressure)

    # Calculates in-situ density from Absolute Salinity and
    # Conservative Temperature, using the computationally-efficient expression
    # for specific volume in terms of SA, CT and p (Roquet et al., 2015).
    density = rho(absolute_salinity, conservative_temperature, pressure)

    return density


def difference(array1, array2):
    """
    Calculate the difference between two equal arrays.

    Bias is the difference between the mean of these estimates and the actual value.

    Args:
        array1 (_type_): Array 1
        array2 (_type_): Array 2

    Returns:
        tuple: difference, bias, rms error
    """
    diff = array1 - array2

    bias = np.round(np.sum(diff).values / diff.shape, 3)
    rms = np.round(rmse(diff), 3)
    return (diff, bias[0], rms.values)


def rmse(array):
    """
    Calculate root mean square error

    Args:
        array (array): _description_

    Returns:
        _type_: _description_
    """
    return np.sqrt(np.mean(array**2))


def lon180to360(array):
    array = np.array(array)
    return np.mod(array, 360)


def lon360to180(array):
    array = np.array(array)
    return np.mod(array + 180, 360) - 180


def find_nearest(array, value):
    """
    Find the index of closest value in array

    Args:
        array (list or np.array): _description_
        value (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx = (np.abs(array - value)).argmin()
    return array.flat[idx], idx


def depth_averaged_current_vectorized(depths, u_currents, v_currents):
    
    # Assuming depths is always the first dimension
    layer_thicknesses = np.diff(depths, axis=0)
    
    # Insert the first layer's thickness
    first_layer_thickness = np.expand_dims(layer_thicknesses[0], axis=0)
    layer_thicknesses = np.concatenate([first_layer_thickness, layer_thicknesses], axis=0)
    
    depth_weighted_u_currents = u_currents * layer_thicknesses[:, np.newaxis, np.newaxis]
    depth_weighted_v_currents = v_currents * layer_thicknesses[:, np.newaxis, np.newaxis]

    depth_averaged_u_current = depth_weighted_u_currents.sum(axis=0) / layer_thicknesses.sum()
    depth_averaged_v_current = depth_weighted_v_currents.sum(axis=0) / layer_thicknesses.sum()
    
    return depth_averaged_u_current, depth_averaged_v_current


# def depth_interpolate(
#     df,
#     depth_var="depth",
#     depth_min=0,
#     depth_max=1000,
#     stride=10,
#     method="linear",
#     index=None,
# ):
#     """_summary_

#     Args:
#         df (pd.DataFrame): Depth profile in the form of a pandas dataframe
#         depth_var (str, optional): Name of the depth variable in the dataframe. Defaults to 'depth'.
#         depth_min (float or string, optional): Shallowest bin depth. Pass 'round' to round to nearest minimumdepth. Defaults to None.
#         depth_max (float or string, optional): Deepest bin depth. Pass 'round' to round to nearest maximum depth. Defaults to None.
#         stride (int, optional): Amount of space between bins. Defaults to 10.
#         method (str, optional): Interpolation type. Defaults to 'linear'.

#     Returns:
#         pd.DataFrame: dataframe with depth interpolated
#     """
#     if df.empty:
#         print("Dataframe empty. Returning to original function")
#         return

#     if isinstance(depth_min, str):
#         if depth_min == "round":
#             depth_min = round(df[depth_var].min())
#         else:
#             depth_min = int(depth_min)

#     if isinstance(depth_min, str):
#         if depth_min == "round":
#             depth_max = round(df[depth_var].max())
#         else:
#             depth_max = int(depth_max)

#     bins = np.arange(depth_min, depth_max + stride, stride)

#     # Create temporary dataframe to interpolate to dz m depths
#     temp = df.set_index(depth_var)  # set index to depth
#     temp = temp[
#         ~temp.index.duplicated()
#     ]  # Remove duplicated indexs (not sure why there would be duplicates)
#     temp = temp.reindex(temp.index.union(bins))  # reindex to depths in bins
#     try:
#         temp = temp.drop("time", axis=1).interpolate(
#             method=method, limit_direction="both"
#         )  # drop time and interpolate new depth indexes
#     except KeyError:
#         temp = temp.interpolate(
#             method=method, limit_direction="both"
#         )  # drop time and interpolate new depth indexes
    
#     temp = temp.reindex(index=bins)  # only want to see new_index data
#     temp = temp.reset_index()  # reset index so you can access the depth variable

#     if index:
#         temp = temp.set_index(index)

#     return temp

from typing import Optional, List


def depth_interpolate(
    df: pd.DataFrame,
    depth_var: str = "depth",
    depth_min: Optional[float] = 0,
    depth_max: Optional[float] = None,
    stride: float = 10,
    bins: Optional[np.ndarray] = None,
    method: str = "linear",
    index: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
    interpolation_direction: str = "both",
) -> pd.DataFrame:
    """
    Interpolates the data along the depth axis for a given dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with depth and associated data.
    depth_var : str
        Column name representing depth.
    depth_min : Optional[float]
        Minimum depth for interpolation. If None, uses the smallest depth in the dataframe.
    depth_max : Optional[float]
        Maximum depth for interpolation. If None, uses the deepest depth in the dataframe.
    stride : float
        Depth increment for interpolation.
    bins : Optional[np.ndarray]
        Custom depth bins to interpolate over. If None, bins are created using depth_min, depth_max, and stride.
    method : str
        Interpolation method. Options include 'linear', 'polynomial', 'cubic', etc.
    index : Optional[str]
        Column to use as the index after interpolation.
    drop_cols : Optional[List[str]]
        Columns to drop before interpolation.
    interpolation_direction : str
        Direction of interpolation. Can be 'forward', 'backward', or 'both'.
    
    Returns:
    --------
    pd.DataFrame
        Interpolated dataframe with the specified depth bins.
    """
    if df.empty:
        raise ValueError("Dataframe is empty")
        
    if depth_var not in df.columns:
        raise ValueError(f"'{depth_var}' is not a valid column in the dataframe")

    if stride <= 0:
        raise ValueError(f"Stride must be positive, but got {stride}")

    # Handle depth_max as optional
    actual_depth_max = df[depth_var].max()
    if depth_max is None:
        depth_max = actual_depth_max
    else:
        depth_max = min(depth_max, actual_depth_max)

    # Ensure depth_min is valid
    actual_depth_min = df[depth_var].min()
    if depth_min is None or depth_min < actual_depth_min:
        depth_min = actual_depth_min

    # Generate bins if not provided
    if bins is None:
        bins = np.arange(depth_min, depth_max + stride, stride)

    # Remove duplicate depth values, handle them as needed (e.g., take the mean)
    df = df.groupby(depth_var).mean().reset_index()

    # Reindex for interpolation
    temp = df.set_index(depth_var)
    temp = temp.reindex(temp.index.union(bins))

    # Optionally drop specified columns before interpolation
    if drop_cols:
        temp = temp.drop(columns=drop_cols, errors='ignore')

    # Interpolate missing values
    temp = temp.interpolate(method=method, limit_direction=interpolation_direction)

    # Reindex back to the bins and reset index
    temp = temp.reindex(index=bins).reset_index()

    # Optionally set the provided index column
    if index:
        temp = temp.set_index(index)

    return temp

def depth_bin(df, depth_var="depth", depth_min=0, depth_max=None, stride=1, aggregation='mean', index_type='mid'):
    """
    Bins a depth profile from a pandas dataframe and computes aggregated values for each bin, with the bin edges or mid-points as the index. Ensures all bins above a specified minimum are included, even if empty.

    Parameters:
    - df (pd.DataFrame): Depth profile data.
    - depth_var (str): Column name for the depth variable. Defaults to "depth".
    - depth_min (int): Minimum depth for binning. Defaults to 0.
    - depth_max (int): Maximum depth for binning. Defaults to max depth in df.
    - stride (int): Interval between depth bins. Defaults to 1.
    - aggregation (str): Aggregation method (e.g., 'mean', 'sum', 'median'). Defaults to 'mean'.
    - index_type (str): Type of index for the bins ('edge' or 'mid'). Defaults to 'mid'.

    Returns:
    - pd.DataFrame: Dataframe with data aggregated into specified depth bins, indexed by bin edges or mid-points.

    Raises:
    - ValueError: If input parameters are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if depth_var not in df.columns:
        raise ValueError(f"{depth_var} column not found in DataFrame")

    if not isinstance(depth_min, (int, float)) or not isinstance(depth_max, (int, float, type(None))) or not isinstance(stride, (int, float)):
        raise ValueError("depth_min, depth_max, and stride must be numbers")

    if depth_max is not None and depth_max <= depth_min:
        raise ValueError("depth_max must be greater than depth_min")

    if index_type not in ['edge', 'mid']:
        raise ValueError("index_type must be 'edge' or 'mid'")

    depth_max = depth_max or max(df[depth_var].max(), depth_min)

    bins = np.arange(depth_min, depth_max + stride, stride)
    cut = pd.cut(df[depth_var], bins, right=False, labels=False, include_lowest=True)

    if aggregation not in ['mean', 'sum', 'median']:
        raise ValueError("Invalid aggregation method. Choose from 'mean', 'sum', 'median'.")

    binned_df = df.groupby(cut).agg(aggregation)

    # Ensure all bins are represented, including empty ones
    binned_df = binned_df.reindex(range(len(bins) - 1))

    # Setting index as bin edges or mid-points
    if index_type == 'edge':
        binned_df.index = pd.IntervalIndex.from_breaks(bins).astype(str)
    else:  # mid
        mid_points = (bins[:-1] + bins[1:]) / 2
        binned_df.index = mid_points

    binned_df.index.name = 'depth'
    return binned_df.drop('depth', axis=1).reset_index()


def calculate_transect(start, end, dist=5000):    
    pts = g.inv_intermediate(start[0], start[1], end[0], end[1], 0, dist) 
    _, _, dist = g.inv(
        np.full(len(pts.lons), start[0]), 
        np.full(len(pts.lons), start[1]), 
        pts.lons, 
        pts.lats
        )
    return np.column_stack([pts.lons, pts.lats]), np.array(dist)


# def calculate_transect(x1, y1, x2, y2, grid_spacing=None):
#     """
#     Calculate longitude and latitude of transect lines
#     :param x1: western longitude
#     :param y1: southern latitude
#     :param x2: eastern longtiude
#     :param y2: northern latitude
#     :return: longitude, latitude, distance along transect (km)
#     """
#     grid_spacing = grid_spacing or 0.05

#     try:
#         # Slope
#         m = (y1 - y2) / (x1 - x2)
#         if np.abs(m) == 0:
#             # Horizontal (W->E) transect
#             X = np.arange(x2, x1, grid_spacing)
#             Y = np.full(X.shape, y1)
#         else:
#             # Intercept
#             b = y1 - m * x1
#             X = np.arange(x1, x2, grid_spacing)
#             Y = b + m * X
#     except ZeroDivisionError:
#         # Vertical (S->N) transect
#         Y = np.arange(y2, y1, grid_spacing)
#         X = np.full(Y.shape, x1)

#     dist = (
#         np.sqrt((X - x1) ** 2 + (Y - y1) ** 2) * 111
#     )  # approx along transect distance in km
#     return X, Y, dist


# decimal degrees to degree-minute-second converter
def dd2dms(vals):
    n = np.empty(np.shape(vals))
    n[:] = False
    n[vals < 0] = True
    vals[n == True] = -vals[n == True]
    d = np.floor(vals)
    rem = vals - d
    rem = rem * 60
    m = np.floor(rem)
    rem -= m
    s = np.round(rem * 60)
    d[n == True] = -d[n == True]
    return d, m, s


def interpolate(xval, df, xcol, ycol):
    """
    Compute xval as the linear interpolation of xval where df is a dataframe and
    from: https://stackoverflow.com/questions/56832608/how-can-i-interpolate-values-in-a-python-dataframe
    :param xval:
    :param df:
    :param xcol:
    :param ycol:
    :return:
    """
    return np.interp([xval], df[xcol], df[ycol])


def inverse_transformation(lons1, lats1, lons2, lats2):
    """
    Inverse computation of bearing and distance given the latitudes and longitudes of an initial and terminus point.

    Args:
        lons1 (array, numpy.ndarray, list, tuple, or scalar): Longitude(s) of initial point(s)
        lats1 (array, numpy.ndarray, list, tuple, or scalar): Latitude(s) of initial point(s)
        lons2 (array, numpy.ndarray, list, tuple, or scalar): Longitude(s) of terminus point(s)
        lats2 (array, numpy.ndarray, list, tuple, or scalar): Latitude(s) of terminus point(s)

    Returns:
       array, numpy.ndarray, list, tuple, or scalar: Forward azimuth(s)
       array, numpy.ndarray, list, tuple, or scalar: Back azimuth(s)
       array, numpy.ndarray, list, tuple, or scalar: Distance(s) between initial and terminus point(s) in kilometers
    """
    # Inverse transformation using pyproj
    # Determine forward and back azimuths, plus distances between initial points and terminus points.
    forward_azimuth, back_azimuth, distance = g.inv(
        lons1, lats1, lons2, lats2)

    forward_azimuth = np.array(forward_azimuth)
    back_azimuth = np.array(back_azimuth)
    distance = np.array(distance)
    forward_azimuth = np.mod(forward_azimuth, 360)
    back_azimuth = np.mod(back_azimuth, 360)

    distance = (
        distance / 1000
    )  # Lets stick with kilometers as the output since RUV ranges are in kilometers

    return forward_azimuth, back_azimuth, distance


def rotate_vector(u, v, theta):
    """Rotate a vector in the counter-clockwise

    Args:
        u (_type_): _description_
        v (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    cos_theta = np.cos(theta/180*np.pi)
    sin_theta = np.sin(theta/180*np.pi)

    # ur = v*sin_theta + u*cos_theta # I believe this is the u-earth component
    # vr = v*cos_theta - u*sin_theta # I believe this is the v-earth component
    
    # ur = u*cos_theta - v*sin_theta
    # vr = u*sin_theta + v*cos_theta
    ur = u*cos_theta + v*sin_theta
    vr = -u*sin_theta + v*cos_theta
    return ur, vr


import matplotlib.colors
import matplotlib.pyplot as plt
def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Expand your colormap by changing the alpha value (opacity) of each color.
    
    Returns a colormap with nc*nsc different colors, 
    where for each category there are nsc colors of same hue.
    
    
    From ImportanceOfBeingErnest
    https://stackoverflow.com/a/47232942/2643708
    
    Args:
        nc (int): number of categories (colors)
        nsc (int): number of subcategories (shades for each color)
        cmap (str, optional): matplotlib colormap. Defaults to "tab10".
        continuous (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: Too many categories for colormap

    Returns:
        object: matplotlib colormap
    """

    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap