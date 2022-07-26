import numpy as np
import pandas as pd
from gsw import SA_from_SP, CT_from_t, rho, p_from_z


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

    cp = 3985 #Heat capacity in J/(kg K)
    temp_mask = temp >= 26 # only want data greater than or equal to 26C
    depth = np.abs(depth) # absolute value of depth

    # Mask the variables based off the temperature mask
    depth_m = depth[temp_mask]
    density_m = density[temp_mask]
    temp_m = temp[temp_mask]

    # If the number of depths do not equal 0
    if len(depth_m) != 0:
        # If the minimum depth is shallower than 10m
        if np.nanmin(depth_m)>10:
            OHC = np.nan
        # If the minimum depth is deeper than 10m
        else:
            rho0 = np.nanmean(density_m) # don't include nans
            OHC = np.abs(cp * rho0 * np.trapz(temp_m - 26, depth_m))
            OHC = OHC * 10**(-7) # in kJ/cm^2
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
    return np.mod(array+180, 360)-180

def find_nearest(array, value):
    """
    Find the index of closest value in array

    Args:
        array (list or np.array): _description_
        value (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx = (np.abs(array-value)).argmin()
    return array.flat[idx], idx


def depth_interpolate(df, depth_var='depth', depth_min=0, depth_max=1000, stride=10, method='linear', index=None):
    """_summary_

    Args:
        df (pd.DataFrame): Depth profile in the form of a pandas dataframe
        depth_var (str, optional): Name of the depth variable in the dataframe. Defaults to 'depth'.
        depth_min (float or string, optional): Shallowest bin depth. Pass 'round' to round to nearest minimumdepth. Defaults to None.
        depth_max (float or string, optional): Deepest bin depth. Pass 'round' to round to nearest maximum depth. Defaults to None.
        stride (int, optional): Amount of space between bins. Defaults to 10.
        method (str, optional): Interpolation type. Defaults to 'linear'.

    Returns:
        pd.DataFrame: dataframe with depth interpolated
    """
    if df.empty:
        print("Dataframe empty. Returning to original function")
        return

    if isinstance(depth_min, str):
        if depth_min == 'round':
            depth_min = round(df[depth_var].min())
        else:
            depth_min = int(depth_min)

    if isinstance(depth_min, str):
        if depth_min == 'round':
            depth_max = round(df[depth_var].max())
        else:
            depth_max = int(depth_max)

    bins = np.arange(depth_min, depth_max+stride, stride)

    # Create temporary dataframe to interpolate to dz m depths
    temp = df.set_index(depth_var)  #set index to depth
    temp = temp[~temp.index.duplicated()]  #Remove duplicated indexs (not sure why there would be duplicates)
    temp = temp.reindex(temp.index.union(bins))  # reindex to depths in bins
    temp = temp.drop('time', axis=1).interpolate(method=method, limit_direction='both')  # drop time and interpolate new depth indexes
    temp = temp.reindex(index=bins)  # only want to see new_index data
    temp = temp.reset_index()  # reset index so you can access the depth variable

    if index:
        temp = temp.set_index(index)

    return temp


def depth_bin(df,  depth_var='depth', depth_min=0, depth_max=None, stride=1):
    """
    This function will

    :param df: depth profile in the form of a pandas dataframe
    :param depth_var: the name of the depth variable in the dataframe
    :param depth_min: the shallowest bin depth
    :param depth_max: the deepest bin depth
    :param stride: the amount of space between each bin
    :return: pandas dataframe where data has been averaged into specified depth bins
    """
    depth_max = depth_max or df[depth_var].max()

    bins = np.arange(depth_min, depth_max+stride, stride)  # Generate array of depths you want to bin at
    cut = pd.cut(df[depth_var], bins)  # Cut/Bin the dataframe based on the bins variable we just generated
    binned_df = df.groupby(cut).mean()  # Groupby the cut and do the mean
    return binned_df


def calculate_transect(x1, y1, x2, y2, grid_spacing=None):
    """
    Calculate longitude and latitude of transect lines
    :param x1: western longitude
    :param y1: southern latitude
    :param x2: eastern longtiude
    :param y2: northern latitude
    :return: longitude, latitude, distance along transect (km)
    """
    grid_spacing = grid_spacing or 0.05

    try:
        # Slope
        m = (y1 - y2) / (x1 - x2)
        if np.abs(m) == 0:
            # Horizontal (W->E) transect
            X = np.arange(x2, x1, grid_spacing)
            Y = np.full(X.shape, y1)
        else:
            # Intercept
            b = y1 - m * x1
            X = np.arange(x1, x2, grid_spacing)
            Y = b + m * X
    except ZeroDivisionError:
        # Vertical (S->N) transect
        Y = np.arange(y2, y1, grid_spacing)
        X = np.full(Y.shape, x1)
    
    dist = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2) * 111  # approx along transect distance in km
    return X, Y, dist


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
