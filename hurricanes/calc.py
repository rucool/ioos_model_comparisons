import numpy as np
import pandas as pd


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


def depth_interpolate(df, depth_var='depth', depth_min=None, depth_max=None, stride=1, method='linear', index=None):
    """_summary_

    Args:
        df (pd.DataFrame): Depth profile in the form of a pandas dataframe
        depth_var (str, optional): Name of the depth variable in the dataframe. Defaults to 'depth'.
        depth_min (_type_, optional): Shallowest bin depth. Defaults to None.
        depth_max (_type_, optional): Deepest bin depth. Defaults to None.
        stride (int, optional): Amount of space between bins. Defaults to 1.
        method (str, optional): Interpolation type. Defaults to 'linear'.

    Returns:
        pd.DataFrame: dataframe with depth interpolated
    """
    if df.empty:
        print("Dataframe empty. Returning to original function")
        return
    depth_min = depth_min or round(df[depth_var].min())
    depth_max = depth_max or round(df[depth_var].max())

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
