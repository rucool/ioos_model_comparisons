import numpy as np
from scipy import spatial
import xarray as xr
import pandas as pd


def depth_interpolate(df, depth_var='depth', depth_min=None, depth_max=None, stride=1, method='linear'):
    """

    :param df: depth profile in the form of a pandas dataframe
    :param depth_var: the name of the depth variable in the dataframe
    :param depth_min: the shallowest bin depth
    :param depth_max: the deepest bin depth
    :param stride: the amount of space between each bin
    :param method: interpolation type: defaults to linear.https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
    :return: pandas dataframe where data has been interpolated to specific depths
    """
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


def convert_ll_to_model_ll(X, Y, model=None):
    """
    Convert from lat
    :param X:
    :param Y:
    :return:
    """

    model = model.lower() or 'rtofs'

    if model.lower() == 'gofs':
        try:
            lon = np.empty((len(X),))
            lon[:] = np.nan
        except TypeError:
            lon = [X]

        for i, ii in enumerate(X):
            if ii < 0:
                lon[i] = 360 + ii
            else:
                lon[i] = ii
        lat = Y
    else:
        lon = X
        lat = Y
    return lon, lat


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


class KDTreeIndex():
    """ A KD-tree implementation for fast point lookup on a 2D grid

    Keyword arguments:
    dataset -- a xarray DataArray containing lat/lon coordinates
               (named 'lat' and 'lon' respectively)

    """

    def transform_coordinates(self, coords):
        """ Transform coordinates from geodetic to cartesian

        Keyword arguments:
        coords - a set of lan/lon coordinates (e.g. a tuple or
                 an array of tuples)
        """
        # WGS 84 reference coordinate system parameters
        A = 6378.137  # major axis [km]
        E2 = 6.69437999014e-3  # eccentricity squared

        coords = np.asarray(coords).astype(np.float)

        # is coords a tuple? Convert it to an one-element array of tuples
        if coords.ndim == 1:
            coords = np.array([coords])

        # convert to radiants
        lat_rad = np.radians(coords[:, 0])
        lon_rad = np.radians(coords[:, 1])

        # convert to cartesian coordinates
        r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
        x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_n * (1 - E2) * np.sin(lat_rad)

        return np.column_stack((x, y, z))

    def __init__(self, dataset):
        # store original dataset shape
        self.shape = dataset.shape

        # reshape and stack coordinates
        coords = np.column_stack((dataset.lat.values.ravel(),
                                  dataset.lon.values.ravel()))

        # construct KD-tree
        self.tree = spatial.cKDTree(self.transform_coordinates(coords))

    def query(self, point):
        """ Query the kd-tree for nearest neighbour.

        Keyword arguments:
        point -- a (lat, lon) tuple or array of tuples
        """
        _, index = self.tree.query(self.transform_coordinates(point))

        # regrid to 2D grid
        index = np.unravel_index(index, self.shape)

        # return DataArray indexers
        return xr.DataArray(index[0], dims='location'), \
               xr.DataArray(index[1], dims='location')