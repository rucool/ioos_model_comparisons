import numpy as np


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
        # Intercept
        b = y1 - m * x1
        X = np.arange(x1, x2, grid_spacing)
        Y = b + m * X
    except ZeroDivisionError:
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