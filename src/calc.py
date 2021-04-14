import numpy as np


def calculate_transect(x1, y1, x2, y2):
    """
    Calculate longitude and latitude of transect lines
    :param x1: wester longitude
    :param y1: southern latitude
    :param x2: eastern longtiude
    :param y2: northern latitude
    :return: longitude, latitude, distance along transect (km)
    """
    # Slope
    m = (y1 - y2) / (x1 - x2)

    # Intercept
    b = y1 - m * x1

    X = np.arange(x1, x2, 0.05)
    Y = b + m * X
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