import xarray as xr
import numpy as np
from ioos_model_comparisons.calc import (
    calculate_transect, 
    lon180to360, 
    lon360to180
    )
import os
import copernicusmarine as cm
from dateutil import parser
import logging
logging.basicConfig(level=logging.INFO)  # or adjust logging level as needed

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


def rtofs(rename=None, source='east'):
    if source == 'east':
        url = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped"
        model = 'RTOFS'
    elif source == 'west':
        url = 'https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_west_scraped'
        model = 'RTOFS (West Coast)'
    elif source == 'parallel':
        url = 'https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_parallel_scraped'
        model = 'RTOFS (Parallel)'
    ds = xr.open_dataset(url)

    ds = ds.rename(
        {'Longitude': 'lon', 
         'Latitude': 'lat',
         'MT': 'time',
         'Depth': 'depth',
         'X': 'x', 
         'Y': 'y'
         }
        )
    ds = ds.set_coords(['lon', 'lat'])
    ds.attrs['model'] = model
    return ds

class RTOFS():
    def __init__(self) -> None:
        self._data_orig = self.load()
        self._data_orig = self._data_orig.set_coords(['u', 'v'])
        self.data = self._data_orig.copy()
        self.x = self.data.x.values
        self.y = self.data.y.values
        self.grid_lons = self.data.lon.values[0,:]
        self.grid_lats = self.data.lat.values[:,0]

    def load(self):
        url = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped"
        ds = xr.open_dataset(url).set_coords(['lon', 'lat'])
        ds.attrs['model'] = 'RTOFS'
        return ds

    def subset(self, extent):
         # Find x, y indexes of the area we want to subset
        lons_ind = np.interp(extent[:2], self.grid_lons, self.x)
        lats_ind = np.interp(extent[2:], self.grid_lats, self.y)

        # Use np.floor on the first index and np.ceiling on  the second index
        # of each slice in order to widen the area of the extent slightly.
        extent = [
            np.floor(lons_ind[0]).astype(int),
            np.ceil(lons_ind[1]).astype(int),
            np.floor(lats_ind[0]).astype(int),
            np.ceil(lats_ind[1]).astype(int)
        ]

        # Use the xarray .isel selector on x/y 
        # since we know the exact indexes we want to slice
        self.data = self._data_orig.isel(
            x=slice(extent[0], extent[1]),
            y=slice(extent[2], extent[3])
            )

    def transect(self, start, end, grid_spacing=5000):
        """
        # Return a transect 

        Args:
            start (array): Start of transect (lon, lat)
            end (array): End of transect (lon, lat)
            grid_spacing (int, optional): Distance (meters) between each point along transect. Defaults to 5000.

        Returns:
            _type_: _description_
        """
        return calculate_transect(start, end, grid_spacing)

    # @classmethod
    def profile(self, lon, lat, method='nearest'):
        # Find x, y indexes of the area we want to subset
        lons_ind = np.interp(lon, self.grid_lons, self.x)
        lats_ind = np.interp(lat, self.grid_lats, self.y)

        if method == 'nearest':
            xds = self._data_orig.sel(x=lons_ind, y=lats_ind, method='nearest')
            # rdsp = rds.sel(time=ctime, method='nearest')
        elif method == 'interp':
            xds = self._data_orig.interp(x=lons_ind, y=lats_ind)
        return xds
        

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

import xarray as xr

class ESPC:
    '''
    Class for handling ESPC data with lazy loading and subsetting for specific times and regions.
    '''

    def __init__(self) -> None:
        '''
        Initialize the ESPC instance by lazily loading the datasets.
        '''
        self.datasets = {}  # Store datasets lazily in a dictionary
        self._load_data()

    def _load_data(self):
        """Load individual datasets lazily and store them."""
        datasets = {
            'temperature': 'https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/t3z/2024',
            'salinity': 'https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/s3z/2024',
            'u': 'https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z/2024',
            'v': 'https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z/2024'
        }
        # Lazy load and store each dataset in a dictionary
        for var, url in datasets.items():
            self.datasets[var] = self._load_single_dataset(url)

    def _load_single_dataset(self, url):
        """Lazy load a single dataset from a given URL."""
        return xr.open_dataset(url, drop_variables='tau')  # Lazy load by default with xarray

    def get_variable(self, var_name):
        """
        Retrieve a variable lazily from the relevant dataset.

        Args:
        - var_name (str): Variable name to retrieve (temperature, salinity, u, v).
        
        Returns:
        - xarray.DataArray: The lazy-loaded variable.
        """
        if var_name in self.datasets:
            return self.datasets[var_name]
        else:
            raise ValueError(f"Variable {var_name} not recognized.")

    def get_subset(self, var_name, lon_extent, lat_extent, time=None):
        """
        Subset the data by variable, longitude/latitude extents, and time (optional).

        Args:
        - var_name (str): The variable to subset (e.g., 'temperature', 'salinity', 'u', 'v').
        - lon_extent (tuple): Longitude range for subsetting.
        - lat_extent (tuple): Latitude range for subsetting.
        - time (datetime): Optional time for subsetting.

        Returns:
        - xarray.DataArray: Subset of the requested variable.
        """
        lon_extent = lon180to360(lon_extent)
        data = self.get_variable(var_name)
        subset = data.sel(lon=slice(lon_extent[0], lon_extent[1]), 
                          lat=slice(lat_extent[0], lat_extent[1]))

        subset['lon'] = lon360to180(subset['lon'])
        if time:
            subset = subset.sel(time=time, method='nearest')
        return subset

    def get_combined_subset(self, lon_extent, lat_extent, time=None):
        """
        Get a combined subset of all variables (temperature, salinity, u, v) for the given extents and time.

        Args:
        - lon_extent (tuple): Longitude range for subsetting.
        - lat_extent (tuple): Latitude range for subsetting.
        - time (datetime): Optional time for subsetting.

        Returns:
        - xarray.Dataset: Dataset with temperature, salinity, u, and v.
        """
        temperature = self.get_subset('temperature', lon_extent, lat_extent, time)
        salinity = self.get_subset('salinity', lon_extent, lat_extent, time)
        u = self.get_subset('u', lon_extent, lat_extent, time)
        v = self.get_subset('v', lon_extent, lat_extent, time)

        # Lazily merge the variables into a single dataset when ready
        ds = xr.merge(
            [
                temperature, 
                salinity, 
                u, 
                v
                ]
            )

        ds = ds.rename(
            {
                'water_u': 'u',
                'water_v': 'v',
                'water_temp': 'temperature'
                }
            )
        # xr.Dataset(
        #     {'temperature': temperature,
        #      'salinity': salinity, 
        #      'u': u,
        #      'v': v}
        # )

        ds.attrs['model'] = 'ESPC'

        return ds

    def get_point(self, lon, lat, time=None, interp=False):
        """
        Retrieve data for a specific longitude and latitude point at a certain time.

        Args:
        - lon (float): Longitude of the point.
        - lat (float): Latitude of the point.
        - time (datetime, optional): Time for subsetting. Defaults to None.

        Returns:
        - xarray.Dataset: Dataset containing temperature, salinity, u, and v at the specific point and time.
        """
        temperature = self.get_variable('temperature')
        salinity = self.get_variable('salinity')
        u = self.get_variable('u')
        v = self.get_variable('v')

        if interp:
            temperature = temperature.interp(time=time, lon=lon, lat=lat)
            salinity = salinity.interp(time=time, lon=lon, lat=lat)
            u = u.interp(time=time, lon=lon, lat=lat)
            v = v.interp(time=time, lon=lon, lat=lat)
        else:
            temperature = temperature.sel(lon=lon, lat=lat, method='nearest')
            salinity = salinity.sel(lon=lon, lat=lat, method='nearest')
            u = u.sel(lon=lon, lat=lat, method='nearest')
            v = v.sel(lon=lon, lat=lat, method='nearest')

        if time:
            temperature = temperature.sel(time=time, method='nearest')
            salinity = salinity.sel(time=time, method='nearest')
            u = u.sel(time=time, method='nearest')
            v = v.sel(time=time, method='nearest')

        # Combine the variables into a single dataset
        ds = xr.merge(
            [
                temperature, 
                salinity, 
                u, 
                v
                ]
            )
        ds.attrs['model'] = 'ESPC'

        ds = ds.rename(
            {
                'water_temp': 'temperature', 
                'water_u': 'u',
                'water_v': 'v'
                }
        )     
        return ds

class CMEMS:
    '''
    Class for handling CMEMS data with lazy loading and subsetting for specific times and regions.
    '''

    def __init__(self, username='maristizabalvar', password='MariaCMEMS2018') -> None:
        '''
        Initialize the CMEMS instance with user credentials.

        Args:
        - username (str): CMEMS username.
        - password (str): CMEMS password.
        '''
        self.username = username or os.getenv('CMEMS_USERNAME')
        self.password = password or os.getenv('CMEMS_PASSWORD')
        self.datasets = {}  # Store datasets lazily in a dictionary
        self._load_data()

    def _load_data(self):
        """Load individual datasets lazily and store them."""
        datasets = {
            'temperature': "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
            'salinity': "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i",
            'currents': "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"
        }
        # Lazy load and store each dataset in a dictionary
        for var, dataset_id in datasets.items():
            self.datasets[var] = self._load_single_dataset(dataset_id)

    def _load_single_dataset(self, dataset_id):
        """Lazy load a single dataset from CMEMS."""
        return cm.open_dataset(
            dataset_id=dataset_id,
            username=self.username,
            password=self.password,
        )  # Lazy load by default

    def get_variable(self, var_name):
        """
        Retrieve a variable lazily from the relevant dataset.

        Args:
        - var_name (str): Variable name to retrieve (temperature, salinity, currents).
        
        Returns:
        - xarray.DataArray: The lazy-loaded variable.
        """
        if var_name == 'temperature':
            return self.datasets['temperature']['thetao']  # Lazy access to temperature
        elif var_name == 'salinity':
            return self.datasets['salinity']['so']  # Lazy access to salinity
        elif var_name in ['uo', 'vo']:
            return self.datasets['currents'][var_name]  # Lazy access to current components
        else:
            raise ValueError(f"Variable {var_name} not recognized.")

    def get_subset(self, var_name, lon_extent, lat_extent, time=None):
        """
        Subset the data by variable, longitude/latitude extents, and time (optional).

        Args:
        - var_name (str): The variable to subset (e.g., 'temperature', 'salinity', 'u', 'v').
        - lon_extent (tuple): Longitude range for subsetting.
        - lat_extent (tuple): Latitude range for subsetting.
        - time (datetime): Optional time for subsetting.

        Returns:
        - xarray.DataArray: Subset of the requested variable.
        """
        data = self.get_variable(var_name)
        subset = data.sel(longitude=slice(lon_extent[0], lon_extent[1]), 
                          latitude=slice(lat_extent[0], lat_extent[1]))
        if time:
            subset = subset.sel(time=time, method='nearest')
        return subset

    def get_combined_subset(self, lon_extent, lat_extent, time=None):
        """
        Get a combined subset of all variables (temperature, salinity, u, v) for the given extents and time.

        Args:
        - lon_extent (tuple): Longitude range for subsetting.
        - lat_extent (tuple): Latitude range for subsetting.
        - time (datetime): Optional time for subsetting.

        Returns:
        - xarray.Dataset: Dataset with temperature, salinity, u, and v.
        """
        temperature = self.get_subset('temperature', lon_extent, lat_extent, time)
        salinity = self.get_subset('salinity', lon_extent, lat_extent, time)
        u = self.get_subset('uo', lon_extent, lat_extent, time)
        v = self.get_subset('vo', lon_extent, lat_extent, time)

        # Lazily merge the variables into a single dataset when ready
        ds = xr.Dataset(
            {'temperature': temperature,
             'salinity': salinity, 
             'u': u,
             'v': v}
            )
        ds.attrs['model'] = 'CMEMS'

        ds = ds.rename(
            {
                'longitude': 'lon',
                'latitude': 'lat'
                }
            )
        return ds

    def get_point(self, lon, lat, time=None, interp=False):
        """
        Retrieve data for a specific longitude and latitude point at a certain time.

        Args:
        - lon (float): Longitude of the point.
        - lat (float): Latitude of the point.
        - time (datetime, optional): Time for subsetting. Defaults to None.

        Returns:
        - xarray.Dataset: Dataset containing temperature, salinity, u, and v at the specific point and time.
        """
        temperature = self.get_variable('temperature')
        salinity = self.get_variable('salinity')
        u = self.get_variable('uo')
        v = self.get_variable('vo')
        
        if interp:
            temperature = temperature.interp(time=time, longitude=lon, latitude=lat)
            salinity = salinity.interp(time=time, longitude=lon, latitude=lat)
            u = u.interp(time=time, longitude=lon, latitude=lat)
            v = v.interp(time=time, longitude=lon, latitude=lat)
        else:
            temperature = temperature.sel(longitude=lon, latitude=lat, method='nearest')
            salinity = salinity.sel(longitude=lon, latitude=lat, method='nearest')
            u = u.sel(longitude=lon, latitude=lat, method='nearest')
            v = v.sel(longitude=lon, latitude=lat, method='nearest')
            
        if time:
            temperature = temperature.sel(time=time, method='nearest')
            salinity = salinity.sel(time=time, method='nearest')
            u = u.sel(time=time, method='nearest')
            v = v.sel(time=time, method='nearest')

        # Combine the variables into a single dataset
        ds = xr.Dataset(
            {'temperature': temperature,
                'salinity': salinity,
                'u': u,
                'v': v}
        )
        ds.attrs['model'] = 'CMEMS'

        ds = ds.rename(
            {
                'longitude': 'lon',
                'latitude': 'lat'
                }
            )

        return ds




def cnaps(rename=False):
    url = "http://3.236.148.88:8080/thredds/dodsC/fmrc/useast_coawst_roms/COAWST-ROMS_SWAN_Forecast_Model_Run_Collection_best.ncd"
    ds = xr.open_dataset(url)
    ds = ds.drop('ocean_time').squeeze()
    ds.attrs['model'] = 'CNAPS'

    if rename:
        ds = ds.rename(
            {
                'temp': 'temperature', 
                'salt': 'salinity',
                'lat_rho': 'lat',
                'lon_rho': 'lon',
                'u_eastward': 'u',
                'v_northward': 'v'
                }
            )
    return ds

if __name__ == '__main__':
    ds = rtofs()