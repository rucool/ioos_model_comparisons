import xarray as xr
import numpy as np
from ioos_model_comparisons.calc import calculate_transect

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


def rtofs(source='east'):
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


# def cmems(rename=False):
#     username = 'maristizabalvar'
#     password = 'MariaCMEMS2018'
#     dataset = 'global-analysis-forecast-phy-001-024'
#     from pydap.client import open_url
#     from pydap.cas.get_cookies import setup_session
#     cas_url = 'https://cmems-cas.cls.fr/cas/login'
#     session = setup_session(cas_url, username, password)
#     session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
#     database = ['my', 'nrt']
#     url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
#     try:
#         data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
#     except:
#         url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
#         data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits

#     # Downloading and reading Copernicus grid
#     ds = xr.open_dataset(data_store, drop_variables='tau')
#     ds.attrs['model'] = 'CMEMS'

#     if rename:
#         ds = ds.rename(
#             {
#                 'thetao': 'temperature', 
#                 'so': 'salinity',
#                 'latitude': 'lat',
#                 'longitude': 'lon',
#                 'uo': 'u',
#                 'vo': 'v'
#                 }
#             )
#     return ds


    # def __init__(self, username='maristizabalvar', password='MariaCMEMS2018') -> None:
import os
# import copernicusmarine as cm
from dateutil import parser
import xarray as xr

class CMEMS:
    '''
    Class for handling Copernicus Marine Environment Monitoring Service (CMEMS) data.
    '''

    def __init__(self, username='maristizabalvar', password='MariaCMEMS2018') -> None:
        '''
        Initialize the CMEMS instance with user credentials.

        Args:
        - username (str): CMEMS username. Defaults to None and reads from environment.
        - password (str): CMEMS password. Defaults to None and reads from environment.
        '''
        self.username = username or os.getenv('CMEMS_USERNAME')
        self.password = password or os.getenv('CMEMS_PASSWORD')
        self.data = None
        self.load_data()

    def load_data(self):
        """Load datasets from CMEMS and merge them into a single xarray Dataset."""
        try:
            datasets = [
                "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
                "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
                "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i"
            ]
            data_list = [self._load_single_dataset(ds_id) for ds_id in datasets]
            self.data = xr.merge(data_list)
            self._rename_vars()
        except Exception as e:
            print(f"Failed to load and merge data: {e}")

    def _load_single_dataset(self, dataset_id):
        return cm.open_dataset(
            dataset_id=dataset_id,
            username=self.username,
            password=self.password,
        )

    def _rename_vars(self):
        rename_dict = {
            'thetao': 'temperature', 
            'so': 'salinity',
            'latitude': 'lat',
            'longitude': 'lon',
            'uo': 'u',
            'vo': 'v'
        }
        existing_vars = set(self.data.variables.keys()) & set(rename_dict.keys())
        self.data = self.data.rename({k: rename_dict[k] for k in existing_vars})

    def subset(self, extent, start_time=None, end_time=None):
        """Return a subset of data based on geographical and optional temporal limits."""
        try:
            if start_time:
                start_time = parser.parse(start_time)
            if end_time:
                end_time = parser.parse(end_time)

            sel_dict = {
                'longitude': slice(extent[0], extent[1]),
                'latitude': slice(extent[2], extent[3])
            }
            if start_time and end_time:
                sel_dict['time'] = slice(start_time, end_time)

            return self.data.sel(**sel_dict)
        except Exception as e:
            print(f"Error subsetting data: {e}")
            return None

      
def cmems(rename=False):
    username = 'maristizabalvar'
    password = 'MariaCMEMS2018'
    from pydap.client import open_url
    from pydap.cas.get_cookies import setup_session
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])

    try:
        url = f'https://my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i'
        data_store1 = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
    except:
        url = f'https://nrt.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i'
        data_store1 = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits

    try:
        url = f'https://my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
        data_store2 = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
    except:
        url = f'https://nrt.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
        data_store2 = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits

    try:
        url = f'https://my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i'
        data_store3 = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
    except:
        url = f'https://nrt.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i'
        data_store3 = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
        

    # # Downloading and reading Copernicus grid
    ds1 = xr.open_dataset(data_store1, drop_variables='tau')
    ds2 = xr.open_dataset(data_store2, drop_variables='tau')
    ds3 = xr.open_dataset(data_store3, drop_variables='tau')
    ds = xr.merge([ds1, ds2, ds3])

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