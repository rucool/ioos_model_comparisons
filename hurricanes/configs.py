from pathlib import Path
from sys import platform

import cartopy.crs as ccrs

# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set root path of where to save the plots
if platform == "linux" or platform == "linux2":
    path_plots = Path("/www/web/rucool/hurricane/model_comparisons")  # server
elif platform == "darwin":
    path_plots = Path.home() / "Documents" / "plots" / "model_comparisons" # local
# elif platform == "win32":
    # Windows...

# Paths to data sources
path_data = current_dir.with_name('data') # data path relative to the toolbox 
# path_gliders = (path_data / "gliders")
# path_argo = (path_data / "argo")
# path_ibtracs = (path_data / "ibtracs/IBTrACS.NA.v04r00.nc")

# Configurations for contour maps 
# (surface_map_comparisons.py, surface_map_rtofs.py, and surface_map_gofs.py
regions = ['gom', 'mab', 'caribbean', 'sab', 'windward', 'yucatan', 'usvi', ]
# regions = ['gom']

days = 2

# Assets
argo = True
gliders = True
search_hours = 24*5
bathy = True

# Plot configurations
dpi = 150
projection = dict(
    map=ccrs.Mercator(), # the projection that you want the map to be in
    data=ccrs.PlateCarree() # the projection that the data is. 
    )
