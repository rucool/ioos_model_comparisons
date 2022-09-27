#!/usr/bin/env python
from ioos_model_comparisons.plotting import export_fig
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
import os

# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_gliders = (path_data / "gliders")
path_impact = (path_data / "impact_metrics")
path_impact_calculated = path_impact / "calculated"
path_impact_model = path_impact / "models"

# Paths to save plots and figures
path_plot = (root_dir / "plots" / script_name.replace(".py", "")) # create plot path

# User defined variables
glid = "ng645-20210613T0000"
offset = 0
combined = f"ng645-20210613T0000_{offset}day_offset_combined.pkl"
freq = None # interval of time for each plot. None for entire record.
extent = [-98, -80, 18, 31] # cartopy extent format
# temp_range = [12, 24, .5] # [min, max, step]
temp_range = [25.5, 31.5, .5] # [min, max, step]
haline_range = [34, 37, .25] # [min, max, step]

# Read glider data pickle file
glider_erddap = pd.read_pickle(path_gliders / f"{glid}_data.pkl")

# Create lon and lat variables for the entire glider track
glider_lon = glider_erddap['longitude (degrees_east)']
glider_lat = glider_erddap['latitude (degrees_north)']

# Create date range from start date to end date of glider data
start_date = glider_erddap.index[0].floor(freq="1D")
end_date = glider_erddap.index[-1].ceil(freq='1D')

if freq:
    ranges = pd.date_range(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), freq=freq)
else:
    ranges = [start_date, end_date]

# Read a precreated pickle file of gofs, rtofs, copernicus, and the glider data
df = pd.read_pickle(path_impact_calculated / "merged" / combined)
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
    # gofs = df[df.source == 'gofs']
    # copernicus = df[df.source == 'copernicus']
    glider = df[df.source == 'ng645']

    for k, v in vars.items():
        var_title = v['name']
        if not v['units'] == ' ':
            var_title += f" ({v['units']})"

        # Create figure 
        fig, ax = plt.subplots(figsize=(13.333, 7.5))

        # Plot each model 
        h1 = ax.plot(glider.index, glider[k], 'b-o', markersize=2, label='NG645')
        # h2 = ax.plot(gofs.index, gofs[k], 'g-', markersize=2, label='GOFS')
        # h3 = ax.plot(rtofs.index, rtofs[k], 'r-o', markersize=2, label='RTOFS')
        h3 = ax.plot(rtofs.index.shift(-offset, freq='D'), rtofs[k], 'r-o', markersize=2, label='RTOFS')
        # h4 = ax.plot(copernicus.index, copernicus[k], 'm-', markersize=2, label='Copernicus')

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
        title = f"{gname} - {t0} to {t1} - {v['name']} - Offset: +{offset} day"
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Add legend
        plt.legend()

        # Save figure 
        savedir = path_plot / f"{glid}"
        export_fig(savedir, f'glider_{k}_{offset}day_offset_{datefmt}.png', dpi=200)
        plt.close()
