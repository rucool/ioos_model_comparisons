import matplotlib.pyplot as plt
import pandas as pd

# rtofs = pd.read_pickle()
gofs = pd.read_pickle('/Users/mikesmith/Documents/uom/calculated_gofs_ng645-20210613T0000_data.pkl')
rtofs = pd.read_pickle('/Users/mikesmith/Documents/uom/calculated_rtofs_ng645-20210613T0000_data.pkl')
glider = pd.read_pickle('/Users/mikesmith/Documents/uom/calculated_ng645-20210613T0000_data.pkl')
copernicus = pd.read_pickle('/Users/mikesmith/Documents/uom/calculated_copernicus_ng645-20210613T0000_data.pkl')
# copernicus = pd.read_csv('/Users/mikesmith/Documents/calculated_copernicus_single_ng645-20210613T0000_data.csv', index_col=0)

gofs.index = pd.to_datetime(gofs.index, utc=True).tz_localize(None)
rtofs.index = pd.to_datetime(rtofs.index, utc=True).tz_localize(None)
copernicus.index = pd.to_datetime(copernicus.index, utc=True).tz_localize(None)

vars = {}
vars['ocean_heat_content'] = dict(name='Ocean Heat Content', units = 'kJ cm-2', flip_y=False, ylim=[0, 130])
vars['mixed_layer_depth_from_temp'] = dict(name='MLD_temp', units='m', flip_y=True, ylim=[0, 55])
vars['mixed_layer_temp_from_temp'] = dict(name='MLT_temp', units='dec C', flip_y=False, ylim=[20, 32])
# vars['mixed_layer_depth_from_density'] = dict(name='MLD_rho', units='m', flip_y=True, ylim=[0, 45.5])
# vars['mixed_layer_temp_from_density'] = dict(name='MLT_rho', units='deg C', flip_y=False, ylim=[25.5, 31.5])
vars['potential_energy_anomaly_100m'] = dict(name='Potential Energy Anomaly (100m)', units='J m-3', flip_y=False, ylim=[100, 650])

vars['salinity_surface'] = dict(name='Surface Salinity', units=' ', flip_y=False, ylim=[28, 38])
vars['salinity_max'] = dict(name='Max Salinity ', units=' ', flip_y=False, ylim=[36, 37.2])
vars['salinity_max_depth'] = dict(name='Max Salinity - Depth', units='m', flip_y=True, ylim=[0, 205])

vars['average_temp_mldt_to_100m'] = dict(name='Mean Temp. (MLDt to 100m)', units='deg C', flip_y=False, ylim=[20, 32])
# vars['average_temp_mlds_to_100m'] = dict(name='Mean Temp. (MLDr to 100m)', units='deg C', flip_y=True, ylim=[20, 32])

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

for k, v in vars.items():
    var_title = v['name']
    if not v['units'] == ' ':
        var_title += f" ({v['units']})"

    # Create figure 
    fig, ax = plt.subplots(figsize=(13.333, 7.5))

    # Plot each model 
    h1 = ax.plot(glider.index, glider[k], 'b-o', markersize=2, label='ng645')
    h2 = ax.plot(gofs.index, gofs[k], 'g-o', markersize=2, label='gofs')
    h3 = ax.plot(rtofs.index, rtofs[k], 'r-o', markersize=2, label='rtofs')
    h4 = ax.plot(copernicus.index, copernicus[k], 'm-o', markersize=2, label='copernicus')

    # Set axis limits 
    ax.set_ylim(v['ylim'])

    # Invert axis if flip_y is True
    if v['flip_y']:
        ax.invert_yaxis()

    # Adjust axes labels
    ax.set_xlabel('Datetime (GMT)', fontsize=14, fontweight='bold')
    ax.set_ylabel(var_title, fontsize=14, fontweight='bold')

    # Create and set title
    title = f"ng645 Model Comparisons\n{v['name']} \n {glider.index.min()} to {glider.index.max()}"
    ax.set_title(title)

    # Add legend
    plt.legend()

    # Save figure 
    plt.savefig(f'/Users/mikesmith/Documents/glider_{k}.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
print()