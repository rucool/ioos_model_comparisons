import seawater
import pandas as pd
import numpy as np
from upper_ocean_metrics.upper_ocean_metrics import mld_temp_crit, ohc_from_profile, \
    mld_dens_crit, temp_average_depth, potential_energy_anomaly100


def calculate_density(temperature, salinity, depth, lat):
    pressure = seawater.eos80.pres(depth, lat)
    density = seawater.eos80.dens(salinity, temperature, pressure)
    return density


def calculate_upper_ocean_metrics(time, temp, salinity, depth, lat, density=None):
    """_summary_

    Args:
        time (_type_): time
        temp (_type_): temperature (c)
        salinity (_type_): salinity (1)
        depth (_type_): depth (m)
        density (_type_, optional): _description_. Defaults to None.
        lat (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if not isinstance(density, pd.Series):
        density = pd.Series(dtype=object)
    
        
    # Reference variables
    dtemp = 0.2
    reference_depth = 10
    threshold_depth = 11
    drho = 0.125

    # Init empty dict 
    ddict = {}
     
    # Calculate density of gofs profile. Converting from depth (m) to pressure 
    # internally.
    if not density.any():
        density = calculate_density(temp, salinity, depth, lat)
    
    # Ocean heat content
    # Depth, temperature, density
    ohc = ohc_from_profile(depth, temp, density)
    ddict['ocean_heat_content'] = ohc

    # Mixed Layer Depth calculated from temperature
    # dtemp, reference depth, depth, and temperature
    depth_mld_t, temp_mld_t, tdepth, rdepth= mld_temp_crit(
        depth, 
        temp, 
        reference_depth, 
        threshold_depth, 
        dtemp)
    ddict['mixed_layer_depth_from_temp'] = depth_mld_t
    ddict['mixed_layer_temp_from_temp'] = temp_mld_t
    ddict['tdepth'] = tdepth
    ddict['rdepth'] = rdepth
    ddict['sdepth'] = depth.min()

    # Mixed Layer Depth calculated from density
    # ddensity, reference depth, depth, temperature, and density
    depth_mld_d, temp_mld_d = mld_dens_crit(drho, reference_depth, depth, temp,  density)
    ddict['mixed_layer_depth_from_density'] = depth_mld_d
    ddict['mixed_layer_temp_from_density'] = temp_mld_d

    # Average temperature in the top 100m 
    # depth, temperature
    ddict['average_temp_mldt_to_100m'] = temp_average_depth(depth, temp, depth_range=[depth_mld_t,100])
    ddict['average_temp_mlds_to_100m'] = temp_average_depth(depth, temp, depth_range=[depth_mld_d,100])
    ddict['average_temp_000m_to_100m'] = temp_average_depth(depth, temp, depth_range=[0,100])
    ddict['average_temp_100m_to_200m'] = temp_average_depth(depth, temp, depth_range=[100,200])
    ddict['average_temp_200m_to_300m'] = temp_average_depth(depth, temp, depth_range=[200,300])
    ddict['average_temp_300m_to_400m'] = temp_average_depth(depth, temp, depth_range=[300,400])
    ddict['average_temp_400m_to_500m'] = temp_average_depth(depth, temp, depth_range=[400,500])
    ddict['average_temp_500m_to_600m'] = temp_average_depth(depth, temp, depth_range=[500,600])   
    ddict['average_temp_600m_to_700m'] = temp_average_depth(depth, temp, depth_range=[600,700])
    ddict['average_temp_700m_to_800m'] = temp_average_depth(depth, temp, depth_range=[700,800])
    ddict['average_temp_800m_to_900m'] = temp_average_depth(depth, temp, depth_range=[800,900])
    ddict['average_temp_900m_to_1000m'] = temp_average_depth(depth, temp, depth_range=[900,1000])


    # Potential Energy Anomaly in the top 100 meters
    # depth, density
    pea = potential_energy_anomaly100(depth, density)
    ddict['potential_energy_anomaly_100m'] = pea

    # Salinity at surface
    # Should this be an average or the very first reading?
    sal_surf_idx = np.nanargmin(depth)
    ddict['salinity_surface'] = salinity[sal_surf_idx]
    # ddict['salinity_surface_depth'] = salinity[sal_surf_idx]

    # Salinity maximum
    try:
        sal_max_idx = np.nanargmax(salinity)
        ddict['salinity_max'] = salinity[sal_max_idx]
        ddict['salinity_max_depth'] = depth[sal_max_idx]
    except ValueError:
        ddict['salinity_max'] = np.nan
        ddict['salinity_max_depth'] = np.nan

    df = pd.DataFrame(data=ddict, index=[pd.to_datetime(time)])
    return df
