import seawater

################################################################################
def ohc_from_profile(depth, temp, dens):
    """
    This function Calculates the ocean heat content from a temperature and
    density profile (Leipper, Dale F., and Douglas Volgenau. "Hurricane heat
    potential of the Gulf of Mexico". Journal of Physical Oceanography 2.3
    (1972): 218-224).
    
    Q = ρ Cp ΔT ΔZ 
    
    Q: Hurricane Heat Potential
    ρ: density (gm cm-3)
    Cp: specific heat at constant pressure (cal cm-3) (C)
    ΔT: Average temperature difference above 26C for a given depth increment
    ΔZ: Depth increment (taken as 500cm)

    Args:
        depth (numpy.ndarray or pandas.Series or xarray.Series): depth (m)
        temp (numpy.ndarray or pandas.Series or xarray.Series): temperature (c)
        dens (numpy.ndarray or pandas.Series or xarray.Series): density (gm/cm^3)

    Returns:
        numpy.ndarray: Ocean heat content of the water column in kJ/cm^2
    """

    cp = 3985 #Heat capacity in J/(kg K)
    ok26 = temp >= 26 
    depth = np.abs(depth) #absolute value of depth

    if len(depth[ok26]) != 0:
        if np.nanmin(depth[ok26])>10:
            OHC = np.nan
        else:
            rho0 = np.nanmean(dens[ok26]) # don't include nans
            OHC = np.abs(cp * rho0 * np.trapz(temp[ok26]-26,depth[ok26]))
            OHC = OHC * 10**(-7) # in kJ/cm^2
    else:
        OHC = np.nan
    return OHC


def mld_temp_crit(depth, temp, ref_depth=10, thres_depth=10, dtemp=.2):
    """
    This function calculates the mixed layer depth and temperature
    based on a temperature criteria: T - T_at_ref_depth <= dtemp

    Args:
        depth (_type_): Profile Depth
        temp (_type_): Profile temperature
        ref_depth (int, optional): Reference depth from the mixed layer depth definition used. 
            Defaults to 10.
        thres_depth (int, optional): Depth threshold.
            Defaults to 10.
        dtemp (float, optional): Delta temperature from the mixed layer depth definition used. 
            Defaults to .2.

    Returns:
        _type_: mixed_layer_depth, mixed_layer_temperature
    """
    if ref_depth > thres_depth:
        print("Threshold depth must be greater (deeper) than the reference depth")
        return

    # MLD and MLT: mixed layer depth and Mixed layer temperature
    # find where profile depths are greater than the reference depth
    ok_ref_depth = np.where(depth >= ref_depth)[0]

    # Get the first depth that is greater than the reference depth
    top_ref_depth = ok_ref_depth[0]
    top_profile_depth = depth[top_ref_depth]

    # Check if the shallowest depth of the profile is deeper than the reference depth
    tol = np.abs(thres_depth - ref_depth)/2
    if np.isclose(top_profile_depth, ref_depth+tol, atol=tol):
        # Get the temperature at the first depth that is greater than the reference depth
        temp_ref_depth = temp[top_ref_depth]

        # subtract every single temperature from the reference depth
        delta_T = temp_ref_depth - temp 
        ok_mld_temp = np.where(delta_T <= dtemp)[0]

        if ok_mld_temp.size == 0:
            MLD = np.nan
            MLT = np.nan
        else:
            MLD = depth[ok_mld_temp[-1]]
            MLT = np.nanmean(temp[ok_mld_temp])
        # print(f"Reference Depth {ref_depth} (m) is deeper than shallowest Profile Depth {top_profile_depth} (m).")
        # print(f"MLD: {MLD}, MLT: {MLT}")
        return MLD, MLT, top_profile_depth, ref_depth
    else:
        print(f"Reference Depth {ref_depth} (m) is shallower than shallowest Profile Depth {top_profile_depth} (m). Skipping this profile.")
        return np.nan, np.nan, top_profile_depth, ref_depth
        # return

################################################################################
def mld_dens_crit(drho, ref_depth, depth, temp, dens):
    """
    This function calculates the mixed layer depth andtemperature
    based on a density criteria: rho_at_ref_depth - rho <= drho

    Args:
        drho (array): delta density from the mixed layer depth definition used
        ref_depth (array): Reference depth from the mixed layer depth definition used
        depth (_type_): Profile Depth
        temp (_type_): Profile temperature
        dens (array): Profile density

    Returns:
        np.ndarray: mixed layer depth, Mixed layer temperature
    """
    ok_ref_depth = np.where(depth >= ref_depth)[0][0]
    rho_ref_depth = dens[ok_ref_depth]
    delta_rho = -(rho_ref_depth - dens)
    ok_mld_rho = np.where(delta_rho <= drho)[0]

    if ok_mld_rho.size == 0:
        MLD = np.nan
        MLT = np.nan
    else:
        MLD = depth[ok_mld_rho[-1]]
        MLT = np.nanmean(temp[ok_mld_rho])

    return MLD, MLT

################################################################################
def temp_average_depth(depth, temp, depth_range=[0, 100]):
    """
    This function calculates the depth average temperature between a minimum
    and maximum depth. The default is set to calculate the average Temperature
    in the top 100 m.

    Args:
        depth (numpy.ndarray or pandas.Series or xarray.Series): depth
        temp (numpy.ndarray or pandas.Series or xarray.Series): temperature

    Returns:
        np.ndarray: depth average temperature in the top 100 meters
    """
    # Get index where depths lie between depth_range
    ind = (depth_range[0] <= np.abs(depth)) & (np.abs(depth) <= depth_range[1])
    
    if len( np.where( np.isnan(temp[ind]) )[0] ) > 10:
        averaged = np.nan
    else:
        averaged = np.nanmean(temp[ind])
    return averaged

################################################################################
def potential_energy_anomaly100(depth, dens):
    """
    This function calculates the potential energy anomaly
    (Simpson J, Brown J, Matthews J, Allen G (1990) Tidal straining, density
    currents and stirring in the control of estuarine stratification.
    Estuaries 13(2):125-132), in the top 100 meters

    Args:
        depth (numpy.ndarray or pandas.Series or xarray.Series): depth
        dens (numpy.ndarray or pandas.Series or xarray.Series): density

    Returns:
        np.ndarray: potential energy anomaly in J/m^3
    """

    g = 9.8 #m/s
    dindex = np.fliplr(np.where(np.asarray(np.abs(depth)) <= 100))[0]
    if len(dindex) == 0:
        PEA = np.nan
    else:
        zz = np.asarray(np.abs(depth[dindex]))
        denss = np.asarray(dens[dindex])
        ok = np.isfinite(denss)
        z = zz[ok]
        densi = denss[ok]
        if len(z)==0 or len(densi)==0 or np.min(zz) > 10 or np.max(zz) < 30:
            PEA = np.nan
        else:
            if z[-1] - z[0] > 0:
                # So PEA is < 0
                # sign = -1
                # Adding 0 to sigma integral is normalized
                z = np.append(0,z)
            else:
                # So PEA is < 0
                # sign = 1
                # Adding 0 to sigma integral is normalized
                z = np.flipud(z)
                z = np.append(0,z)
                densit = np.flipud(densi)

            # adding density at depth = 0
            densitt = np.interp(z,z[1:],densit)
            density = np.flipud(densitt)

            # defining sigma
            max_depth = np.nanmax(zz[ok])
            sigma = -1*z/max_depth
            sigma = np.flipud(sigma)

            rhomean = np.trapz(density,sigma,axis=0)
            drho = rhomean - density
            torque = drho * sigma
            PEA = g * max_depth * np.trapz(torque,sigma,axis=0)

    return PEA


def calculate_density(temperature, salinity, depth, lat):
    pressure = seawater.eos80.pres(depth, lat)
    density = seawater.eos80.dens(salinity, temperature, pressure)
    return density


def calculate_upper_ocean_metrics(time, temp, salinity, depth, density=None):
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
    # if not isinstance(density, pd.Series):
        # density = pd.Series(dtype=object)
        
    # Reference variables
    dtemp = 0.2
    reference_depth = 10
    threshold_depth = 11
    drho = 0.125

    # Init empty dict 
    ddict = {}
     
    # Calculate density of gofs profile. Converting from depth (m) to pressure 
    # internally.
    
    # if not density.any():
        # density = calculate_density(temp, salinity, depth, lat)
    
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
