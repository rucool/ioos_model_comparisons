import numpy as np

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
