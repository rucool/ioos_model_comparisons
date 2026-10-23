"""Step 2: per-profile metrics for each glider/model pair -> <region>/<glider>/calculated/.

OHC (ohc_with_qc), salinity max, 2 m salinity, 26 C isotherm depth and mixed layer depth (de Boyer Montegut 2007).
Adapted from oneoffs/load_glider_and_model_profiles.py.

usage: python step2_calculate_metrics.py <region> <model>
"""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

from ioos_model_comparisons.calc import density, depth_interpolate
from season_config import ROOT
import numpy as np
from gsw import p_from_z, SA_from_SP, pot_rho_t_exact

models = ['espc', 'rtofs', 'cmems', 'parallel']

def isotherm_depth(temp, depth, target=26.0):
    """
    Return the depth at which temperature == target (°C), by linear interpolation.
    If the profile never reaches target, returns np.nan.
    """
    t = np.asarray(temp)
    z = np.asarray(depth)

    # sort by temperature so np.interp works on ascending t
    idx = np.argsort(t)
    t_sorted = t[idx]
    z_sorted = z[idx]

    if target < t_sorted.min() or target > t_sorted.max():
        return np.nan

    return np.interp(target, t_sorted, z_sorted)


def ohc_with_qc(depth, temp, dens, max_top=10.0, dz=1.0):
    """
    Ocean heat content (kJ/cm^2) above the 26 C isotherm, computed the same way for glider and model.

    The profile is linearly interpolated onto a 1 m grid from the surface to its deepest sample, so the
    integral ends at the actual 26 C crossing (not at the last sample >= 26 C, which costs coarse model
    levels several kJ/cm^2). Above the shallowest sample the top value is held constant (mixed layer).

    qc: 1 good (0.0 when the whole column is below 26 C)
        2 profile starts deeper than max_top m (misses the warm surface layer)
        3 profile never gets below 26 C (integral would be truncated)
        4 fewer than 2 valid samples
    """
    df = pd.DataFrame({'d': np.asarray(depth, float), 't': np.asarray(temp, float),
                       'r': np.asarray(dens, float)}).dropna(subset=['d', 't'])
    df = df.groupby('d', as_index=False).mean().sort_values('d')   # np.interp needs unique, increasing depths
    if len(df) < 2:
        return np.nan, 4
    d, t, r = df.d.values, df.t.values, df.r.values
    if d[0] > max_top:
        return np.nan, 2
    if np.all(t < 26):
        return 0.0, 1
    if t[-1] >= 26:
        return np.nan, 3
    z = np.arange(0.0, d[-1] + dz, dz)
    excess = np.clip(np.interp(z, d, t) - 26, 0, None)
    rho = np.nanmean(r[t >= 26]) if np.isfinite(r[t >= 26]).any() else 1024.0
    return 3985 * rho * np.trapezoid(excess, z) * 1e-7, 1


def calc_mld_upper(df,ref_depth,deltaT,_z,_temp,_sal,lat,lon):
    #find the depth index closest to reference depth
    zdiff = _z - ref_depth
    dref=np.where((zdiff>0) & (zdiff<=1) & (~np.isnan(zdiff)))[0]
    
    # dref=np.argmin(np.abs(np.abs(_z)-ref_depth))
    if (dref.size!=0) and (~np.isnan(_z[dref[0]])):
        dref=dref[0]
    # if (np.abs(_z[dref]-ref_depth)<=1) and (~np.isnan(_z[dref])):
        
        pres=p_from_z(-_z,lat)
        SA = SA_from_SP(_sal,pres,lon,lat)
        #calculate intial rho
        ini_rho=pot_rho_t_exact(SA,_temp,pres,0)
        if (~np.isnan(SA[dref])) and (~np.isnan(_temp[dref])) and (~np.isnan(pres[dref])):
            # ini_rho=pot_rho_t_exact(SA[dref:],_temp[dref:],pres[dref:],0)
            #calculate rho profile using reference depth
            refdepth_rho=pot_rho_t_exact(SA[dref],_temp[dref],pres[dref],0)
            #calculate rho -deltaT reference depth
            refdepth_rho_deltaT =pot_rho_t_exact(SA[dref],_temp[dref]-deltaT,pres[dref],0)
            #calculate delta sigma
            delta_sig=np.abs(refdepth_rho_deltaT - refdepth_rho)
            #calculate difference between rho profile and rho at reference depth
            dens_diff=np.abs(ini_rho[dref:] - refdepth_rho)
            #grab first index where this is true
            if np.where(dens_diff>=delta_sig)[0].size>0:
                
                mld_idx=np.where(dens_diff>=delta_sig)[0][0]
                mldU=_z[dref:][mld_idx]
                qcU=1
                
            else:
                mldU=df.depth.iloc[-1]
                qcU=2 #profile doesn't surpass delta_sigma threshold
        else:
            mldU=np.nan
            qcU= 5 #one of T S P being nan cuased related density vars to be nan       
    else:
        mldU=np.nan
        qcU= 3 #profile doesn't have depth close enough to ref_depth or depth[ref_depth] is nan
    
    return mldU,qcU


def calc_mld_lower(df,deltaT,_z,_temp,_sal,lat,lon):
    bot_depth = df.depth.iloc[-1]
    dref=np.where(_z==bot_depth)[0]
    if (dref.size!=0) and (~np.isnan(dref[-1])):
        dref=dref[-1]
        pres=p_from_z(-_z,lat)
        SA = SA_from_SP(_sal,pres,lon,lat)
        #calculate intial rho
        ini_rho=pot_rho_t_exact(SA,_temp,pres,0)
        if (~np.isnan(SA[dref])) and (~np.isnan(_temp[dref])) and (~np.isnan(pres[dref])):
            
            #calculate rho profile using reference depth
            refdepth_rho=pot_rho_t_exact(SA[dref],_temp[dref],pres[dref],0)
            #calculate rho -deltaT reference depth
            refdepth_rho_deltaT =pot_rho_t_exact(SA[dref],_temp[dref]-deltaT,pres[dref],0)
            #calculate delta sigma
            delta_sig=np.abs(refdepth_rho_deltaT - refdepth_rho)
            #calculate difference between rho profile and rho at reference depth
            dens_diff=np.abs(ini_rho - refdepth_rho)
            # dens_diff=np.abs(ini_rho - ini_rho[-1])
            #grab first index where this is true
            if np.where(dens_diff>delta_sig)[0].size>0:
                
                mld_idx=np.where(dens_diff>delta_sig)[0][-1]
                mldL=_z[mld_idx]
                qcL=1
                
            else:
                mldL=df.depth.iloc[-1]
                qcL=2 #profile doesn't surpass delta_sigma threshold
        else:
            mldL=np.nan
            qcL=5 #one of T S P being nan cuased related density vars to be nan    
    else:
        mldL=np.nan
        qcL=3 #ref depth is nan
    return mldL,qcL

def calc_mld(df,ref_depth,deltaT):
    """
    based off of de Boyer Montegut (2007) and Rudzin (2017)
    df - single glider profile
    ref_depth - reference depth to calculate potential desnity at 
    deltaT - change in temperature threshold
    qc: 1 - profile is good and has mld
        2 - profile doesn't surpass delta_signma threshold
        3 - profile doesn't have a depth close enough to ref_depth
        4 - profile is less than 10m long
        5 - T, S, P being nan cuased related density vars to be nan
        6 - mldL < mldU, relatively unstratified
    """
    ttdf=df.sort_values('depth',ascending=True)
    tdf=depth_interpolate(ttdf,depth_var='depth',depth_min=ttdf.depth.min(),depth_max=ttdf.depth.max(),stride=0.25,method='linear')
    # print(tdf.gliders.iloc[0],tdf.time.iloc[0])
    depth_diff=tdf.depth.iloc[-1]-tdf.depth.iloc[0]
    if (depth_diff > 10) or (np.isnan(depth_diff)):
        _z=tdf.depth.values
        _temp=tdf.temperature.values
        _sal=tdf.salinity.values
        lat=tdf.lat.values
        lon=tdf.lon.values
    
        if (~np.isnan(_sal).all()) and (~np.isnan(_temp).all()) and (~np.isnan(_z).all()):
            mldU,qcU = calc_mld_upper(tdf,ref_depth,deltaT,_z,_temp,_sal,lat,lon)
            mldL,qcL = calc_mld_lower(tdf,deltaT,_z,_temp,_sal,lat,lon)
            if mldL < mldU:
                qcU=6
                qcL=6
                mldU=df.depth.iloc[-1]
                mldL=df.depth.iloc[-1]
        else:
            mldU=np.nan
            mldL=np.nan
            qcU=5 # one of T S P being nan cuased related density vars to be nan
            qcL=5
    else:
        mldU=np.nan
        mldL=np.nan
        qcU=4 #profile less than 10m
        qcL=4
    return mldU, mldL, qcU, qcL

def load_profiles(region_dir, model='rtofs'):
    region_path = Path(region_dir)
    glider_dirs = [d for d in region_path.iterdir() if d.is_dir()]
    
    matched_profiles = []

    for glider_dir in tqdm(glider_dirs, desc="Crawling glider directories"):
        glider_name = glider_dir.name
        all_csvs = list(glider_dir.glob(f"{glider_name}_*.csv"))

        for glider_file in all_csvs:
            fname = glider_file.name

            # # Only use glider_names that have 'ng278' or 'ng656' in the fname
            # if not any(glider_name in fname for glider_name in ['ng278', 'ng656']):
            #     continue

            # Skip files that are not glider profiles
            models = ['espc', 'rtofs', 'cmems', 'parallel']      
            stem = glider_file.stem  # e.g., 'sp070-20240912T1432_20241012T225615Z_rtofs'
            if any(stem.endswith(f"_{m}") for m in models):
                continue  # Skip model profile

            # Load glider profile
            try:
                glider_df = pd.read_csv(glider_file)
            except Exception as e:
                print(f"⚠️ Error reading glider profile {fname}: {e}")
                continue

            # Sort the glider dataframe by depth
            glider_df = glider_df.sort_values(by='depth', ascending=True).reset_index(drop=True)

            # Extract latitude and longitude from the glider dataframe
            lat = glider_df['lat'].iloc[0]
            lon = glider_df['lon'].iloc[0]

            # Remove NaNs
            valid = glider_df[['depth', 'salinity']].dropna()
            glider_df_sorted = valid.sort_values('depth')

            depths = glider_df_sorted['depth'].values
            salinities = glider_df_sorted['salinity'].values

            salinity_2m = np.nan  # default if we can't calculate

            # depths = glider_df['depth'].values
            # salinities = glider_df['salinity'].values

            # Interpolate salinity to 2m depth using np.interp
            if len(depths) >= 2:
                if 2.0 < depths[0]:
                    x0, x1 = depths[0], depths[1]
                    y0, y1 = salinities[0], salinities[1]
                    if x1 != x0:
                        slope = (y1 - y0) / (x1 - x0)
                        salinity_2m = y0 + slope * (2.0 - x0)
                elif 2.0 > depths[-1]:
                    # Optional: extrapolate deep
                    salinity_2m = np.nan
                else:
                    salinity_2m = np.interp(2.0, depths, salinities)


            # Calculate ocean heat content for glider profile
            density_glider = density(glider_df['temperature'].values, -glider_df['depth'].values, glider_df['salinity'].values, lat, lon)
            ohc_glider, ohc_qc_glider = ohc_with_qc(glider_df['depth'].values, glider_df['temperature'].values, density_glider)

            # Add ocean heat content to glider dataframe
            glider_df['glider_ohc'] = ohc_glider
            glider_df['glider_ohc_qc'] = ohc_qc_glider

            # Find the salinity maximum in the glider profile  
            salinity_max_glider = glider_df['salinity'].max()  
            salinity_max_depth_glider = glider_df['depth'].iloc[glider_df['salinity'].argmax()]
            salinity_surface_glider = glider_df['salinity'].iloc[0]  # Surface salinity

            # Find the salinity surface (at 2m) in the glider profile
            # if salinity_2m is not None:
            # salinity_surface_glider = salinity_2m


            # Add salinity maximum information to glider dataframe
            glider_df['glider_salinity_max'] = salinity_max_glider
            glider_df['glider_salinity_max_depth'] = salinity_max_depth_glider
            glider_df['glider_salinity_surface'] = salinity_2m

            # compute 26 °C isotherm depth for the glider
            iso26_glider = isotherm_depth(
                glider_df['temperature'].values,
                glider_df['depth'].values,
                target=26.0
            )
            glider_df['glider_iso26_depth'] = iso26_glider

            # Calculate mld for glider profile
            # mld_df=model_df.groupby('Profile_ID').apply(calc_mld,ref_depth=4,deltaT=0.5).reset_index()
            # mld_df[['mldU','mldL','qcU', 'qcL']] = mld_df[0].apply(pd.Series)
            # mld_df=mld_df.drop(0, axis=1)
            mldU, mldL, qcU, qcL = calc_mld(glider_df.drop(['glider', 'time'], axis=1), ref_depth=4, deltaT=0.5)

            # Add mld information to glider dataframe
            glider_df['glider_mldU'] = mldU
            glider_df['glider_mldL'] = mldL
            glider_df['glider_qcU'] = qcU
            glider_df['glider_qcL'] = qcL
# 
            # Construct model filename
            base_name = fname.replace(".csv", f"_{model}.csv")
            model_file = glider_file.parent / base_name

            if not model_file.exists():
                print(f"❌ Missing model file for {fname}")
                continue

            # Load model profile
            try:
                model_df = pd.read_csv(model_file)
            except Exception as e:
                print(f"⚠️ Error reading model profile {model_file.name}: {e}")
                continue

            # Calculate ocean heat content for the model profile
            density_model = density(model_df['temperature'].values, -model_df['depth'].values, model_df['salinity'].values, lat, lon)
            ohc_model, ohc_qc_model = ohc_with_qc(model_df['depth'].values, model_df['temperature'].values, density_model)

            # Find the salinity maximum in the model profile
            salinity_max_model = model_df['salinity'].max()
            salinity_max_depth_model = model_df['depth'].iloc[model_df['salinity'].argmax()]

            # Interpolate salinity to 2m depth for the model profile
            depths_model = model_df['depth'].values
            salinities_model = model_df['salinity'].values
            salinity_2m_model = np.nan  # default if we can't calculate
            if len(depths_model) >= 2:
                if 2.0 < depths_model[0]:
                    x0, x1 = depths_model[0], depths_model[1]
                    y0, y1 = salinities_model[0], salinities_model[1]
                    if x1 != x0:
                        slope = (y1 - y0) / (x1 - x0)
                        salinity_2m_model = y0 + slope * (2.0 - x0)
                elif 2.0 > depths_model[-1]:
                    # Optional: extrapolate deep
                    salinity_2m_model = np.nan
                else:
                    salinity_2m_model = np.interp(2.0, depths_model, salinities_model)
            
            salinity_surface_model = salinity_2m_model  # Surface salinity for model

            # Add ocean heat content and salinity maximum information to model dataframe
            model_df['model_ohc'] = ohc_model
            model_df['model_ohc_qc'] = ohc_qc_model
            model_df['model_salinity_max'] = salinity_max_model
            model_df['model_salinity_max_depth'] = salinity_max_depth_model
            model_df['model_salinity_surface'] = salinity_surface_model

            # compute 26 °C isotherm depth for the model
            iso26_model = isotherm_depth(
                model_df['temperature'].values,
                model_df['depth'].values,
                target=26.0
            )
            model_df['model_iso26_depth'] = iso26_model

            # Calculate mld for model profile

            mldU, mldL, qcU, qcL = calc_mld(model_df.drop('time', axis=1), ref_depth=4, deltaT=0.5)

            # Add mld information to model dataframe
            model_df['model_mldU'] = mldU
            model_df['model_mldL'] = mldL
            model_df['model_qcU'] = qcU
            model_df['model_qcL'] = qcL

            matched_profiles.append((glider_name, glider_df, model_df))

            # # Save the matched profiles to a new CSV file
            # output_file = glider_file.parent / f"{glider_name}_matched_{model}.csv"
            # try:
            #     combined_df = pd.concat([glider_df, model_df], axis=1)
            #     combined_df.to_csv(output_file, index=False)
            #     print(f"✅ Saved matched profiles to {output_file}")
            # except Exception as e:
            #     print(f"⚠️ Error saving matched profiles to {output_file}: {e}")

            # Save the glider and model profiles as CSV files in the 'calculated' subdirectory
            calculated_dir = glider_file.parent / 'calculated'
            calculated_dir.mkdir(exist_ok=True)

            # # Create a new filename for the glider and model profiles
            # glider_file.rename(calculated_dir / glider_file.name)
            # model_file.rename(calculated_dir / model_file.name)

            # Save the glider and model profiles as separate CSV files
            glider_df = glider_df.reset_index(drop=True)
            model_df = model_df.reset_index(drop=True)

            glider_df.to_csv(calculated_dir / f"{glider_file.name}", index=False)
            model_df.to_csv(calculated_dir / f"{model_file.name}", index=False)
            # print(f"✅ Saved glider and model profiles to {glider_file.parent / f'{glider_name}_glider.csv'} and {glider_file.parent / f'{glider_name}_model.csv'}")



    return matched_profiles

region, model = sys.argv[1], sys.argv[2]
matched = load_profiles(os.path.join(ROOT, region), model=model)
print(f"✅ Loaded {len(matched)} matched profile pairs.")
