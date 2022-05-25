import datetime as dt
import os
import matplotlib.pyplot as plt

import hurricanes.gliders as gld
import numpy as np

## Peak Finder
from scipy.signal import find_peaks
from scipy.signal import peak_prominences


glider = 'ng645-20210613T0000'
# glider = 'ng657-20211002T0000'
save_dir = '/Users/mikesmith/Documents/'
t0 = dt.datetime(2021, 8, 28, 0, 0)
t1 =  dt.datetime(2021, 8, 31, 0, 0)
ylims = [150, 1]
temp_xlim = [15, 31]
salinity_xlim = [34, 36.75]
density_xlim = [1021, 1028]

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = t0  # False
gargs['time_end'] =t1
gargs['filetype'] = 'dataframe'

sdir_glider = os.path.join(save_dir, glider, 'transects', 'transect-ribbons')
os.makedirs(sdir_glider, exist_ok=True)
glider_df = gld.glider_dataset(glider, **gargs)

glider_df = glider_df[glider_df.depth < 150]

grouped = glider_df.groupby('time')

ts_time = [] # time for mixed layer
ts_mld = [] # mixed layer depth

for group in grouped:
    tstr = group[0].strftime("%Y-%m-%dT%H%M%SZ")
    print(tstr)

    count = 0

    # Set up for vertical binning
    dz = 1  # 1m resolution
    bins = np.arange(0, round(group[1].depth.max()), dz)
    # bins = -np.arange(1, 101, dz).astype(float)

    # Create temporary dataframe to interpolate to dz m depths
    temp = group[1].set_index('depth') #set index to depth
    temp = temp[~temp.index.duplicated()] #Remove duplicated indexs (not sure why there would be duplicates)
    temp = temp.reindex(temp.index.union(bins)) # reindex to depths in bins
    temp = temp.drop('time', axis=1).interpolate('index') # drop time and interpolate new depth indexes
    temp = temp.reindex(index=bins) # only want to see new_index data
    temp = temp.reset_index() # reset index so you can access the depth variable

    max_dTdz = 0
    d = 5  # distance between peaks

    # Calculate dT/dz from binned temperature profile
    # bin_dTdz = np.append(np.diff(binT) / np.diff(bins), np.nan) #Sams way
    # bin_dTdz = np.diff(temp['temperature']) / np.diff(bins) #Sams way modified
    bin_dTdz = temp['density'].diff() / temp['depth'].diff()

    # Find where the peaks in the signal occur, excluding negative peaks, making sure they are a distance 'd' away from each other
    peaks, prop = find_peaks(bin_dTdz, distance=d, threshold=(0, np.nanmax(bin_dTdz)))

    # Determine the magnitude of these peaks
    prom = peak_prominences(bin_dTdz, peaks)[0]  # the amount the peak stands out from its lowest contour

    # Sort by peak magnitude
    sort_ind = np.argsort(prom)
    sort_peaks = peaks[sort_ind]
    sort_prom = prom[sort_ind]

    # Find the two strongest peaks
    peak_peak_dep = sort_peaks[-2:]
    peak_peak_prom = sort_prom[-2:]

    # Sort by index at which they occur
    sort_peak_peak_ind = np.argsort(peak_peak_dep)
    sort_peak_peak_dep = peak_peak_dep[sort_peak_peak_ind]
    sort_peak_peak_prom = peak_peak_prom[sort_peak_peak_ind]

    # # Shallowest peak goes to surface layer and deeper peak goes to deep layer
    # try:
    #     ts_mld.append(sort_peak_peak_dep[0])
    #     ts_time.append(group[0])
    # except:
    #     print('skipping')

    # Salinity profile
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 10),
        # constrained_layout=True
    )

    h = ax.scatter(
        temp['temperature'],
        temp['depth'],
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )
    plt.axhline(sort_peak_peak_dep[0])

    ax.legend()
    ax.set_title(group[0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

    # for axs in ax.flat:
    ax.set_ylim([150, 1])
    # ax.set_xlim([1020, 1028])
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Temperature (f)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')
    plt.savefig(f'/Users/mikesmith/Documents/{glider}-profile_comparison-{tstr}.png')
    plt.close()

    # plt.show()

    # max_dT_depth[count] = -sort_peak_peak_dep[0]
    # deep_max_dT_depth[count] = -sort_peak_peak_dep[-1]

    # Save the height of the peaks as well for a time series
    # max_dT_peakprom[count] = sort_peak_peak_prom[0]
    # deep_max_dT_peakprom[count] = sort_peak_peak_prom[-1]

    # count += 1
    #
    # if np.any(np.nanmax(bin_dTdz) > max_dTdz):
    #     max_dTdz = np.nanmax(bin_dTdz)
# print()
#
# ###########################
# ## TS diagrams
# # again, g* are 1D arrays of glider data
# # Jaden deos something like this with a hex map function and it looks a bit better. Travis doesn't like the bricks in mine and we are playing with alternate ideas.
#
# import gsw
# from scipy import stats
#
#
# def plot_TS_contours(T, S):
#     mint = np.nanmin(T)
#     maxt = np.nanmax(T)
#     mins = np.nanmin(S)
#     maxs = np.nanmax(S)
#     tempL = np.linspace(mint - 1, maxt + 1, 156)
#     salL = np.linspace(mins - 1, maxs + 1, 156)
#     Tg, Sg = np.meshgrid(tempL, salL)
#     sigma_theta = gsw.sigma0(Sg, Tg)  # ignore effects of pressure on density
#     cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), 156)
#     cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', zorder=1)
#     cl = plt.clabel(cs, fontsize=10, inline=True, fmt='%.1f')
#     return cs
#
#
# # heat map of all data
# sBins = np.arange(np.nanmin(temp['salinity']), np.nanmax(temp['salinity']), 0.1)
# tBins = np.arange(np.nanmin(temp['temperature']), np.nanmax(temp['temperature']), 0.05)
# sGrid, tempGrid = np.meshgrid(sBins, tBins)
# out = stats.binned_statistic_2d(temp['salinity'], temp['temperature'], None, statistic='count',
#                                 bins=(sBins, tBins))  # Counts how many data values fall in each bin in the grid
#
# fig, ax = plt.subplots(figsize=(10, 10))
# import cmocean
# kwargs = {'vmin': 0, 'vmax': 100, 'cmap': cmocean.cm.speed}
# plt.pcolormesh(sGrid[:-1, :-1], tempGrid[:-1, :-1], out.statistic.T, **kwargs)
#
#
