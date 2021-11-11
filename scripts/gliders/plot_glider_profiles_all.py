import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gsw
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import cmocean
from matplotlib import colors as c
from hurricanes.gliders import glider_dataset
from hurricanes.calc import depth_bin

## Peak Finder
from scipy.signal import find_peaks
from scipy.signal import peak_prominences

glider = 'ng645-20210613T0000'
# glider = 'ng657-20211002T0000'
save_dir = '/Users/mikesmith/Documents/'
t0 = dt.datetime(2021, 8, 28, 12, 0)
t1 =  dt.datetime(2021, 8, 28, 13, 0)
ylims = [150, 1]
temp_xlim = [15, 31]
salinity_xlim = [34, 36.75]
density_xlim = [1021, 1028]

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = t0  # False
gargs['time_end'] =t1
gargs['filetype'] = 'dataframe'

glider_df = glider_dataset(glider, **gargs)
# glider_df = glider_df[glider_df.depth < 150]

grouped = glider_df.groupby('time')

prev = pd.DataFrame()

for group in grouped:
    tstr = group[0].strftime("%Y-%m-%dT%H%M%SZ")
    mint = np.min(group[1]['temperature'])
    maxt = np.max(group[1]['temperature'])
    mins = np.min(group[1]['salinity'])
    maxs = np.max(group[1]['salinity'])
    tempL = np.linspace(mint-1, maxt+1, 156)
    salL = np.linspace(mins-1, maxs+1, 156)
    Tg, Sg = np.meshgrid(tempL, salL)
    sigma_theta = gsw.sigma0(Sg, Tg)
    count = 0

    # Set up for vertical binning
    dz = 1  # 1m resolution
    bins = np.arange(0, round(group[1].depth.max()), dz)
    # bins = -np.arange(1, 101, dz).astype(float)

    # Create temporary dataframe to interpolate to dz m depths
    # temp = depth_interpolate(group[1])
    temp = depth_bin(group[1])
    # temp = group[1].set_index('depth') #set index to depth
    # temp = temp[~temp.index.duplicated()] #Remove duplicated indexs (not sure why there would be duplicates)
    # temp = temp.reindex(temp.index.union(bins)) # reindex to depths in bins
    # temp = temp.drop('time', axis=1).interpolate('index') # drop time and interpolate new depth indexes
    # temp = temp.reindex(index=bins) # only want to see new_index data
    # temp = temp.reset_index() # reset index so you can access the depth variable

    max_dTdz = 0
    d = 1  # distance between peaks

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

    fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    # plt.rcParams['figure.constrained_layout.use'] = True
    grid = plt.GridSpec(2, 2, figure=fig)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, 0])
    ax4 = plt.subplot(grid[1, 1])
    # plt.show()

    # Temperature Profiles (Panel 1)
    if not prev.empty:
        # Temperature Profiles (Panel 1)
        h = ax1.scatter(
            prev['temperature'],
            prev['depth'],
            c='silver',
            s=75,
            label=group[0].strftime('%H:%M:%S'),
            edgecolor='black'
        )

    h = ax1.scatter(
        temp['temperature'],
        temp['depth'],
        c='red',
        s=100,
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )
    ax1.axhline(sort_peak_peak_dep[0])

    # axs[0, 0].legend()

    # for axs in ax.flat:
    ax1.set_ylim(ylims)
    ax1.set_xlim(temp_xlim)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_xlabel('Temperature (°C)', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontsize=22, fontweight='bold')

    # Salinity Profiles (Panel 2)
    if not prev.empty:
        # Temperature Profiles (Panel 1)
        h = ax2.scatter(
            prev['salinity'],
            prev['depth'],
            c='silver',
            s=75,
            label=group[0].strftime('%H:%M:%S'),
            edgecolor='black'
        )

    h = ax2.scatter(
        temp['salinity'],
        temp['depth'],
        c='green',
        s=100,
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )
    ax2.axhline(sort_peak_peak_dep[0])

    # axs[0, 1].legend()
    # axs[0, 1].set_title(group[0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')
    # axs[0, 1].set_title('Salinity Profile')

    # for axs in ax.flat:
    ax2.set_ylim(ylims)
    ax2.set_xlim(salinity_xlim)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_xlabel('Salinity', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Depth (m)', fontsize=2, fontweight='bold')

    # Density Profiles (Panel 3)
    if not prev.empty:
        # Temperature Profiles (Panel 1)
        h = ax3.scatter(
            prev['density'],
            prev['depth'],
            c='silver',
            s=75,
            label=group[0].strftime('%H:%M:%S'),
            edgecolor='black'
        )

    h = ax3.scatter(
        temp['density'],
        temp['depth'],
        c='blue',
        s=100,
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )
    ax3.axhline(sort_peak_peak_dep[0])


    # axs[1, 0].legend()
    # axs[1, 0].set_title(group[0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')
    # axs[1, 0].set_title('Density Profile')

    # for axs in ax.flat:
    ax3.set_ylim(ylims)
    ax3.set_xlim(density_xlim)
    ax3.grid(True, linestyle='--', linewidth=0.5)
    ax3.tick_params(axis='x', labelsize=20)
    ax3.tick_params(axis='y', labelsize=20)
    ax3.set_xlabel('Density', fontsize=22, fontweight='bold')
    ax3.set_ylabel('Depth (m)', fontsize=22, fontweight='bold')

    # T-S Diagrams (Panel 4)

    cs = ax4.contour(Sg, Tg, sigma_theta, colors='grey', linestyles='dashed')
    cl = plt.clabel(cs, inline=1, fontsize=20, fmt='%0.1f')

    N = 19
    cmap = cmocean.cm.dense

    colors = cmap(np.linspace(.1, 1, 19))
    cmap = c.LinearSegmentedColormap.from_list('mycmap', colors)
    cmdict = cmocean.tools.get_dict(cmap, N=19)
    cmap = LinearSegmentedColormap(cmap, segmentdata=cmdict, N=N)
    cmap.set_under('black')

    # define the bins and normalize
    bounds = np.linspace(0, 150, 6)
    norm = c.BoundaryNorm(bounds, cmap.N)

    if not prev.empty:
        sc = ax4.scatter(prev['salinity'], prev['temperature'], c=prev['depth'], cmap=cmap, s=100, norm=norm)
    sc = ax4.scatter(group[1]['salinity'], group[1]['temperature'], c=group[1]['depth'], cmap=cmap, s=100, norm=norm, edgecolor='black'
)

    cb = plt.colorbar(sc, aspect=50, extend='max')
    # axs[1, 1].set_xlim(ts_xlim)
    ax4.set_xlim(salinity_xlim)
    ax4.set_ylim(temp_xlim)
    ax4.set_xlabel('Salinity', fontsize=22, fontweight='bold')
    ax4.set_ylabel('Temperature (°C)', fontsize=22, fontweight='bold')
    # axs[1, 1].set_title('Temperature-Salinity Diagram')
    ax4.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax4.tick_params(direction='out', labelsize=20)
    # cb.ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # cb.ax.set_xticklabels(['zero', 'two', 'four', 'six'])
    cb.ax.tick_params(direction='out', labelsize=19)
    cb.set_label('Depth (m)', fontsize=22, fontweight='bold')
    cb.ax.invert_yaxis()
    title_str = f'{glider} Profiles\n{tstr}'
    plt.suptitle(title_str, fontsize=30, fontweight='bold')

    # cb.set_label('Density[kg m$^{-3}$]')
    # plt.tight_layout()
    plt.savefig(f'/Users/mikesmith/Documents/{glider}-profile_comparison-{tstr}.png')
    plt.close()
    # plt.show()
    # print()
    prev = group[1]


