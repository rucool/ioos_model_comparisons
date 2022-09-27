import datetime as dt
import numpy as np
import os
import matplotlib.pyplot as plt
import ioos_model_comparisons.gliders as gld
import gsw
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import cmocean

glider = 'ng645-20210613T0000'
# glider = 'ng657-20211002T0000'
save_dir = '/Users/mikesmith/Documents/'
title_str = 'ng657 T-S (Temperature and salinity) Diagram\n{}'

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = dt.datetime(2021, 8, 28, 0, 0)  # False
gargs['time_end'] = dt.datetime(2021, 8, 30, 0, 0)
# gargs['time_start'] = dt.datetime(2021, 6, 17, 3, 0)  # False
# gargs['time_end'] = dt.datetime(2021, 6, 24, 9, 0)
gargs['filetype'] = 'dataframe'

y_limits = [-20, 0]  # None
c_limits = dict(temp=dict(shallow=np.arange(25, 30, .25)),
                salt=dict(shallow=np.arange(34, 37, .25)))

sdir_glider = os.path.join(save_dir, glider, 'transects', 'transect-ribbons')
os.makedirs(sdir_glider, exist_ok=True)
glider_df = gld.glider_dataset(glider, **gargs)

glider_df = glider_df[glider_df.depth < 150]

tdf = glider_df.sort_values('temperature', ascending=True)

grouped = tdf.groupby('time')

for group in grouped:
    mint = np.min(group[1]['temperature'])
    maxt = np.max(group[1]['temperature'])
    mins = np.min(group[1]['salinity'])
    maxs = np.max(group[1]['salinity'])
    tempL = np.linspace(mint-1, maxt+1, 156)
    salL = np.linspace(mins-1, maxs+1, 156)
    Tg, Sg = np.meshgrid(tempL, salL)
    sigma_theta = gsw.sigma0(Sg, Tg)
    # cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), 156)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(16, 12),
        # constrained_layout=True
    )

    cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', linestyles='dashed')
    cl = plt.clabel(cs, inline=1, fontsize=20, fmt='%0.1f')

    N = 19
    # density = cm.get_cmap(cmocean.cm.dense, N)
    # density.set_under('red')
    # kwargs['cmap'] = viridis

    # min_val, max_val = .1, 1
    # n = 19
    from matplotlib import colors as c
    cmap = cmocean.cm.dense

    colors = cmap(np.linspace(.1, 1, 19))
    cmap = c.LinearSegmentedColormap.from_list('mycmap', colors)
    cmdict = cmocean.tools.get_dict(cmap, N=19)
    cmap = LinearSegmentedColormap(cmap, segmentdata=cmdict, N=N)

    # define the bins and normalize
    bounds = np.linspace(0, 150, 6)
    norm = c.BoundaryNorm(bounds, cmap.N)

    sc = ax.scatter(group[1].salinity, group[1].temperature, c=group[1].depth, cmap=cmap, s=50, norm=norm)

    cb = plt.colorbar(sc)
    ax.set_xlim([34.25, 37])
    ax.set_ylim([15, 31])
    ax.set_xlabel('Salinity', fontsize=22, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=22, fontweight='bold')
    title_str = f'ng645 T-S (Temperature and Salinity) Diagram\n{group[0].strftime("%Y-%m-%dT%H:%M:%SZ")}'
    ax.set_title(title_str, fontsize=30, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(direction='out', labelsize=20)
    # cb.ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # cb.ax.set_xticklabels(['zero', 'two', 'four', 'six'])
    cb.ax.tick_params(direction='out', labelsize=19)
    cb.set_label('Depth (m)', fontsize=22, fontweight='bold')

    # cb.set_label('Density[kg m$^{-3}$]')
    plt.tight_layout()
    plt.savefig(f'/Users/mikesmith/Documents/ng657-ts-{group[0].strftime("%Y-%m-%dT%H%M%SZ")}.png')
    plt.close()
    # plt.show()

    # Temperature profile
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 10),
        # constrained_layout=True
    )

    h = ax.scatter(
        group[1]['temperature'],
        group[1]['depth'],
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )

    ax.legend()
    ax.set_title(group[0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

    # for axs in ax.flat:
    ax.set_ylim([150, 1])
    ax.set_xlim([5, 30])
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')

    # cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', linestyles='dashed')
    # cl = plt.clabel(cs, inline=1, fontsize=20, fmt='%0.1f')

    plt.tight_layout()
    plt.savefig(f'/Users/mikesmith/Documents/ng657-temperature-{group[0].strftime("%Y-%m-%dT%H%M%SZ")}.png')
    plt.close()

    # Salinity profile
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 10),
        # constrained_layout=True
    )

    h = ax.scatter(
        group[1]['salinity'],
        group[1]['depth'],
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )

    ax.legend()
    ax.set_title(group[0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

    # for axs in ax.flat:
    ax.set_ylim([500, 1])
    ax.set_xlim([34, 36.5])
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Salinity', fontsize=16, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')

    # cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', linestyles='dashed')
    # cl = plt.clabel(cs, inline=1, fontsize=20, fmt='%0.1f')

    plt.tight_layout()
    plt.savefig(f'/Users/mikesmith/Documents/ng657-density-{group[0].strftime("%Y-%m-%dT%H%M%SZ")}.png')
    plt.close()

    # Salinity profile
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 10),
        # constrained_layout=True
    )

    h = ax.scatter(
        group[1]['density'],
        group[1]['depth'],
        label=group[0].strftime('%H:%M:%S'),
        edgecolor='black'
    )

    ax.legend()
    ax.set_title(group[0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

    # for axs in ax.flat:
    ax.set_ylim([150, 1])
    ax.set_xlim([1020, 1028])
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Density', fontsize=16, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')

    # cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', linestyles='dashed')
    # cl = plt.clabel(cs, inline=1, fontsize=20, fmt='%0.1f')

    plt.tight_layout()
    plt.savefig(f'/Users/mikesmith/Documents/ng657-density-{group[0].strftime("%Y-%m-%dT%H%M%SZ")}.png')
    plt.close()

    N = 19
    # density = cm.get_cmap(cmocean.cm.dense, N)
    # density.set_under('red')
    # kwargs['cmap'] = viridis

    # min_val, max_val = .1, 1
    # n = 19
    from matplotlib import colors as c

    colors = cmap(np.linspace(.1, 1, 19))
    cmap = c.LinearSegmentedColormap.from_list('mycmap', colors)
    cmap = cmocean.cm.dense
    cmdict = cmocean.tools.get_dict(cmap, N=19)
    cmap = LinearSegmentedColormap(cmap, segmentdata=cmdict, N=N)

    # define the bins and normalize
    bounds = np.linspace(0, 500, 11)
    norm = c.BoundaryNorm(bounds, cmap.N)

    sc = ax.scatter(group[1].salinity, group[1].temperature, c=group[1].depth, cmap=cmap, s=50, norm=norm)

    cb = plt.colorbar(sc)
    ax.set_xlim([34, 37.5])
    ax.set_ylim([7, 31])
    ax.set_xlabel('Salinity', fontsize=22, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=22, fontweight='bold')
    title_str = f'ng645 T-S (Temperature and Salinity) Diagram\n{group[0].strftime("%Y-%m-%dT%H:%M:%SZ")}'
    ax.set_title(title_str, fontsize=30, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(direction='out', labelsize=20)
    # cb.ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # cb.ax.set_xticklabels(['zero', 'two', 'four', 'six'])
    cb.ax.tick_params(direction='out', labelsize=19)
    cb.set_label('Depth (m)', fontsize=22, fontweight='bold')

    # cb.set_label('Density[kg m$^{-3}$]')
    plt.tight_layout()
    plt.savefig(f'/Users/mikesmith/Documents/ng645-{group[0].strftime("%Y-%m-%dT%H%M%SZ")}.png')
    plt.close()

# import plotly.graph_objects as go
# import numpy as np
#
# # Create figure
# fig = go.Figure()
#
# times = glider_df['time'].unique()
#
# # Add traces, one for each slider step
# for step in times:
#     tdf = glider_df[glider_df['time'] == step]
#     fig.add_trace(
#         go.Scatter(
#             x=tdf.salinity,
#             y=tdf.temperature,
#             mode='markers',
#             marker=dict(
#                 size=10,
#                 color=tdf.density,
#                 colorscale='Viridis',
#                 showscale=True
#             )
#         )
#     )
#
# # Make 10th trace visible
# # fig.data[10].visible = True
# import pandas as pd
# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "ng645 timestamp: " + pd.Timestamp(times[i]).strftime('%Y-%m-%dT%H:%M:%SZ')}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)
#
# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Dive: "},
#     pad={"t": 50},
#     steps=steps
# )]
#
# fig.update_layout(
#     sliders=sliders
# )
#
# fig.show()