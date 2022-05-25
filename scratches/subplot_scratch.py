# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

# %%
        
grid = """
RG
LL
"""

axd = plt.figure(constrained_layout=True, figsize=(16,10)).subplot_mosaic(grid,
                                subplot_kw={'projection': ccrs.Mercator()},
                                gridspec_kw={
                                    # set the height ratios between the rows
                                    "height_ratios": [1, .5],
                                    # set the width ratios between the columns
                                    # "width_ratios": [1],
                                    },
                                )

identify_axes(axd)

plt.show()

# %%
fig, axs = plt.subplots(3, 3, constrained_layout=True)
for ax in axs.flat:
    pcm = ax.pcolormesh(np.random.random((20, 20)))

fig.colorbar(pcm, ax=axs[0, :2], shrink=0.6, location='bottom')
fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
fig.colorbar(pcm, ax=axs[1:, :], location='right', shrink=0.6)
fig.colorbar(pcm, ax=[axs[2, 1]], location='left')
plt.show()

# %%
import matplotlib.pyplot as plt

fig, ax_dict = plt.subplot_mosaic([['left', 'right'], ['left', 'right']],
                                  empty_sentinel="BLANK")
ax_dict['left'].plot([1, 2, 3], label="test1")
ax_dict['left'].plot([3, 2, 1], label="test2")

# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
ax_dict['left'].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=2, mode="expand", borderaxespad=0.)

ax_dict['right'].plot([1, 2, 3], label="test1")
ax_dict['right'].plot([3, 2, 1], label="test2")

# Place a legend to the right of this smaller subplot.
ax_dict['right'].legend(bbox_to_anchor=(1.02, 0, 1, 1),
                         loc='upper left')

plt.show()
# %%
