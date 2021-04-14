#!/usr/bin/env python
import numpy as np
import Nio
import Ngl

import xarray as xr
import datetime as dt
from glob import glob
import os
import pandas as pd

# # Directories for my computer
# main_dir = '/Users/mikesmith/Documents/projects/swarm/data/totals/'
# save_dir = '/Users/mikesmith/Documents/projects/swarm/data/images/'
# shape_file = '/Users/mikesmith/Documents/projects/swarm/plotAntarcticTotals/antarctica_shape/cst00_polygon_wgs84.shp'

# Directories for swarm computer
main_dir = '/Users/codar/Desktop/data/totals/'
save_dir = '/Users/codar/Desktop/data/images/totals/'
shape_file = '/Users/codar/Desktop/scripts/toolboxes/totals_toolbox/coast_files/antarctica_shape/cst00_polygon_wgs84.shp'

# Previous hours to plot
hours = 24

# Open shapefile
shpf = Nio.open_file(shape_file)

lon = np.ravel(shpf.variables['x'][:])
lat = np.ravel(shpf.variables['y'][:])
segments = shpf.variables['segments'][:, 0]

color = True  # Plot color vectors
defaultWksType = "png"  # Default workstation type (available: x11, ps, eps, epsi, pdf, png)
colormap = 'viridis'
height = 6000
width = 6000
velocity_min = 0  # m/s
velocity_max = 0.3  # m/s

# Set up NSEW bounds:
nLat = -64 - 2/3
sLat = -65 - 1/12
wLon = -64 - 2/3
eLon = -63 - 2/3

# Site Longitudes and Latitudes
joub = [-64.3604167, -64.7871667]
wauw = [-64.0446333, -64.9183167]
palm = [-64.049417, -64.775417]

north_cross_canyon = [[-64 - 14.222/60, -64 - 1.307/60], [-64 - 48.354/60, -64 - 52.163/60]]
along_canyon = [[-64 - 5.507/60, -64 - 19.018/60], [-64 - 49.777/60, -64 - 57.328/60]]
south_cross_canyon = [[-64 - 27.427/60, -64 - 11.480/60], [-64 - 54.947/60, -64 - 59.409/60]]

mooring1 = [-64 - 12.714/60, -64 - 48.744/60]
mooring2 = [-64 - 7.038/60, -64 - 50.616/60]
mooring3 = [-64 - 2.256/60, -64 - 51.840/60]


def main(fname, save_dir):
    # Open netCDF file
    with xr.open_dataset(fname) as ds:

        # Setup resource for colormap.
        color = Ngl.Resources()
        color.wkColorMap = colormap
        color.wkWidth = width
        color.wkHeight = height

        if 'oi' in fname:
            ds = ds.where(ds.u_err <= 0.6)
            ds = ds.where(ds.v_err <= 0.6)

            vcMinDistanceF = 0.015  # spacing between vectors (larger value = less vectors)

            if 'ideal' in fname:
                save_path = os.path.join(save_dir, 'oi', 'ideal')
                save_name = os.path.join(save_path, os.path.basename(fname))
            elif 'measured' in fname:
                save_path = os.path.join(save_dir, 'oi', 'measured')
                save_name = os.path.join(save_path, os.path.basename(fname))
            elif 'bestchoice' in fname:
                save_path = os.path.join(save_dir, 'oi', 'best')
                save_name = os.path.join(save_path, os.path.basename(fname))
        elif 'lsq' in fname:
            vcMinDistanceF = 0.0125  # spacing between vectors (larger value = less vectors)
            if 'ideal' in fname:
                save_path = os.path.join(save_dir, 'lsq', 'ideal')
                save_name = os.path.join(save_path, os.path.basename(fname))
            elif 'measured' in fname:
                save_path = os.path.join(save_dir, 'lsq', 'measured')
                save_name = os.path.join(save_path, os.path.basename(fname))
            elif 'bestchoice' in fname:
                save_path = os.path.join(save_dir, 'lsq', 'best')
                save_name = os.path.join(save_path, os.path.basename(fname))
        else:
            return

        os.makedirs(save_path, exist_ok=True)

        # Retrieve the lat and lon arrays
        lats = ds.lat
        lons = ds.lon

        # Get the u/v variables
        u = ds.u.squeeze()/100
        v = ds.v.squeeze()/100

        wks = Ngl.open_wks(defaultWksType, save_name, color)

        # Vector map plot resource settings
        uvres = Ngl.Resources()
        uvres.nglFrame = False

        uvres.vfXArray, uvres.vfYArray = np.meshgrid(lons, lats)

        # Map settings
        # uvres.mpOceanFillColor = "Transparent"
        # uvres.mpLandFillColor = "gray" # Map fill color is gray
        # uvres.mpDataBaseVersion = "HighRes" # Medium resolution map
        # uvres.mpDataResolution = 'Finest'
        uvres.mpProjection = "Mercator" # Mercator projection
        uvres.mpGeophysicalLineThicknessF = 0.
        uvres.pmTitleDisplayMode = "Always"
        uvres.mpOutlineOn = False  # Turn off the default map outlines for pyngl

        # Grid settings
        uvres.mpGridAndLimbOn = True  # Turn on map grid
        uvres.mpGridSpacingF = 5/60  # 1/2 degree intervals
        uvres.mpGridLineDashPattern = 2  # dashed line
        uvres.mpGridLineThicknessF = 0.75

        # Domain
        uvres.mpLimitMode = "LatLon"
        uvres.mpMinLatF = sLat
        uvres.mpMaxLatF = nLat
        uvres.mpMinLonF = wLon
        uvres.mpMaxLonF = eLon

        # Vector styling
        uvres.vcGlyphStyle = "CurlyVector"
        uvres.vcRefMagnitudeF = 0.3  # Magnitude(m/s) of reference vector.  Any vector of this speed will be rendered at the length of vcRefLength
        uvres.vcRefLengthF = 0.06  # vector scaling factor (larger value = longer vectors).
        uvres.vcMinDistanceF = vcMinDistanceF  # spacing between vectors (larger value = less vectors)
        uvres.vcLineArrowThicknessF = 5.0
        uvres.vcMonoLineArrowColor = False  # Set to colored vectors

        # Specify the min/max values and color intervals
        uvres.vcLevelSelectionMode = "ManualLevels"
        uvres.vcMinLevelValF = velocity_min  # Min value for first color (cm/s)
        uvres.vcMaxLevelValF = velocity_max  # Max value for last color (cm/s)
        uvres.vcLevelSpacingF = 0.02

        # LabelBar/Colorbar resources (lb)
        uvres.lbLabelStride = 2  # Make sure colorbar labels don't overlap
        uvres.lbBoxLinesOn = False  # No lines on colorbar intervals
        uvres.lbOrientation = "Vertical"  # Place the label bar on the right side of the plot
        uvres.lbRasterFillOn = True  # faster drawing of label bar
        uvres.lbLabelFontHeightF = 0.012  # Colorbar values font size
        uvres.lbTitleString = "Current Velocity (m/s)"  # Colorbar title
        uvres.lbTitleFontHeightF = 0.012  # Colorbar title font size
        uvres.lbTitlePosition = "Right"  # Colorbar title to the right of the colorbar
        uvres.lbTitleDirection = "Across"  # Title direction
        uvres.lbTitleAngleF = 90.0  # Rotate title 90 degrees
        uvres.lbLeftMarginF = -0.1  #Label bar horizontal positioning
        uvres.lbRightMarginF = 0.2  #Label bar horizontal positioning

        uvres.vcRefAnnoOn = False  # Don't draw the ref vector
        uvres.vcExplicitLabelBarLabelsOn = True

        uvres.lbAutoManage = False
        uvres.lbLabelJust = "CenterLeft"

        # Plot annotation resources
        uvres.tiMainString = "Palmer Deep, Antarctica Totals ~C~   {}".format(pd.to_datetime(str(ds.time.data[0])).strftime('%Y-%m-%d %H:%M:%S GMT'))
        uvres.tiMainFontHeightF = 0.015

        # Plot the vectors
        vPlot = Ngl.vector_map(wks, u, v, uvres)

        # Plot the antarctica coastline from shape file
        plres = Ngl.Resources()
        plres.gsEdgesOn = True
        plres.gsEdgeThicknessF = 0.2
        plres.gsEdgeColor = 'black'
        plres.gsFillColor = 'tan'
        plres.gsSegments = segments
        Ngl.add_polygon(wks, vPlot,  lon, lat, plres)

        # Plot the hf radar sites
        hfres = Ngl.Resources()
        hfres.gsMarkerIndex = 16  # dots
        hfres.gsMarkerColor = "Red"
        hfres.gsMarkerSizeF = 0.01  # twice normal size

        textres = Ngl.Resources()
        textres.txFontHeightF = .015

        # Add site marker and label to map for JOUB, WAUW, and PALM
        Ngl.add_polymarker(wks, vPlot, joub[0], joub[1], hfres)
        Ngl.add_polymarker(wks, vPlot, wauw[0], wauw[1], hfres)
        Ngl.add_polymarker(wks, vPlot, palm[0], palm[1], hfres)

        Ngl.add_text(wks, vPlot, 'JOUB', joub[0] - 3.1/60, joub[1] + .5/60, textres)
        Ngl.add_text(wks, vPlot, 'WAUW', wauw[0] + 4/60, wauw[1], textres)
        Ngl.add_text(wks, vPlot, 'PALM', palm[0] + 3/60, palm[1] + .5/60, textres)

        # Plot the mooring sites
        moores = Ngl.Resources()
        moores.gsMarkerIndex = 16  # dots
        moores.gsMarkerColor = "Black"
        moores.gsMarkerSizeF = 0.01  # twice normal size

        Ngl.add_polymarker(wks, vPlot, mooring1[0], mooring1[1], moores)
        Ngl.add_polymarker(wks, vPlot, mooring2[0], mooring2[1], moores)
        Ngl.add_polymarker(wks, vPlot, mooring3[0], mooring3[1], moores)

        # Ngl.add_text(wks, vPlot, 'AMLR1', mooring1[0], mooring1[1], textres)
        # Ngl.add_text(wks, vPlot, 'AMLR2', mooring2[0], mooring2[1], textres)
        # Ngl.add_text(wks, vPlot, 'LR', mooring3[0], mooring3[1], textres)

        # Plot the canyon lines
        lineres = Ngl.Resources()
        lineres.gsLineColor = "Red"
        lineres.gsLineThicknessF = 6.0  # thrice thickness

        Ngl.add_polyline(wks, vPlot, north_cross_canyon[0], north_cross_canyon[1], lineres)
        Ngl.add_polyline(wks, vPlot, along_canyon[0], along_canyon[1], lineres)
        Ngl.add_polyline(wks, vPlot, south_cross_canyon[0], south_cross_canyon[1], lineres)


        Ngl.draw(vPlot)
        Ngl.frame(wks)
        Ngl.destroy(wks)



end_time = dt.datetime.utcnow()
start_time = end_time - dt.timedelta(hours=hours)

file_list = glob(os.path.join(main_dir, '**', dt.datetime.utcnow().strftime('%Y_%m'), '**', '*.nc'), recursive=True)
df = pd.DataFrame(sorted(file_list), columns=['file'])
df['time'] = df['file'].str.extract(r'(\d{8}T\d{6})')
df['time'] = df['time'].apply(lambda x: pd.datetime.strptime(x, '%Y%m%dT%H%M%S'))
df = df.set_index(['time']).sort_index()
df = df[start_time: end_time]

for row in df.itertuples():
    main(row.file, save_dir)

Ngl.end()
