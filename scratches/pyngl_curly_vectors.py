import numpy as np
import Ngl
import Nio
import xarray as xr

# ds = xr.open_dataset('/Users/mikesmith/Documents/data/codar/totals/nc/RU_SWARM_20191125T070000Z.nc')
# shape_file = '/Users/mikesmith/Documents/projects/swarm/plotAntarcticTotals/antarctica_shape/cst00_polygon_wgs84.shp'

ds = xr.open_dataset('/Users/mikesmith/Documents/github/rucool/hfradarpy/hfradar/data/totals/oi/nc/hourly/RU_MARA_20190101T000000Z.nc')
color = True # Plot color vectors
defaultWksType = "ps" # Default workstation type (available: x11, ps, eps, epsi, pdf, png)

# Domain plotting bounds - comment out to select default values from the dataset
defaultSLat = 35.0
defaultNLat = 42.25
defaultWLon = -77.0
defaultELon = -68.0

# defaultSLat = -65.1
# defaultNLat = -64.6
# defaultWLon = -64.7
# defaultELon = -63.7


# State boundaries ascii file
# slFile = "/www/home/kerfoot/public_html/shapefiles/usgs/state_boundaries/sblines.dat"
# ============================================================================

# Set up NSEW bounds: ismissing(PARAM) returns true if the variable was not
# set explicitly with a float.  If this is the case, use the default value
# from the dataset.
nLat = defaultNLat
sLat = defaultSLat
wLon = defaultWLon
eLon = defaultELon

# Retrieve the lat and lon arrays
lats = ds.lat
lons = ds.lon
u = ds.u.squeeze()/100
v = ds.v.squeeze()/100

uMax = u.max()
vMax = v.max()

# Display the min and max values of u and v
print("------------------------------------------------------------------------------")
print("Max u velocity: " + str(uMax) + " cm/s")
print("Max v velocity: " + str(vMax) + " cm/s")
print("------------------------------------------------------------------------------")

color = Ngl.Resources()
color.wkColorMap = 'viridis'

wks = Ngl.open_wks(defaultWksType, '/Users/mikesmith/Documents/pyngl_test', color)

# Vector map plot resource settings
uvres = Ngl.Resources()
uvres.nglFrame = False

uvres.vfXArray, uvres.vfYArray = np.meshgrid(lons, lats)

# Maximize the size of the plot on an 8x11 sheet of paper
# uvres.gsnMaximize = True

# Display best tick mark labeling
# uvres.pmTickMarkDisplayMode = "Always"

# Map settings
# uvres.gsnAddCyclic = False
# uvres.mpFillOn = True
uvres.mpOceanFillColor = "Transparent"
# uvres.mpLandFillColor = "gray" # Map fill color is gray
# uvres.mpDataBaseVersion = "HighRes" # Medium resolution map
# uvres.mpDataResolution = 'Finest'
uvres.mpProjection = "Mercator" # Mercator projection

# Grid settings
uvres.mpGridAndLimbOn = True # Turn on map grid
uvres.mpGridSpacingF = 5/60 # 1/2 degree intervals
uvres.mpGridLineDashPattern = 2 # dashed line
uvres.mpGridLineThicknessF = 0.25
uvres.mpLimitMode            = "LatLon"

# Domain
uvres.mpLimitMode = "LatLon"
uvres.mpMinLatF = sLat
uvres.mpMaxLatF = nLat
uvres.mpMinLonF = wLon
uvres.mpMaxLonF = eLon

# Vector styling
uvres.vcGlyphStyle = "CurlyVector"
uvres.vcMinDistanceF = 0.015 # spacing between vectors (larger value = less vectors)
uvres.vcRefLengthF = 0.06 # vector scaling factor (larger value = longer vectors). (NDC units: http://www.ncl.ucar.edu/Document/Graphics/ndc.shtml)
uvres.vcRefMagnitudeF = 0.5 # Length (in uv units) of reference vector. Any vector of this size will be rendered at the length of vcRefLength
uvres.vcLineArrowThicknessF = 0.5

# uvres.vcRefAnnoOrthogonalPosF = -0.475 # Y-dir position of reference vector
# uvres.vcRefAnnoParallelPosF = 0.95 # X-dir position of reference vector
# uvres.vcMagnitudeFormat = ".0.2f" # '%02.2f'
# uvres.vcRefAnnoString1 = "$VMG$ " + u.units#

# Determine color or black and white vectors
if color:
    print("Plotting colored vectors.")
    uvres.vcMonoLineArrowColor = False # Set to colored vectors


# define colormap
# gsn_define_colormap(wks, "sst")
# uvres.gsnSpreadColors = True
# uvres.gsnSpreadColorStart = 2
# uvres.gsnSpreadColorEnd = -4

# Specify the min/max values and color intervals
uvres.vcLevelSelectionMode = "ManualLevels"
uvres.vcMinLevelValF = 0 # Min value for first color (cm/s)
uvres.vcMaxLevelValF = 0.20 # Max value for last color (cm/s)
uvres.vcLevelSpacingF = 0.02

# LabelBar/Colorbar resources (lb)
uvres.lbLabelStride = 2 # Make sure colorbar labels don't overlap
uvres.lbBoxLinesOn = False # No lines on colorbar intervals
uvres.lbOrientation = "Vertical" # Place the label bar on the right side of the plot
uvres.lbRasterFillOn = True # faster drawing of label bar
uvres.lbLabelFontHeightF = 0.012 # Colorbar values font size
uvres.lbTitleString = "Current Velocity (m/s)"# Colorbar title
uvres.lbTitleFontHeightF = 0.012 # Colorbar title font size
uvres.lbTitlePosition = "Right" # Colorbar title to the right of the colorbar
uvres.lbTitleDirection = "Across" # Title direction
uvres.lbTitleAngleF = 90.0 # Rotate title 90 degrees
uvres.lbLeftMarginF = -0.1 #Label bar horizontal positioning
uvres.lbRightMarginF = 0.2 #Label bar horizontal positioning


uvres.vcRefAnnoOn = False # Don't draw the ref vector
uvres.vcRefAnnoArrowUseVecColor = False # Don't use color for reference vector
uvres.vcRefAnnoArrowLineColor = "black" # Color the reference vector black
uvres.vcExplicitLabelBarLabelsOn = True

uvres.lbAutoManage = False
uvres.lbLabelJust = "CenterLeft"

# Plot annotation resources
uvres.tiMainString = ""
# uvres.gsnLeftString = ""
# uvres.gsnRightString = ""
# uvres.gsnCenterStringFontHeightF = 0.012
# uvres.gsnCenterString = "MARACOOS Hourly Surface Current Field: " + tsLabel

# Tickmarks
# uvres.tmXBLabelFontHeightF = 0.01 # 1/2 default height
# uvres.tmYLLabelFontHeightF = 0.01 # 1/2 default height

# Plot the vectors
vPlot = Ngl.vector_map(wks, u, v, uvres)

# # Open shapefile
# shpf = Nio.open_file(shape_file)
#
# lon = np.ravel(shpf.variables['x'][:])
# lat = np.ravel(shpf.variables['y'][:])
# segments = shpf.variables['segments'][:,0]

plres = Ngl.Resources()
plres.gsEdgesOn = True
plres.gsEdgeThicknessF = 0.2
plres.gsEdgeColor = 'black'
plres.gsFillColor = 'tan'
# plres.gsSegments = segments
# id = Ngl.add_polygon(wks, vPlot,  lon, lat, plres)

# Read in state line boundaries from ascii file
# Number of rows in the file
# rows = numAsciiRow(slFile)#

# Read the data into a 2-column array
# gps = asciiread(slFile, (/rows, 2/), "float")#
# gps._FillValue = -999.0#
# boundaries = gsn_add_polyline(wks, vPlot, gps(:,0), gps(:,1), False)#
Ngl.draw(vPlot)
Ngl.frame(wks)
Ngl.end()
print("Done.")