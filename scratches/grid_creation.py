import shapely.geometry
import pyproj

# Set up transformers, EPSG:3857 is metric, same as EPSG:900913
to_proxy_transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
to_original_transformer = pyproj.Transformer.from_crs('epsg:3857', 'epsg:4326')# Create corners of rectangle to be transformed to a grid

southwest = (-89, 19+25/60)
northeast = (-85+30/60, 23+30/60)
km = 6
sname = 'yucatan_6km.csv'

# 6km grid
sw = shapely.geometry.Point(southwest)
ne = shapely.geometry.Point(northeast)
stepsize = km*1000 # grid step size in meters

# Project corners to target projection
transformed_sw = to_proxy_transformer.transform(sw.x, sw.y) # Transform NW point to 3857
transformed_ne = to_proxy_transformer.transform(ne.x, ne.y) # .. same for SE

# Iterate over 2D area
gridpoints = []
x = transformed_sw[0]
while x < transformed_ne[0]:
    y = transformed_sw[1]
    while y < transformed_ne[1]:
        p = shapely.geometry.Point(to_original_transformer.transform(x, y))
        gridpoints.append(p)
        y += stepsize
    x += stepsize


with open(sname, 'w') as of:
    of.write('lon,lat\n')
    for p in gridpoints:
        of.write('{:f},{:f}\n'.format(p.x, p.y))