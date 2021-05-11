#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/11/2021
Last modified: Lori Garzio on 5/11/2021
Wrapper script for model-glider comparisons: glider track, surface maps, and transects
"""
import datetime as dt
import scripts.surface_maps as surface_maps

glider_deployments = ['ru30-20210503T1929']
sdir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
bathy = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
# bathy = False
land_color = 'none'
model_t0 = dt.datetime(2021, 5, 10, 0, 0)  # False
model_t1 = False
glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
glider_t1 = dt.datetime(2021, 5, 10, 0, 0)
line_transect = False  # True or False  # get a straight line transect, rather than a transect along the glider track
curr_location = False  # indicate the current glider location with a triangle marker

# make a map of the glider track
surface_maps.glider_track.main(glider_deployments, sdir, bathy, land_color, glider_t0, glider_t1, line_transect,
                               curr_location)

# create surface maps of GOFS and RTOFS temperature and salinity overlaid with the glider track and optional transect
surface_maps.gofs_glider_surface_maps.main(glider_deployments, sdir, bathy, model_t0, model_t1, glider_t0, glider_t1,
                                           line_transect, curr_location)
surface_maps.rtofs_glider_surface_maps.main(glider_deployments, sdir, bathy, model_t0, model_t1, glider_t0, glider_t1,
                                            line_transect, curr_location)