from collections import OrderedDict

    
def region_config(regions=None, model=None):
    """
    return extent and other variable limits of certain regions 
    :param model: rtofs or gofs
    :param regions: list containing regions you want to plot
    :return: dictionary containing limits
    """

    model = model or 'rtofs'
    regions = regions or ['gom']
    # ['gom', 'sab', 'mab', 'caribbean', 'windward', 'nola',  'usvi', 'north_atlantic', 'west_indies', 'yucatan', 'yucatan_caribbean_expanded']]

    # Create new dictionary for selected model. Needs to be done because the variable names are different in each model
    # initialize empty dictionary for limits
    limits = OrderedDict()

    # Specify common variable and region limits for both gofs and rtofs
    # To add different depths for each variable, append to the specific variable list the following format:
    # dict(depth=n, limits=[min, max, stride])  

    key = "ng645"
    if key in regions:
        # Yucatan/Caribbean
        name = "ng645-20211010T0000"
        extent = [-92, -76, 22, 34]
        sea_water_temperature = [dict(depth=0, limits=[24.25, 28, .25]), dict(depth=200, limits=[14, 23, .5])]
        salinity = [dict(depth=0, limits=[35.7, 36.2, .05])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=2, gofs=3),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )   
        figure = dict(
            legend = dict(columns=5),
            # figsize
            )  
  
    key = "yucatan"
    if key in regions:
        # Yucatan Limits
        name = "Yucatan"
        extent = [-90, -78, 18, 26]
        sea_water_temperature = [
            dict(depth=0, limits=[24, 29.5, .5]),
            dict(depth=200, limits=[14, 23, .5])
            ]
        salinity = [
            dict(depth=0, limits=[35.8, 36.6, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=5, gofs=6),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            # figsize
            )
        
    key = "north_atlantic"
    if key in regions:
        # North Alantic Limits
        name = "North Atlantic"
        extent = [-80, 0, 0, 50]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[33, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=7, gofs=8),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=5),
            # figsize
            )

    key = "nola"
    if key in regions:
        # GOM - New Orleans Area
        name = 'Gulf of Mexico/New Orleans'
        extent = [-94, -84, 25.5, 31]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[33, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=7, gofs=8),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=5),
            # figsize
            )

    key = "prvi"
    if key in regions:
        # USVI Limits
        name = "Puerto Rico/Virgin Islands"
        extent = [-72, -64, 15, 20]
        sea_water_temperature = [dict(depth=0, limits=[26, 27.5, .1])]
        salinity = [
            dict(depth=0, limits=[35.5, 36.2, .1]),
            dict(depth=100, limits=[35.5, 36.2, .1]),
            dict(depth=150, limits=[35.5, 36.2, .1]),
            dict(depth=200, limits=[35.5, 36.2, .1])
            ]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=1, gofs=1),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            # figsize
            )

    key = "usvi"
    if key in regions:
        # USVI Limits
        name = "Virgin Islands"
        extent = [-66.26, -62.61, 16.5, 19]
        sea_water_temperature = [
            dict(depth=0, limits=[26.6, 28.1, .1])
            ]
        salinity = [
            dict(depth=0, limits=[35.5, 36.3, .05])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=2, gofs=3),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            # figsize
            )

    key = "west_indies"
    if key in regions:
        # West Indies
        name = "West Indies"
        extent = [-67, -61, 14, 19]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 29.5, .25])]
        salinity = [dict(depth=0, limits=[34, 36.5, .25 ])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=2, gofs=3),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=40,
                # headwidth=5,
                # headlength=5,
                # headaxislength=4.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            # figsize
            )

    key = "gom"
    if key in regions:
        # Gulf of Mexico Limits
        name = "Gulf of Mexico"
        extent = [-99, -79, 18, 31]
        sea_water_temperature = [
            dict(depth=0, limits=[26, 30, .25]), 
            dict(depth=200, limits=[12, 24, .5])
            ]
        salinity = [
            dict(depth=0, limits=[34, 36.6, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=7, gofs=8),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            # figsize
            )

    key = "sab"
    if key in regions:
        # South Atlantic Bight Limits
        name = "South Atlantic Bight"
        extent = [-82, -64, 25, 36]
        sea_water_temperature = [
            dict(depth=0, limits=[22, 29, .5])
            ]
        salinity = [
            dict(depth=0, limits=[36, 36.9, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=7, gofs=8),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=5),
            # figsize
            )

    key = "mab"
    if key in regions:
        # Mid Atlantic Bight Limits
        name = 'Mid Atlantic Bight'
        extent = [-77, -67, 35, 43]
        sea_water_temperature = [
            dict(depth=0, limits=[10, 25.5, .5]),
            dict(depth=100, limits=[12.5, 22, .5])
            ]
        salinity = [
            dict(depth=0, limits=[31, 37, .25])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=5, gofs=6),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=4,
                # headlength=4,
                # headaxislength=3.5
                )
            )
        figure = dict(
            legend = dict(columns=5),
            figsize = (12, 9)
            )

    key = "caribbean"
    if key in regions:
        # Caribbean Limits
        name = "Caribbean Sea"
        extent = [-89, -58, 7, 23]
        sea_water_temperature = [
            dict(depth=0, limits=[26, 29.25, .25])
            ]
        salinity = [
            dict(depth=0, limits=[34.6, 36.8, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            coarsen=dict(rtofs=11, gofs=12),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=4,
                # headlength=4,
                # headaxislength=3.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            figsize = (16,7)
            )

    key = "windward"
    if key in regions:
        # Windward Islands imits
        name = 'Windward Islands'
        extent = [-68.2, -56.4, 9.25, 19.75]
        sea_water_temperature = [
            dict(depth=0, limits=[25.25, 28.25, .25]),
            ]
        salinity = [
            dict(depth=0, limits=[34.3, 36.6, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            # coarsen=dict(rtofs=None, gofs=None),
            coarsen=dict(rtofs=5, gofs=6),
            kwargs=dict(
                ptype="streamplot",
                color="dimgray"
                # scale=60,
                # headwidth=3,
                # headlength=3,
                # headaxislength=2.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            figsize = (13, 9)
            )

    # Create subdirectory for data variables
    vars = {}
    vars.update(salinity=salinity)
    vars.update(temperature=sea_water_temperature)
    
    # Update Dictionary with limits defined above
    limits.update(name=name)
    limits.update(extent=extent)
    limits.update(currents=currents)
    limits.update(figure=figure)
    limits.update(sea_surface_height=sea_surface_height)
    limits.update(variables=vars)
    return limits

