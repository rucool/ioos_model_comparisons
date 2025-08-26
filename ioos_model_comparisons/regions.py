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

    # Defaults
    eez = False
    ocean_heat_content = False
    salinity_max = False
    currents = False

    key = "mastr"
    if key in regions:
        # Yucatan Limits
        name = "Yucatan/Mastr"
        folder = "mastr"
        extent = [-89, -84, 17.75, 22]
        sea_water_temperature = [
            # dict(depth=0, limits=[24.5, 27.75, .25]),
            # dict(depth=0, limits=[27.5, 31, .25]),
            # dict(depth=150, limits=[18, 25.5, .5]),
            # dict(depth=200, limits=[14, 23, .5])
            ]
        salinity = [
            # dict(depth=0, limits=[35.8, 36.6, .1]),
            dict(depth=150, limits=[35.6, 37.1, .1]),
            # dict(depth=200, limits=[36, 36.8, .05]), 
            ]
        sea_surface_height = [
            # dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            # figsize=(14, 6.5),
            # limits=[36, 37, .1]
            )
        ocean_heat_content = dict(
            # limits= [0, 140, 10]
            )
        currents = dict(
            bool=True,
            # depths = [0, 150, 300, 600, 900],
            depths = [0],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=5, espc=6),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        eez = True
        figure = dict(
            legend = dict(columns=8),
            figsize=(10,7)
            )

    key = "yucatan"
    if key in regions:
        # Yucatan Limits
        name = "Yucatan"
        folder = "yucatan"
        extent = [-90, -78, 17, 29]
        sea_water_temperature = [
            dict(depth=0, limits=[24.5, 27.75, .25]),
            # dict(depth=0, limits=[27.5, 31, .25]),
            dict(depth=150, limits=[18, 25.5, .5]),
            # dict(depth=200, limits=[14, 23, .5])
            ]
        salinity = [
            dict(depth=0, limits=[35.8, 36.6, .1]),
            dict(depth=150, limits=[36, 37, .05]),
            # dict(depth=200, limits=[36, 36.8, .05]), 
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            # figsize=(14, 6.5),
            limits=[36, 37, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 150],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=5, espc=6),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        eez = True
        figure = dict(
            legend = dict(columns=8),
            figsize=(10,7)
            )

    key = "leeward"
    if key in regions:
        # USVI Limits
        name = "Caribbean: Leeward Islands"
        folder = "caribbean-leeward"
        extent = [-68.5, -61, 15, 19]
        sea_water_temperature = [
            dict(depth=0, limits=[28, 29.3, .1]),
            dict(depth=150, limits=[20, 25, .5]),
            dict(depth=200, limits=[15, 23, .5])
            ]
        salinity = [
            dict(depth=0, limits=[34, 35.6, .1]),
            dict(depth=150, limits=[36, 37.3, .1]),
            dict(depth=200, limits=[36.1, 37.3, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 5.5),
            limits=[36, 37.5, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 150, 200],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=1, espc=1),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        eez = True
        figure = dict(
            legend = dict(columns=5),
            figsize = (11.25,5.5),
            )

    key = "loop_current"
    if key in regions:
        # Gulf of Mexico Limits
        name = "Loop Current Eddy"
        folder = "loop_current_eddy"
        extent = [-93, -83, 24, 30]
        sea_water_temperature = [
            dict(depth=0, limits=[27, 32, .5]),
            dict(depth=150, limits=[14, 26, .5]),
            dict(depth=200, limits=[12, 23, .5])
            ]
        salinity = [
            dict(depth=0, limits=[34, 36.7, .1]), 
            dict(depth=150, limits=[35.9, 36.7, .1]),
            dict(depth=200, limits=[35.7, 36.8, .1]),
            ]
        salinity_max = dict(
            figsize=(14, 6.5),
            limits=[36, 37, .1]
        )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 30],
            limits = [0, 2.0, .1],
            coarsen=dict(rtofs=7, espc=8),
            contours=[0.771667], # m/s
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        
        figure = dict(
            legend = dict(columns=7),
            figsize = (13, 7.5)
            )

    key = "gom"
    if key in regions:
        # Gulf of Mexico Limits
        name = "Gulf of Mexico"
        folder = "gulf_of_mexico"
        extent = [-99, -78, 18, 33]
        # extent = [-100, -40, 5, 35]
        # extent = [-91, -81, 20, 28]
        # extent = [-91, -79, 18, 32]

        sea_water_temperature = [
            dict(depth=0, limits=[25, 31.5, .5]),
            dict(depth=150, limits=[14, 26, .5]),
            dict(depth=200, limits=[12, 23, .5])
            ]
        salinity = [
            dict(depth=0, limits=[32.0, 36.6, .1]), 
            dict(depth=150, limits=[35.6, 37.1, .1]),
            dict(depth=200, limits=[35.7, 37.1, .1]),
            ]
        sea_surface_height = [
            # dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 6.5),
            limits=[36, 37, .1]
        )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0],
            limits = [0, 1.5, .1],
            # depths = [0,],
            # limits = [0, 1.5, .1],
            coarsen=dict(rtofs=7, espc=8, hafs=7, cmems=8),
            kwargs=dict(
                ptype="streamplot",
                color="black",
                density=3
                )
            )
        eez=False
        figure = dict(
            legend = dict(columns=10),
            figsize = (12.125, 7.5)
            )
        
    key = "grase"
    if key in regions:
        # Gulf of Mexico Limits
        name = "Gulf of Mexico"
        folder = "grase"
        # extent = [-99, -78, 18, 33]
        # extent = [-100, -40, 5, 35]
        extent = [-91, -80, 17.75, 29]

        sea_water_temperature = [
            dict(depth=0, limits=[25, 31.5, .5]),
            dict(depth=150, limits=[14, 26, .5]),
            dict(depth=200, limits=[12, 23, .5])
            ]
        salinity = [
            dict(depth=0, limits=[32.0, 36.6, .1]), 
            dict(depth=150, limits=[35.6, 37.1, .1]),
            dict(depth=200, limits=[35.7, 37.1, .1]),
            ]
        sea_surface_height = [
            # dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 6.5),
            limits=[36, 37, .1]
        )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 80, 1500],
            # depths = [0],
            limits = [0, 1.5, .1],
            # depths = [0,],
            # limits = [0, 1.5, .1],
            coarsen=dict(rtofs=7, espc=8, hafs=7, cmems=8),
            kwargs=dict(
                ptype="streamplot",
                color="black",
                density=3
                )
            )
        eez=False
        figure = dict(
            legend = dict(columns=10),
            # figsize = (12.125, 7.5)
            figsize = (8,8)
            )
        
        

    key = "sab"
    if key in regions:
        # South Atlantic Bight Limits
        name = "South Atlantic Bight"
        folder = "south_atlantic_bight"
        extent = [-82, -64, 25, 36]
        sea_water_temperature = [
            dict(depth=0, limits=[23, 30.5, .5]),
            dict(depth=150, limits=[15, 22.5, .5]),
            # dict(depth=200, limits=[15, 21, .5])
            ]
        salinity = [
            dict(depth=0, limits=[36, 36.9, .1]),
            dict(depth=150, limits=[36, 37.05, .05]),
            # dict(depth=200, limits=[35.8, 36.8, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 6.5),
            limits=[36, 37, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 150, 200],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=7, espc=8),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        figure = dict(
            legend = dict(columns=7),
            figsize = (12, 7)
            )

    key = "mab"
    if key in regions:
        # Mid Atlantic Bight Limits
        name = 'Mid Atlantic Bight'
        folder = "mid_atlantic_bight"
        extent = [-77, -63, 35, 44]
        sea_water_temperature = [
            dict(depth=0, limits=[15, 29, 1]),
            dict(depth=30, limits=[7, 25, 1]),
            dict(depth=100, limits=[12, 23, 1]),
            dict(depth=150, limits=[11, 22, 1]),
            dict(depth=200, limits=[9, 21, 1])
            ]
        salinity = [
            dict(depth=0, limits=[31, 36.5, .25]),
            dict(depth=30, limits=[33, 36.75, .25]),
            dict(depth=100, limits=[34.7, 36.7, .1]),
            dict(depth=150, limits=[35.3, 36.6, .1]),
            dict(depth=200, limits=[35.2, 36.6, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1]),
            ]
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        salinity_max = dict(
            figsize=(14, 8.5),
            limits=[34.4, 37, .2]
            )
        currents = dict(
            bool=True,
            depths = [0, 100, 150, 200],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=5, espc=6),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        figure = dict(
            legend = dict(columns=5),
            figsize = (10.5, 7.5)
            )
    
    key = "west_florida_shelf"
    if key in regions:
        name = "West Florida Shelf"
        folder = "west_florida_shelf"
        # extent = [-83.2, -82.4, 27, 27+30/60]
        extent = [-87.5, -80, 22.5, 30.5]
        sea_water_temperature = [
            dict(depth=0, limits=[29, 31.3, .1]),
            dict(depth=100, limits=[18, 27.5, .5]),
            dict(depth=200, limits=[13, 24, .5]),
            ]
        salinity = [
            dict(depth=0, limits=[34.5, 36.5 , .1]),
            dict(depth=100, limits=[36.1, 36.5, .025]),
            dict(depth=200, limits=[35.6, 37, .1]),
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 8),
            limits=[36, 37, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=11, espc=12),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        figure = dict(
            legend = dict(columns=5),
            figsize = (10,7.75)
            )
    
    key = "caribbean"
    if key in regions:
        # Caribbean Limits
        name = "Caribbean"
        folder = "caribbean"
        extent = [-89, -58, 7, 23.5]
        sea_water_temperature = [
            dict(depth=0, limits=[28, 31.75, .25]),
            dict(depth=150, limits=[17, 25, .5]),
            dict(depth=200, limits=[14, 22.5, .5])
            ]
        salinity = [
            dict(depth=0, limits=[34.6, 36.8, .1]),
            dict(depth=150, limits=[35.7, 37.4, .1]),
            dict(depth=200, limits=[35.6, 37.1, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 5),
            limits= [36, 37.5, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 150, 200],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=11, espc=12),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        eez = False
        figure = dict(
            legend = dict(columns=8),
            figsize = (12.5,6.5)
            )

    key = "windward"
    if key in regions:
        # Windward Islands imits
        name = 'Caribbean: Windward Islands'
        folder = "caribbean-windward"
        extent = [-68.2, -56.4, 9.25, 19.75]
        sea_water_temperature = [
            dict(depth=0, limits=[27, 29.3, .1]),
            dict(depth=150, limits=[16.5, 24.5, .5]),
            # dict(depth=200, limits=[14, 22, .5])
            ]
        salinity = [
            dict(depth=0, limits=[33, 36.1, .1]),
            dict(depth=150, limits=[35.5, 37.4, .1]),
            #dict(depth=200, limits=[35.3, 37.1, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            figsize=(14, 8),
            limits=[36, 37.5, .1]
        )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            # coarsen=dict(rtofs=None, gofs=None),
            depths = [0, 150],
            limits = [0, 1.3, .1],
            coarsen=dict(rtofs=5, espc=6),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        eez = True
        figure = dict(
            legend = dict(columns=10),
            figsize = (11.5, 8)
            )
        
    key = "amazon"
    if key in regions:
        # Amazon Plume limits
        name = 'Amazon Plume'
        folder = "amazon_plume"
        extent = [-70, -43, 0, 20]
        sea_water_temperature = [
            dict(depth=0),
            dict(depth=150),
            # dict(depth=200)
            ]
        salinity = [
            dict(depth=0, limits=[33.8, 37.1, .1]),
            dict(depth=150),
            # dict(depth=200)
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            # coarsen=dict(rtofs=None, gofs=None),
            depths = [0, 160, 200],
            coarsen=dict(rtofs=5, espc=6),
            kwargs=dict(
                ptype="streamplot",
                color="black"
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

    key = "hurricane"
    if key in regions:
        # Caribbean Limits
        name = "Hurricane Alley"
        folder = "hurricane_alley"
        extent = [-89, -12, 0, 20]
        sea_water_temperature = [
            dict(depth=0, limits=[20 , 29.25, .25]),
            dict(depth=150, limits=[17, 24.5, .5]),
            # dict(depth=200, limits=[14, 22, .5])
            ]
        salinity = [
            dict(depth=0, limits=[34.6, 37.2, .1]),
            dict(depth=150, limits=[35.7, 36.4, .05]),
            # dict(depth=200, limits=[35.5, 37, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        currents = dict(
            bool=True,
            depths = [0, 150, 200],
            coarsen=dict(rtofs=11, espc=12),
            kwargs=dict(
                ptype="streamplot",
                color="black", 
                density=4,
                linewidth=.5
                )
            )
        figure = dict(
            legend = dict(columns=9),
            )

    key = "tropical_western_atlantic"
    if key in regions:
        # Caribbean Limits
        name = "Tropical Western Atlantic"
        folder = "tropical_western_atlantic"
        extent = [-70, -40.7, 0, 25]
        sea_water_temperature = [
            dict(depth=0, limits=[25, 29.5, .5]),
            dict(depth=150, limits=[11, 23, 1]),
            dict(depth=200, limits=[10, 22, 1])
            ]
        salinity = [
            dict(depth=0, limits=[32, 37.6, .1]),
            dict(depth=150, limits=[35.2, 37.6, .1]),
            dict(depth=200, limits=[35.2, 37.3, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            limits= [36, 37.5, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 100, 150, 200],
            limits = [0, 1.6, .1],
            coarsen=dict(
                rtofs=14,
                espc=15,
                cmems=14,
                amseas=13
                ),
            kwargs=dict(
                ptype="streamplot",
                color="black", 
                density=2.25,
                linewidth=.5
                )
            )
        figure = dict(
            figsize = (12, 8.5),
            legend = dict(columns=9),
            )
        
    key = "passengers"
    if key in regions:
        # Mid Atlantic Bight Limits
        name = 'Atlantis II Seamounts'
        folder = "passengers"
        # 38.404525, -63.994369
        # 37.514475, -62.8
        # 39.316911, -62.852942
        # 38.406367, -61.714686

        extent = [-68, -58, 36, 41]
        sea_water_temperature = [
            dict(depth=0, limits=[15, 28, 1]),
            dict(depth=150, limits=[11, 22, 1]),
            dict(depth=300, limits=[9, 21, 1])
            ]
        salinity = [
            dict(depth=0, limits=[31, 36.5, .25]),
            dict(depth=150, limits=[35.3, 36.6, .1]),
            dict(depth=300, limits=[35.2, 36.6, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1]),
            ]
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        salinity_max = dict(
            figsize=(14, 8.5),
            limits=[34.4, 37, .2]
            )
        currents = dict(
            bool=True,
            depths = [0, 150, 300],
            limits = [0, 1.5, .1],
            coarsen=dict(rtofs=5, espc=6),
            kwargs=dict(
                ptype="streamplot",
                color="black"
                )
            )
        figure = dict(
            legend = dict(columns=5),
            figsize = (16, 9)
            )

    key = "mexico_pacific"
    if key in regions:
        # Caribbean Limits
        name = "Eastern Pacific - Mexico"
        folder = "mexico_pacific"
        extent = [-126, -97, 8, 33]
        sea_water_temperature = [
            dict(depth=0, limits=[20, 32, 1]),
            dict(depth=100, limits=[12, 20, 1]),
            dict(depth=200, limits=[8, 13, .5])
            ]
        salinity = [
            dict(depth=0, limits=[33, 34.7, .1]),
            dict(depth=100, limits=[33.5, 34.9, .1]),
            dict(depth=200, limits=[33.8, 35.0, .1])
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            limits= [36, 37.5, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 100, 200],
            limits = [0, 1, .1],
            coarsen=dict(
                rtofs=14,
                espc=15,
                cmems=14,
                amseas=13
                ),
            kwargs=dict(
                ptype="streamplot",
                color="black", 
                density=2.25,
                linewidth=.5
                )
            )
        figure = dict(
            figsize = (12, 8),
            legend = dict(columns=9),
            )

    key = "hawaii"
    if key in regions:
        # Caribbean Limits
        name = "Hawaii"
        folder = "hawaii"
        extent = [-167, -138, 10, 27]
        sea_water_temperature = [
            dict(depth=0, limits=[23.5, 29, .5]),
            dict(depth=100, limits=[14, 26, 1]),
            dict(depth=200, limits=[10, 21, 1])
            ]
        salinity = [
            dict(depth=0, limits=[34.0, 35.3, .1]),
            dict(depth=100, limits=[34.5, 35.3, .1]),
            dict(depth=200, limits=[34.1, 35, .1])
            # dict(depth=150, limits=[35.8, 36.3, .05]),
            ]
        sea_surface_height = [
            dict(depth=0, limits=[-.6, .7, .1])
            ]
        salinity_max = dict(
            limits= [36, 37.5, .1]
            )
        ocean_heat_content = dict(
            limits= [0, 160, 10]
            )
        currents = dict(
            bool=True,
            depths = [0, 100, 200],
            limits = [0, 1.0, .1],
            coarsen=dict(
                rtofs=14,
                espc=15,
                cmems=14,
                amseas=13
                ),
            kwargs=dict(
                ptype="streamplot",
                color="black", 
                density=2.25,
                linewidth=.5
                )
            )
        figure = dict(
            figsize = (12, 7),
            legend = dict(columns=9),
            )

    # Create subdirectory for data variables
    vars = {}
    vars.update(salinity=salinity)
    vars.update(temperature=sea_water_temperature)
    
    # Update Dictionary with limits defined above
    limits.update(name=name)
    limits.update(folder=folder)
    limits.update(extent=extent)
    limits.update(figure=figure)
    limits.update(sea_surface_height=sea_surface_height)
    limits.update(eez=eez)
    limits.update(variables=vars)
    if currents:
        limits.update(currents=currents)
    if salinity_max:
        limits.update(salinity_max=salinity_max)
    if ocean_heat_content:
        limits.update(ocean_heat_content=ocean_heat_content)
    return limits

