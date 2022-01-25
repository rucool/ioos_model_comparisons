import numpy as np

def limits_regions(model=None, regions=None):
    """
    return extent and other variable limits of certain regions for rtofs or gofs
    :param model: rtofs or gofs
    :param regions: list containing regions you want to plot
    :return: dictionary containing limits
    """

    model = model or 'rtofs'
    regions = regions or ['gom', 'sab', 'mab', 'carib', 'wind', 'nola', 'ng645', 'ng738', 'usvi', 'yucatan']

    # Create new dictionary for selected model. Needs to be done because the variable names are different in each model
    # initialize empty dictionary for limits
    limits = dict()

    # Specify common variable and region limits for both gofs and rtofs
    # To add different depths for each variable, append to the specific variable list the following format:
    # dict(depth=n, limits=[min, max, stride])

    if 'yucatan' in regions:
        alias = limits['Yucatan'] = dict()

        # Limits
        extent = [-90, -82, 18, 24]
        sea_water_temperature = [dict(depth=0, limits=[24.25, 28, .25]), dict(depth=200, limits=[14, 23, .5])]
        salinity = [dict(depth=0, limits=[35.7, 36.2, .05])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True, coarsen=2)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'north_atlantic' in regions:
        alias = limits['North Atlantic'] = dict()

        # Limits
        extent = [-80, 0, 0, 50]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[33, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=False, coarsen=8)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity) 
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'nola' in regions:
        alias = limits['Gulf of Mexico/New Orleans'] = dict()

        # Limits
        extent = [-94, -84, 25.5, 31]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[33, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=False, coarsen=8)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'usvi' in regions:
        alias = limits['Virgin Islands'] = dict()

        # Limits
        extent = [-66.26, -62.61, 16.5, 19]
        # 19.027909879161452, -66.26100277351196
        # 16.568466806347633, -62.61079850383202
        sea_water_temperature = [dict(depth=0, limits=[26, 27.5, .1])]
        salinity = [dict(depth=0, limits=[35.5, 36.2, .1])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True,
                        coarsen=1,
                        scale=60,
                        headwidth=3,
                        headlength=3,
                        headaxislength=2.5
                        )

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'west_indies' in regions:
        alias = limits['West Indies'] = dict()

        # Limits
        extent = [-67, -61, 14, 19]
        # 19.027909879161452, -66.26100277351196
        # 16.568466806347633, -62.61079850383202
        sea_water_temperature = [dict(depth=0, limits=[27.5, 29.5, .25])]
        salinity = [dict(depth=0, limits=[34, 36.5, .25 ])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True,
                        coarsen=3,
                        scale=40,
                        headwidth=5,
                        headlength=5,
                        headaxislength=4.5
                        )

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'gom' in regions:
        alias = limits['Gulf of Mexico'] = dict()

        # Limits
        extent = [-100, -80, 18, 31]
        sea_water_temperature = [dict(depth=0, limits=[26, 31, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[34, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=False, coarsen=8)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'sab' in regions:
        alias = limits['South Atlantic Bight'] = dict()

        # Limits
        extent = [-82, -64, 25, 36]
        sea_water_temperature = [dict(depth=0, limits=[24, 30, .5])]
        salinity = [dict(depth=0, limits=[36, 37, .1])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True, coarsen=7)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'mab' in regions:
        alias = limits['Mid Atlantic Bight'] = dict()

        # Limits
        extent = [-77, -67, 35, 43]
        sea_water_temperature = [dict(depth=0, limits=[15, 29, .5]), dict(depth=100, limits=[10, 20, .5])]
        salinity = [dict(depth=0, limits=[31, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True, coarsen=6)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'carib' in regions:
        # Caribbean
        alias = limits['Caribbean'] = dict()

        # Limits
        extent = [-89, -55, 7, 23]
        sea_water_temperature = [dict(depth=0, limits=[26, 31.5, .5])]
        salinity = [dict(depth=0, limits=[34.6, 37, .1])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True,
                        coarsen=12,
                        scale=60,
                        headwidth=4,
                        headlength=4,
                        headaxislength=3.5
                        )

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'wind' in regions:
        # Windward Islands
        alias = limits['Windward Islands'] = dict()

        # Limits
        extent = [-68.2, -56.4, 9.25, 19.75]
        sea_water_temperature = [dict(depth=0, limits=[26, 30, .5])]
        salinity = [dict(depth=0, limits=[34.6, 37, .1])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True, coarsen=6)

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    if 'ng645' in regions:
        alias = limits['ng645'] = dict()

        # Limits
        extent = [-83.5, -81.5, 23.1, 25]
        sea_water_temperature = [dict(depth=0, limits=[23, 28, .5])]
        salinity = [dict(depth=0, limits=[35.6, 36.5, .1])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        # currents = [dict(depth=0, bool=True, coarsen=1), dict(depth=200, bool=True, coarsen=1)]
        currents = dict(bool=True, coarsen=1)


        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)
    
    if 'ng738' in regions:
        alias = limits['ng738'] = dict()

        # Limits
        extent = [-72, -69, 34.4, 36.5]
        sea_water_temperature = [dict(depth=0, limits=[21.5, 22.5, .1]), dict(depth=200, limits=[19, 21.4, .1])]
        salinity = [dict(depth=0, limits=[36.3, 36.8, .05])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True, coarsen=1)
        # currents = [dict(depth=0, bool=True, coarsen=1), dict(depth=200, bool=True, coarsen=1)] 

        # Update Dictionary with limits defined above
        alias.update(lonlat=extent)
        alias.update(salinity=salinity)
        alias.update(temperature=sea_water_temperature)
        alias.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            alias.update(sea_surface_height=sea_surface_height)

    return limits

def transects():
    # transect coordinates and variable limits
    # mesoscale features (mostly brought in by altimetry)
    transects = dict(
        virgin_islands_1=dict(
            xaxis='latitude',
            region='Virgin Islands',
            extent=[-64-45/60, 18+15/60, -64-45/60, 17+45/60],
            limits=dict(
                temperature=dict(
                    deep=np.arange(13, 27.5),
                    shallow=np.arange(13, 27.5),
                    isobath=[26]
                ),
                salinity=dict(
                    deep=np.arange(35.5, 36.2, 0.1),
                    shallow=np.arange(35.5, 36.2, 0.1)
                    ),
                u=dict(
                    deep=np.arange(-.3, .4, 0.05),
                    shallow=np.arange(-3, .4, 0.05)
                    ),
                v=dict(
                    deep=np.arange(-.3, .4, 0.05),
                    shallow=np.arange(-.3, .4, 0.05)
                    )
                )
        ),
        # virgin_islands_2=dict(
        #     xaxis='latitude',
        #     region='Virgin Islands',
        #     extent=[-64-15/60, 18+15/60, -64-15/60, 17+45/60],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(13, 27.5),
        #             shallow=np.arange(13, 27.5),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(35.5, 36.2, 0.1),
        #             shallow=np.arange(35.5, 36.2, 0.1)
        #             ),
        #         u=dict(
        #             deep=np.arange(-.3, .3, 0.1),
        #             shallow=np.arange(-3, .3, 0.1)
        #             ),
        #         v=dict(
        #             deep=np.arange(-3, .3, 0.1),
        #             shallow=np.arange(-3, .3, 0.1)
        #             )
        #         )
        # ),
        # ng645=dict(
        #     xaxis='latitude',
        #     region='ng645',
        #     extent=[-82-7.5/60, 24+30/60, -82-7.5/60, 23+10/60],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 27),
        #             shallow=np.arange(10, 27),
        #             isobath=[26]
        #             ),
        #         salinity=dict(
        #             deep=np.arange(34.75, 37, 0.1),
        #             shallow=np.arange(34.75, 37, 0.1)
        #             ),
        #         u=dict(
        #             deep=np.arange(-1, 1.1, 0.1),
        #             shallow=np.arange(-1, 1.1, 0.1)
        #             ),
        #         v=dict(
        #             deep=np.arange(-1, 1.1, 0.1),
        #             shallow=np.arange(-1, 1.1, 0.1)
        #             ),
        #         )
        # ),
        # ng738_sargasso_eastwest=dict(
        #     xaxis='longitude',
        #     region='ng738',
        #     extent=[-69-30/60, 35+25/60, -71-30/60, 35+25/60],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(13, 23),
        #             shallow=np.arange(13, 23),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(36, 36.6, 0.1),
        #             shallow=np.arange(36, 36.6, 0.1)
        #             ),
        #         u=dict(
        #             deep=np.arange(-1, 1.1, 0.1),
        #             shallow=np.arange(-1, 1.1, 0.1)
        #             ),
        #         v=dict(
        #             deep=np.arange(-1, 1.1, 0.1),
        #             shallow=np.arange(-1, 1.1, 0.1)
        #             )
        #         )
        # ),
        # ng738_sargasso_northsouth=dict(
        #     xaxis='latitude',
        #     region='ng738',
        #     extent=[-70-15/60, 36+30/60, -70-15/60, 34+30/60],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(13, 23),
        #             shallow=np.arange(13, 23),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(36, 36.6, 0.1),
        #             shallow=np.arange(36, 36.6, 0.1)
        #             ),
        #         u=dict(
        #             deep=np.arange(-1, 1.1, 0.1),
        #             shallow=np.arange(-1, 1.1, 0.1)
        #             ),
        #         v=dict(
        #             deep=np.arange(-1, 1.1, 0.1),
        #             shallow=np.arange(-1, 1.1, 0.1)
        #             )
        #         )
        # ),
        # grace_path_carib=dict(
        #     xaxis='longitude',
        #     region='Gulf of Mexico',
        #     extent=[-86.76, 21.05, -71.75, 17.35],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 34),
        #             shallow=np.arange(10, 34),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(34.75, 37, 0.1),
        #             shallow=np.arange(34.75, 37, 0.1)))
        # ),
        # grace_path_gom=dict(
        #     xaxis='longitude',
        #     region='Gulf of Mexico',
        #     extent=[-97.5, 21, -90.45, 21],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 34),
        #             shallow=np.arange(10, 34),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(34.75, 37, 0.1),
        #             shallow=np.arange(34.75, 37, 0.1)))
        # ),
        # henri_path_mab_overview=dict(
        #     xaxis='latitude',
        #     region='Mid Atlantic Bight',
        #     extent=[-71.5, 41.5, -71.5, 36.5],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 34),
        #             shallow=np.arange(10, 34),
        #             isobath=[15, 26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(32, 37, 0.25),
        #             shallow=np.arange(32, 37, 0.25)
        #         )
        #     )
        # ),
        # henri_path_mab_shelf=dict(
        #     xaxis='latitude',
        #     region='Mid Atlantic Bight',
        #     extent=[-71.5, 41.5, -71.5, 40],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 34),
        #             shallow=np.arange(10, 34),
        #             isobath=[15, 26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(32, 35.6, 0.25),
        #             shallow=np.arange(32, 35.6, 0.25)
        #         )
        #     )
        # ),
    )
    return transects

