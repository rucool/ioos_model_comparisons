import numpy as np

def limits_regions(model=None, regions=None):
    """
    return extent and other variable limits of certain regions for rtofs or gofs
    :param model: rtofs or gofs
    :param regions: list containing regions you want to plot
    :return: dictionary containing limits
    """

    model = model or 'rtofs'
    regions = regions or ['gom', 'sab', 'mab', 'carib', 'wind', 'nola', 'ng645', 'ng738']

    # Create new dictionary for selected model. Needs to be done because the variable names are different in each model
    # initialize empty dictionary for limits
    limits = dict()

    # Specify common variable and region limits for both gofs and rtofs
    # To add different depths for each variable, append to the specific variable list the following format:
    # dict(depth=n, limits=[min, max, stride])

    if 'north_atlantic' in regions:
        # Gulf of Mexico
        limits['North Atlantic'] = dict()
        nola = limits['North Atlantic']

        # Limits
        extent = [-80, 0, 0, 50]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[33, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=False, coarsen=8)

        # Update Dictionary with limits defined above
        nola.update(lonlat=extent)
        nola.update(salinity=salinity)
        nola.update(temperature=sea_water_temperature)
        nola.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            nola.update(sea_surface_height=sea_surface_height)

    if 'nola' in regions:
        # Gulf of Mexico
        limits['Gulf of Mexico'] = dict()
        nola = limits['Gulf of Mexico']

        # Limits
        extent = [-94, -84, 25.5, 31]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[33, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=False, coarsen=8)

        # Update Dictionary with limits defined above
        nola.update(lonlat=extent)
        nola.update(salinity=salinity)
        nola.update(temperature=sea_water_temperature)
        nola.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            nola.update(sea_surface_height=sea_surface_height)

    if 'usvi' in regions:
        # Gulf of Mexico
        limits['Virgin Islands'] = dict()
        vi = limits['Virgin Islands']

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
        vi.update(lonlat=extent)
        vi.update(salinity=salinity)
        vi.update(temperature=sea_water_temperature)
        vi.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            vi.update(sea_surface_height=sea_surface_height)

    if 'west_indies' in regions:
        # Gulf of Mexico
        limits['West Indies'] = dict()
        wi = limits['West Indies']

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
        wi.update(lonlat=extent)
        wi.update(salinity=salinity)
        wi.update(temperature=sea_water_temperature)
        wi.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            nola.update(sea_surface_height=sea_surface_height)

    if 'gom' in regions:
        # Gulf of Mexico
        limits['Gulf of Mexico'] = dict()
        gom = limits['Gulf of Mexico']

        # Limits
        gom_extent = [-100, -80, 18, 31]
        gom_sea_water_temperature = [dict(depth=0, limits=[26, 31, .5]), dict(depth=200, limits=[12, 24, .5])]
        gom_salinity = [dict(depth=0, limits=[34, 37, .25])]
        gom_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        gom_currents = dict(bool=False, coarsen=8)

        # Update Dictionary with limits defined above
        gom.update(lonlat=gom_extent)
        gom.update(salinity=gom_salinity)
        gom.update(temperature=gom_sea_water_temperature)
        gom.update(currents=gom_currents)

        # GOFS has sea surface height
        if model == 'gofs':
            gom.update(sea_surface_height=gom_sea_surface_height)

    if 'sab' in regions:
        # South Atlantic Bight
        limits['South Atlantic Bight'] = dict()
        sab = limits['South Atlantic Bight']

        # Limits
        sab_extent = [-82, -64, 25, 36]
        sab_sea_water_temperature = [dict(depth=0, limits=[24, 30, .5])]
        sab_salinity = [dict(depth=0, limits=[36, 37, .1])]
        sab_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        sab_currents = dict(bool=True, coarsen=7)

        # Update Dictionary with limits defined above
        sab.update(lonlat=sab_extent)
        sab.update(salinity=sab_salinity)
        sab.update(temperature=sab_sea_water_temperature)
        sab.update(currents=sab_currents)

        # GOFS has sea surface height
        if model == 'gofs':
            sab.update(sea_surface_height=sab_sea_surface_height)

    if 'mab' in regions:
        # Mid Atlantic Bight
        limits['Mid Atlantic Bight'] = dict()
        mab = limits['Mid Atlantic Bight']

        # Limits
        mab_extent = [-77, -67, 35, 43]
        mab_sea_water_temperature = [dict(depth=0, limits=[15, 29, .5]), dict(depth=100, limits=[10, 20, .5])]
        mab_salinity = [dict(depth=0, limits=[31, 37, .25])]
        mab_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        mab_currents = dict(bool=True, coarsen=6)

        # Update Dictionary with limits defined above
        mab.update(lonlat=mab_extent)
        mab.update(salinity=mab_salinity)
        mab.update(temperature=mab_sea_water_temperature)
        mab.update(currents=mab_currents)

        # GOFS has sea surface height
        if model == 'gofs':
            mab.update(sea_surface_height=mab_sea_surface_height)

    if 'carib' in regions:
        # Caribbean
        limits['Caribbean'] = dict()
        carib = limits['Caribbean']

        # Limits
        carib_extent = [-89, -55, 7, 23]
        carib_sea_water_temperature = [dict(depth=0, limits=[26, 31.5, .5])]
        carib_salinity = [dict(depth=0, limits=[34.6, 37, .1])]
        carib_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True,
                        coarsen=12,
                        scale=60,
                        headwidth=4,
                        headlength=4,
                        headaxislength=3.5
                        )

        # Update Dictionary with limits defined above
        carib.update(lonlat=carib_extent)
        carib.update(salinity=carib_salinity)
        carib.update(temperature=carib_sea_water_temperature)
        carib.update(currents=currents)

        # GOFS has sea surface height
        if model == 'gofs':
            carib.update(sea_surface_height=carib_sea_surface_height)

    if 'wind' in regions:
        # Windward Islands
        limits['Windward Islands'] = dict()
        wind = limits['Windward Islands']

        # Limits
        wind_extent = [-68.2, -56.4, 9.25, 19.75]
        wind_sea_water_temperature = [dict(depth=0, limits=[26, 30, .5])]
        wind_salinity = [dict(depth=0, limits=[34.6, 37, .1])]
        wind_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        wind_currents = dict(bool=True, coarsen=6)

        # Update Dictionary with limits defined above
        wind.update(lonlat=wind_extent)
        wind.update(salinity=wind_salinity)
        wind.update(temperature=wind_sea_water_temperature)
        wind.update(currents=wind_currents)

        # GOFS has sea surface height
        if model == 'gofs':
            wind.update(sea_surface_height=wind_sea_surface_height)

    if 'ng645' in regions:
        # ng645 Islands
        limits['ng645'] = dict()
        alias = limits['ng645']

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
        # ng645 Islands
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
        ng738_sargasso_northsouth=dict(
            xaxis='latitude',
            region='ng738',
            extent=[-70-15/60, 36+30/60, -70-15/60, 34+30/60],
            limits=dict(
                temperature=dict(
                    deep=np.arange(13, 23),
                    shallow=np.arange(13, 23),
                    isobath=[26]
                ),
                salinity=dict(
                    deep=np.arange(36, 36.6, 0.1),
                    shallow=np.arange(36, 36.6, 0.1)
                    ),
                u=dict(
                    deep=np.arange(-1, 1.1, 0.1),
                    shallow=np.arange(-1, 1.1, 0.1)
                    ),
                v=dict(
                    deep=np.arange(-1, 1.1, 0.1),
                    shallow=np.arange(-1, 1.1, 0.1)
                    )
                )
        ),
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

