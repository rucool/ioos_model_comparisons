import numpy as np


def transects():
    # transect coordinates and variable limits
    # mesoscale features (mostly brought in by altimetry)
    transects = dict(
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
        henri_path_mab_overview=dict(
            xaxis='latitude',
            region='Mid Atlantic Bight',
            extent=[-71.5, 41.5, -71.5, 36.5],
            limits=dict(
                temperature=dict(
                    deep=np.arange(4, 34),
                    shallow=np.arange(10, 34),
                    isobath=[15, 26]
                ),
                salinity=dict(
                    deep=np.arange(32, 37, 0.25),
                    shallow=np.arange(32, 37, 0.25)
                )
            )
        ),
        henri_path_mab_shelf=dict(
            xaxis='latitude',
            region='Mid Atlantic Bight',
            extent=[-71.5, 41.5, -71.5, 40],
            limits=dict(
                temperature=dict(
                    deep=np.arange(4, 34),
                    shallow=np.arange(10, 34),
                    isobath=[15, 26]
                ),
                salinity=dict(
                    deep=np.arange(32, 35.6, 0.25),
                    shallow=np.arange(32, 35.6, 0.25)
                )
            )
        ),
    )
    return transects


def limits_regions(model=None, regions=None):
    """
    return extent and other variable limits of certain regions for rtofs or gofs
    :param model: rtofs or gofs
    :param regions: list containing regions you want to plot
    :return: dictionary containing limits
    """

    model = model or 'rtofs'
    regions = regions or ['gom', 'sab', 'mab', 'carib', 'wind', 'nola']

    # Create new dictionary for selected model. Needs to be done because the variable names are different in each model
    # initialize empty dictionary for limits
    limits = dict()

    # Specify common variable and region limits for both gofs and rtofs
    # To add different depths for each variable, append to the specific variable list the following format:
    # dict(depth=n, limits=[min, max, stride])

    if 'nola' in regions:
        # Gulf of Mexico
        limits['Gulf of Mexico'] = dict()
        nola = limits['Gulf of Mexico']

        # Limits
        extent = [-93, -87, 26, 31]
        sea_water_temperature = [dict(depth=0, limits=[27.5, 32, .5]), dict(depth=200, limits=[12, 24, .5])]
        salinity = [dict(depth=0, limits=[34, 37, .25])]
        sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]
        currents = dict(bool=True, coarsen=8)

        # Update Dictionary with limits defined above
        nola.update(lonlat=extent)
        nola.update(salinity=salinity)
        nola.update(temperature=sea_water_temperature)
        nola.update(currents=currents)

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
        gom_currents = dict(bool=True, coarsen=8)

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
        carib_currents = dict(bool=True, coarsen=12)

        # Update Dictionary with limits defined above
        carib.update(lonlat=carib_extent)
        carib.update(salinity=carib_salinity)
        carib.update(temperature=carib_sea_water_temperature)
        carib.update(currents=carib_currents)

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

    return limits