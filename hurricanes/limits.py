import numpy as np

def transects():
    # transect coordinates and variable limits
    # mesoscale features (mostly brought in by altimetry)
    transects = dict( 
                    #21.472000, -86.788000 MX
                    #21.855367829243697, -84.94756372687907 CUBA
        # contoy_to_cuba=dict(
        #     xaxis='longitude',
        #     region='Yucatan Strait',
        #     extent=[-86.78, 21.47, -84.95, 21.9],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 15),
        #             shallow=np.arange(4, 15),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(32, 35.6, 0.25),
        #             shallow=np.arange(32, 35.6, 0.25)
        #             ),
        #         u=dict(
        #             deep=np.arange(-.3, .4, 0.05),
        #             shallow=np.arange(-3, .4, 0.05)
        #             ),
        #         v=dict(
        #             deep=np.arange(-.3, .4, 0.05),
        #             shallow=np.arange(-.3, .4, 0.05)
        #             )
        #         )
        # ),
        contoy_east_perpendicular=dict(
            xaxis='longitude',
            region='Yucatan Strait',
            extent=[-84.95, 21.47, -86.78, 21.47,],
            limits=dict(
                temperature=dict(
                    deep=np.arange(4, 15),
                    shallow=np.arange(4, 15),
                    isobath=[26]
                ),
                salinity=dict(
                    deep=np.arange(32, 35.6, 0.25),
                    shallow=np.arange(32, 35.6, 0.25)
                    ),
                u=dict(
                    deep=np.arange(-1.0, 1.1, 0.2),
                    shallow=np.arange(-1.0, 1.1, 0.2)
                    ),
                v=dict(
                    deep=np.arange(-1.0, 1.1, 0.2),
                    shallow=np.arange(-1.0, 1.1, 0.2)
                    )
                )
        ),
        # nj_cross_shelf_ru34=dict(
        #     xaxis='longitude',
        #     region='Mid Atlantic Bight',
        #     # 39 +17/60, -74-22/60
        #     # 38+22/60, -73
        #     extent=[-74.4, 39.3, -73, 38.3],
        #     limits=dict(
        #         temperature=dict(
        #             deep=np.arange(4, 15),
        #             shallow=np.arange(4, 15),
        #             isobath=[26]
        #         ),
        #         salinity=dict(
        #             deep=np.arange(32, 35.6, 0.25),
        #             shallow=np.arange(32, 35.6, 0.25)
        #             ),
        #         u=dict(
        #             deep=np.arange(-.3, .4, 0.05),
        #             shallow=np.arange(-3, .4, 0.05)
        #             ),
        #         v=dict(
        #             deep=np.arange(-.3, .4, 0.05),
        #             shallow=np.arange(-.3, .4, 0.05)
        #             )
        #         )
        # ),
        # virgin_islands_1=dict(
        #     xaxis='latitude',
        #     region='Virgin Islands',
        #     extent=[-64-45/60, 18+15/60, -64-45/60, 17+45/60],
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
        #             deep=np.arange(-.3, .4, 0.05),
        #             shallow=np.arange(-3, .4, 0.05)
        #             ),
        #         v=dict(
        #             deep=np.arange(-.3, .4, 0.05),
        #             shallow=np.arange(-.3, .4, 0.05)
        #             )
        #         )
        # ),
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

