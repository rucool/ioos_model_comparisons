#!/usr/bin/env python
"""
@author Mike Smith
@email michaesm@marine.rutgers.edu
@purpose Parse CODAR radial files utilizing the Radial subclass and run class defined quality control (QC) methods
"""

import logging
import os
import sys
import glob
import datetime as dt
from hfradar.src.radials import Radial

# Set up the parse_wave_files logger
logger = logging.getLogger(__name__)
log_level = 'INFO'
log_format = '%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
logging.basicConfig(stream=sys.stdout, format=log_format, level=log_level)


def main(radial_file, save_path, qc_values, export_type='radial'):
    """
    Main function to parse and qc radial files
    :param radial_file: Path to radial file
    :param save_path: Path to save quality controlled radial file
    :param qc_values: Dictionary containing thresholds for each QC test
    """
    try:
        r = Radial(radial_file, mask_over_land=False)
    except Exception as err:
        logging.error('{} - {}'.format(radial_file, err))
        return

    if r.is_valid():
        t0 = r.time - dt.timedelta(hours=1)
        previous_radial = '{}_{}'.format('_'.join(r.file_name.split('_')[:2]), t0.strftime('%Y_%m_%d_%H00.ruv'))
        previous_full_file = os.path.join(os.path.dirname(r.full_file), previous_radial)

        # run high frequency radar qartod tests on open radial file
        r.initialize_qc()
        r.qc_qartod_syntax()
        r.qc_qartod_maximum_velocity(**qc_values['qc_qartod_maximum_velocity'])
        r.qc_qartod_valid_location()
        r.qc_qartod_radial_count(**qc_values['qc_qartod_radial_count'])
        r.qc_qartod_spatial_median(**qc_values['qc_qartod_spatial_median'])

        if os.path.exists(previous_full_file):
            r.qc_qartod_temporal_gradient(previous_full_file)
        else:
            logging.error('{} does not exist at specified location. Bypassing temporal gradient test'.format(previous_full_file))
        # r.qc_qartod_avg_radial_bearing(**qc_values['qc_qartod_avg_radial_bearing'])
        r.qc_qartod_primary_flag()

        # Export radial file to either a radial or netcdf
        try:
            r.export(os.path.join(save_path, r.file_name), export_type)
        except ValueError as err:
            logging.error('{} - {}'.format(radial_file, err))
            pass


if __name__ == '__main__':
    export_type = 'radial'

    # # JOUB - QC - Start #
    # radial_path = '/Users/mikesmith/Documents/Work/codar/swarm/radials/JOUB/'
    # save_path = '/Users/mikesmith/Documents/Work/codar/swarm/radials_qc/JOUB/'
    #
    # qc_values = dict(
    #     qc_qartod_radial_count=dict(radial_min_count=121.5, radial_low_count=364.5),
    #     qc_qartod_maximum_velocity=dict(radial_max_speed=150, radial_high_speed=175.0),
    #     qc_qartod_spatial_median=dict(radial_smed_range_cell_limit=2.1, radial_smed_angular_limit=10, radial_smed_current_difference=30),
    #     qc_qartod_temporal_gradient=dict(gradient_temp_fail=16, gradient_temp_warn=11)
    # )
    # # JOUB - QC - End #

    # # PALM - QC - Start #
    # radial_path = '/Users/mikesmith/Documents/Work/codar/swarm/radials/PALM/'
    # save_path = '/Users/mikesmith/Documents/Work/codar/swarm/radials_qc/PALM/'
    #
    # qc_values = dict(
    #     qc_qartod_radial_count=dict(radial_min_count=148.75, radial_low_count=446.25),
    #     qc_qartod_maximum_velocity=dict(radial_max_speed=300, radial_high_speed=150.0),
    #     qc_qartod_spatial_median=dict(radial_smed_range_cell_limit=2.1, radial_smed_angular_limit=10, radial_smed_current_difference=30),
    #     qc_qartod_temporal_gradient=dict(gradient_temp_fail=17, gradient_temp_warn=12)
    # )
    # # PALM - QC - End #

    # WAUW - QC - Start #
    radial_path = '/Users/mikesmith/Documents/Work/codar/swarm/radials/WAUW/'
    save_path = '/Users/mikesmith/Documents/Work/codar/swarm/radials_qc/WAUW/'

    qc_values = dict(
        qc_qartod_radial_count=dict(radial_min_count=121.5, radial_low_count=364.5),
        qc_qartod_maximum_velocity=dict(radial_max_speed=150, radial_high_speed=175.0),
        qc_qartod_spatial_median=dict(radial_smed_range_cell_limit=2.1, radial_smed_angular_limit=10, radial_smed_current_difference=30),
        qc_qartod_temporal_gradient=dict(gradient_temp_fail=18.5, gradient_temp_warn=13)
    )
    # WAUW - QC - End #

    radials = glob.glob(os.path.join(radial_path, '*.ruv'))

    for radial in sorted(radials):
        main(radial, save_path, qc_values, export_type)
