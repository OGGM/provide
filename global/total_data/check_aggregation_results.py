# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:oggm_env]
#     language: python
#     name: conda-env-oggm_env-py
# ---

# # Imports

import os
import sys
import json
import geopandas as gpd
import xarray as xr
import numpy as np
import time

# +
# go up until we are in the project base directory
base_dir = os.getcwd()
while base_dir.split('/')[-1] != 'provide':
    base_dir = os.path.normpath(os.path.join(base_dir, '..'))

# add paths for tools and data
things_to_add = ['general_tools', 'aggregation_tools', 'general_data_for_aggregation']
for thing in things_to_add:
    sys.path.append(os.path.join(base_dir, thing))

from general_tools import check_if_notebook, mkdir
from oggm_result_filepath_and_realisations import scenarios_mesmer
from aggregation_plots import plot_map, plot_timeseries, plot_unavoidable_risk
# -

# Use this to conditionally execute tests/debugging
if check_if_notebook():
    is_notebook = True
else:
    is_notebook = False

# # define inputs

resolution_dir = 'total_data'
input_folder = 'aggregated_data'
output_folder = 'aggregated_result_plots'

# +
preprocess_region_dict_outpath = os.path.join(base_dir, 'global', resolution_dir)

with open(os.path.join(preprocess_region_dict_outpath, "preprocessed_region_grids.json"), 'r') as f:
    region_structure_dict = json.load(f)

# +
regions_data_dir = os.path.join(base_dir, 'global', 'data')
regions_file = 'global_shapefile.shp'
gdf_regions = gpd.read_file(os.path.join(regions_data_dir, regions_file))
name_col_regions = 'full_name'

gdf_regions[name_col_regions] = ['global']


# -

# # plot for all regions and scenarios

# +
def plot_scenario_results_region(region_name, scenario, input_folder, output_folder,
                                 add_map_plot=True
                                ):
    plot_output_folder = os.path.join(output_folder, region_name)
    mkdir(plot_output_folder)
    region = gdf_regions[gdf_regions[name_col_regions] == region_name]

    if add_map_plot:
        plot_map(region, region_name, scenario, input_folder, figsize=(12, 12),
                 save_plot=plot_output_folder)

    plot_timeseries(region_name, scenario, input_folder, figsize=(5, 9),
                    save_plot=plot_output_folder)

def plot_unavoidable_risk_for_all_scenarios(region_name, scenarios, input_folder,
                                            output_folder):
    plot_output_folder = os.path.join(output_folder, region_name)
    mkdir(plot_output_folder)

    plot_unavoidable_risk(region_name, scenarios, input_folder, figsize=(5, 15),
                          save_plot=plot_output_folder)


# -

# ## testing in notebook

if is_notebook:
    test_output_folder = 'aggregated_result_plots_test'
    mkdir(test_output_folder)

    test_region = 'ISL'
    test_scenario = scenarios_mesmer[0]

    plot_scenario_results_region(test_region, test_scenario,
                                  input_folder, test_output_folder)

    plot_unavoidable_risk_for_all_scenarios(test_region, scenarios_mesmer,
                                            input_folder, test_output_folder)

# ## code for cluster

if not is_notebook:
    # save results on cluster and copy at the end in run_slurm-file
    working_dir_cluster = os.environ.get('OGGM_WORKDIR', None)

    output_folder = os.path.join(working_dir_cluster,
                                 output_folder)
    mkdir(output_folder)

    start_time = time.time()
    for region in region_structure_dict:
        print(f'Start plotting {region}:')
        print(f'    unavoidable risk ({time.time() - start_time:.1f} s)')
        plot_unavoidable_risk_for_all_scenarios(region, scenarios_mesmer,
                                                input_folder, output_folder)
        for scenario in scenarios_mesmer:
            print(f'    {scenario} plots ({time.time() - start_time:.1f} s)')
            plot_scenario_results_region(region, scenario,
                                         input_folder, output_folder,
                                         add_map_plot=False,
                                        )

# # count files of each region

if is_notebook:
    nr_files_ref = None
    for region in region_structure_dict:
        path_region = os.path.join(output_folder,
                                    region)
        nr_files_region = len([file for file in os.listdir(path_region)
                               if os.path.isfile(os.path.join(path_region,file))])
        if nr_files_ref is None:
            nr_files_ref = nr_files_region
        elif nr_files_ref != nr_files_region:
            print(f'!!!{region} {nr_files_region} files, reference {nr_files_ref}!!!')
        else:
            print(f'{region} {nr_files_region} files')

# # check output files for consistancy

if is_notebook:
    # same ref values for each scenario?
    for region in region_structure_dict:
        print(f'Checking ref values {region}')
        ref_volume = None
        ref_area = None
        ref_runoff = None
        for scenario in scenarios_mesmer:
            with xr.open_dataset(
                os.path.join(input_folder,
                             region,
                             f'{region}_{scenario}_timeseries.nc')) as ds_time:
                if ref_volume is None:
                    ref_volume = ds_time.volume.reference_2020_km3
                else:
                    if not np.isclose(ds_time.volume.reference_2020_km3,
                                      ref_volume,
                                      #rtol=0.01,
                                      #atol=30
                                     ):
                        print(f'{region}/{scenario}: volume NOT close to reference '
                              f'(given {ds_time.volume.reference_2020_km3:.1f}, '
                              f'reference {ref_volume:.1f})')

                if ref_area is None:
                    ref_area = ds_time.area.reference_2020_km2
                else:
                    if not np.isclose(ds_time.area.reference_2020_km2,
                                      ref_area,
                                      #rtol=0.01,
                                      #atol=80
                                     ):
                        print(f'{region}/{scenario}: area NOT close to reference '
                              f'(given {ds_time.area.reference_2020_km2:.1f}, '
                              f'reference {ref_area:.1f})')

                if ref_runoff is None:
                    ref_runoff = ds_time.runoff.reference_2000_2019_Mt_per_yer
                else:
                    if not np.isclose(ds_time.runoff.reference_2000_2019_Mt_per_yer,
                                      ref_runoff,
                                      #rtol=0.01,
                                      #atol=80
                                     ):
                        print(f'{region}/{scenario}: runoff NOT close to reference '
                              f'(given {ds_time.runoff.reference_2000_2019_Mt_per_yer:.1f}, '
                              f'reference {ref_runoff:.1f})')



