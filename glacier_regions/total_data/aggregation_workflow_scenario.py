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
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# +
# go up until we are in the project base directory
base_dir = os.getcwd()
while base_dir.split('/')[-1] != 'provide':
    base_dir = os.path.normpath(os.path.join(base_dir, '..'))

# add paths for tools and data
things_to_add = ['general_tools', 'aggregation_tools', 'general_data_for_aggregation']
for thing in things_to_add:
    sys.path.append(os.path.join(base_dir, thing))

# import stuff we need
from general_tools import check_if_notebook, mkdir
from oggm_result_filepath_and_realisations import (gcms_mesmer, quantiles_mesmer,
    scenarios_mesmer, oggm_result_dir, provide_regions, raw_oggm_output_file)
from aggregation_preprocessing import open_grid_from_dict
from actual_aggregation import open_files_and_aggregate_on_map, aggregating_scenario, check_slurm_done
# -

# Use this to conditionally execute tests/debugging
if check_if_notebook():
    is_notebook = True
else:
    is_notebook = False

len(gcms_mesmer)

# # Define directories

resolution_dir = 'total_data'

preprocess_region_dict_outpath = os.path.join(base_dir, 'glacier_regions', resolution_dir)
mkdir(preprocess_region_dict_outpath);

# # Open data

with open(os.path.join(preprocess_region_dict_outpath, "preprocessed_region_grids.json"), 'r') as f:
    region_structure_dict = json.load(f)

# # Code for running on cluster

# ## define experiments for multiprocessing

# +
# create dict for sbatch --array=
slurm_arrays = {}

count = 1
for region in region_structure_dict.keys():
    for scenario in scenarios_mesmer:
        slurm_arrays[str(count)] = (region, scenario)
        count += 1

if is_notebook:
    print(slurm_arrays)
# -

# ## run current experiment

if not is_notebook:
    slurm_id = os.environ.get('ARRAY_ID', None)
    # convert slurm array to region and scenario
    region, scenario = slurm_arrays[slurm_id]

    # save results on cluster and copy at the end in run_slurm-file
    working_dir_cluster = os.environ.get('OGGM_WORKDIR', None)

    aggregated_data_outpath = os.path.join(
        working_dir_cluster, 'aggregated_data')
    mkdir(aggregated_data_outpath);

    aggregated_data_intermediate_outpath = os.path.join(
        preprocess_region_dict_outpath,
        'aggregated_data_intermediate')
    mkdir(aggregated_data_intermediate_outpath);

    aggregating_scenario(
            target_name=region,
            target_structure_dict=region_structure_dict,
            scenario=scenario,
            output_folder=aggregated_data_outpath,
            oggm_result_dir=oggm_result_dir,
            raw_oggm_output_file=raw_oggm_output_file,
            intermediate_data_folder=aggregated_data_intermediate_outpath,
            variables=['volume', 'area', 'thinning_rate', 'runoff'],
            risk_variables=['volume', 'area'],
            risk_thresholds=np.append(np.arange(10, 91, 10), [99]),  # in % melted of 2020, 10% means 10% of 2020 melted
            time_steps=np.arange(2015, 2101, 5),
            reset_files=False
        )

# ## check which experiments failed for rerunning

if is_notebook:
    check_slurm_done(465474)

if is_notebook:
    check_slurm_done(466392)

if is_notebook:
    check_slurm_done(466486)

if is_notebook:
    check_slurm_done(436405)


