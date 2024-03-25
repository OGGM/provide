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

# ## general packages

import glob
import os
import json
import time
import xarray as xr
import numpy as np
import sys

# ## project tools and data

# +
base_path = os.getcwd()

# go up until we are in the project base directory
path_to_add = base_path
while path_to_add.split('/')[-1] != 'provide':
    path_to_add = os.path.normpath(os.path.join(path_to_add, '..'))

# add paths for tools and data
things_to_add = ['general_tools', 'aggregation_tools', 'general_data_for_aggregation']
for thing in things_to_add:
    sys.path.append(os.path.join(path_to_add, thing))

# import stuff we need
from general_tools import check_if_notebook
from oggm_result_filepath_and_realisations import (gcms_mesmer, quantiles_mesmer,
    scenarios_mesmer, oggm_result_dir, provide_regions, raw_oggm_output_file)
# -

# Use this to conditionally execute tests/debugging
if check_if_notebook():
    is_notebook = True
else:
    is_notebook = False

# # define directories for results

output_dir_region = os.path.join(base_path, 'common_running_glaciers_slurm')
output_dir_all = os.path.join(path_to_add, 'general_data_for_aggregation')


# # Acutal work is done here

# ## help functions for getting batched-files

# +
def get_batches_of_region(region):
    files = glob.glob(os.path.join(oggm_result_dir, region, raw_oggm_output_file))
    batches = []
    for file in files:
        parts = os.path.basename(file).split('_')
        batches.append(f"{parts[-2]}_{parts[-1].replace('.nc','')}")
    return list(np.unique(batches))


def get_filename_with_batch(batch):
    return raw_oggm_output_file[::-1].replace('*', batch[::-1], 1)[::-1]


# -

# ## open multiple files, extract commonly running glaciers and store

def preprocess(ds):
    """
    Preprocess function to extract model, scenario, and quantile from the filename
    and add them as coordinates to the dataset.
    """
    # only need volume here
    ds = ds[['volume']]

    # Extract model, scenario, and quantile from the filename
    filename = ds.encoding['source']
    parts = os.path.basename(filename).split('_')
    scenario = parts[5]
    gcm = parts[6]
    quantile = float(parts[7].replace('q', ''))
    ds = ds.expand_dims({'gcm': [gcm], 'scenario': [scenario], 'quantile': [quantile]})
    
    return ds


if not is_notebook:
    # slurm comment: sbatch --array=1-12 run_slurm_common_running_glaciers.sh
    
    # extract provide region from execution
    region = os.environ.get('PROVIDE_REG', None)
    if region is None:
        raise ValueError('Need a Provide region to start!')

    # for all gcms
    start_time = time.time()
    commonly_running_glaciers_all = []
    #for region in provide_regions:
    print(f'Start region {region} for all.')
    region_batches = get_batches_of_region(region)

    for batch in region_batches:
        print(f'batch {batch}')
        files = glob.glob(
            os.path.join(oggm_result_dir,
                         region,
                         get_filename_with_batch(batch)
                        )
        )
        
        combined_ds = xr.open_mfdataset(files,
                                        preprocess=preprocess,
                                        combine='by_coords',
                                        parallel=False)

        commonly_running_glaciers_all.extend(
            list(combined_ds.volume.dropna(
                dim='rgi_id', how='any').rgi_id.values)
        )
        combined_ds.close()

    with open(os.path.join(output_dir_region,
                           f"commonly_running_glaciers_all_{region}.json"), "w") as outfile: 
        json.dump(commonly_running_glaciers_all, outfile)

    print(f'Time needed for all gcms {time.time() - start_time:.1f} s')
    print('')

    # excluding IPSL-CM6A-LR
    start_time = time.time()
    commonly_running_glaciers_without_IPSL = []
    #for region in provide_regions:
    print(f'Start region {region} without IPSL.')
    region_batches = get_batches_of_region(region)
    for batch in region_batches:
        print(f'batch {batch}')
        files = glob.glob(
            os.path.join(oggm_result_dir,
                         region,
                         get_filename_with_batch(batch)
                        )
        )
        files = [file for file in files if 'IPSL-CM6A-LR' not in file]
        combined_ds = xr.open_mfdataset(files,
                                        preprocess=preprocess,
                                        combine='by_coords',
                                        parallel=False)
        commonly_running_glaciers_without_IPSL.extend(
            list(combined_ds.volume.dropna(
                dim='rgi_id', how='any').rgi_id.values)
        )
        combined_ds.close()
    with open(os.path.join(output_dir_region,
                           f"commonly_running_glaciers_without_IPSL_{region}.json"), "w") as outfile: 
        json.dump(commonly_running_glaciers_without_IPSL, outfile)
    print(f'Time needed without IPSL {time.time() - start_time:.1f} s')

if is_notebook:
    test_region = provide_regions[7]
    files = glob.glob(os.path.join(oggm_result_dir, test_region, raw_oggm_output_file))
    test_gcms = [gcms_mesmer[0], gcms_mesmer[1]]
    test_scenarios_mesmer = [scenarios_mesmer[0], scenarios_mesmer[1]]
    # Function to filter long strings
    def filter_long_strings(long_strings, list1, list2):
        # List to store the result
        filtered_strings = []
        
        # Iterate through each long string
        for long_string in long_strings:
            # Check if the long string contains at least one string from each list
            if any(short_string in long_string for short_string in list1) and \
               any(short_string in long_string for short_string in list2):
                # If both conditions are met, add the long string to the result list
                filtered_strings.append(long_string)
        
        return filtered_strings
    test_files = filter_long_strings(files, test_gcms, test_scenarios_mesmer)
    start_time = time.time()
    combined_ds = xr.open_mfdataset(test_files,
                                    preprocess=preprocess,
                                    combine='by_coords',
                                    parallel=False)
    print(f'Time needed {time.time() - start_time:.1f} s')

# # Merge all list into single list after cluster run

if is_notebook:
    common_running_list_dir = output_dir_region
    raw_filename = 'commonly_running_glaciers_all_{}.json'

    all_common_running_glaciers = []
    for region in provide_regions:
        with open(os.path.join(common_running_list_dir,
                               raw_filename.format(region)), 'r') as f:
            all_common_running_glaciers.extend(json.load(f))

    with open(os.path.join(output_dir_all,
                           f"commonly_running_glaciers.json"), "w") as outfile: 
        json.dump(all_common_running_glaciers, outfile)

# # Look at results

if is_notebook:
    from oggm import utils
    import pandas as pd

    frgi = utils.file_downloader('https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62_stats.h5')
    df_rgi = pd.read_hdf(frgi, index_col=0)

    fp_rgi_prov_region = 'rgi_ids_per_provide_region.json'
    with open(os.path.join(output_dir_all, fp_rgi_prov_region), 'r') as f:
        dict_rgis_preg = json.load(f)

if is_notebook:

    #test_region = 'P07'
    for test_region in provide_regions:
        file_all = f"commonly_running_glaciers_all_{test_region}.json"
        file_without_IPSL = f"commonly_running_glaciers_without_IPSL_{test_region}.json"
    
        with open(os.path.join(output_dir_region, file_all), 'r') as f:
            rgi_ids_all_test = json.load(f)
    
        with open(os.path.join(output_dir_region, file_without_IPSL), 'r') as f:
            rgi_ids_without_IPSL_test = json.load(f)

        # all rgi_ids of region
        rgi_ids_region_all = list(set(dict_rgis_preg[test_region]))
        if test_region == 'P03': # omit connectiity level 2 from P03 (i.e., Greenland)
            odf_preg = df_rgi.loc[rgi_ids_region_all]
            odf_preg_sel = odf_preg.loc[odf_preg['Connect'] != 2]
            rgi_ids_region_all = odf_preg_sel.index

        all_area = df_rgi.loc[rgi_ids_region_all].Area.sum()
        call_area = df_rgi.loc[rgi_ids_all_test].Area.sum()
        ipsl_area = df_rgi.loc[rgi_ids_without_IPSL_test].Area.sum()
        print(f'Region {test_region}:')
        print(f'  ALL nr RGI-IDS: {len(rgi_ids_region_all)}')
        print(f' CAll nr RGI-IDS: {len(rgi_ids_all_test)}')
        print(f' IPSL nr RGI-IDS: {len(rgi_ids_without_IPSL_test)}')
        print(f'     CALL - IPSL: {len(rgi_ids_all_test) - len(rgi_ids_without_IPSL_test)}')
        print(f'      ALL - CALL: {len(rgi_ids_region_all)- len(rgi_ids_all_test)}')
        print(f'        ALL area: {all_area:.1f} km2, {100 / all_area * all_area:.1f} %')
        print(f'       CAll area: {call_area:.1f} km2, {100 / all_area * call_area:.3f} %')
        print(f'       IPSL area: {ipsl_area:.1f} km2, {100 / all_area * ipsl_area:.3f} %')
        print(f'     CALL - IPSL: '
              f'{df_rgi.loc[rgi_ids_all_test].Area.sum() -df_rgi.loc[rgi_ids_without_IPSL_test].Area.sum():.1f}'
              f' km2')
        print(f'      ALL - CALL: '
              f'{df_rgi.loc[rgi_ids_region_all].Area.sum() - df_rgi.loc[rgi_ids_all_test].Area.sum():.1f}'
              f' km2')  
