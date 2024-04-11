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

# +
import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import sys
import time

from aggregation_preprocessing import open_grid_from_dict

# +
# go up until we are in the project base directory
base_dir = os.getcwd()
while base_dir.split('/')[-1] != 'provide':
    base_dir = os.path.normpath(os.path.join(base_dir, '..'))

# add paths for tools and data
things_to_add = ['general_data_for_aggregation']
for thing in things_to_add:
    sys.path.append(os.path.join(base_dir, thing))
    
from oggm_result_filepath_and_realisations import gcms_mesmer, quantiles_mesmer


# -

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


# # Tools for interacting with preprocessed gridded structure

def flatten_ds_var(ds_var):
    """extracte a pure list of a variable which is aggregted on a grid.
    """
    if isinstance(ds_var, xr.core.dataarray.DataArray):
        nested_list = ds_var.values.tolist()
    else:
        nested_list = ds_var
    # The flattened list to be returned
    flattened = []
    for item in nested_list:
        if item is None:
            continue  # Skip None values
        elif isinstance(item, list):
            # If the item is a list, recursive call
            flattened.extend(flatten_ds_var(item))
        else:
            # If the item is not a list, add it to the flattened list
            flattened.append(item)
    return flattened


# # Tools for opening batched result files

# +
def get_filename_with_batch(raw_file, batch):
    return raw_file[::-1].replace('*', batch[::-1], 1)[::-1]


def get_all_files_from_result_batches(path, raw_file, needed_files):
    """ Takes the raw file structure of a batched files and 
    and get all files corresponding to the given batches.
    """
    resulting_filenames = []
    for file in needed_files:
        provide_region, batch = file.split('/')
        resulting_filenames.extend(
            glob.glob(
                os.path.join(
                    path,
                    provide_region,
                    get_filename_with_batch(raw_file, batch)
                )
            )
        )
    return resulting_filenames


# -

# # Smoothing and size threshold

def apply_size_threshold_and_smoothing(ds, smoothing_window=5, start_year_of_smoothing=2020):
    # define thersholds
    area_threshold = 0.01 * 1e6  # m²
    # VAS following Marzeion et. al., 2012
    volume_threshold = 0.1912 * area_threshold ** 1.375  # m³

    # apply thresholds
    ds['volume'] = ds['volume'] - volume_threshold
    ds['area'] = ds['area'] - area_threshold
    below_threshold_mask = (ds['volume'] < 0) | (ds['area'] < 0)
    ds[['area', 'volume']] = xr.where(below_threshold_mask, 0, ds[['area', 'volume']])

    # once zero always zero
    non_zero = ds['volume'] != 0
    mask = non_zero.cumprod('time')
    ds[['area', 'volume']] = ds[['area', 'volume']] * mask

    
    # rolling mean smoothing
    ds_return = ds.copy()
    ds_return[['area', 'volume']] = ds[['area', 'volume']].rolling(
        min_periods=1, time=smoothing_window, center=True).mean()

    # set all years before the start year of smoogthing back to the original values
    dates_before_year = ds_return['time'].values <= start_year_of_smoothing
    indices_before_year = np.nonzero(dates_before_year)[0]
    for var_name in ['area', 'volume']:
        ds_return[var_name][{'time': indices_before_year}] = ds[var_name].isel(time=indices_before_year)

    return ds_return


# # Preprocessing during opening of files

def preprocess_ds_during_opening(ds, rgi_ids_per_batch, variables, time_steps=None):
    """
    Preprocess function to extract model, scenario, and quantile from the filename
    and add them as coordinates to the dataset.
    """
    # Extract model, scenario, and quantile from the filename
    filename = ds.encoding['source']
    parts = os.path.basename(filename).split('_')
    scenario = parts[5]
    gcm = parts[6]
    quantile = float(parts[7].replace('q', ''))
    batch_start = parts[-2]
    batch_end = parts[-1].replace('.nc', '')
    region = filename.split('/')[-2]

    # only keep needed rgi_ids
    file_batch_key = f'{region}/{batch_start}_{batch_end}'
    ds = ds.sel(rgi_id=ds.rgi_id.isin(rgi_ids_per_batch[file_batch_key]))

    # only keep variables of interest
    ds = ds[variables]

    # only keep time_steps of interest
    if time_steps is not None:
        ds = ds.loc[{'time': time_steps}]

    # apply size threshold and smoothing
    ds = apply_size_threshold_and_smoothing(ds)

    # finally add new dimension
    ds = ds.expand_dims({'gcm': [gcm], 'scenario': [scenario], 'quantile': [quantile]})
    
    return ds


# # Tools for calculating glacier variables

# ## Normalize Volume and Area

# +
def get_ref_value(ds, var):
    if 'lat' in ds.coords:
        ds = ds.sum(dim=['lat', 'lon'])
    return ds.loc[{'time': 2020,
                   'quantile': 0.5}][var].values.flatten()[0]

def normalize_var(ds, var):
    reference_value = get_ref_value(ds, var)
    if var == 'volume':
        unit = 'km3'
        unit_conversion = 1e9
    elif var == 'area':
        unit = 'km2'
        unit_conversion = 1e6
    elif var == 'thickness':
        unit = 'meter_w.e.'
        unit_conversion = 1
    else:
        raise NotImplementedError()
    ds[var] = ds[var] / reference_value * 100
    ds[var].attrs[f'unit'] = f'in % relative to total value of 2020 (see reference_2020_{unit})'
    ds[var].attrs[f'reference_2020_{unit}'] = reference_value / unit_conversion


# -

# ## Thickness

# +
def add_thickness_unit(ds):
    ds['thickness'].attrs['unit'] = 'meter water equivalent (m w.e.)'


def add_thickness(ds):
    ds['thickness'] = xr.where(ds.area > 0,
                               ds.volume / ds.area,
                               ds.area * 0) * 900 / 1000  # m w.e.
    add_thickness_unit(ds)


# -

# ## Thinning_Rate

# +
def add_thinning_rate_unit(ds):
    ds['thinning_rate'].attrs['unit'] = 'meter water equivalent per year (m w.e. yr-1)'


def add_thinning_rate(ds):
    dv = ds.volume.diff(dim='time', label='upper')  # m3
    a_shifted = ds.area.shift(time=1)
    a_mean = (ds.area + a_shifted).where(a_shifted.notnull()) / 2  # m2
    dt = ds.time.diff(dim='time').values[0]  # yr
    rho = 900  # kg m-3
    ds['thinning_rate'] = dv * rho / a_mean / dt / 1000  # 0.001 kg m-2 yr-1 == m w.e. m-2 yr-1
    add_thinning_rate_unit(ds)


# -

# # Aggregate data on maps

def aggregate_data_on_map(ds_data, ds_structure):
    # Create a mapping from lon, lat to rgi_id using ds_structure
    mapping_df = ds_structure.to_dataframe().reset_index().explode('rgi_ids')
    
    # We exclude depenent varialbes during aggretation and add again at the end
    dependent_coords = [coord for coord in ds_data.coords
                        if not ds_data.coords[coord].dims == (coord,)]
    
    # Convert ds_data to DataFrame excluding dependent coordinates
    data_df = ds_data.drop_vars(list(dependent_coords)).to_dataframe().reset_index()
    
    # Actual mapping is happening here
    merged_df = pd.merge(mapping_df, data_df,
                         left_on='rgi_ids', right_on='rgi_id',
                         how='left')
    merged_df = merged_df.drop(columns=['rgi_id', 'rgi_ids'])
    
    # We only want to aggregate the data variables
    data_vars = list(ds_data.data_vars)
    agg_dims = [dim for dim in merged_df.columns if dim not in data_vars]
    
    # Now, aggregate by the new dimensions (including lat and lon)
    result_df = merged_df.groupby(agg_dims, dropna=False).sum(min_count=1).reset_index()
    
    # Convert back to xarray Dataset
    ds_result = result_df.set_index(agg_dims).to_xarray()
    
    # Re-add dependent coordinates
    for coord in ds_data.coords:
        if coord == 'rgi_id':
            continue
        if coord not in ds_result.coords:
            ds_result = ds_result.assign_coords({coord: ds_data.coords[coord]})

    # keep some attributes from structure
    ds_result.attrs['grid_resolution'] = ds_structure.attrs['resolution']
    ds_result.attrs['grid_points_with_data'] = ds_structure.attrs['grid_points_with_data']

    return ds_result


# # Function for getting the weighted quantiles

def get_weighted_quantiles(ds, q_to_return=None,
                           return_weighted_total_quantiles=True):
    # Define the repetition scheme for each quantile
    quantile_repetitions = {
        0.05: 6,
        0.25: 9,
        0.50: 10,
        0.75: 9,
        0.95: 6
    }
    if q_to_return is None:
        q_to_return = list(quantile_repetitions.keys())
    
    duplicated_data_arrays = []
    
    # Iterate over each quantile, duplicating the data points as specified
    for quantile, repetitions in quantile_repetitions.items():
        quantile_data = ds.sel(quantile=quantile)
        for _ in range(repetitions):
            duplicated_data_arrays.append(quantile_data)
    
    combined_data = xr.concat(duplicated_data_arrays, dim='extra_quantiles')

    def add_attr(ds_return, ds):
        ds_return.attrs['grid_resolution'] = ds_structure.attrs['resolution']
        ds_result.attrs['grid_points_with_data'] = ds_structure.attrs['grid_points_with_data']

    if return_weighted_total_quantiles:
        # here we return the quantiles drawn from all realisations
        ds_return = combined_data.stack(
            sample=('gcm', 'extra_quantiles')).chunk({'sample': -1}).quantile(
            q_to_return, dim='sample')

    else:
        # here we return the quantiles for each gcm
        ds_return = combined_data.chunk({'extra_quantiles': -1}).quantile(
            q_to_return, dim='extra_quantiles')

    if 'grid_resolution' in ds.attrs:
        ds_return.attrs['grid_resolution'] = ds.attrs['grid_resolution']
        ds_return.attrs['grid_points_with_data'] = ds.attrs['grid_points_with_data']

    return ds_return


# # Main aggregation function, doing aggregation step by step

# ## opening all files and aggregate for map and total

def open_files_and_aggregate(gcm_use,
                             quantile_use,
                             all_files_target,
                             scenario,
                             start_time,
                             ds_grid_structure,
                             variables_to_open,
                             variables,
                             time_steps,
                             target_name,
                             map_data_folder,
                             total_data_folder,
                             reset_files,
                            ):
    # start opening gcm and quantiles, aggregate and save
    files_for_merging_map_data = []
    files_for_merging_total_data = []
    for gcm in gcm_use:
        gcm_files = [file for file in all_files_target if f'{gcm}_' in file]
        for quant in quantile_use:
            print(f'Opening {scenario}, {gcm}, {quant} ({time.time() - start_time:.1f} s)')
            files_to_use = [file for file in gcm_files if quant in file]

            tmp_map_filepath = os.path.join(
                    map_data_folder,
                    f'{target_name}_{scenario}_{gcm}_{quant}_map_data.nc'
                )
            tmp_total_filepath = os.path.join(
                    total_data_folder,
                    f'{target_name}_{scenario}_{gcm}_{quant}_total_data.nc'
                )

            if not reset_files:
                if os.path.exists(tmp_map_filepath) and os.path.exists(tmp_total_filepath):

                    files_for_merging_map_data.append(tmp_map_filepath)
                    files_for_merging_total_data.append(tmp_total_filepath)

                    print(f'{target_name}_{scenario}_{gcm}_{quant} files already exist!')
                    continue

            with xr.open_mfdataset(files_to_use,
                                   preprocess=lambda x: preprocess_ds_during_opening(
                                       x,
                                       rgi_ids_per_batch=ds_grid_structure.result_batches,
                                       variables=variables_to_open,
                                       time_steps=time_steps),
                                   combine='nested',
                                   parallel=False,
                                   engine='netcdf4',
                                  ) as ds_use:
                print('Files opened, start aggregation on map')

                if not reset_files and os.path.exists(tmp_map_filepath):
                    files_for_merging_map_data.append(tmp_map_filepath)
                    print('Aggregated map data already exist!')
                    
                else:
                    # aggregate on map
                    ds_map = aggregate_data_on_map(ds_use, ds_grid_structure)
    
                    # add additional variables
                    if 'volume' in variables:
                        ds_map['volume'].attrs['unit'] = 'm3'
                    if 'area' in variables:
                        ds_map['area'].attrs['unit'] = 'm2'
                    if 'thickness' in variables:
                        add_thickness(ds_map)
                    if 'thinning_rate' in variables:
                        add_thinning_rate(ds_map)
                        # first timestep is only for calculation of mass
                        ds_map = ds_map.sel({'time': time_steps[1:]})
    
                    ds_map.to_netcdf(tmp_map_filepath)
                    files_for_merging_map_data.append(tmp_map_filepath)
    
                    # delete variable, because for big countries all the memory is needed
                    del ds_map

                print('Start total aggregation')

                if not reset_files and os.path.exists(tmp_total_filepath):
                    files_for_merging_total_data.append(tmp_total_filepath)
                    print('Aggregated total data already exist!')
                else:
                    # aggregate for whole target
                    ds_total = ds_use.sum(dim='rgi_id')
    
                    # add additional variables
                    if 'volume' in variables:
                        ds_total['volume'].attrs['unit'] = 'm3'
                    if 'area' in variables:
                        ds_total['area'].attrs['unit'] = 'm2'
                    if 'thickness' in variables:
                        add_thickness(ds_total)
                    if 'thinning_rate' in variables:
                        add_thinning_rate(ds_total)
                        # first timestep is only for calculation of mass
                        ds_total = ds_total.sel({'time': time_steps[1:]})
    
                    ds_total.to_netcdf(tmp_total_filepath)
                    files_for_merging_total_data.append(tmp_total_filepath)

    print(f'Finished opening of all raw result files ({time.time() - start_time:.1f} s)')

    return files_for_merging_map_data, files_for_merging_total_data


# ## merge map data including quantiles

def merge_map_data_with_quantiles(files_for_merging_map_data,
                                  result_folder,
                                  target_name,
                                  variables,
                                  scenario,
                                  start_time,
                                  reset_files,
                                 ):

    print(f'Start merging all gcms and raw quantiles and calculate weighted quantiles')

    result_path = os.path.join(
            result_folder,
            f'{target_name}_{scenario}_map.nc'
        )
    if not reset_files:
        if os.path.exists(result_path):
            print('Merged map file already exists!')
            return

    with xr.open_mfdataset(files_for_merging_map_data,
                           combine='by_coords',
                           parallel=False,
                           engine='netcdf4') as ds_map:
        print(f'Finished opening map files, start calculation of weighted quantiles '
              f'({time.time() - start_time:.1f} s)')
        ds_map = get_weighted_quantiles(ds_map, q_to_return=None)
        if 'volume' in variables:
            normalize_var(ds_map, 'volume')
        if 'area' in variables:
            normalize_var(ds_map, 'area')
        if 'thickness' in variables:
            add_thickness_unit(ds_map)
        if 'thinning_rate' in variables:
            add_thinning_rate_unit(ds_map)
        ds_map.to_netcdf(result_path)
        print(f'Finished calculation of weighted quantiles for map ({time.time() - start_time:.1f} s)')


# ## merge total data with quantiles

def merge_total_data_with_quantiles(files_for_merging_total_data,
                                    result_folder,
                                    target_name,
                                    variables,
                                    scenario,
                                    start_time,
                                    reset_files,
                                   ):

    result_path = os.path.join(
            result_folder,
            f'{target_name}_{scenario}_timeseries.nc'
        )
    if not reset_files:
        if os.path.exists(result_path):
            print('Merged timeseries file already exists!')
            return

    with xr.open_mfdataset(files_for_merging_total_data,
                           combine='by_coords',
                           parallel=False,
                           engine='netcdf4') as ds_total:
        print(f'Finished opening target files, start calculation of weighted quantiles '
              f'({time.time() - start_time:.1f} s)')
        ds_total = get_weighted_quantiles(ds_total, q_to_return=None)
        if 'volume' in variables:
            normalize_var(ds_total, 'volume')
        if 'area' in variables:
            normalize_var(ds_total, 'area')
        if 'thickness' in variables:
            add_thickness_unit(ds_total)
        if 'thinning_rate' in variables:
            add_thinning_rate_unit(ds_total)
        print('Start saving')
        ds_total.to_netcdf(result_path)
        print(f'Finished calculation of weighted quantiles for timeseries '
              f'({time.time() - start_time:.1f} s)')


# ## merge risk data with quantiles

def merge_risk_data_with_quantiles(files_for_merging_total_data,
                                   result_folder,
                                   target_name,
                                   risk_variables,
                                   risk_thresholds,
                                   scenario,
                                   start_time,
                                   reset_files,
                                  ):

    result_path = os.path.join(
            result_folder,
            f'{target_name}_{scenario}_unavoidable_risk.nc'
        )
    if not reset_files:
        if os.path.exists(result_path):
            print('Merged risk file already exists!')
            return

    with xr.open_mfdataset(files_for_merging_total_data,
                           combine='by_coords',
                           parallel=False,
                           engine='netcdf4') as ds_risk:
        print(f'Finished opening risk files, start calculation of risk '
              f'({time.time() - start_time:.1f} s)')

        # only keep variables needed for risk calculation
        for var in ds_risk.data_vars:
            if var not in risk_variables:
                ds_risk = ds_risk.drop_vars(var)

        # get quantiles for each gcm
        ds_risk = get_weighted_quantiles(ds_risk, q_to_return=[0.1, 0.3, 0.5, 0.7, 0.9],
                                         return_weighted_total_quantiles=False)
        if 'volume' in risk_variables:
            normalize_var(ds_risk, 'volume')
        if 'area' in risk_variables:
            normalize_var(ds_risk, 'area')
        if 'thickness' in risk_variables:
            add_thickness_unit(ds_risk)
            normalize_var(ds_risk, 'thickness')
        # calculate for each threshhold 
        ds_risk_threshold = []
        for threshold in risk_thresholds:
            ds_risk_threshold.append(
                (xr.where(ds_risk <= (100 - threshold), 1, 0
                         ).sum(dim=['gcm', 'quantile']) / 
                 (len(ds_risk['gcm']) * len(ds_risk['quantile']))
                ).expand_dims({'risk_threshold': [threshold]})
            )
        ds_risk_threshold = xr.merge(ds_risk_threshold)
        ds_risk_threshold.risk_threshold.attrs['unit'] = '% of 2020 total value'

        ds_risk_threshold.to_netcdf(result_path)
        print(f'Finished calculation of weighted quantiles for risk ({time.time() - start_time:.1f} s)')


# ## Main function

# ### only opening

def open_files_and_aggregate_on_map(
    target_name, target_structure_dict,
    scenario, output_folder,
    oggm_result_dir, raw_oggm_output_file,
    intermediate_data_folder=None,
    variables=['volume', 'area', 'thickness', 'thinning_rate'],
    time_steps=np.arange(2015, 2101, 5),
    gcm_test=None, quantile_test=None,
    reset_files=False
):
    start_time = time.time()
    print(f'Starting openening and aggregation on map for {target_name} and {scenario}')

    # create a folder for saving the results and inbetween computations
    result_folder = os.path.join(output_folder,
                                 target_name)
    mkdir(result_folder)

    if intermediate_data_folder is None:
        intermediate_data_folder = result_folder
    else:
        intermediate_data_folder = os.path.join(
            intermediate_data_folder,
            target_name)
        mkdir(intermediate_data_folder)
    
    map_data_folder = os.path.join(intermediate_data_folder,
                                   'map_data')
    mkdir(map_data_folder)

    total_data_folder = os.path.join(intermediate_data_folder,
                                       'total_data')
    mkdir(total_data_folder)

    # open grids for structure
    ds_grid_structure = open_grid_from_dict(target_structure_dict[target_name])

    # get all files for this target, only keep the once from current scenario
    all_files_target = get_all_files_from_result_batches(
        oggm_result_dir,
        raw_oggm_output_file,
        list(ds_grid_structure.result_batches.keys()))
    all_files_target = [file for file in all_files_target if scenario in file]

    # variable to open
    variables_to_open = [var for var in variables if var in ['volume', 'area']]

    # check if this is a test
    gcm_use = gcms_mesmer if gcm_test is None else gcm_test
    quantile_use = quantiles_mesmer if quantile_test is None else quantile_test

    # open files and merge for each gcm and quantile
    files_for_merging_map_data, files_for_merging_total_data = open_files_and_aggregate(
        gcm_use,
        quantile_use,
        all_files_target,
        scenario,
        start_time,
        ds_grid_structure,
        variables_to_open,
        variables,
        time_steps,
        target_name,
        map_data_folder,
        total_data_folder,
        reset_files,
    )

    print(f'Finished openening and aggregation on map for {target_name} and {scenario}')


# ### only aggregating scenario

def aggregating_scenario(
    target_name, target_structure_dict,
    scenario, output_folder,
    oggm_result_dir, raw_oggm_output_file,
    intermediate_data_folder=None,
    variables=['volume', 'area', 'thickness', 'thinning_rate'],
    risk_variables=['volume', 'area', 'thickness'],
    risk_thresholds=np.append(np.arange(10, 91, 10), [99]),  # in % melted of 2020, 10% means 10% of 2020 melted
    time_steps=np.arange(2015, 2101, 5),
    gcm_test=None, quantile_test=None,
    reset_files=False
):
    start_time = time.time()
    print(f'Starting aggregation scenario for {target_name} and {scenario}')

    # create a folder for saving the results and inbetween computations
    result_folder = os.path.join(output_folder,
                                 target_name)
    mkdir(result_folder)

    if intermediate_data_folder is None:
        intermediate_data_folder = result_folder
    else:
        intermediate_data_folder = os.path.join(
            intermediate_data_folder,
            target_name)
        mkdir(intermediate_data_folder)
    
    map_data_folder = os.path.join(intermediate_data_folder,
                                   'map_data')
    mkdir(map_data_folder)

    total_data_folder = os.path.join(intermediate_data_folder,
                                     'total_data')
    mkdir(total_data_folder)

    files_for_merging_map_data = glob.glob(
        os.path.join(
            map_data_folder,
            f'{target_name}_{scenario}_*_map_data.nc'
        )
    )

    files_for_merging_total_data = glob.glob(
        os.path.join(
            total_data_folder,
            f'{target_name}_{scenario}_*_total_data.nc'
        )
    )

    # now merge all gcm and quantiles into one file and save
    # also normalize volume and area here
    # map data
    merge_map_data_with_quantiles(files_for_merging_map_data,
                                  result_folder,
                                  target_name,
                                  variables,
                                  scenario,
                                  start_time,
                                  reset_files,
                                 )

    # total data
    merge_total_data_with_quantiles(files_for_merging_total_data,
                                    result_folder,
                                    target_name,
                                    variables,
                                    scenario,
                                    start_time,
                                    reset_files,
                                   )

    # risk data
    merge_risk_data_with_quantiles(files_for_merging_total_data,
                                   result_folder,
                                   target_name,
                                   risk_variables,
                                   risk_thresholds,
                                   scenario,
                                   start_time,
                                   reset_files,
                                  )

    print(f'Finished aggregation scenario for {target_name} and {scenario}')


# ### opening and aggregation at once

def aggregate_data_step_by_step(
    target_name, target_structure_dict,
    scenario, output_folder,
    oggm_result_dir, raw_oggm_output_file,
    intermediate_data_folder=None,
    variables=['volume', 'area', 'thickness', 'thinning_rate'],
    risk_variables=['volume', 'area', 'thickness'],
    risk_thresholds=np.append(np.arange(10, 91, 10), [99]),  # in % melted of 2020, 10% means 10% of 2020 melted
    time_steps=np.arange(2015, 2101, 5),
    gcm_test=None, quantile_test=None,
    reset_files=False
):
    start_time = time.time()
    print(f'Starting aggregation for {target_name} and {scenario}')

    # create a folder for saving the results and inbetween computations
    result_folder = os.path.join(output_folder,
                                 target_name)
    mkdir(result_folder)

    if intermediate_data_folder is None:
        intermediate_data_folder = result_folder
    else:
        intermediate_data_folder = os.path.join(
            intermediate_data_folder,
            target_name)
        mkdir(intermediate_data_folder)
    
    map_data_folder = os.path.join(intermediate_data_folder,
                                   'map_data')
    mkdir(map_data_folder)

    total_data_folder = os.path.join(intermediate_data_folder,
                                       'total_data')
    mkdir(total_data_folder)

    # open grids for structure
    ds_grid_structure = open_grid_from_dict(target_structure_dict[target_name])

    # get all files for this target, only keep the once from current scenario
    all_files_target = get_all_files_from_result_batches(
        oggm_result_dir,
        raw_oggm_output_file,
        list(ds_grid_structure.result_batches.keys()))
    all_files_target = [file for file in all_files_target if scenario in file]

    # variable to open
    variables_to_open = [var for var in variables if var in ['volume', 'area']]

    # check if this is a test
    gcm_use = gcms_mesmer if gcm_test is None else gcm_test
    quantile_use = quantiles_mesmer if quantile_test is None else quantile_test

    # open files and merge for each gcm and quantile
    files_for_merging_map_data, files_for_merging_total_data = open_files_and_aggregate(
        gcm_use,
        quantile_use,
        all_files_target,
        scenario,
        start_time,
        ds_grid_structure,
        variables_to_open,
        variables,
        time_steps,
        target_name,
        map_data_folder,
        total_data_folder,
        reset_files,
    )

    # now merge all gcm and quantiles into one file and save
    # also normalize volume and area here
    # map data
    merge_map_data_with_quantiles(files_for_merging_map_data,
                                  result_folder,
                                  target_name,
                                  variables,
                                  scenario,
                                  start_time,
                                  reset_files,
                                 )

    # total data
    merge_total_data_with_quantiles(files_for_merging_total_data,
                                    result_folder,
                                    target_name,
                                    variables,
                                    scenario,
                                    start_time,
                                    reset_files,
                                   )

    # risk data
    merge_risk_data_with_quantiles(files_for_merging_total_data,
                                   result_folder,
                                   target_name,
                                   risk_variables,
                                   risk_thresholds,
                                   scenario,
                                   start_time,
                                   reset_files,
                                  )


# # Check which slurm runs failed

def check_slurm_done(job_id):
    # Pattern to match the output files for the given job ID
    pattern = f"slurm-{job_id}_*.out"
    files = glob.glob(pattern)

    result_str = 'sbatch --array='
    
    if not files:
        print(f"No output files found for job ID {job_id}.")
        return

    files_without_done = []
    
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            # Check if 'SLURM DONE' is in the file content
            if 'SLURM DONE' not in content:
                result_str += f"{file.split('_')[-1].replace('.out', '')},"
                files_without_done.append(file)
    
    if files_without_done:
        print("Files without 'SLURM DONE':")
        print(result_str[:-1] + ' run_slurm_aggregation_workflow.sh')
        #for file in files_without_done:
        #    print(file)
    else:
        print("All files contain 'SLURM DONE'.")
