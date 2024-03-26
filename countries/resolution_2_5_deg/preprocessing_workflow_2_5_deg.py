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
from oggm import utils
import time
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point

# +
# go up until we are in the project base directory
base_dir = os.getcwd()
while base_dir.split('/')[-1] != 'provide':
    base_dir = os.path.normpath(os.path.join(base_dir, '..'))

# add paths for tools and data
things_to_add = ['general_tools', 'aggregation_tools']
for thing in things_to_add:
    sys.path.append(os.path.join(base_dir, thing))

# import stuff we need
from general_tools import check_if_notebook
from aggregation_preprocessing import (get_global_grid, get_squared_grid_for_shape,
    _lonlat_grid_from_dataset, assign_rgi_ids_to_grid_points, plot_local_grid_with_glaciers,
    get_result_batches_of_glaciers, merge_result_structures, serialize_dataset,
    open_grid_from_dict)
# -

# Use this to conditionally execute tests/debugging
if check_if_notebook():
    is_notebook = True
else:
    is_notebook = False

# # Define directories

resolution_dir = 'resolution_2_5_deg'

preprocess_country_dict_outpath = os.path.join(base_dir, 'countries', resolution_dir)
utils.mkdir(preprocess_country_dict_outpath);

grid_plots_outpath = os.path.join(base_dir, 'countries', resolution_dir, 'grid_plots')
utils.mkdir(grid_plots_outpath);

aggregated_data_outpath = os.path.join(base_dir, 'countries', resolution_dir, 'aggregated_data')
utils.mkdir(aggregated_data_outpath);

countries_data_dir = os.path.join(base_dir, 'countries', 'data')
general_data_dir = os.path.join(base_dir, 'general_data_for_aggregation')

# # Open data

countries_file = 'ne_10m_admin_0_countries_deu_CA.shp'
gdf_countries = gpd.read_file(os.path.join(countries_data_dir, countries_file))
name_col_countries = 'ADM0_A3_DE'

# open dict_country_to_rgi_ids from data
with open(os.path.join(countries_data_dir,
                       "dict_country_to_rgi_ids.json"), "r") as f:
    dict_country_to_rgi_ids = json.load(f)

# this list is already cleaned from connectivity 2, RGI19 and only common running glaciers
fp_rgi_result_batch = "rgi_ids_to_result_batch.json"
with open(os.path.join(general_data_dir, fp_rgi_result_batch), 'r') as f:
    dict_rgis_to_batch = json.load(f)

df_rgi = pd.read_csv(os.path.join(general_data_dir, 'rgi_position_terminus_backdrop_centroid.csv'),
                     index_col=0)
# only keep rgi_ids which are assigned to a result file
# (excluding connectivity 2 and RGI19, in future also only commonly running glaciers)
df_rgi = df_rgi.loc[list(dict_rgis_to_batch.keys())]
df_rgi['geometry'] = df_rgi.apply(lambda row: Point(row['position_lon'], row['position_lat']), axis=1)
gdf_rgi_ids = gpd.GeoDataFrame(df_rgi, geometry='geometry', crs='EPSG:4326')

# # Settings for this run

resolution = 2.5
save_local_grid_plots = True

# # Workflow for testing in notebook

# ## define test country

if is_notebook:
    test_country_index = gdf_countries[gdf_countries['NAME'] == 'Austria'].index[0]
    test_country = gdf_countries.iloc[test_country_index: test_country_index + 1]

# ## get raw grid for country

if is_notebook:
    test_ds_global_grid = get_global_grid(resolution)
    test_ds_local = get_squared_grid_for_shape(test_ds_global_grid, test_country)

    # small test plot
    fig, ax = plt.subplots(1, 1)
    
    for test_ds_loc in test_ds_local:
        _lonlat_grid_from_dataset(test_ds_loc).to_geometry().plot(ax=ax,
                                             edgecolor='yellow',
                                             facecolor='green',
                                             lw=2)
    
    test_country.plot(ax=ax, color='blue')

# ## assign rgi_ids to grid points

if is_notebook:
    assign_rgi_ids_to_grid_points(test_ds_local,
                                  test_country[name_col_countries].values.item(),
                                  dict_country_to_rgi_ids,
                                  gdf_rgi_ids)
    plot_local_grid_with_glaciers(gdf_rgi_ids,
                                  test_country, test_ds_local,
                                  title=f'{test_country[name_col_countries].values[0]}, resolution '
                                  f'{test_ds_local[0].resolution}°', outpath=None)

# ## add files which needed to be opened for aggregation

if is_notebook:
    get_result_batches_of_glaciers(test_ds_local, dict_rgis_to_batch)

    for i, grid_loc in enumerate(test_ds_local):
        print(f'{i}. grid files: {list(grid_loc.result_batches.keys())}')

# Figure for debugging if provide regions where correctly selected
# ![provide_regions_overview.png](attachment:ce462378-d352-4cf2-a5f8-2a399f3d14ca.png)

# ## merge multiple files to one

# can be tested using Russia
if is_notebook:
    if len(test_ds_local) > 1:
        test_ds_local = merge_result_structures(test_ds_local)
    else:
        test_ds_local = test_ds_local[0]

# ## converting final structure to dictionary for saving and reconverting it to ds again

if is_notebook:
    save_sample_data = False  # for testing the saving
    test_ds_local_dict = serialize_dataset(test_ds_local).to_dict()
    test_key = test_country[name_col_countries].values[0]
    final_dict = {test_key: test_ds_local_dict}

    if save_sample_data:
        with open("prepocessed_country_grids_sample_data.json", "w") as outfile: 
            json.dump(final_dict, outfile)

    # check that we can open the final data
    local_grid_reloaded = open_grid_from_dict(final_dict[test_key])
    print('Original ds:')
    print(test_ds_local)
    print('\nReopened ds:')
    print(local_grid_reloaded)

# # Workflow for running on cluster

# here is the actual code which is running on the cluster for all target shapes
# could use multiprocessing here if it takes too long, but lets see first
# (only takess 10 min for all countries -> multiprocessing not needed)
if not is_notebook:

    start_time = time.time()
    final_dict = {}
    countries_with_no_glaciers = []

    ds_global_grid_resolution = get_global_grid(resolution)

    # loop through all countries and add local grid to final dict
    nr_of_countries = len(gdf_countries)
    for index, row in gdf_countries.iterrows():
        print(f'Starting country number {index + 1} of {nr_of_countries}.')
        country = gpd.GeoDataFrame([row], crs=gdf_countries.crs)

        ds_local_grid = get_squared_grid_for_shape(ds_global_grid_resolution,
                                                   country)

        for grid in ds_local_grid:
            grid.attrs['country'] = country[name_col_countries].values[0]
        assign_rgi_ids_to_grid_points(ds_local_grid,
                                      country[name_col_countries].values.item(),
                                      dict_country_to_rgi_ids,
                                      gdf_rgi_ids)
        if save_local_grid_plots:
            plot_local_grid_with_glaciers(
                gdf_rgi_ids, country, ds_local_grid,
                title=f'{country[name_col_countries].values[0]}, '
                f'{ds_local_grid[0].resolution} °',
                outpath=os.path.join(grid_plots_outpath,
                                     f'{country[name_col_countries].values[0]}.png'))
        get_result_batches_of_glaciers(ds_local_grid, dict_rgis_to_batch)
        if len(ds_local_grid) > 1:
            ds_local_grid = merge_result_structures(ds_local_grid)
        else:
            ds_local_grid = ds_local_grid[0]

        if ds_local_grid.grid_points_with_data == 0:
            # ok this is a country with no glaciers
            countries_with_no_glaciers.append(country[name_col_countries].values[0])
        else:
            # we have some glaciers so let's add
            ds_local_grid_dict = serialize_dataset(ds_local_grid).to_dict()
            final_dict[country[name_col_countries].values[0]] = ds_local_grid_dict

        print(f'Finished country number {index + 1} of {nr_of_countries}.')

    # save final preprocessed file in input dir
    with open(os.path.join(preprocess_country_dict_outpath,
                           "preprocessed_country_grids.json"), "w") as outfile: 
        json.dump(final_dict, outfile)

    # save list of countries with no glaciers
    with open(os.path.join(aggregated_data_outpath,
                           "countries_with_no_glaciers.json"), "w") as outfile: 
        json.dump(countries_with_no_glaciers, outfile)

    print(f'Time needed for {resolution}°: {time.time() - start_time:.1f} s\n\n')

# # Tests after running on cluster

# ## check for missing glaciers

if is_notebook:
    final_filepath = os.path.join(preprocess_country_dict_outpath,
                                  "preprocessed_country_grids.json")

    with open(final_filepath, 'r') as f:
        dict_final_grids = json.load(f)

    # check that all rgi_ids are assigned to a geometry
    def flatten_ds_var(ds_var):
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

    assigned_rgi_ids = []
    for key in dict_final_grids:
        try:
            ds_tmp = open_grid_from_dict(dict_final_grids[key])
            assigned_rgi_ids.extend(flatten_ds_var(ds_tmp.rgi_ids))
        except ValueError:
            print(key)

    given_rgi_ids = list(dict_rgis_to_batch.keys())

    assert len(given_rgi_ids) == len(np.unique(given_rgi_ids))
    assert len(assigned_rgi_ids) == len(np.unique(assigned_rgi_ids))
    print(f'Given rgi_ids: {len(given_rgi_ids)}')
    print(f'Assigned rgi_ids: {len(assigned_rgi_ids)}')
    print(f'Diff: {len(given_rgi_ids) - len(assigned_rgi_ids)}')
    
    # plot missing glaciers on map
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_countries.plot(ax=ax, color='blue', edgecolor='black', linewidth=1,
                       alpha=0.5)
    gdf_missing_rgi_ids = gdf_rgi_ids[~gdf_rgi_ids.index.isin(assigned_rgi_ids)]
    if len(gdf_missing_rgi_ids) != 0:
        gdf_missing_rgi_ids.plot(ax=ax, color='red', markersize=2)
    ax.set_title(f'Glaciers not attributed to a country '
                 f'(Missing nr. {len(given_rgi_ids) - len(assigned_rgi_ids)})') 
    plt.show()

# ## check for uniqueness of rgi_ids in batches

if is_notebook:
    # check uniqueness of rgi_ids in batches
    for key in dict_final_grids:
        ds_tmp = open_grid_from_dict(dict_final_grids[key])
        raw_rgi_id_batch_list = []
        for batch in ds_tmp.result_batches:
            raw_rgi_id_batch_list.extend(ds_tmp.result_batches[batch])

        assert len(raw_rgi_id_batch_list) == len(np.unique(raw_rgi_id_batch_list))

# ## check that countries with glaciers and countries with no glaciers add up to total number

if is_notebook:
    countries_with_glaciers_filepath = os.path.join(
        preprocess_country_dict_outpath,"preprocessed_country_grids.json"
    )
    with open(countries_with_glaciers_filepath, 'r') as f:
        countries_with_glaciers = json.load(f)

    countries_with_no_glaciers_filepath = os.path.join(
        aggregated_data_outpath, "countries_with_no_glaciers.json"
    )
    with open(countries_with_no_glaciers_filepath, 'r') as f:
        countries_with_no_glaciers = json.load(f)

    total_nr_of_countries = len(gdf_countries)
    assigned_countries = (countries_with_no_glaciers + 
                          list(countries_with_glaciers.keys()))

    assert len(np.unique(assigned_countries)) == total_nr_of_countries

    print(f'Countries with glaciers: {len(list(countries_with_glaciers.keys()))}')
    print(f'Countries without glaciers: {len(countries_with_no_glaciers)}')
