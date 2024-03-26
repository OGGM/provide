import geopandas as gpd
import numpy as np
import pandas as pd
from salem import utils as salem_utils
from salem import gis as salem_gis
from salem import wgs84
import xarray as xr
import matplotlib.pyplot as plt
import json


def assign_rgi_ids_to_target_shapes(gdf_rgi_ids, target_shapes, name_col_target, do_plot=False):
    """ Assign rgi_is to target shapes for more efficient aggregation of data.
    
    Takes geopandas dataframe gdf_rgi_ids and assigns all containing glaciers to the provided
    target_shapes. This should only consists of common running glaciers. 'name_col_target'
    should be a unique column in target_shapes and is used as the identifier for the resulting
    dictionary.

    In a first round all rgi_ids which are intersecting with a target_shape are assigned. In
    a second round the rest is assigned depending on the shortest distance to the target shapes.
    If setting do_plot to True a plot for these glaciers assinged in the second round is shown.
    """
    # first assign all rgi_ids which are in target shapes
    sjoin_country_rgi = gpd.sjoin(gdf_rgi_ids,
                                  target_shapes.to_crs(gdf_rgi_ids.crs),
                                  how='inner', predicate='intersects')
    dict_target_to_rgis = sjoin_country_rgi.groupby(
        name_col_target).apply(lambda x: list(x.index)).to_dict()

    # now check if we have some missing rgi ids outside of target shapes
    missing_rgi_ids = gdf_rgi_ids.index.difference(sjoin_country_rgi.index)

    if len(missing_rgi_ids) == 0:
        # nothing missing, we are done
        return dict_target_to_rgis
    else:
        # ok we need to assign missing to closest target geometries
        # for this we use a projection which preserves distances
        gdf_missing_rgi_ids = gdf_rgi_ids.loc[missing_rgi_ids].to_crs(epsg=4087)
        target_shapes = target_shapes.to_crs(epsg=4087)

        def find_closest_polygon_name(point, polygons, name_col):
            distances = polygons.geometry.distance(point)
            closest_polygon_idx = distances.idxmin()
            return polygons.loc[closest_polygon_idx, name_col]

        # Apply the function to each point
        gdf_missing_rgi_ids['closest_target_name'] = gdf_missing_rgi_ids.geometry.apply(
            find_closest_polygon_name, args=(target_shapes, name_col_target))

        if do_plot:
            gdf_missing_rgi_ids.plot(column='closest_target_name', legend=True)

        # add to existing dictionary
        dict_missing_rgi_ids = gdf_missing_rgi_ids.groupby('closest_target_name').apply(
            lambda x: list(x.index)).to_dict()
        for target, rgi_ids in dict_missing_rgi_ids.items():
            if target in dict_target_to_rgis:
                dict_target_to_rgis[target].extend(rgi_ids)
            else:
                dict_target_to_rgis[target] = rgi_ids

        # test if everything is assigned
        assigned_rgi_ids = []
        for target, rgi_ids in dict_target_to_rgis.items():
            assigned_rgi_ids.extend(rgi_ids)
        assert len(assigned_rgi_ids) == len(np.unique(assigned_rgi_ids))
        assert len(assigned_rgi_ids) == len(gdf_rgi_ids)

        return dict_target_to_rgis


def generate_global_coordinates(resolution):
    # Check if resolution divides into 360 and 180 without remainder
    if 360 % resolution != 0 or 180 % resolution != 0:
        raise InvalidParameterError(
            f"Resolution {resolution}° does not allow for even coverage!")
    
    # Generate longitude and latitude coordinates
    longitudes = [i - resolution / 2 for i in np.arange(-180 + resolution, 181, resolution)]
    latitudes = [i - resolution / 2 for i in np.arange(-90 + resolution, 91, resolution)]

    return latitudes, longitudes


def get_global_grid(resolution):
    """ Creates a global grid with the given reolution and returns it as a xr.Dataset.
    """
    latitudes, longitudes = generate_global_coordinates(resolution)
    ds = xr.Dataset(coords={'lat': latitudes,
                            'lon': longitudes},
                    attrs={'resolution': resolution})
    return ds


def _lonlat_grid_from_dataset(ds):
    """Adapted from salem.sio._lonlat_grid_from_dataset
    Seek for longitude and latitude coordinates.
    Also works for sinlge point, where resolution is extracted from ds.
    """

    # Do we have some standard names as variable?
    vns = ds.variables.keys()
    xc = salem_utils.str_in_list(vns, salem_utils.valid_names['x_dim'])
    yc = salem_utils.str_in_list(vns, salem_utils.valid_names['y_dim'])

    # Sometimes there are more than one coordinates, one of which might have
    # more dims (e.g. lons in WRF files): take the first one with ndim = 1:
    x = None
    for xp in xc:
        if len(ds.variables[xp].shape) == 1:
            x = xp
    y = None
    for yp in yc:
        if len(ds.variables[yp].shape) == 1:
            y = yp
    if (x is None) or (y is None):
        return None

    # OK, get it
    lon = ds.variables[x][:]
    lat = ds.variables[y][:]

    # double check for dubious variables
    if not salem_utils.str_in_list([x], salem_utils.valid_names['lon_var']) or \
            not salem_utils.str_in_list([y], salem_utils.valid_names['lat_var']):
        # name not usual. see if at least the range follows some conv
        if (np.max(np.abs(lon)) > 360.1) or (np.max(np.abs(lat)) > 90.1):
            return None

    # Make the grid
    try:
        dx = lon[1]-lon[0]
        dy = lat[1]-lat[0]
    except IndexError:
        dx = ds.resolution
        dy = ds.resolution
    args = dict(nxny=(lon.shape[0], lat.shape[0]), proj=wgs84, dxdy=(dx, dy),
                x0y0=(lon[0], lat[0]))
    return salem_gis.Grid(**args)


def get_squared_grid_for_shape(ref_grid, target_shape):
    # This function also deals with geometries crossing +/-180° line,
    # for this case two squared grids are defined

    # convert reference grid into geometry for spatial joint
    gdf_ref_grid = _lonlat_grid_from_dataset(ref_grid).to_geometry()

    # here only get grid points which intersects with target shape
    selected_grid = gpd.sjoin(gdf_ref_grid,
                              target_shape.to_crs(gdf_ref_grid.crs),
                              how='inner', predicate='intersects')

    def get_ds_from_coordinate_limits(lat_min, lat_max,
                                      lon_min, lon_max,
                                      resolution):
        lat_local = np.arange(lat_min, lat_max + resolution, resolution)
        lon_local = np.arange(lon_min, lon_max + resolution, resolution)
        
        return xr.Dataset(coords={'lat': lat_local,
                                  'lon': lon_local},
                          attrs={'resolution': resolution})

    if len(ref_grid.lon) - 1  == selected_grid.i.max() and 0 == selected_grid.i.min():
        # special case if geomtery crosses -/+180° longitude, we create two grids
        # we take the largest gap between lon indices as position where to split
        selected_lon_grid_points = np.unique(selected_grid.i)
        selected_lon_grid_gap_start = np.argmax(np.diff(selected_lon_grid_points))
        selected_lon_grid_gap_end = selected_lon_grid_gap_start + 1
        lon_index_of_max_gap_start = selected_lon_grid_points[selected_lon_grid_gap_start]
        lon_index_of_max_gap_end = selected_lon_grid_points[selected_lon_grid_gap_end]
        
        # create two grids, one from 0 - splitting_index,
        # and one from splitting_index - last_index
        lat_max = ref_grid.lat.values[selected_grid.j.max()]
        lat_min = ref_grid.lat.values[selected_grid.j.min()]
        ds_local = []
        for i_min, i_max in [(0, lon_index_of_max_gap_start),
                             (lon_index_of_max_gap_end, selected_grid.i.max())]:
            lon_min = ref_grid.lon.values[i_min]
            lon_max = ref_grid.lon.values[i_max]
            ds_local.append(get_ds_from_coordinate_limits(lat_min, lat_max,
                                                          lon_min, lon_max,
                                                          ref_grid.resolution))

    else:
        lat_max = ref_grid.lat.values[selected_grid.j.max()]
        lat_min = ref_grid.lat.values[selected_grid.j.min()]
        lon_max = ref_grid.lon.values[selected_grid.i.max()]
        lon_min = ref_grid.lon.values[selected_grid.i.min()]
    
        ds_local = [get_ds_from_coordinate_limits(lat_min, lat_max,
                                                  lon_min, lon_max,
                                                  ref_grid.resolution)]

    return ds_local


def assign_rgi_ids_to_grid_points(ds_grids, target_shape_name,
                                  dict_target_to_rgi_ids, gdf_rgi_ids):

    # first check if their are glaciers in target shape
    if target_shape_name not in dict_target_to_rgi_ids:
        for ds_grid in ds_grids:
            ds_grid['rgi_ids'] = (('lat', 'lon'),
                                  np.full((ds_grid.dims['lat'],
                                          ds_grid.dims['lon']),
                                          None))
            ds_grid.attrs['grid_points_with_data'] = 0
        return None

    # select only rgi_ids which lies inside of target_shape
    selected_rgi_ids = gdf_rgi_ids.loc[dict_target_to_rgi_ids[target_shape_name]]

    for ds_grid in ds_grids:
        # get list of rgi_ids for each grid point
        gdf_grid = _lonlat_grid_from_dataset(ds_grid).to_geometry()
        rgi_ids_per_grid_point = gpd.sjoin(gdf_grid,
                                           selected_rgi_ids.to_crs(gdf_grid.crs),
                                           how='inner', predicate='intersects')

        # identify duplicates and just keep first (if glacier center lies on grid boundary)
        duplicates = rgi_ids_per_grid_point['index_right'].value_counts()
        duplicates = duplicates[duplicates > 1].index.tolist()
        if len(duplicates) > 0:
            dup_assignments = rgi_ids_per_grid_point[rgi_ids_per_grid_point['index_right'].isin(duplicates)]
            resolved_duplicates = dup_assignments.sort_values(
                by=['j', 'i', 'index_right']).drop_duplicates(subset=['index_right'], keep='first')
        
            # take away duplicates and add resolved version
            rgi_ids_per_grid_point = rgi_ids_per_grid_point[~rgi_ids_per_grid_point['index_right'].isin(duplicates)]
            rgi_ids_per_grid_point = pd.concat([rgi_ids_per_grid_point, resolved_duplicates])

        rgi_ids_per_grid_point = rgi_ids_per_grid_point.groupby(['j', 'i']).apply(
            lambda x: list(x['index_right']))

        if len(rgi_ids_per_grid_point) == 0:
            ds_grid['rgi_ids'] = (('lat', 'lon'),
                                  np.full((ds_grid.dims['lat'],
                                          ds_grid.dims['lon']),
                                          None))
        else:
            # create new variable in ds_grid containing rgi_id-lists per grid point
            # Extract coordinate mappings from grid
            lon_coord_map = pd.Series(ds_grid.lon.values,
                                      index=pd.RangeIndex(start=0,
                                                          stop=len(ds_grid.lon.values)))
            lat_coord_map = pd.Series(ds_grid.lat.values,
                                      index=pd.RangeIndex(start=0,
                                                          stop=len(ds_grid.lat.values)))
            # Use map to convert index positions to coordinate values
            df_rgi_ids = pd.DataFrame(index=rgi_ids_per_grid_point.index)
            df_rgi_ids['lat'] = df_rgi_ids.index.get_level_values(0).map(lat_coord_map)
            df_rgi_ids['lon'] = df_rgi_ids.index.get_level_values(1).map(lon_coord_map)
            df_rgi_ids['rgi_ids'] = rgi_ids_per_grid_point.values
            # Could also convert it to np.array, but keep pyhton list for saving as json later
            #df['rgi_ids'] = df['rgi_ids'].apply(lambda x: np.array(x, dtype=object))
            data_array_rgi_ids = df_rgi_ids.set_index(
                ['lat', 'lon']).to_xarray().fillna(None)['rgi_ids']
            # Ensure alignment; missing values will be automatically filled with None
            # Convert to native list, for later saving as dictionary
            ds_grid['rgi_ids'] = data_array_rgi_ids.reindex_like(
                                      ds_grid,
                                      method='nearest',
                                      tolerance=1e-2,
                                      fill_value=None)
            
        # add number of not None values (maybe useful to see which resolution we can use for the dashboard)
        ds_grid.attrs['grid_points_with_data'] = ds_grid['rgi_ids'].notnull().sum().item()


def plot_local_grid_with_glaciers(gdf_rgi_ids,
                                  country, ds_local, title=None, outpath=None):
    selected_rgi_ids = gpd.sjoin(gdf_rgi_ids,
                                 country.to_crs(gdf_rgi_ids.crs),
                                 how='inner', predicate='intersects')
    selected_rgi_ids = gpd.GeoDataFrame(selected_rgi_ids.geometry,
                                        index=selected_rgi_ids.index)
    if np.isclose(len(selected_rgi_ids), 0):
        return None

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    country.plot(ax=ax, alpha=0.5)
    lon_min = None
    lon_max = None
    #ds_local = [xr.merge(ds_local)]
    for local_grid in ds_local:
        if lon_min is None:
            lon_min = local_grid.lon.min().item()
        else:
            if lon_min > local_grid.lon.min().item():
                lon_min = local_grid.lon.min().item()

        if lon_max is None:
            lon_max = local_grid.lon.max().item()
        else:
            if lon_max < local_grid.lon.max().item():
                lon_max = local_grid.lon.max().item()

        list_length_da = xr.apply_ufunc(lambda x: float(len(x)) if isinstance(x, list) else np.nan,
                                    local_grid.rgi_ids,
                                    vectorize=True)

        # if only one lat or lon coordinate we add one artifically for plotting
        def adjust_dataarray_for_plotting(da, delta_lat=1, delta_lon=1):
            for coord, delta in [('lat', delta_lat), ('lon', delta_lon)]:
                if coord in da.coords and da.coords[coord].size == 1:  # Check if only one value exists for the coordinate
                    # Create new coordinate values
                    new_coord_value = da[coord].values[0] + delta
                    
                    # Prepare new coordinates for the DataArray
                    new_coords = {c: da[c] for c in da.coords if c != coord}
                    new_coords[coord] = ([coord], [new_coord_value])  # Update the adjusted coordinate
                    
                    # Prepare a DataArray filled with NaNs, keeping other dimensions intact
                    new_shape = [len(da[c].values) for c in da.dims if c != coord] + [1]
                    new_data = np.full(shape=new_shape, fill_value=np.nan)
                    new_da = xr.DataArray(new_data, dims=[c for c in da.dims if c != coord] + [coord], coords=new_coords)
                    
                    # Concatenate along the adjusted dimension
                    da = xr.concat([da, new_da], dim=coord)
                    da = da.sortby(coord)  # Sort by the adjusted coordinate to maintain order
        
            return da

        list_length_da = adjust_dataarray_for_plotting(
            list_length_da,
            delta_lat=local_grid.resolution,
            delta_lon=local_grid.resolution)

        try:
            list_length_da.plot(ax=ax, alpha=0.8,
                                cbar_kwargs={'label': 'Number of glaciers'})
        except TypeError:
            # if their is nothing to plot just continue
            pass
    try:
        selected_rgi_ids.plot(ax=ax, color='red', markersize=0.5)
    except ValueError:
        # if no rgi_ids selected just continue
        pass
    if title is not None:
        plt.gca().set_title(title)
    plt.gca().set_xlim([lon_min, lon_max])

    if outpath is not None:
        plt.savefig(outpath,
                    bbox_inches='tight',
                   )


def get_result_batches_of_glaciers(ds_grids, dict_rgis_to_batch):
    for ds_grid in ds_grids:
        flattened_rgi_ids = pd.Series(ds_grid.rgi_ids.values.ravel()
                                     ).dropna().explode().reset_index(drop=True)
        
        # Use vectorized operations to split rgi_ids and map to provide regions
        missing_value = 'None'
        result_batches = flattened_rgi_ids.map(
            lambda rgi_id: dict_rgis_to_batch.get(rgi_id, missing_value))

        df_rgi_id_to_file_batch = pd.DataFrame(
            {'rgi_id': flattened_rgi_ids,
             'file_batch': result_batches})

        result_batches = df_rgi_id_to_file_batch.groupby(
            'file_batch', group_keys=False)['rgi_id'].apply(list).to_dict()

        ds_grid.attrs['result_batches'] = result_batches


def merge_result_structures(result_structures):
    structures_merged = xr.merge(result_structures, fill_value=None)

    # also merge attributes
    combined_result_batches = {}
    combined_grid_points_with_data = 0
    for single_struct in result_structures:
        single_result_batches = single_struct.result_batches
        for batch in single_result_batches:
            if batch in combined_result_batches:
                combined_result_batches[batch].extend(single_result_batches[batch])
            else:
                combined_result_batches[batch] = single_result_batches[batch]
        combined_grid_points_with_data += single_struct.grid_points_with_data
    structures_merged.attrs['result_batches'] = combined_result_batches
    structures_merged.attrs['grid_points_with_data'] = combined_grid_points_with_data
    return structures_merged


def serialize_dataset(ds):
    """Serializes all variables in the xarray dataset to JSON strings."""
    serialized_ds = ds.copy()
    for var_name in ds.data_vars:
        # Apply serialization to each element
        serialized_data = np.vectorize(json.dumps)(ds[var_name].values)
        serialized_ds[var_name] = (ds[var_name].dims, serialized_data)
    return serialized_ds


# the function for deserialization after opening
def deserialize_dataset(ds):
    """Deserializes all variables in the xarray dataset from JSON strings, handling sequences."""
    deserialized_ds = ds.copy()
    for var_name in ds.data_vars:
        # Deserialize data into a list of lists to handle varying data structures
        deserialized_data = [json.loads(item) for item in np.ravel(ds[var_name].values)]
        
        # Determine if the deserialized data can be represented as a uniform NumPy array
        try:
            # Attempt to create a NumPy array, which will work for uniform data structures
            deserialized_array = np.array(deserialized_data, dtype=object).reshape(ds[var_name].shape)
        except ValueError:
            # Fallback to a list of lists for non-uniform data structures
            deserialized_array = deserialized_data
            
        deserialized_ds[var_name] = (ds[var_name].dims, deserialized_array)
    return deserialized_ds


def open_grid_from_dict(grid):
    return deserialize_dataset(xr.Dataset.from_dict(grid))
