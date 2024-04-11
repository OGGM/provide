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

import xarray as xr
import numpy as np
import os
from shapely.ops import transform
import matplotlib.pyplot as plt


# # Map plot

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


def plot_map(target, target_name, scenario, input_dir, figsize=(12, 7),
             save_plot=False):
    # Function to adjust longitude values within a geometry
    def adjust_longitude(geometry):
        def _adjust_lon(lon, lat):
            """Adjust longitude values to 0° - 360° range."""
            return (lon + 360) % 360 if lon < 0 else lon, lat
        
        if geometry.is_empty:
            return geometry
        else:
            return transform(lambda x, y: _adjust_lon(x, y), geometry)

    target.loc[:, 'geometry'] = target['geometry'].apply(adjust_longitude)

    with xr.open_dataset(
        os.path.join(input_dir,
                     target_name,
                     f'{target_name}_{scenario}_map.nc')) as ds:
        # if we have negative lon we convert to 0-360
        ds['lon'] = xr.where(ds['lon'] < 0,
                             ds['lon'] + 360,
                             ds['lon'])
        ds = ds.sortby('lon')
        ds_plot = ds

    times_to_show = {0: 2020, 1: 2040, 2: 2060, 3: 2080}
    cbar_labels = {'volume': 'Volume in\n% of 2020 total',
                   'area': 'Area in\n% of 2020 total',
                   'thickness': 'Thickness in m w.e.',
                   'thinning_rate': 'Thinning Rate in\nm w.e. yr-1'}
    ref_values = {'volume': 'reference_2020_km3',
                  'area': 'reference_2020_km2'}
    ref_unit = {'volume': 'km3',
                'area': 'km2'}

    fig, axs = plt.subplots(len(ds_plot.data_vars), 4, figsize=figsize)

    for nr_var, var in enumerate(ds_plot.data_vars):
        cbar_label = cbar_labels[var]
        axs_var = axs[nr_var, :]

        vmin = ds_plot.loc[{'scenario': scenario,
                          'quantile': 0.5}][var].min()
        vmax = ds_plot.loc[{'scenario': scenario,
                          'quantile': 0.5}][var].max()
        
        for i, ax in enumerate(axs_var):
            (adjust_dataarray_for_plotting(
                ds_plot.loc[{'scenario': scenario,
                             'time': times_to_show[i],
                             'quantile': 0.5}][var],
                delta_lat=ds_plot.grid_resolution,
                delta_lon=ds_plot.grid_resolution)
            ).plot(
                ax=ax, vmin=vmin, vmax=vmax, cbar_kwargs={'label': cbar_label})
            target.plot(ax=ax, facecolor='none', edgecolor='black')
            title = f'{var}\n{scenario}, Year {times_to_show[i]}'
            if var in ['volume', 'area']:
                title += f'\nref. {ds[var].attrs[ref_values[var]]:.3f} {ref_unit[var]}'
            ax.set_title(title)
    fig.suptitle(f'{target_name}')
    fig.tight_layout()
    if save_plot:
        plt.savefig(
            os.path.join(save_plot,
                         f'{target_name}_{scenario}_map.png'),
            bbox_inches='tight',
        )
    else:
        plt.show()
    plt.close(fig)


# # Timeseries plot

def plot_timeseries(target_name, scenario, input_dir, figsize=(5, 9),
                    save_plot=False):
    with xr.open_dataset(
        os.path.join(input_dir,
                     target_name,
                     f'{target_name}_{scenario}_timeseries.nc')) as ds:
        ds_plot = ds

    variables = ['volume', 'area', 'thickness', 'thinning_rate']
    ylabels = {'volume': 'Volume in % of 2020 total',
               'area': 'Area in % of 2020 total',
               'thickness': 'Thickness in m w.e.',
               'thinning_rate': 'Thinning Rate in m w.e. yr-1'}
    ref_values = {'volume': 'reference_2020_km3',
                  'area': 'reference_2020_km2'}
    ref_unit = {'volume': 'km3',
                'area': 'km2'}

    fig, axs = plt.subplots(len(ds_plot.data_vars), 1, figsize=figsize)

    for nr_var, var in enumerate(ds_plot.data_vars):
        ylabel = ylabels[var]
        ax = axs[nr_var]
        
        (ds_plot.loc[{'scenario': scenario,
                      'quantile': 0.5}][var]).plot(ax=ax, label='quantile 0.5')
        (ds_plot.loc[{'scenario': scenario,
                      'quantile': 0.05}][var]).plot(ax=ax, label='quantile 0.05')
        (ds_plot.loc[{'scenario': scenario,
                      'quantile': 0.95}][var]).plot(ax=ax, label='quantile 0.95')
        
        ax.legend()
        title = f'{target_name}, {scenario},  {var}'
        if var in ['volume', 'area']:
            title += f', ref. {ds[var].attrs[ref_values[var]]:.3f} {ref_unit[var]}'
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    fig.tight_layout()

    if save_plot:
        plt.savefig(
            os.path.join(save_plot,
                         f'{target_name}_{scenario}_timeseries.png'),
            bbox_inches='tight',
        )
    else:
        plt.show()

    plt.close(fig)


# # Risk plot

def plot_unavoidable_risk(target_name, scenarios, input_dir, figsize=(4, 15),
                          save_plot=False):
    ds_plots = []
    colors = []
    for i, scenario in enumerate(scenarios):
        with xr.open_dataset(
            os.path.join(input_dir,
                         target_name,
                         f'{target_name}_{scenario}_unavoidable_risk.nc')) as ds:
            ds_plots.append(ds)
            colors.append(f'C{i}')

    middle_ax_nr = int(len(ds_plots[0].risk_threshold) / 2)
    
    for var in ds_plots[0].data_vars:
        fig, axs = plt.subplots(len(ds_plots[0].risk_threshold), 1, figsize=figsize,
                                gridspec_kw={'hspace': 0.8})
        for ax_nr, (ax, threshold) in enumerate(zip(axs, ds_plots[0].risk_threshold)):
            for color, ds_plot in zip(colors, ds_plots):
                ds_plot.loc[{'risk_threshold': threshold}][var].plot(
                    ax=ax,
                    linestyle='', marker='o', label=ds_plot.scenario.item(), color=color,
                    alpha=0.5
                )
            
            ax.set_ylabel('Probability of\nexceeding')
            ax.set_xlabel('')
            ax.set_title (f'{target_name}, less than {threshold.item()}% of 2020 {var} remaining')
            if ax_nr == middle_ax_nr:
                ax.legend(loc='center left',
                          bbox_to_anchor=(1.1, 0.5))
            ax.set_ylim([-0.05, 1.05])

        if save_plot:
            plt.savefig(
                os.path.join(save_plot,
                             f'{target_name}_{var}_unavoidable_risk.png'),
                bbox_inches='tight'
            )
        else:
            plt.show()

        plt.close(fig)


