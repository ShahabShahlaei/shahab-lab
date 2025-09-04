import os
from common_imports import *
def plot_data_map(data, model, region, variable_name="Variable", save_dir="pic", cmap='turbo', unit='', experiment=""):
    # Ensure data is a DataArray
    if isinstance(data, dict):
        raise ValueError("data should be a DataArray, not a dictionary")

    # Average over time if needed
    if "time" in data.dims:
        data = data.mean(dim="time")

    # Detect lat/lon
    if 'lat' in data.dims and 'lon' in data.dims:
        lat = data.lat.values
        lon = data.lon.values
    elif 'j' in data.dims and 'i' in data.dims:
        lat = data.latitude.values
        lon = data.longitude.values
    else:
        raise ValueError("Data does not have recognized lat/lon dimensions.")

    # Ensure meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Mask invalid data
    masked_data = np.ma.masked_invalid(data.values)

    # Set projection
    if region.lower() == "arctic":
        projection = ccrs.NorthPolarStereo(central_longitude=0.0)
    elif region.lower() == "antarctic":
        projection = ccrs.SouthPolarStereo(central_longitude=0.0)
    else:
        projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=projection)
    if region.lower() == "global":
        ax.set_global()

    # Auto colorbar type based on data
    if np.nanmin(masked_data) >= 0:
        mesh = ax.pcolormesh(lon_grid, lat_grid, masked_data, cmap=cmap, shading='auto', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.07, shrink=0.5)
    else:
        vmax = np.nanmax(np.abs(masked_data))
        vmin = -vmax
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        mesh = ax.pcolormesh(lon_grid, lat_grid, masked_data, cmap=cmap, shading='auto', transform=ccrs.PlateCarree(), norm=norm)
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.07, shrink=0.5, extend='both')

    cbar.set_label(f'{variable_name} ({unit})')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    if region.lower() == "global":
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    plt.title(f"{model} - {region} - {variable_name} ({experiment})")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model}_{region}_{variable_name}_{experiment}.png".replace(" ", "_")
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()



def plot_all_data_maps(data_dict, variable_name="ERF", save_dir="images", experiment="", **plot_kwargs):
    """
    Plot maps for all models and regions in the provided data dictionary, or a single dataset.
    """
    if isinstance(data_dict, dict):
        for model, experiments in data_dict.items():
            if not experiments:
                continue
            # Check if experiments level exists
            if 'control' in experiments or '2xDMS' in experiments:
                # Nested
                for experiment_name, regions in experiments.items():
                    for region, data in regions.items():
                        if data is None or (isinstance(data, xr.DataArray) and np.isnan(data).all()):
                            print(f"⚠ Skipping {model} - {experiment_name} - {region} (empty or NaN)")
                            continue
                        if data.ndim < 2:
                            print(f"⚠ Skipping {model} - {experiment_name} - {region} (Data has less than 2 dimensions)")
                            continue
                        print(f"Plotting data for {model} - {experiment_name} - {region}")
                        plot_data_map(
                            data=data,
                            model=model,
                            region=region,
                            variable_name=variable_name,
                            save_dir=save_dir,
                            experiment=experiment_name,
                            **plot_kwargs
                        )
            else:
                # Flat
                for region, data in experiments.items():
                    if data is None or (isinstance(data, xr.DataArray) and np.isnan(data).all()):
                        print(f"⚠ Skipping {model} - {region} (empty or NaN)")
                        continue
                    if data.ndim < 2:
                        print(f"⚠ Skipping {model} - {region} (Data has less than 2 dimensions)")
                        continue
                    print(f"Plotting data for {model} - {region}")
                    plot_data_map(
                        data=data,
                        model=model,
                        region=region,
                        variable_name=variable_name,
                        save_dir=save_dir,
                        experiment=experiment,
                        **plot_kwargs
                    )
    elif isinstance(data_dict, xr.DataArray):
        if np.isnan(data_dict).all():
            print("⚠ Skipping data (All values are NaN)")
            return
        if data_dict.ndim < 2:
            print("⚠ Skipping data (Data has less than 2 dimensions)")
            return
        plot_data_map(
            data=data_dict,
            model=experiment,
            region="global",
            variable_name=variable_name,
            save_dir=save_dir,
            experiment=experiment,
            **plot_kwargs
        )
    else:
        raise ValueError(f"Unexpected type for data_dict: {type(data_dict)}")
