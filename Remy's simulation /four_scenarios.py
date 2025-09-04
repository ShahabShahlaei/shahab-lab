import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from plot_helpers import _prepare_lon_lat, _circular_polar_boundary, _add_region_labels, _add_land_ocean


# Set font to Times New Roman globally
plt.rcParams['font.family'] = 'DejaVu Serif'  # Added to set font

def plot_four_scenarios(
        scenario_data: dict,
        model: str,
        region: str = "Arctic",
        variable_name: str = "EMIDMS",
        unit: str = "",
        cmap: str = "turbo",
        smooth: bool = False,
        dpi: int = 300,
        add_area_names: bool = True,
        save_dir: str = "figures",
        fname: str = None):
    """
    Plot 4 scenarios in a 2×2 layout with a common colour scale,
    masking data outside the selected region (Arctic/Antarctic).
    """
    n = len(scenario_data)
    if n != 4:
        raise ValueError("Exactly 4 scenarios are required for a 2×2 panel.")

    # Compute common colour range across all scenarios
    all_masked = []
    for da in scenario_data.values():
        masked = np.ma.masked_invalid(da.mean("time") if "time" in da.dims else da)
        all_masked.append(masked)
    data_concat = np.ma.concatenate([m.compressed() for m in all_masked])
    if np.nanmin(data_concat) >= 0:
        vmin, vmax = 0, np.nanmax(data_concat)  # unipolar
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmax = np.nanpercentile(np.abs(data_concat), 99)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Prepare figure and axes
    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": {
        "arctic":   ccrs.NorthPolarStereo(),
        "antarctic": ccrs.SouthPolarStereo(),
    }.get(region.lower(), ccrs.PlateCarree())},
                             figsize=(12, 12), dpi=dpi)
    axs = axes.ravel()

    for ax, (label, da) in zip(axs, scenario_data.items()):
        lon2d, lat2d = _prepare_lon_lat(da)
        data = da.mean("time") if "time" in da.dims else da
        masked = np.ma.masked_invalid(data)

        reg = region.lower()
        if reg == "arctic":
            # Mask latitudes south of 50° N (adjust threshold if needed)
            mask = lat2d < 66.0
            masked = np.ma.masked_where(mask, masked)
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
            _circular_polar_boundary(ax)
        elif reg == "antarctic":
            # Mask latitudes north of −50° S
            mask = lat2d > -50.0
            masked = np.ma.masked_where(mask, masked)
            ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
            _circular_polar_boundary(ax)
        else:
            ax.set_global()

        _add_land_ocean(ax, land_color="white", ocean_color="#D6ECFF")

        shade_kw = dict(shading="gouraud") if smooth else dict(shading="auto")
        mesh = ax.pcolormesh(
            lon2d, lat2d, masked,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            **shade_kw,
        )

        if add_area_names:
            _add_region_labels(ax, reg, transform=ccrs.PlateCarree())

        ax.gridlines(draw_labels=False, linewidth=0.4, color="gray",
                     alpha=0.35, linestyle="--")
        ax.set_title(f"{label}")

    fig.suptitle(f"{model} – {region} – {variable_name}", fontsize=14)

    # Create a standalone mappable for the shared colourbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Adjusted colorbar position to be further from subplots
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.01])  # Changed y0 from 0.05 to 0.03
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{variable_name} ({unit})")

    # Adjusted rect to accommodate lower colorbar
    # Adjust figure margins to add space below colorbar and maintain subplot spacing
    fig.subplots_adjust(bottom=0.12, hspace=0.2, wspace=0.2)  # Modified: increased bottom margin, set explicit subplot spacing

    import os
    os.makedirs(save_dir, exist_ok=True)
    if fname is None:
        fname = f"{model}_{region}_{variable_name}_4panel"
    fig.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=dpi)
    plt.close(fig)