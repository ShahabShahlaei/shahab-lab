from common_imports import *
import os, numpy as np, xarray as xr
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.path as mpath
from cartopy.util import add_cyclic_point
from plot_siconc import _region_proj, _coords_2d, _to_percent, _savefig

def _set_polar_circle_boundary(ax, npts=720):
    theta = np.linspace(0, 2*np.pi, npts)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.cos(theta), np.sin(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

def _maybe_add_cyclic_from_da(da_mean, arr):
    """
    If data is on a regular 1-D lon grid, append a cyclic column to avoid a seam.
    Otherwise, return original lon2d/lat2d.
    """
    lon_name = None; lat_name = None
    for lo in ("lon","longitude"): 
        if lo in da_mean.dims and getattr(da_mean[lo], "ndim", 1) == 1: lon_name = lo; break
    for la in ("lat","latitude"):
        if la in da_mean.dims and getattr(da_mean[la], "ndim", 1) == 1: lat_name = la; break

    if lon_name and lat_name:
        lon = da_mean[lon_name].values
        lat = da_mean[lat_name].values
        arr_c, lon_c = add_cyclic_point(arr, coord=lon, axis=-1)
        lon2d, lat2d = np.meshgrid(lon_c, lat)
        return lon2d, lat2d, arr_c
    # fallback: use whatever grid we already built upstream
    lon2d, lat2d = _coords_2d(da_mean)
    return lon2d, lat2d, arr

def plot_siconc_panel_side_by_side(
    siconc_dict, *,
    region="Arctic",
    exp="piControl",
    models=None,                 # e.g., ["EC-Earth3-AerChem", "GISS-E2-1-H", "NorESM2-LM"]
    save_dir="plots_siconc",
    filename=None,
    cmap="Blues",
    vmin=0.0, vmax=100.0,
    dpi=300,
    smooth=True,
    add_ice_edge=True,
    view_lat_min=50.0,          # <<< how far down from pole to display
    mask_land=True,             # <<< draw land in white on top
    land_color="white",
    ocean_color="#eaf2ff"       # light ocean outside plotted field
):
    """
    One figure, multiple models for the same region/experiment, ONE shared colorbar.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Which models
    all_models = list(siconc_dict.keys()) if models is None else list(models)
    selected = []
    for m in all_models:
        da = siconc_dict.get(m, {}).get(exp, {}).get(region, None)
        if (da is not None) and not (np.isnan(da).all()):
            selected.append((m, da))
    if not selected:
        print(f"⚠️ No valid SICONC for region={region}, exp={exp}."); return

    ncols = len(selected)
    proj   = _region_proj(region)
    fig    = plt.figure(figsize=(4.8 * ncols, 5.8), dpi=dpi)

    axes, meshes = [], []
    shade_kw = dict(shading="gouraud") if smooth else dict(shading="auto")

    for i, (model, da) in enumerate(selected, start=1):
        ax = fig.add_subplot(1, ncols, i, projection=proj)
        axes.append(ax)
        ax.set_facecolor(ocean_color)

        # Camera/extent + circular boundary for polar views
        if region.lower() == "global":
            ax.set_global()
        elif region.lower() == "arctic":
            ax.set_extent([-180, 180, view_lat_min, 90], ccrs.PlateCarree())
            _set_polar_circle_boundary(ax)
            ax.patch.set_edgecolor(ax.get_facecolor())  # hide ring
        elif region.lower() == "antarctic":
            ax.set_extent([-180, 180, -90, -view_lat_min], ccrs.PlateCarree())
            _set_polar_circle_boundary(ax)
            ax.patch.set_edgecolor(ax.get_facecolor())

        # Time-mean and convert to %
        da_mean = da.mean("time") if "time" in da.dims else da
        arr_pct = np.ma.masked_invalid(_to_percent(da_mean).values)

        # Use cyclic lon for regular grids → no meridian seam
        lon2d, lat2d, arr2 = _maybe_add_cyclic_from_da(da_mean, arr_pct)

        # Plot
        mesh = ax.pcolormesh(
            lon2d, lat2d, arr2, cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), **shade_kw
        )
        # Clip to the circle in polar views
        if region.lower() in ("arctic","antarctic"):
            try: mesh.set_clip_path(ax.patch)
            except Exception: pass
        meshes.append(mesh)

        # 15% ice edge
        if add_ice_edge:
            try:
                ax.contour(lon2d, lat2d, arr2, levels=[15.0],
                           transform=ccrs.PlateCarree(), linewidths=0.8, colors="#5e2d79")
            except Exception:
                pass

        # Mask land in white (clean land/sea separation)
        if mask_land:
            land = cfeature.NaturalEarthFeature("physical","land","50m",
                                                facecolor=land_color, edgecolor="none")
            ax.add_feature(land, zorder=3)

        ax.coastlines(resolution="50m", linewidth=0.6, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, zorder=4)
        ax.set_title(f"{model} ({exp}) – {region}")

    # ONE shared colorbar
    cbar = fig.colorbar(meshes[0], ax=axes, orientation="horizontal",
                        pad=0.06, fraction=0.03, shrink=0.9)
    cbar.set_label("Sea-ice concentration (%)")

    # Save
    if filename is None:
        filename = f"SICONC_panel_{region}_{exp}.png".replace(" ", "_")
    outpath = os.path.join(save_dir, filename)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved panel: {outpath}")
