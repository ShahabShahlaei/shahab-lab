# panel_erf_2x2.py
# Four Arctic ERF maps in a 2×2 layout with ONE shared colorbar.
# Style: DejaVu, circular polar, land masked (white), ocean light blue.

import os
import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.path as mpath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# ---- Global style ----
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
})

# ---- helpers ----
def _find_lat_lon_names(da):
    cand_lat = ["lat","latitude","nav_lat","j","y"]
    cand_lon = ["lon","longitude","nav_lon","i","x"]
    la = next((n for n in cand_lat if (n in da.dims or n in da.coords)), None)
    lo = next((n for n in cand_lon if (n in da.dims or n in da.coords)), None)
    if la is None or lo is None:
        raise ValueError(f"lat/lon not found on {da.name or 'DataArray'}; dims={list(da.dims)} coords={list(da.coords)}")
    return la, lo

def _lonlat_2d(da):
    la, lo = _find_lat_lon_names(da)
    latc = da.coords.get(la); lonc = da.coords.get(lo)
    # 2-D curvilinear
    if getattr(latc, "ndim", 1) == 2 and getattr(lonc, "ndim", 1) == 2:
        return lonc.values, latc.values, "curvi"
    # 1-D regular
    if (la in da.dims) and (lo in da.dims):
        lat1d = latc.values; lon1d = lonc.values
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d, "regular"
    # last resort (coords exist but not dims)
    if latc.ndim == 1 and lonc.ndim == 1:
        lon2d, lat2d = np.meshgrid(lonc.values, latc.values)
        return lon2d, lat2d, "regular"
    raise ValueError("Cannot build 2-D lon/lat for plotting.")

def _maybe_add_cyclic(da, arr):
    """For 1-D lon/lat grids, add a cyclic column to avoid the 0/360° seam."""
    try:
        la, lo = _find_lat_lon_names(da)
        if la in da.dims and lo in da.dims and da.coords[lo].ndim == 1:
            lon = da.coords[lo].values
            lat = da.coords[la].values
            arr_c, lon_c = add_cyclic_point(arr, coord=lon, axis=-1)
            lon2d, lat2d = np.meshgrid(lon_c, lat)
            return lon2d, lat2d, arr_c
    except Exception:
        pass
    lon2d, lat2d, _ = _lonlat_2d(da)
    return lon2d, lat2d, arr

def _proj_for_region(region):
    r = (region or "").lower()
    if r == "arctic":
        return ccrs.NorthPolarStereo(central_longitude=0.0)
    if r == "antarctic":
        return ccrs.SouthPolarStereo(central_longitude=0.0)
    return ccrs.PlateCarree()

def _set_polar_circle_boundary(ax, npts=720):
    """Circular boundary for polar panels (removes square frame & corners)."""
    theta = np.linspace(0, 2*np.pi, npts)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.cos(theta), np.sin(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

def _format_cbar(cb):
    cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()

def _robust_limits(arrays, symmetric=True, q=98.0):
    """Robust, optionally symmetric (±) limits across multiple arrays."""
    vec = []
    for a in arrays:
        v = np.asarray(a).ravel()
        v = v[np.isfinite(v)]
        if v.size: vec.append(v)
    if not vec: return None, None
    allv = np.concatenate(vec)
    if symmetric:
        rad = np.nanpercentile(np.abs(allv), q)
        rad = float(max(rad, 1e-20))
        return -rad, rad
    return np.nanpercentile(allv, 100-q), np.nanpercentile(allv, q)

def _add_ice_edge(ax, siconc_da, level=15.0):
    """Optional: overlay the 15% SICONC edge (MIZ) if a DA is provided."""
    if siconc_da is None:
        return
    sic = siconc_da.mean("time") if "time" in siconc_da.dims else siconc_da
    vmax = float(np.nanmax(sic.values))
    if vmax <= 1.01:  # convert to %
        sic = sic * 100.0
    lo2d, la2d, _ = _lonlat_2d(sic)
    try:
        ax.contour(lo2d, la2d, sic.values, levels=[level],
                   colors="purple", linewidths=0.8, transform=ccrs.PlateCarree(), zorder=5)
    except Exception:
        pass

# ---- main panel ----
def plot_erf_panel_2x2(
    erf_data,
    *,
    models,                        # e.g. ["UKESM","NorESM","GISS","EC-AEREarth3"]
    region="Arctic",
    save_dir="images_erf",
    filename=None,
    cmap="RdBu_r",                 # diverging; use "turbo" if you want sequential
    vmin=None, vmax=None,          # None → robust symmetric shared across all four
    view_lat_min=50.0,             # polar footprint
    mask_land=True,
    land_color="white",
    ocean_color="#eaf2ff",
    use_contourf=False,
    siconc_data=None               # optional: dict like {model:{experiment:{region:DA}}} for 15% edge
):
    """
    Make a 2×2 panel of ERF for the given 4 models (all at `region`) with ONE shared colorbar.
    `erf_data` must look like: {model: {region: DataArray}}.
    """
    os.makedirs(save_dir, exist_ok=True)
    if len(models) != 4:
        raise ValueError("Please pass exactly 4 models for a 2×2 panel.")

    # Collect fields
    fields = []
    lonlats = []
    titles = []
    for mdl in models:
        da = erf_data.get(mdl, {}).get(region, None)
        if da is None or not isinstance(da, xr.DataArray) or da.ndim < 2 or np.isnan(da).all():
            raise ValueError(f"Missing/invalid ERF for {mdl} – {region}")
        if "time" in da.dims:
            da = da.mean("time")
        lo2d_raw, la2d_raw, grid_kind = _lonlat_2d(da)
        arr = np.ma.masked_invalid(da.values)
        # add cyclic if regular
        if grid_kind == "regular":
            lo2d, la2d, arrp = _maybe_add_cyclic(da, arr)
        else:
            lo2d, la2d, arrp = lo2d_raw, la2d_raw, arr
        fields.append(arrp)
        lonlats.append((lo2d, la2d))
        titles.append(mdl)

    # Shared normalization (symmetric by default—ERF is signed)
    if vmin is None or vmax is None:
        vmin_r, vmax_r = _robust_limits(fields, symmetric=True, q=98.0)
        vmin = vmin if vmin is not None else vmin_r
        vmax = vmax if vmax is not None else vmax_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Figure & axes
    proj = _proj_for_region(region)
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.04)
    axes = []
    mappables = []

    for idx, (arrp, (lon2d, lat2d), mdl) in enumerate(zip(fields, lonlats, titles)):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, c], projection=proj)
        axes.append(ax)

        # background & framing
        ax.set_facecolor(ocean_color)
        if region.lower() == "arctic":
            ax.set_extent([-180, 180, view_lat_min, 90], ccrs.PlateCarree())
            _set_polar_circle_boundary(ax)
            ax.patch.set_edgecolor(ax.get_facecolor())
        elif region.lower() == "antarctic":
            ax.set_extent([-180, 180, -90, -view_lat_min], ccrs.PlateCarree())
            _set_polar_circle_boundary(ax)
            ax.patch.set_edgecolor(ax.get_facecolor())
        else:
            ax.set_global()

        # field
        if use_contourf:
            levels = np.linspace(vmin, vmax, 21)
            mesh = ax.contourf(
                lon2d, lat2d, arrp, levels=levels, cmap=cmap, norm=norm,
                extend="both", transform=ccrs.PlateCarree(), zorder=1
            )
        else:
            mesh = ax.pcolormesh(
                lon2d, lat2d, arrp,
                cmap=cmap, norm=norm,
                shading="nearest",
                edgecolors="none", linewidth=0.0,
                antialiased=False, rasterized=True,
                transform=ccrs.PlateCarree(), zorder=1
            )
        # clip to circular boundary
        if region.lower() in ("arctic", "antarctic"):
            try: mesh.set_clip_path(ax.patch)
            except Exception: pass
        mappables.append(mesh)

        # land/lines
        if mask_land:
            land = cfeature.NaturalEarthFeature("physical", "land", "50m",
                                                facecolor=land_color, edgecolor="none")
            ax.add_feature(land, zorder=3)
        ax.coastlines(resolution="50m", linewidth=0.6, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, zorder=4)

        # optional: 15% ice edge (from SICONC if supplied)
        if siconc_data is not None:
            sic_da = None
            # prefer control if available
            if "control" in siconc_data.get(mdl, {}):
                sic_da = siconc_data[mdl]["control"].get(region, None)
            elif "2xDMS" in siconc_data.get(mdl, {}):
                sic_da = siconc_data[mdl]["2xDMS"].get(region, None)
            _add_ice_edge(ax, sic_da, level=15.0)

        # title
        ax.set_title(mdl, pad=4)

    # Shared colorbar (thin, full width)
    cax = fig.add_axes([0.12, 0.06, 0.76, 0.028])
    cb = fig.colorbar(mappables[0], cax=cax, orientation="horizontal")
    _format_cbar(cb)
    cb.set_label("ERF (W m$^{-2}$)")

    # Save
    if filename is None:
        tag = "_".join(models)
        filename = f"panel_ERF_{region}_{tag}.png".replace(" ", "_")
    out = os.path.join(save_dir, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")
