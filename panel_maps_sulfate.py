# panel_maps_sulfate.py
# 2x2 panel for sulfate (mmrso4) with shared colorbar, academic styling.
# Rows: control, 2xDMS.  Columns: GISS, NorESM.
# Polar view is circular (no square frame), land is white, ocean is light blue,
# and the visible domain extends down to view_lat_min (default 50°N).

import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath  # for circular boundary

# =========================
# ---- Global style --------
# =========================
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

# =========================
# ---- Small helpers -------
# =========================
def _find_lat_lon_names(da):
    cand_lat = ["lat", "latitude", "nav_lat", "y", "j"]
    cand_lon = ["lon", "longitude", "nav_lon", "x", "i"]
    lat_name = next((n for n in cand_lat if (n in da.dims or n in da.coords)), None)
    lon_name = next((n for n in cand_lon if (n in da.dims or n in da.coords)), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find lat/lon on {da.name or 'DataArray'} "
                         f"(dims={list(da.dims)}, coords={list(da.coords)})")
    return lat_name, lon_name

def _lonlat_2d(da):
    la, lo = _find_lat_lon_names(da)
    latc = da.coords.get(la)
    lonc = da.coords.get(lo)

    # 2-D curvilinear
    if (latc is not None and getattr(latc, "ndim", 1) == 2) and (lonc is not None and getattr(lonc, "ndim", 1) == 2):
        return lonc.values, latc.values

    # 1-D → tile
    if (la in da.dims) and (lo in da.dims):
        lat1d = latc.values if latc is not None else np.arange(da.sizes[la])
        lon1d = lonc.values if lonc is not None else np.arange(da.sizes[lo])
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d

    # last resort if coords exist but not dims
    if (latc is not None and latc.ndim == 1) and (lonc is not None and lonc.ndim == 1):
        lon2d, lat2d = np.meshgrid(lonc.values, latc.values)
        return lon2d, lat2d

    raise ValueError("Cannot construct 2-D lon/lat for plotting.")

def _proj_for_region(region):
    r = (region or "").lower()
    if r == "arctic":
        return ccrs.NorthPolarStereo(central_longitude=0.0)
    if r == "antarctic":
        return ccrs.SouthPolarStereo(central_longitude=0.0)
    return ccrs.PlateCarree()

def _robust_limits(arrays, qlow=2.0, qhigh=98.0):
    """Compute robust vmin/vmax over many arrays (avoid outliers)."""
    vec = []
    for a in arrays:
        v = np.asarray(a.values).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            vec.append(v)
    if not vec:
        return None, None
    allv = np.concatenate(vec)
    return (np.nanpercentile(allv, qlow),
            np.nanpercentile(allv, qhigh))

def _two_slope_norm_from_arrays(arrays, center=0.0, q=98.0):
    """Symmetric limits around center using high quantile of |values - center|."""
    vec = []
    for a in arrays:
        v = np.asarray(a.values).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            vec.append(np.abs(v - center))
    if not vec:
        return None
    rad = np.nanpercentile(np.concatenate(vec), q)
    rad = float(max(rad, 1e-20))
    return mcolors.TwoSlopeNorm(vmin=center - rad, vcenter=center, vmax=center + rad)

def _format_cbar(cb):
    cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()

def _add_ice_edge(ax, siconc_da, level=15.0):
    """Optional 15% SIC contour if siconc_da provided for this panel."""
    if siconc_da is None:
        return
    sic = siconc_da.mean("time") if "time" in siconc_da.dims else siconc_da
    vmax = float(np.nanmax(sic.values))
    if vmax <= 1.01:
        sic = sic * 100.0

    lon2d, lat2d = _lonlat_2d(sic)
    try:
        cs = ax.contour(lon2d, lat2d, sic.values, levels=[level],
                        transform=ccrs.PlateCarree(), linewidths=0.8)
        cs.collections[0].set_label("15% ice edge")
    except Exception:
        pass

# Circular boundary for polar views (removes square frame)
def _set_polar_circle_boundary(ax, npts=720):
    theta = np.linspace(0, 2*np.pi, npts)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.cos(theta), np.sin(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

# =========================================
# ---- Main panel function (2×2 layout) ----
# =========================================
def plot_mmrso4_panel_2x2(
    mmrso4_data,
    *,
    region="Arctic",
    models=("GISS", "NorESM"),
    experiments=("control", "2xDMS"),
    save_dir="images",
    filename=None,
    unit="kg kg⁻¹",
    cmap="viridis",
    vmin=None, vmax=None,        # if None, uses robust quantiles over all four panels
    use_log=False,               # set True if you prefer LogNorm (positive-only)
    siconc_data=None,            # optional 15% ice-edge overlay, same structure
    add_grid_global=False,
    mask_land=True,
    land_color="white",
    ocean_color="#dbe8ff",       # light blue ocean everywhere outside data
    view_lat_min=50.0            # <<< NEW: how far down from the pole to show (deg N/S)
):
    """
    Make a 2×2 panel for sulfate (mmrso4) with shared colorbar.
    mmrso4_data structure:
        {model: {experiment: {region: DataArray}}}
    """
    os.makedirs(save_dir, exist_ok=True)

    # Collect the 4 fields (time-mean if needed) in row-major order
    fields = []
    titles = []
    lonlats = []
    for exp in experiments:
        for mdl in models:
            da = None
            try:
                da = mmrso4_data[mdl][exp][region]
            except Exception:
                pass
            if da is None:
                da = xr.DataArray(np.full((2,2), np.nan), dims=("lat","lon"))
                titles.append(f"{mdl} ({exp}) – {region} (missing)")
                lonlats.append((None, None))
                fields.append(da)
                continue

            if "time" in da.dims:
                da = da.mean("time")

            lon2d, lat2d = _lonlat_2d(da)
            lonlats.append((lon2d, lat2d))
            titles.append(f"{mdl} ({exp}) – {region}")
            fields.append(da)

    # Decide normalization / limits shared across panels
    any_negative = any(np.nanmin(f.values) < 0 for f in fields)
    if use_log:
        if vmin is None or vmin <= 0:
            vmin_auto, vmax_auto = _robust_limits(fields, 2.0, 98.0)
            vmin = max(vmin or 0, vmin_auto or 1e-20, 1e-20)
        if vmax is None:
            _, vmax_auto = _robust_limits(fields, 2.0, 98.0)
            vmax = vmax_auto if vmax_auto is not None else np.nanmax([np.nanmax(f.values) for f in fields])
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if any_negative:
            norm = _two_slope_norm_from_arrays(fields, center=0.0, q=98.0)
        else:
            if vmin is None or vmax is None:
                vmin_r, vmax_r = _robust_limits(fields, 2.0, 98.0)
                vmin = vmin if vmin is not None else vmin_r
                vmax = vmax if vmax is not None else vmax_r
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Figure & axes (2 rows × 2 cols)
    proj = _proj_for_region(region)
    fig = plt.figure(figsize=(9.6, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.04)
    mappables = []

    for r in range(2):
        for c in range(2):
            idx = r*2 + c
            da = fields[idx]
            lon2d, lat2d = lonlats[idx]

            ax = fig.add_subplot(gs[r, c], projection=proj)

            # Ocean everywhere outside data
            ax.set_facecolor(ocean_color)

            # Camera / domain
            if region.lower() == "global":
                ax.set_global()
            elif region.lower() == "arctic":
                ax.set_extent([-180, 180, view_lat_min, 90], ccrs.PlateCarree())
                _set_polar_circle_boundary(ax)
                ax.patch.set_edgecolor(ax.get_facecolor())  # hide circle edge
            elif region.lower() == "antarctic":
                ax.set_extent([-180, 180, -90, -view_lat_min], ccrs.PlateCarree())
                _set_polar_circle_boundary(ax)
                ax.patch.set_edgecolor(ax.get_facecolor())

            # Data values
            A = np.ma.masked_invalid(da.values)

            # Pseudocolor mesh
            if lon2d is None or lat2d is None:
                mesh = ax.pcolormesh(A, cmap=cmap, norm=norm)
            else:
                mesh = ax.pcolormesh(lon2d, lat2d, A,
                                     cmap=cmap, norm=norm,
                                     shading="auto",
                                     transform=ccrs.PlateCarree())
                # Clip to circular boundary on polar views (prevents square frame)
                if region.lower() in ("arctic", "antarctic"):
                    try:
                        mesh.set_clip_path(ax.patch)
                    except Exception:
                        pass

            mappables.append(mesh)

            # Land mask in white (on top)
            if mask_land:
                land = cfeature.NaturalEarthFeature("physical", "land", "50m",
                                                    facecolor=land_color, edgecolor="none")
                ax.add_feature(land, zorder=3)

            # Coastlines/Borders above land
            ax.coastlines(resolution="50m", linewidth=0.6, zorder=4)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, zorder=4)

            # Optional 15% SIC edge per panel
            if siconc_data is not None:
                sic_da = None
                try:
                    sic_da = siconc_data[models[c]][experiments[r]][region]
                except Exception:
                    sic_da = None
                _add_ice_edge(ax, sic_da, level=15.0)

            # Titles only on the top row
            if r == 0:
                ax.set_title(f"{models[c]} ({experiments[r]}) – {region}", pad=4)

            # Clean spines
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    # Row labels on the left margin
    for r, label in enumerate(experiments):
        fig.text(0.008, 0.75 - 0.5*r, label, rotation=90, va="center", ha="left", fontsize=10)

    # Shared colorbar (thin, full-length)
    cax = fig.add_axes([0.12, 0.06, 0.76, 0.028])  # [left, bottom, width, height]
    cb = fig.colorbar(mappables[0], cax=cax, orientation="horizontal")
    _format_cbar(cb)
    cb.set_label(f"Sulfate mass mixing ratio (mmrso4) [{unit}]")

    # Save
    if filename is None:
        models_tag = "_".join(models)
        filename = f"panel_mmrso4_{region}_{models_tag}_{experiments[0]}_{experiments[1]}.png".replace(" ", "_")
    out = os.path.join(save_dir, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")
