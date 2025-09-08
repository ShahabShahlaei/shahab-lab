# plot_erf_maps.py
# Publication-style ERF maps (no experiment layer) aligned with your other figures.

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
        return lonc.values, latc.values, "curvi"

    # 1-D → mesh
    if (la in da.dims) and (lo in da.dims):
        lat1d = latc.values if latc is not None else np.arange(da.sizes[la])
        lon1d = lonc.values if lonc is not None else np.arange(da.sizes[lo])
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d, "regular"

    # last resort if coords exist but not dims
    if (latc is not None and latc.ndim == 1) and (lonc is not None and lonc.ndim == 1):
        lon2d, lat2d = np.meshgrid(lonc.values, latc.values)
        return lon2d, lat2d, "regular"

    raise ValueError("Cannot construct 2-D lon/lat for plotting.")


def _maybe_add_cyclic(da, arr):
    """
    If DA has 1-D lon/lat dims, append a cyclic column at 360° to avoid the seam.
    Returns (lon2d, lat2d, arr2).
    """
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
    """True circular boundary for polar projections (removes square frame)."""
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


def _robust_limits(arrays, qlow=2.0, qhigh=98.0, symmetric=False):
    """Robust vmin/vmax over arrays; symmetric → center on 0 with max abs percentile."""
    vec = []
    for a in arrays:
        v = np.asarray(a).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            vec.append(v)
    if not vec:
        return None, None
    allv = np.concatenate(vec)
    if symmetric:
        rad = np.nanpercentile(np.abs(allv), qhigh)
        rad = float(max(rad, 1e-20))
        return -rad, rad
    vmin = np.nanpercentile(allv, qlow)
    vmax = np.nanpercentile(allv, qhigh)
    return vmin, vmax


# =========================
# ---- Core plotter --------
# =========================
def plot_erf_map(
    da,
    *,
    model,
    region,
    variable_name="ERF",
    unit="W m$^{-2}$",
    save_dir="images_erf",
    filename=None,
    cmap="RdBu_r",               # diverging default for signed ERF; pass "turbo" if you want
    vmin=None, vmax=None,        # None → robust & symmetric for signed fields
    view_lat_min=50.0,           # polar footprint
    mask_land=True,
    land_color="white",
    ocean_color="#eaf2ff",
    add_grid_global=False,
    use_contourf=False
):
    """
    Plot ONE ERF map with academic styling (polar circle, land mask, cyclic lon).
    Accepts 2-D or 3-D xarray DataArray (averages over time if needed).
    """
    if not isinstance(da, xr.DataArray):
        raise ValueError("da must be an xarray.DataArray")

    # Time-mean if needed
    if "time" in da.dims:
        da = da.mean("time")

    # Build lon/lat
    lon2d_raw, lat2d_raw, grid_kind = _lonlat_2d(da)

    # Values
    arr = np.ma.masked_invalid(da.values)

    # Cyclic only for regular (1-D) grids
    if grid_kind == "regular":
        lon2d, lat2d, arr_plot = _maybe_add_cyclic(da, arr)
    else:
        lon2d, lat2d, arr_plot = lon2d_raw, lat2d_raw, arr

    # Projection & figure
    proj = _proj_for_region(region)
    fig = plt.figure(figsize=(9.6, 5.8))
    ax  = plt.axes(projection=proj)
    ax.set_facecolor(ocean_color)

    # Camera / extent + circular boundary
    rl = region.lower()
    if rl == "global":
        ax.set_global()
    elif rl == "arctic":
        ax.set_extent([-180, 180, view_lat_min, 90], ccrs.PlateCarree())
        _set_polar_circle_boundary(ax)
        ax.patch.set_edgecolor(ax.get_facecolor())
    elif rl == "antarctic":
        ax.set_extent([-180, 180, -90, -view_lat_min], ccrs.PlateCarree())
        _set_polar_circle_boundary(ax)
        ax.patch.set_edgecolor(ax.get_facecolor())

    # Decide norm (ERF often signed → use symmetric robust by default)
    finite_vals = np.asarray(arr_plot)[np.isfinite(arr_plot)]
    any_negative = finite_vals.size > 0 and np.nanmin(finite_vals) < 0
    if vmin is None or vmax is None:
        if any_negative:
            vmin_r, vmax_r = _robust_limits([finite_vals], symmetric=True)
            vmin = vmin if vmin is not None else vmin_r
            vmax = vmax if vmax is not None else vmax_r
        else:
            vmin_r, vmax_r = _robust_limits([finite_vals], 2.0, 98.0, symmetric=False)
            vmin = vmin if vmin is not None else vmin_r
            vmax = vmax if vmax is not None else vmax_r
    norm = (mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            if any_negative else mcolors.Normalize(vmin=vmin, vmax=vmax))

    # Draw field
    if use_contourf:
        levels = np.linspace(vmin, vmax, 21 if any_negative else 17)
        mesh = ax.contourf(
            lon2d, lat2d, arr_plot,
            levels=levels, cmap=cmap, norm=norm, extend="both",
            transform=ccrs.PlateCarree(), zorder=1
        )
    else:
        mesh = ax.pcolormesh(
            lon2d, lat2d, arr_plot,
            cmap=cmap, norm=norm,
            shading="nearest",
            edgecolors="none", linewidth=0.0,
            antialiased=False, rasterized=True,
            transform=ccrs.PlateCarree(), zorder=1
        )

    # Clip to circular boundary on polar views
    if rl in ("arctic", "antarctic"):
        try:
            mesh.set_clip_path(ax.patch)
        except Exception:
            pass

    # Land mask on top
    if mask_land:
        land = cfeature.NaturalEarthFeature("physical", "land", "50m",
                                            facecolor=land_color, edgecolor="none")
        ax.add_feature(land, zorder=3)

    # Coastlines/Borders above land
    ax.coastlines(resolution="50m", linewidth=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, zorder=4)

    if rl == "global" and add_grid_global:
        ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.4, linestyle=":")

    # Title
    ax.set_title(f"{model} – {region} – {variable_name}", pad=4)

    # Thin colorbar
    cax = fig.add_axes([0.12, 0.08, 0.76, 0.028])
    cb  = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    _format_cbar(cb)
    cb.set_label(f"{variable_name} ({unit})")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        safe_var = variable_name.replace(" ", "_")
        filename = f"{model}_{region}_{safe_var}.png".replace(" ", "_")
    out = os.path.join(save_dir, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


# =========================
# ---- Batch helper --------
# =========================
def plot_all_erf_maps(
    erf_data,
    *,
    variable_name="ERF",
    unit="W m$^{-2}$",
    save_dir="images_erf",
    models=None,                 # None → all models
    regions=None,                # None → infer per model
    cmap="RdBu_r",
    view_lat_min=50.0,
    mask_land=True,
    land_color="white",
    ocean_color="#eaf2ff",
    use_contourf=False,
    share_norm_by_region=True    # True → same vmin/vmax across models for each region
):
    """
    Plot ERF maps for a dict shaped {model: {region: DataArray}}.
    If share_norm_by_region=True, all models for a given region use the same color scale.
    """
    if not isinstance(erf_data, dict):
        raise ValueError("erf_data must be {model → {region → DataArray}}")

    model_keys = models if models is not None else list(erf_data.keys())

    # Precompute shared scales per region
    region_scales = {}
    if share_norm_by_region:
        # collect finite values by region
        vals_by_region = {}
        for mdl in model_keys:
            for reg, da in erf_data.get(mdl, {}).items():
                if regions is not None and reg not in regions:
                    continue
                if not isinstance(da, xr.DataArray) or da.ndim < 2:
                    continue
                arr = da.mean("time").values if "time" in da.dims else da.values
                fv = np.asarray(arr, dtype=float)
                fv = fv[np.isfinite(fv)]
                if fv.size:
                    vals_by_region.setdefault(reg, []).append(fv)
        # compute symmetric robust limits per region (good for signed ERF)
        for reg, chunks in vals_by_region.items():
            allv = np.concatenate(chunks)
            vmin_r, vmax_r = _robust_limits([allv], symmetric=True)
            region_scales[reg] = (vmin_r, vmax_r)

    for mdl in model_keys:
        exps = erf_data.get(mdl, {})
        reg_keys = regions if regions is not None else list(exps.keys())
        for reg in reg_keys:
            da = exps.get(reg, None)
            if not isinstance(da, xr.DataArray) or da.ndim < 2 or np.isnan(da).all():
                print(f"⚠ Skipping {mdl} – {reg} (empty or not 2-D)")
                continue
            vmin = vmax = None
            if share_norm_by_region and reg in region_scales:
                vmin, vmax = region_scales[reg]

            plot_erf_map(
                da, model=mdl, region=reg,
                variable_name=variable_name, unit=unit,
                save_dir=save_dir, filename=None,
                cmap=cmap, vmin=vmin, vmax=vmax,
                view_lat_min=view_lat_min, mask_land=mask_land,
                land_color=land_color, ocean_color=ocean_color,
                use_contourf=use_contourf
            )
