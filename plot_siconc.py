# siconc_plots_v2.py
# Drop-in replacement that keeps your public function name: plot_siconc_maps
from common_imports import *
import matplotlib as mpl
import os, numpy as np, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

# ===========================
# ----- Global plot style ----
# ===========================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ===========================
# ----- Small utilities  -----
# ===========================
def _coords_2d(da):
    """Return lon2d, lat2d and CRS-aware info for regular or curvilinear grids."""
    dims = set(da.dims)
    if {"lat", "lon"} <= dims:  # regular
        lat, lon = da.lat, da.lon
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lon2d, lat2d
    # Curvilinear (common dim names: (j,i) or (y,x); coords often named latitude/longitude)
    for la_name in ["lat", "latitude"]:
        for lo_name in ["lon", "longitude"]:
            if la_name in da.coords and lo_name in da.coords:
                lat = da.coords[la_name]
                lon = da.coords[lo_name]
                if lat.ndim == 2 and lon.ndim == 2:
                    return lon.values, lat.values
    raise ValueError("Unrecognised grid; need 2-D latitude/longitude on coords.")

def _region_proj(region):
    r = region.lower()
    if r == "arctic":
        return ccrs.NorthPolarStereo()
    if r == "antarctic":
        return ccrs.SouthPolarStereo()
    return ccrs.PlateCarree()

def _mask_region(lat2d, region, arctic_lat0=66.0, antarctic_lat0=-60.0):
    r = region.lower()
    if r == "arctic":
        return lat2d >= arctic_lat0
    if r == "antarctic":
        return lat2d <= antarctic_lat0
    return np.ones_like(lat2d, dtype=bool)

def _to_percent(da):
    """Ensure SICONC in % (0..100). If already in fraction, convert."""
    v = da.values
    vmax = np.nanmax(v)
    if vmax <= 1.001:  # fraction 0..1
        return da * 100.0
    return da

def _open_water_fraction(siconc_percent):
    """Return open water fraction (0..1) from SICONC in %."""
    return (100.0 - siconc_percent) / 100.0

def _area_weights(lat2d):
    """Cos(lat) weights; not exact areas, but OK for regional weighting."""
    return np.cos(np.deg2rad(lat2d)).clip(min=0.0)

def _regrid_like(src_da, target_da):
    """Lightweight regrid via xarray.interp onto target grid (lon/lat 1D or 2D)."""
    # Try shared 1D lat/lon first:
    if {"lat","lon"} <= set(src_da.dims) and {"lat","lon"} <= set(target_da.dims):
        if np.allclose(src_da.lat, target_da.lat) and np.allclose(src_da.lon, target_da.lon):
            return src_da
        return src_da.interp(lat=target_da.lat, lon=target_da.lon)
    # Fallback: if target has 2D lat/lon coords
    # Build a dataset with target coords and interpolate
    for la in ["lat","latitude"]:
        for lo in ["lon","longitude"]:
            if la in target_da.coords and lo in target_da.coords:
                tgt = xr.Dataset({la: target_da.coords[la], lo: target_da.coords[lo]})
                return src_da.interp({la: tgt[la], lo: tgt[lo]})
    raise ValueError("Could not regrid; missing compatible lat/lon coordinates.")

def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# ==================================
# ----- Core map plotting (SIC) -----
# ==================================
def _draw_single_siconc(
    da, *, model, region, save_dir,
    unit="%", dpi=300, smooth=True, title_extra="",
    cmap="Blues", vmin=0.0, vmax=100.0, add_ice_edge=True
):
    """Low-memory plot of a **time-mean** SICONC field with fixed, comparable colorbar."""
    # Prepare grid & data
    lon2d, lat2d = _coords_2d(da)
    arr = np.ma.masked_invalid(_to_percent(da).values)  # to % and mask
    proj = _region_proj(region)

    fig = plt.figure(figsize=(7.2, 6.2), dpi=dpi)
    ax = plt.axes(projection=proj)
    if region.lower() == "global":
        ax.set_global()

    shade_kw = dict(shading="gouraud") if smooth else dict(shading="auto")
    mesh = ax.pcolormesh(lon2d, lat2d, arr, cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree(), **shade_kw)
    # 15% ice edge
    if add_ice_edge:
        try:
            cs = ax.contour(lon2d, lat2d, arr, levels=[15.0], transform=ccrs.PlateCarree(),
                            linewidths=0.8)
            cs.collections[0].set_label("15% ice edge")
        except Exception:
            pass

    # Style
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, lw=0.3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.4)

    # Colorbar (fixed 0..100%)
    cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.7)
    cb.set_label(f"Sea-ice concentration ({unit})")

    # Title
    title = f"{model} – {region} {title_extra}".strip()
    ax.set_title(title)

    # Save
    fname = f"{model}_{region}_siconc{title_extra}.png".replace(" ", "_")
    _savefig(fig, os.path.join(save_dir, fname))

# ==========================================
# ----- STEP 1: emidms vs open water    -----
# ==========================================
def step1_emidms_vs_openwater(
    siconc, emidms, *, region="Arctic", save_dir="plots_siconc", label="(Step1)", arctic_lat0=60.0
):
    """
    Area-weighted linear fit of emidms vs open-water fraction (1 - siconc).
    Both inputs can be monthly time series; will align time & regrid as needed.
    Returns slope, intercept (ordinary least squares).
    """
    # Regrid emidms onto siconc grid if needed
    emidms_rg = _regrid_like(emidms, siconc)

    # Align time (inner join)
    siconc_, emidms_ = xr.align(siconc, emidms_rg, join="inner")

    # Convert to open water (0..1)
    ow = _open_water_fraction(_to_percent(siconc_))

    # Build 2D coords & mask
    lon2d, lat2d = _coords_2d(siconc_)
    Rmask = _mask_region(lat2d, region, arctic_lat0=arctic_lat0)
    W = _area_weights(lat2d) * Rmask

    # Flatten space, keep time
    def _aw_mean(da):
        A = np.ma.masked_invalid(da.values)
        ww = np.ma.masked_array(W, mask=np.isnan(A[0]))  # mask where data are NaN
        num = np.ma.sum(A * ww, axis=(-2, -1))
        den = np.ma.sum(ww, axis=(-2, -1))
        return (num / den).filled(np.nan)

    # Monthly (or whatever) regional time series
    ow_ts = _aw_mean(ow)
    em_ts = _aw_mean(emidms_)

    # OLS fit: em_ts = a * ow_ts + b
    m = np.isfinite(ow_ts) & np.isfinite(em_ts)
    if m.sum() < 3:
        slope, intercept = np.nan, np.nan
    else:
        X = np.vstack([ow_ts[m], np.ones(m.sum())]).T
        slope, intercept = np.linalg.lstsq(X, em_ts[m], rcond=None)[0]

    # Plot scatter + fit
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(111)
    ax.scatter(ow_ts, em_ts, s=6, alpha=0.5, rasterized=True)
    if np.isfinite(slope):
        xs = np.linspace(np.nanmin(ow_ts), np.nanmax(ow_ts), 100)
        ax.plot(xs, slope * xs + intercept, lw=1.5)
    ax.set_xlabel("Open-water fraction (1 − siconc)")
    ax.set_ylabel("DMS emissions (emidms)")
    ax.set_title(f"{region} {label}: emidms vs open water\nslope={slope:.3g}, intercept={intercept:.3g}")
    _savefig(fig, os.path.join(save_dir, f"step1_emidms_vs_openwater_{region}.png"))
    return float(slope), float(intercept)

# ==========================================
# ----- STEP 2: ERF share by SIC regime -----
# ==========================================
def step2_erf_partition_by_sic(
    siconc, erf, *, region="Arctic", save_dir="plots_siconc", label="(Step2)",
    bins=(0.0, 15.0, 80.0, 100.0), use_abs=True, arctic_lat0=60.0
):
    """
    Partition ERF into sea-ice regimes by SICONC (%):
      Open water: <15%, MIZ: 15–80%, Pack: >80% (default).
    Report % share of total |ERF| (or signed ERF if use_abs=False).
    """
    erf_rg = _regrid_like(erf, siconc)
    siconc_, erf_ = xr.align(_to_percent(siconc), erf_rg, join="inner")

    lon2d, lat2d = _coords_2d(siconc_)
    Rmask = _mask_region(lat2d, region, arctic_lat0=arctic_lat0)
    W = _area_weights(lat2d) * Rmask

    # Time-mean fields for partitioning
    sic_mean = np.nanmean(siconc_.values, axis=0) if "time" in siconc_.dims else siconc_.values
    erf_mean = np.nanmean(erf_.values, axis=0) if "time" in erf_.dims else erf_.values

    VAL = np.abs(erf_mean) if use_abs else erf_mean
    weights = W * np.isfinite(sic_mean) * np.isfinite(VAL)

    # Bins
    edges = np.array(bins)
    labels = ["Open water (<15%)", "MIZ (15–80%)", "Pack ice (>80%)"]
    shares = []

    total = np.ma.sum(np.ma.masked_invalid(VAL) * weights)
    for i in range(len(edges) - 1):
        mask = (sic_mean >= edges[i]) & (sic_mean < edges[i + 1])
        num = np.ma.sum(np.ma.masked_invalid(VAL) * (weights * mask))
        share = float(num / total) * 100.0 if total != 0 else np.nan
        shares.append(share)

    # Bar plot
    fig = plt.figure(figsize=(5.0, 3.6))
    ax = fig.add_subplot(111)
    ax.bar(labels, shares)
    ax.set_ylabel("% of total |ERF|" if use_abs else "% of total ERF")
    ax.set_title(f"{region} {label}: ERF by SIC regime")
    for i, v in enumerate(shares):
        ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom")
    plt.xticks(rotation=10)
    _savefig(fig, os.path.join(save_dir, f"step2_erf_partition_{region}.png"))

    # Also return a dict you can print/log
    return {labels[i]: shares[i] for i in range(3)}

# ===================================================
# ----- PUBLIC: draw maps from your siconc_dict  -----
# ===================================================
def plot_siconc_maps(
    siconc_dict, *,
    save_dir="plots_siconc", dpi=300, smooth=True,
    cmap="Blues", vmin=0.0, vmax=100.0, add_ice_edge=True,
    step1_inputs=None,   # dict like {"emidms": xr.DataArray, "region": "Arctic"}
    step2_inputs=None    # dict like {"erf": xr.DataArray, "region": "Arctic"}
):
    """
    Draw SICONC maps from `siconc_dict` produced by your pipeline, but with:
      - fixed colorbar [0..100%] for apples-to-apples comparison
      - 15% ice-edge contour
      - optional Step 1 and Step 2 analyses

    siconc_dict structure:
      {model: {exp: {region: Da(time?, y, x)}}}

    step1_inputs: {"emidms": DataArray, "region": "Arctic"|"Antarctic"|"Global"}
    step2_inputs: {"erf": DataArray,    "region": "Arctic"|"Antarctic"|"Global"}
    """
    os.makedirs(save_dir, exist_ok=True)
    for model, exps in siconc_dict.items():
        for exp, regions in exps.items():
            for region, da in regions.items():
                if da is None or np.isnan(da).all():
                    print(f"⚠️  {model} – {exp} – {region}: only NaNs, skip.")
                    continue
                print(f"🔷 {model} – {exp} – {region}")
                # Time mean for the map (keeps your low-memory intent)
                da_mean = da.mean("time") if "time" in da.dims else da
                _draw_single_siconc(
                    da_mean, model=f"{model} ({exp})", region=region,
                    save_dir=save_dir, dpi=dpi, smooth=smooth,
                    title_extra=f"(SImon)", cmap=cmap, vmin=vmin, vmax=vmax,
                    add_ice_edge=add_ice_edge
                )

                # Optional Step 1 & 2
                if isinstance(step1_inputs, dict) and "emidms" in step1_inputs:
                    try:
                        slope, intercept = step1_emidms_vs_openwater(
                            siconc=da, emidms=step1_inputs["emidms"],
                            region=step1_inputs.get("region", region),
                            save_dir=save_dir, label=f"(Step1 {model} {exp})"
                        )
                        print(f"   Step1 slope={slope:.3g}, intercept={intercept:.3g}")
                    except Exception as e:
                        print(f"   ⚠️ Step1 failed for {model}-{exp}-{region}: {e}")

                if isinstance(step2_inputs, dict) and "erf" in step2_inputs:
                    try:
                        shares = step2_erf_partition_by_sic(
                            siconc=da, erf=step2_inputs["erf"],
                            region=step2_inputs.get("region", region),
                            save_dir=save_dir, label=f"(Step2 {model} {exp})"
                        )
                        print(f"   Step2 shares: {shares}")
                    except Exception as e:
                        print(f"   ⚠️ Step2 failed for {model}-{exp}-{region}: {e}")
