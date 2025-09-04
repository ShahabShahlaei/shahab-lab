import os, numpy as np, xarray as xr
import matplotlib.pyplot as plt
import numpy as np, xarray as xr, os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree



from analyze_erf_emidms_siconc import analyze_erf_emidms_siconc  # uses your existing function
import matplotlib as mpl

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


# ----------------- small helpers -----------------
def _binned_percentiles(x, y, *, nbins=30, q_low=25, q_med=50, q_high=75):
    """
    Bin x into 'nbins' bins and compute y percentiles per bin.
    Returns (x_mid, y_p25, y_p50, y_p75) with NaNs where not enough points.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return (np.array([]),)*4

    xb = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), nbins + 1)
    idx = np.digitize(x[m], xb) - 1
    x_mid = 0.5 * (xb[:-1] + xb[1:])
    p25 = np.full(nbins, np.nan)
    p50 = np.full(nbins, np.nan)
    p75 = np.full(nbins, np.nan)
    for i in range(nbins):
        yi = y[m][idx == i]
        if yi.size >= 10:  # need enough points for a stable percentile
            p25[i] = np.nanpercentile(yi, q_low)
            p50[i] = np.nanpercentile(yi, q_med)
            p75[i] = np.nanpercentile(yi, q_high)
    return x_mid, p25, p50, p75

def _to_percent(sic):
    """Ensure SICONC is in % (0..100)."""
    vmax = float(np.nanmax(sic.values))
    return sic * 100.0 if vmax <= 1.01 else sic

def _open_water_fraction(sic_percent):
    """Return open-water fraction (0..1) from SICONC in %."""
    return (100.0 - sic_percent) / 100.0
# --- NEW: find the actual lat/lon coordinate names on any DA ---
def _find_lat_lon_names(da):
    cand_lat = ["lat", "latitude", "nav_lat", "y", "j"]
    cand_lon = ["lon", "longitude", "nav_lon", "x", "i"]
    lat_name = next((n for n in cand_lat if (n in da.dims or n in da.coords)), None)
    lon_name = next((n for n in cand_lon if (n in da.dims or n in da.coords)), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find lat/lon names on DA with dims={list(da.dims)} coords={list(da.coords)}")
    return lat_name, lon_name

def _latlon_2d(da):
    """Return (lat2d, lon2d) arrays for any grid (1-D or 2-D coords)."""
    la, lo = _find_lat_lon_names(da)
    latc = da.coords.get(la, None)
    lonc = da.coords.get(lo, None)

    # 2-D curvilinear
    if (latc is not None and getattr(latc, "ndim", 1) == 2) and (lonc is not None and getattr(lonc, "ndim", 1) == 2):
        return latc.values, lonc.values

    # 1-D lat/lon → tile
    if (la in da.dims) and (lo in da.dims):
        lat1d = (latc.values if latc is not None else np.arange(da.sizes[la]))
        lon1d = (lonc.values if lonc is not None else np.arange(da.sizes[lo]))
        lat2d = np.tile(lat1d.reshape(-1,1), (1, lon1d.size))
        lon2d = np.tile(lon1d.reshape(1,-1), (lat1d.size, 1))
        return lat2d, lon2d

    # Last resort: coords exist but not dims (rare)
    if (latc is not None and latc.ndim == 1) and (lonc is not None and lonc.ndim == 1):
        lat2d = np.tile(latc.values.reshape(-1,1), (1, lonc.size))
        lon2d = np.tile(lonc.values.reshape(1,-1), (latc.size, 1))
        return lat2d, lon2d

    raise ValueError("Cannot build 2-D lat/lon for this DataArray.")
def _lat2d_from_da(da):
    """Back-compat shim: return only the 2-D latitude array."""
    lat2d, _ = _latlon_2d(da)
    return lat2d

def _area_weights_like(da):
    lat2d = _lat2d_from_da(da)
    return np.cos(np.deg2rad(lat2d)).clip(min=0.0)


def _ll_to_unitvec(lat_deg, lon_deg):
    """lat/lon (deg) → 3D unit vectors for great-circle nearest-neighbor."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)

def _regrid_to_target_nearest(src_da, tgt_da, k=4):
    """
    Nearest-neighbor regrid of src_da onto tgt_da's grid (works for 1-D or 2-D lat/lon).
    Uses spherical KD-Tree; tries up to k neighbors to avoid NaN picks.
    """
    # Build source KD-Tree in 3D unit-sphere space
    s_lat2d, s_lon2d = _latlon_2d(src_da)
    s_xyz = _ll_to_unitvec(s_lat2d.ravel(), s_lon2d.ravel())
    tree = cKDTree(s_xyz)

    # Target points
    t_lat2d, t_lon2d = _latlon_2d(tgt_da)
    t_xyz = _ll_to_unitvec(t_lat2d.ravel(), t_lon2d.ravel())

    # Query k nearest
    dist, idx = tree.query(t_xyz, k=min(k, s_xyz.shape[0]))
    if np.ndim(idx) == 1:
        idx = idx[:, None]  # shape (M,1)

    # Pull source data
    s_vals = src_da.values
    if s_vals.ndim != 2:
        # Expect 2-D time-mean fields here; if extra dims, squeeze the last 2
        s_vals = np.squeeze(s_vals)
    s_flat = s_vals.reshape(-1)

    # choose first finite among k
    picked = np.full(idx.shape[0], np.nan)
    for j in range(idx.shape[1]):
        cand = s_flat[idx[:, j]]
        need = ~np.isfinite(picked)
        picked[need] = cand[need]
        if not np.isnan(picked).any():
            break

    out = picked.reshape(t_lat2d.shape)

    # Build DataArray on tgt grid/dims/coords
    out_da = xr.DataArray(
        out,
        dims=tgt_da.dims,
        coords={d: tgt_da.coords[d] for d in tgt_da.dims if d in tgt_da.coords},
        attrs=src_da.attrs
    )
    out_da.name = src_da.name
    return out_da

def _align_to(src_da, tgt_da):
    """
    Align src_da to tgt_da's grid.
    - If both are 1-D lat/lon dims, use xarray.interp (bilinear).
    - Otherwise, do spherical nearest-neighbor (KD-Tree).
    """
    if src_da.shape == tgt_da.shape:
        return src_da

    try:
        s_la, s_lo = _find_lat_lon_names(src_da)
        t_la, t_lo = _find_lat_lon_names(tgt_da)
    except Exception:
        # If we cannot even find coords, return as-is
        return src_da

    # 1-D → use interp
    if (s_la in src_da.dims and s_lo in src_da.dims and
        t_la in tgt_da.dims and t_lo in tgt_da.dims and
        getattr(src_da.coords.get(s_la, None), "ndim", 1) == 1 and
        getattr(src_da.coords.get(s_lo, None), "ndim", 1) == 1 and
        getattr(tgt_da.coords.get(t_la, None), "ndim", 1) == 1 and
        getattr(tgt_da.coords.get(t_lo, None), "ndim", 1) == 1):
        try:
            same_lat = np.allclose(src_da.coords[s_la], tgt_da.coords[t_la])
            same_lon = np.allclose(src_da.coords[s_lo], tgt_da.coords[t_lo])
        except Exception:
            same_lat = same_lon = False
        if same_lat and same_lon:
            return src_da
        return src_da.interp({s_la: tgt_da.coords[t_la], s_lo: tgt_da.coords[t_lo]})

    # Else → curvilinear or mixed → nearest-neighbor
    return _regrid_to_target_nearest(src_da, tgt_da, k=4)

def _clean_common_mask(*arrays):
    """Apply a joint finite mask to multiple numpy arrays (same shape)."""
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return [np.where(mask, a, np.nan) for a in arrays]

# ----------------- STEP 1: spatial regression -----------------
def run_step1_spatial(siconc_da, emidms_da, *, region_label="Arctic",
                      save_dir="plots_steps", fname_prefix="step1"):
    """
    Spatial (grid-cell) WLS: emidms ~ a * open_water + b.
    Plot: hexbin density (no overplot clutter) + binned-median trend with IQR.
    """
    os.makedirs(save_dir, exist_ok=True)

    sic_pct = _to_percent(siconc_da)
    ow = _open_water_fraction(sic_pct)
    emidms_aligned = _align_to(emidms_da, sic_pct)

    W = _area_weights_like(sic_pct)
    x = ow.values
    y = emidms_aligned.values
    w = W

    x, y, w = _clean_common_mask(x, y, w)
    m = np.isfinite(x) & np.isfinite(y) & (w > 0)
    npts = int(m.sum())
    if npts < 10:
        return float("nan"), float("nan")

    # Weighted least squares (numeric output only)
    X = np.vstack([x[m], np.ones(npts)]).T
    sw = np.sqrt(w[m])
    Xw = X * sw[:, None]
    yw = y[m] * sw
    slope, intercept = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    # Build plotting sample
    xs = x[m].ravel()
    ys = y[m].ravel()

    # Hexbin density + binned median/IQR
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    hb = ax.hexbin(xs, ys, gridsize=45, mincnt=5, bins='log')  # density, log counts
    cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.04, aspect=30)
    cb.set_label("log10(count)")

    # Binned medians (trend) + IQR band
    x_mid, p25, p50, p75 = _binned_percentiles(xs, ys, nbins=30)
    if x_mid.size:
        ax.plot(x_mid, p50, lw=1.7, label="Median (binned)")
        ax.fill_between(x_mid, p25, p75, alpha=0.25, linewidth=0, label="IQR (25–75%)")

    # Cosmetics
    ax.set_xlim(0, 1)  # open-water fraction
    ax.set_xlabel("Open-water fraction (1 − siconc)")
    ax.set_ylabel("DMS emissions (emidms)")
    ax.set_title(f"{region_label}: emidms vs open water\n"
                 f"slope={slope:.3g}, intercept={intercept:.3g}")
    ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = os.path.join(save_dir, f"{fname_prefix}_{region_label}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

    return float(slope), float(intercept)



# ----------------- STEP 2: ERF partition -----------------
def run_step2_partition(siconc_da, erf_da, *, region_label="Arctic",
                        bins=(0.0, 15.0, 80.0, 100.0), use_abs=True,
                        save_dir="plots_steps", fname_prefix="step2"):
    """
    Partition ERF into SICONC regimes (in %): <15, 15–80, >80.
    Plot: tidy bars with headroom and labels above bars.
    """
    os.makedirs(save_dir, exist_ok=True)

    sic_pct = _to_percent(siconc_da)
    erf_on_sic = _align_to(erf_da, sic_pct)

    W = _area_weights_like(sic_pct)
    sic = sic_pct.values
    erf = np.abs(erf_on_sic.values) if use_abs else erf_on_sic.values

    sic, erf, W = _clean_common_mask(sic, erf, W)
    total = np.nansum(erf * W)
    labels = ["Open water (<15%)", "MIZ (15–80%)", "Pack ice (>80%)"]

    if not np.isfinite(total) or total == 0:
        shares = [np.nan, np.nan, np.nan]
    else:
        edges = np.array(bins)
        shares = []
        for i in range(len(edges) - 1):
            mask = (sic >= edges[i]) & (sic < edges[i+1])
            num = np.nansum(erf * W * mask)
            shares.append(float(num / total * 100.0))

    # Plot (with headroom and clean labels)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    bars = ax.bar(labels, shares, width=0.6)
    y_max = np.nanmax(shares) if np.isfinite(shares).any() else 1.0
    ax.set_ylim(0.0, y_max * 1.18)  # headroom so labels don't collide
    ax.set_ylabel("% of total |ERF|" if use_abs else "% of total ERF")
    ax.set_title(f"{region_label}: ERF by sea-ice regime")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Labels above bars (not congested)
    lab = [f"{v:.1f}%" if np.isfinite(v) else "—" for v in shares]
    ax.bar_label(bars, labels=lab, padding=3, fontsize=9)

    plt.xticks(rotation=10, ha="center")
    out = os.path.join(save_dir, f"{fname_prefix}_{region_label}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

    return {labels[i]: shares[i] for i in range(3)}



# ----------------- Orchestrator -----------------
def run_steps_with_analyzer(model_data, *, time_slice=(None, None),
                            region="Arctic", preferred_em_exp="control",
                            save_dir="plots_steps"):
    """
    1) Calls your analyzer once.
    2) For each model, runs Step 1 (spatial regression) and Step 2 (ERF partition).
    3) Returns a results dict and writes PNGs under `save_dir`.

    Notes:
      • We use SICONC from 'control' if available; if not, fallback to '2xDMS'.
      • Step 1 uses EMIDMS from `preferred_em_exp` (default 'control'),
        falling back to whichever experiment exists.
      • Step 2 uses ERF from analyzer (already 2xDMS − control).
    """
    (erf_results, erf_data,
     emidms_results, emidms_data,
     siconc_results, siconc_data,
     mmrso4_results, mmrso4_data) = analyze_erf_emidms_siconc(model_data, time_slice=time_slice)

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for model in siconc_data.keys():
        # --- pick SICONC
        sic_da = None
        if "control" in siconc_data[model] and region in siconc_data[model]["control"]:
            sic_da = siconc_data[model]["control"][region]
        elif "2xDMS" in siconc_data[model] and region in siconc_data[model]["2xDMS"]:
            sic_da = siconc_data[model]["2xDMS"][region]

        if sic_da is None:
            print(f"⚠️ {model}: no SICONC for {region}; skip both steps.")
            continue

        # --- pick EMIDMS for Step 1
        em_da = None
        if preferred_em_exp in emidms_data.get(model, {}) and region in emidms_data[model][preferred_em_exp]:
            em_da = emidms_data[model][preferred_em_exp][region]
        else:
            # fallback to whichever exp exists
            for exp in ("control", "2xDMS"):
                if exp in emidms_data.get(model, {}) and region in emidms_data[model][exp]:
                    em_da = emidms_data[model][exp][region]; break

        # --- pick ERF for Step 2 (analyzer stores by model→region)
        erf_da = erf_data.get(model, {}).get(region, None)

        # Run steps
        step1 = (np.nan, np.nan)
        if em_da is not None:
            try:
                step1 = run_step1_spatial(sic_da, em_da,
                                          region_label=f"{model} {region}",
                                          save_dir=save_dir,
                                          fname_prefix=f"step1_{model}_{region}")
            except Exception as e:
                print(f"⚠️ Step1 failed for {model}: {e}")

        step2 = None
        if erf_da is not None:
            try:
                step2 = run_step2_partition(sic_da, erf_da,
                                            region_label=f"{model} {region}",
                                            save_dir=save_dir,
                                            fname_prefix=f"step2_{model}_{region}")
            except Exception as e:
                print(f"⚠️ Step2 failed for {model}: {e}")

        results[model] = {"step1_slope_intercept": step1, "step2_shares": step2}

    return results
