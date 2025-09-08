# linkage_sulfate_graphs.py
# Academic 2×2 summary of the DMS → sulfate linkage, per model.
# Robust to mixed grids: everything is aligned to the SICONC grid first.

import os, math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.spatial import cKDTree
import importlib, linkage_sulfate_graphs as lsg
importlib.reload(lsg)

# ---------- global style ----------
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

# ---------- coord helpers ----------
def _find_lat_lon_names(da):
    cand_lat = ["lat","latitude","nav_lat","j","y"]
    cand_lon = ["lon","longitude","nav_lon","i","x"]
    la = next((n for n in cand_lat if (n in da.dims or n in da.coords)), None)
    lo = next((n for n in cand_lon if (n in da.dims or n in da.coords)), None)
    if la is None or lo is None:
        raise ValueError(f"lat/lon not found on {da.name or 'DataArray'}; dims={list(da.dims)} coords={list(da.coords)}")
    return la, lo

def _latlon_2d(da):
    la, lo = _find_lat_lon_names(da)
    latc = da.coords.get(la); lonc = da.coords.get(lo)
    # 2-D curvilinear
    if getattr(latc, "ndim", 1) == 2 and getattr(lonc, "ndim", 1) == 2:
        return latc.values, lonc.values
    # 1-D
    if (la in da.dims) and (lo in da.dims) and latc.ndim == 1 and lonc.ndim == 1:
        lat1d = latc.values; lon1d = lonc.values
        lat2d = np.tile(lat1d.reshape(-1,1), (1, lon1d.size))
        lon2d = np.tile(lon1d.reshape(1,-1), (lat1d.size, 1))
        return lat2d, lon2d
    # last resort
    if (latc is not None and getattr(latc, "ndim", 1) == 1) and (lonc is not None and getattr(lonc, "ndim", 1) == 1):
        lat2d = np.tile(latc.values.reshape(-1,1), (1, lonc.size))
        lon2d = np.tile(lonc.values.reshape(1,-1), (latc.size, 1))
        return lat2d, lon2d
    raise ValueError("Cannot build 2-D lat/lon for this DataArray.")

def _weights_like(da):
    lat2d, _ = _latlon_2d(da)
    return np.cos(np.deg2rad(lat2d)).clip(min=0.0)

def _to_percent(sic):
    vmax = float(np.nanmax(sic.values))
    return sic * 100.0 if vmax <= 1.01 else sic

# ---------- regridding ----------
def _ll_to_unitvec(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    cl = np.cos(lat)
    return np.stack([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], axis=-1)

def _regrid_to_target_nearest(src_da, tgt_da, k=4):
    """Nearest-neighbor on the sphere using KDTree (works for curvilinear)."""
    s_lat2d, s_lon2d = _latlon_2d(src_da)
    t_lat2d, t_lon2d = _latlon_2d(tgt_da)
    s_xyz = _ll_to_unitvec(s_lat2d.ravel(), s_lon2d.ravel())
    t_xyz = _ll_to_unitvec(t_lat2d.ravel(), t_lon2d.ravel())
    tree = cKDTree(s_xyz)
    dist, idx = tree.query(t_xyz, k=min(k, s_xyz.shape[0]))
    if np.ndim(idx) == 1: idx = idx[:, None]
    s_vals = np.squeeze(src_da.values)
    s_flat = s_vals.reshape(-1)
    picked = np.full(idx.shape[0], np.nan)
    for j in range(idx.shape[1]):
        cand = s_flat[idx[:, j]]
        need = ~np.isfinite(picked)
        picked[need] = cand[need]
        if not np.isnan(picked).any(): break
    out = picked.reshape(t_lat2d.shape)
    return xr.DataArray(out, dims=tgt_da.dims,
                        coords={d: tgt_da.coords[d] for d in tgt_da.dims if d in tgt_da.coords},
                        attrs=src_da.attrs, name=src_da.name)

def _align_to(src_da, tgt_da):
    """Align src_da to tgt_da's grid (prefer bilinear for 1-D, else nearest)."""
    try:
        sla, slo = _find_lat_lon_names(src_da)
        tla, tlo = _find_lat_lon_names(tgt_da)
    except Exception:
        return src_da  # fall back; better than crashing
    # both 1-D → bilinear
    if (sla in src_da.dims and slo in src_da.dims and
        tla in tgt_da.dims and tlo in tgt_da.dims and
        getattr(src_da.coords.get(sla), "ndim", 1) == 1 and
        getattr(src_da.coords.get(slo), "ndim", 1) == 1 and
        getattr(tgt_da.coords.get(tla), "ndim", 1) == 1 and
        getattr(tgt_da.coords.get(tlo), "ndim", 1) == 1):
        try:
            return src_da.interp({sla: tgt_da.coords[tla], slo: tgt_da.coords[tlo]})
        except Exception:
            pass
    # otherwise → nearest neighbor on sphere
    return _regrid_to_target_nearest(src_da, tgt_da, k=4)

# ---------- statistics ----------
def _binned_percentiles(x, y, *, nbins=30, q_low=25, q_med=50, q_high=75):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.array([]), np.array([]), np.array([]), np.array([])
    xb = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), nbins + 1)
    idx = np.digitize(x[m], xb) - 1
    x_mid = 0.5 * (xb[:-1] + xb[1:])
    p25 = np.full(nbins, np.nan); p50 = np.full(nbins, np.nan); p75 = np.full(nbins, np.nan)
    for i in range(nbins):
        yi = y[m][idx == i]
        if yi.size >= 10:
            p25[i] = np.nanpercentile(yi, q_low)
            p50[i] = np.nanpercentile(yi, q_med)
            p75[i] = np.nanpercentile(yi, q_high)
    return x_mid, p25, p50, p75

def _regime_shares(val2d, sic_percent2d, w2d,
                   bins=(0.0, 15.0, 80.0, 100.0), use_abs=True):
    """Return shares (%) across SIC regimes that sum to ~100%."""
    v = np.abs(val2d) if use_abs else val2d
    sic = sic_percent2d
    W = w2d

    # joint finite mask (must have SIC too)
    m = np.isfinite(v) & np.isfinite(W) & np.isfinite(sic)
    if m.sum() == 0:
        return [np.nan, np.nan, np.nan]

    v = v[m]; W = W[m]; sic = np.clip(sic[m], 0.0, 100.0)

    total = np.nansum(v * W)
    if not np.isfinite(total) or total == 0:
        return [np.nan, np.nan, np.nan]

    # 0: <15, 1: 15–80, 2: ≥80 (incl. 100)
    codes = np.digitize(sic, [15.0, 80.0], right=False)
    shares = []
    for code in (0, 1, 2):
        sel = codes == code
        shares.append(float(100.0 * np.nansum(v[sel] * W[sel]) / total))

    # tiny renormalization for rounding drift
    s = np.nansum(shares)
    if np.isfinite(s) and 99.5 <= s <= 100.5:
        shares = [float(x * 100.0 / s) for x in shares]

    return shares

# ---------- core data prep ----------
def compute_linkage_arrays(mmrso4_data, siconc_data, emidms_data, erf_data,
                           *, model, region="Arctic", em_exp="control"):
    """
    Returns dict with arrays on the SICONC grid (exact same 2-D shape):
      ow2d  : open-water fraction (0..1)
      em2d  : EMIDMS (aligned) or None
      dmm2d : Δmmrso4 = (2xDMS - control) aligned
      erf2d : ERF aligned or None
      w2d   : area weights
      sicp2d: SICONC in %
    """
    # --- SICONC (prefer control)
    sic = (siconc_data.get(model, {}).get("control", {}) or
           siconc_data.get(model, {}).get("2xDMS", {})).get(region, None)
    if sic is None:
        raise ValueError(f"No SICONC for {model}/{region}")
    sic  = sic.mean("time") if "time" in sic.dims else sic
    sicp = _to_percent(sic)
    w2d  = _weights_like(sicp)

    # --- EMIDMS
    em2d = None
    if em_exp in emidms_data.get(model, {}) and region in emidms_data[model][em_exp]:
        em = emidms_data[model][em_exp][region]
    else:
        em = None
        for e in ("control","2xDMS"):
            if e in emidms_data.get(model, {}) and region in emidms_data[model][e]:
                em = emidms_data[model][e][region]; break
    if em is not None:
        em = em.mean("time") if "time" in em.dims else em
        em2d = _align_to(em, sicp)

    # --- sulfate
    mm_c = mmrso4_data[model]["control"][region]
    mm_2 = mmrso4_data[model]["2xDMS"][region]
    mm_c = mm_c.mean("time") if "time" in mm_c.dims else mm_c
    mm_2 = mm_2.mean("time") if "time" in mm_2.dims else mm_2
    mm_c = _align_to(mm_c, sicp)
    mm_2 = _align_to(mm_2, sicp)
    dmm2d = (mm_2 - mm_c)

    # --- ERF (optional)
    erf2d = None
    if erf_data is not None:
        erf = erf_data.get(model, {}).get(region, None)
        if erf is not None:
            erf = erf.mean("time") if "time" in erf.dims else erf
            erf2d = _align_to(erf, sicp)

    ow2d = (100.0 - sicp.values) / 100.0
    return dict(ow2d=ow2d, em2d=None if em2d is None else em2d.values,
                dmm2d=dmm2d.values, erf2d=None if erf2d is None else erf2d.values,
                w2d=w2d, sicp2d=sicp.values)

# ---------- plotting ----------
def plot_sulfate_linkage_model(mmrso4_data, siconc_data, emidms_data, erf_data=None,
                               *, model="GISS", region="Arctic",
                               em_exp="control", save_dir="images",
                               fname=None, unit_mmr="kg kg$^{-1}$"):
    """
    Draw a 2×2 academic summary for `model`:
      (1) EMIDMS vs open-water (hexbin + median/IQR)
      (2) Δmmrso4 vs open-water (hexbin + median/IQR)
      (3) |Δmmrso4| shares by SICONC regime
      (4) |ERF| shares by SICONC regime (if available)
    """
    os.makedirs(save_dir, exist_ok=True)
    D = compute_linkage_arrays(mmrso4_data, siconc_data, emidms_data, erf_data,
                               model=model, region=region, em_exp=em_exp)

    # scale Δmmrso4 axis label nicely
    finite = np.abs(D["dmm2d"][np.isfinite(D["dmm2d"])])
    scale = 1.0; pow10 = 0
    if finite.size:
        m = np.nanmax(finite)
        pow10 = int(math.floor(math.log10(m))) if m > 0 else 0
        # snap to nearest 3-decade step for readability
        pow10 = int(np.floor(pow10 / 3) * 3)
        scale = 10.0 ** (-pow10)
    dmm_scaled = D["dmm2d"] * scale
    dmm_label  = rf"$\Delta$mmrso4 [{10**pow10:.0e} {unit_mmr}]" if pow10 != 0 else rf"$\Delta$mmrso4 [{unit_mmr}]"

    fig, axs = plt.subplots(2, 2, figsize=(8.8, 6.6), constrained_layout=True)

    # 1) EMIDMS vs open-water (only if EMIDMS available)
    ax = axs[0,0]
    if D["em2d"] is not None:
        x = D["ow2d"].ravel()
        y = D["em2d"].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        hb = ax.hexbin(x[m], y[m], gridsize=45, mincnt=5, bins='log')
        cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.04, aspect=30); cb.set_label("log10(count)")
        xm, p25, p50, p75 = _binned_percentiles(x[m], y[m], nbins=30)
        if xm.size:
            ax.plot(xm, p50, lw=1.8, label="Median")
            ax.fill_between(xm, p25, p75, alpha=0.25, linewidth=0, label="IQR")
        ax.set_title(f"{model} – {region}\nEMIDMS vs open water")
        ax.set_xlabel("Open-water fraction (1 − SICONC)")
        ax.set_ylabel("EMIDMS")
        ax.set_xlim(0, 1); #ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No EMIDMS available", ha="center", va="center"); ax.set_axis_off()

    # 2) Δmmrso4 vs open-water
    ax = axs[0,1]
    x = D["ow2d"].ravel()
    y = dmm_scaled.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(x[m], y[m], gridsize=45, mincnt=5, bins='log')
    cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.04, aspect=30); cb.set_label("log10(count)")
    xm, p25, p50, p75 = _binned_percentiles(x[m], y[m], nbins=30)
    if xm.size:
        ax.plot(xm, p50, lw=1.8, label="Median")
        ax.fill_between(xm, p25, p75, alpha=0.25, linewidth=0, label="IQR")
    ax.set_title(r"$\Delta$mmrso4 vs open water")
    ax.set_xlabel("Open-water fraction (1 − SICONC)"); ax.set_ylabel(dmm_label)
    ax.set_xlim(0, 1); #ax.legend(frameon=False, fontsize=8, loc="lower right")

    # 3) |Δmmrso4| regime shares
    ax = axs[1,0]
    shares_mmr = _regime_shares(D["dmm2d"], D["sicp2d"], D["w2d"], use_abs=True)
    labels = ["Open (<15%)", "MIZ (15–80%)", "Pack (>80%)"]
    bars = ax.bar(labels, shares_mmr, width=0.6)
    y_max = np.nanmax(shares_mmr) if np.isfinite(shares_mmr).any() else 1.0
    ax.set_ylim(0, y_max * 1.18)
    ax.set_ylabel("% of total |Δmmrso4|")
    ax.set_title("Contribution by sea-ice regime")
    ax.bar_label(bars, labels=[f"{v:.1f}%" if np.isfinite(v) else "—" for v in shares_mmr],
                 padding=3, fontsize=9)
    for s in ("top","right"): ax.spines[s].set_visible(False)

    # 4) |ERF| regime shares (if ERF provided)
    ax = axs[1,1]
    if D["erf2d"] is not None:
        shares_erf = _regime_shares(D["erf2d"], D["sicp2d"], D["w2d"], use_abs=True)
        bars = ax.bar(labels, shares_erf, width=0.6, color="#6c8ebf")
        y_max2 = np.nanmax(shares_erf) if np.isfinite(shares_erf).any() else 1.0
        ax.set_ylim(0, y_max2 * 1.18)
        ax.set_ylabel("% of total |ERF|")
        ax.set_title("ERF concentration by regime")
        ax.bar_label(bars, labels=[f"{v:.1f}%" if np.isfinite(v) else "—" for v in shares_erf],
                     padding=3, fontsize=9)
        for s in ("top","right"): ax.spines[s].set_visible(False)
    else:
        ax.text(0.5, 0.5, "ERF field unavailable", ha="center", va="center"); ax.set_axis_off()

    if fname is None:
        fname = f"linkage_{model}_{region}.png".replace(" ", "_")
    out = os.path.join(save_dir, fname)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Saved: {out}")
