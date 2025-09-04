from common_imports import *          # xarray, numpy, etc.
def analyze_erf_emidms_siconc(model_data, time_slice=(None, None)):
    """
    Return six dictionaries:
      erf_results / erf_data / emidms_results / emidms_data /
      siconc_results / siconc_data / mmrso4_data / mmrso4_results
    Each holds   {model → … → region}.
    """

    regions = {"Global": (-90,  90),
               "Arctic": ( 66,  90),
               "Antarctic": (-90, -66)}

    erf_results = {}; erf_data = {}
    emidms_results = {}; emidms_data = {}
    siconc_results = {}; siconc_data = {}
    mmrso4_results = {}; mmrso4_data = {}
    

    # ------------------------------------------------------------------
    for model, mdl in model_data.items():
        erf_results   [model] = {}
        erf_data      [model] = {}
        emidms_results[model] = {}
        emidms_data   [model] = {}
        siconc_results[model] = {}
        siconc_data   [model] = {}
        mmrso4_results[model] = {}                                  
        mmrso4_data   [model] = {}

        # loop control / 2xDMS
        for exp in ("control", "2xDMS"):
            if exp not in mdl:
                continue

            # =============================== EMIDMS
            if "emidms" in mdl[exp]:
                em = mdl[exp]["emidms"]
                if time_slice is not None and "time" in em.dims:
                    em = em.isel(time=slice(*time_slice))
                em = em.mean("time") if "time" in em.dims else em

                emidms_results[model][exp] = {}
                emidms_data   [model][exp] = {}

                for reg, (lat0, lat1) in regions.items():
                    em_reg = em.where((em.lat >= lat0) & (em.lat <= lat1), drop=True)
                    emidms_data[model][exp][reg] = em_reg
                    w = np.cos(np.deg2rad(em_reg.lat))
                    emidms_results[model][exp][reg] = em_reg.weighted(w).mean(("lat", "lon")).item()

            # =============================== mmrso4
            if "mmrso4" in mdl[exp]:
                so4 = mdl[exp]["mmrso4"]
                # select surface level in 4D
                if "lev" in so4.dims:
                    so4 = so4.isel(lev=0).drop_vars("lev")
                if time_slice is not None and "time" in so4.dims:
                    so4 = so4.isel(time=slice(*time_slice))
                so4 = so4.mean("time") if "time" in so4.dims else so4

                mmrso4_results[model][exp] = {}
                mmrso4_data   [model][exp] = {}

                for reg, (lat0, lat1) in regions.items():
                    so4_reg = so4.where((so4.lat >= lat0) & (so4.lat <= lat1), drop=True)
                    mmrso4_data[model][exp][reg] = so4_reg
                    w = np.cos(np.deg2rad(so4_reg.lat))
                    # choose only the dims that are really present
                    dims_to_mean = tuple(d for d in ("lat", "lon") if d in so4_reg.dims)
                    mmrso4_results[model][exp][reg] = (so4_reg.weighted(w).mean(dims_to_mean).item()
)
        

            # =============================== SICONC
            if "siconc" in mdl[exp]:
                sic = mdl[exp]["siconc"]
                if time_slice is not None and "time" in sic.dims:
                    sic = sic.isel(time=slice(*time_slice))
                sic = sic.mean("time") if "time" in sic.dims else sic

                # grid discovery
                if {"j", "i"}.issubset(sic.dims) and "latitude" in sic.coords:
                    lat2d, grid = sic["latitude"], "curvi"
                elif {"lat", "lon"}.issubset(sic.dims):
                    lat2d, grid = sic["lat"], "regular"
                else:
                    print(f"⚠ {model}/{exp}: cannot locate latitude → skip SICONC")
                    continue

                siconc_results[model][exp] = {}
                siconc_data   [model][exp] = {}

                for reg, (lat0, lat1) in regions.items():
                    print(f"🔷 {model} – {exp} – {reg}")

                    if grid == "curvi":
                        lat_clean = lat2d.where((lat2d > -90) & (lat2d < 90))
                        mask      = (lat_clean >= lat0) & (lat_clean <= lat1)
                        sic_reg   = sic.where(mask)
                        w         = np.cos(np.deg2rad(lat_clean)).where(mask).fillna(0)
                    else:
                        mask      = (lat2d >= lat0) & (lat2d <= lat1)
                        sic_reg   = sic.where(mask)
                        w         = np.cos(np.deg2rad(lat2d)).where(mask).fillna(0)

                    # 🔸 NEW — if *within the mask* everything is NaN, set to 0
                    if np.isnan(sic_reg).all():
                        sic_reg = xr.zeros_like(sic_reg)

                    siconc_data[model][exp][reg] = sic_reg

                    mu = sic_reg.weighted(w).mean(("j", "i") if grid == "curvi" else ("lat", "lon"))
                    siconc_results[model][exp][reg] = mu.item()

        # =============================== ERF
        if "lat" not in mdl["2xDMS"]["rlut"].dims:
            print(f"Skipping ERF for {model} (no lat dim)")
            continue

        for reg, (lat0, lat1) in regions.items():
            erf, mu = calculate_mean_erf(
                rlut_2xdms = mdl["2xDMS"]["rlut"],
                rsut_2xdms = mdl["2xDMS"]["rsut"],
                rsdt_2xdms = mdl["2xDMS"]["rsdt"],
                rlut_control = mdl["control"]["rlut"],
                rsut_control = mdl["control"]["rsut"],
                rsdt_control = mdl["control"]["rsdt"],
                lat_min = lat0, lat_max = lat1)
            erf_results[model][reg] = mu.item()
            erf_data   [model][reg] = erf

    return (erf_results, erf_data,
            emidms_results, emidms_data,
            siconc_results, siconc_data,
            mmrso4_results, mmrso4_data)



def print_results(results, label="ERF", unit="W/m²", precision=20, auto_prefix=False):
    """
    Print results for each model and experiment, optionally auto-scaling to an SI prefix.

    Parameters
    ----------
    results : dict
        Either {model: {region: value}} or {model: {experiment: {region: value}}}
    label : str
        Name of the variable (e.g. "ERF")
    unit : str
        Physical unit string (e.g. "W/m²")
    precision : int
        Number of decimal places in the mantissa
    auto_prefix : bool
        If True, automatically choose an SI prefix (n, µ, m, k, etc.)
    """
    import math

    # ——— embedded SI table ———
    si = {
        -24: "y",  # yocto
        -21: "z",  # zepto
        -18: "a",  # atto
        -15: "f",  # femto
        -12: "p",  # pico
         -9: "n",  # nano
         -6: "µ",  # micro
         -3: "m",  # milli
           0: "",   # (none)
          3: "k",  # kilo
          6: "M",  # mega
          9: "G",  # giga
         12: "T",  # tera
         15: "P",  # peta
         18: "E",  # exa
         21: "Z",  # zetta
         24: "Y",  # yotta
    }

    # decide scaling
    if auto_prefix:
        # gather all finite values
        vals = []
        for mdl, exps in results.items():
            if any(isinstance(v, dict) for v in exps.values()):
                for rd in exps.values():
                    vals += [v for v in rd.values() if math.isfinite(v)]
            else:
                vals += [v for v in exps.values() if math.isfinite(v)]

        if vals:
            m = max(abs(v) for v in vals)
            exp3 = int(math.floor(math.log10(m)/3)*3) if m>0 else 0
            # snap to nearest available exponent
            choices = sorted(si)
            exp3 = min(choices, key=lambda e: abs(e - exp3))
        else:
            exp3 = 0

        prefix = si[exp3]
        scale  = 10.0**(-exp3)
        unit   = f"{prefix}{unit}"
    else:
        scale = 1.0

    # printing
    for mdl, exps in results.items():
        print(f"\nModel: {mdl}")
        # nested? experiments→regions
        if isinstance(exps, dict) and any(isinstance(v, dict) for v in exps.values()):
            for exp_name, rd in exps.items():
                print(f"  Experiment: {exp_name}")
                for region, val in rd.items():
                    v = val * scale
                    print(f"    {region} {label}: {v:.{precision}e} {unit}")
        else:
            # flat: regions only
            for region, val in exps.items():
                v = val * scale
                print(f"  {region} {label}: {v:.{precision}e} {unit}")
