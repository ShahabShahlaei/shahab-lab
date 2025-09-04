from Common_import2 import *
def analyze_emidms(model_data: dict,
                   time_slice = (None, None),
                   regions    = {"Global":    (-90,  90),
                                 "Arctic":     (66,  90),
                                 "Antarctic": (-90, -66)},
                   var        = "emidms"):
    """
    Return two dicts (same structure):
        emidms_results   {model → exp → region → number}
        emidms_data      {model → exp → region → DataArray}
    """
    res:  dict[str, dict] = {}
    data: dict[str, dict] = {}

    for mdl, exps in model_data.items():
        res [mdl] = {}
        data[mdl] = {}

        for exp, fields in exps.items():
            if var not in fields:
                continue

            da = fields[var]

            # optional time slicing
            if time_slice is not None and "time" in da.dims:
                da = da.isel(time=slice(*time_slice))
            da = da.mean("time") if "time" in da.dims else da

            # weights: cos(lat)
            lat = da["lat"] if "lat" in da.coords else da["latitude"]

            res [mdl][exp] = {}
            data[mdl][exp] = {}

            for reg, (lo, hi) in regions.items():
                sub = da.where((lat >= lo) & (lat <= hi), drop=True)
                w   = np.cos(np.deg2rad(lat)).where((lat >= lo)&(lat<=hi)).fillna(0)
                mu  = sub.weighted(w).mean(("lat", "lon")).compute().values.item()

                res [mdl][exp][reg] = mu
                data[mdl][exp][reg] = sub

    return res, data


# ---------------------------------------------------------------
# 2)   pretty printer   (auto-scales to n/µ/m/k/M/…)
# ---------------------------------------------------------------
_SI = {e: p for e, p in zip(
    range(-24, 25, 3),
    "y z a f p n µ m  k M G T P E Z Y".split())}

def _choose_scale(values):
    vals = [abs(v) for v in values if math.isfinite(v) and v != 0]
    if not vals:
        return 1.0, ""
    exp3 = int(math.floor(math.log10(max(vals))/3)*3)
    nearest = min(_SI, key=lambda e: abs(e-exp3))
    return 10.0**(-nearest), _SI[nearest]

def print_emidms(res, unit="kg m⁻² s⁻¹", prec=4, auto=True):
    """Nicely print the results dict produced above."""
    scale, prefix = (1.0, "") if not auto else _choose_scale(
        v for mdl in res.values()
          for exp in mdl.values()
          for v   in exp.values())
    unit2 = f"{prefix}{unit}"

    for mdl, exps in res.items():
        print(f"\nModel: {mdl}")
        for exp, rd in exps.items():
            print(f"  Experiment: {exp}")
            for reg, val in rd.items():
                print(f"    {reg:<9}: {(val*scale):.{prec}e} {unit2}")
