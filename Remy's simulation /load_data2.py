from Common_import2 import *
DL_ROOT = "./emidms_download"           
MODEL_KEY = "EC-AEREarth3"              
EXPS      = ["baseline", "future_ssp585",
             "seaice_contrib_ssp585", "SST_contrib_ssp585"]

def open_emidms(exp: str,
                dl_root: str,
                var: str = "emidms",
                model: str = "EC-Earth") -> xr.DataArray:
    """
    Open *one* experiment and return its EMIDMS DataArray (lazy / dask).

    Parameters
    ----------
    exp      : experiment name (“baseline”, “future_ssp585”, …)
    dl_root  : directory containing  dl_root/exp/*.nc
    var      : variable name in the NetCDF (default “emidms” – lower-case!)
    model    : model tag inside the filename (default “EC-Earth”)

    Raises
    ------
    FileNotFoundError – if no files match
    KeyError          – if *var* not inside dataset
    """
    pat   = os.path.join(dl_root,
                         exp,
                         f"EMIDMS_{model}_{exp}_monthly_singlelevel_*.nc")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No NetCDFs matched\n    {pat}")

    ds = xr.open_mfdataset(files,
                           combine="by_coords",
                           parallel=True,
                           chunks={"time": -1})      # one chunk = one file

    if var not in ds.data_vars:
        raise KeyError(f"{var!r} not in dataset. Available: {list(ds.data_vars)}")

    return ds[var]   # -> DataArray(time, lat, lon)


# -------------------------------------------------------------------
def build_emidms_dict(*,
                      dl_root: str,
                      model_key: str,
                      exps: tuple[str, ...] | list[str],
                      var: str = "emidms") -> dict[str, dict]:
    """
    Load **all** requested experiments and return an independent dictionary:

        {model_key:
            {exp:
                {var: DataArray}}}

    Example
    -------
    emidms_dict = build_emidms_dict(
        dl_root   = "./emidms_download",
        model_key = "EC-AEREarth3",
        exps      = ("baseline", "future_ssp585",
                     "seaice_contrib_ssp585", "SST_contrib_ssp585"))
    """
    out: dict[str, dict] = {model_key: {}}

    for exp in exps:
        da = open_emidms(exp,
                         dl_root = dl_root,
                         var     = var,
                         model   = model_key)
        out[model_key][exp] = {var: da}

        nt, ny, nx = da.sizes.values()
        print(f"✅ loaded {model_key}/{exp}: "
              f"{nt} mon × {ny} × {nx} (lazy)")

    return out