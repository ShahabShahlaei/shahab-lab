from common_imports import *
def calculate_mean_erf(rlut_2xdms, rsut_2xdms,
                  rsdt_2xdms, rlut_control, rsut_control,
                  rsdt_control, time_slice=(None,None), lat_min=-90, lat_max=90):
    import numpy as np
    import xarray as xr

    # Apply time slice if provided
    if time_slice is not None:
        rlut_2xdms = rlut_2xdms.isel(time=slice(*time_slice))
        rsut_2xdms = rsut_2xdms.isel(time=slice(*time_slice))
        rsdt_2xdms = rsdt_2xdms.isel(time=slice(*time_slice))
        rlut_control = rlut_control.isel(time=slice(*time_slice))
        rsut_control = rsut_control.isel(time=slice(*time_slice))
        rsdt_control = rsdt_control.isel(time=slice(*time_slice))

    # ✅ Align time dimension only, and ignore spatial mismatch for now
    rlut_2xdms, rsut_2xdms, rsdt_2xdms = xr.align(rlut_2xdms, rsut_2xdms, rsdt_2xdms, join='inner')
    rlut_control, rsut_control, rsdt_control = xr.align(rlut_control, rsut_control, rsdt_control, join='inner')

    # ✅ Broadcast control runs to 2xDMS spatial grid if needed (safe auto-match)
    rlut_control = rlut_control.broadcast_like(rlut_2xdms)
    rsut_control = rsut_control.broadcast_like(rsut_2xdms)
    rsdt_control = rsdt_control.broadcast_like(rsdt_2xdms)

    # Calculate net fluxes
    net_2xdms = rsdt_2xdms - rsut_2xdms - rlut_2xdms
    net_control = rsdt_control - rsut_control - rlut_control
    erf = net_2xdms - net_control

    # --- Latitude filtering ---
    erf = erf.where(
        (erf.lat >= lat_min) & (erf.lat <= lat_max),
        drop=True
    )

    # Average over time if time dimension exists
    if 'time' in erf.dims:
        erf = erf.mean(dim='time')

    weights = np.cos(np.deg2rad(erf.lat))
    mean_erf = erf.weighted(weights).mean(dim=['lat', 'lon'])

    return erf, mean_erf

