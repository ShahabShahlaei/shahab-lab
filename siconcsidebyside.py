from common_imports import *
import os
from plot_siconc import _region_proj, _coords_2d, _to_percent, _savefig
def plot_siconc_panel_side_by_side(
    siconc_dict, *,
    region="Arctic",
    exp="piControl",
    models=None,                 # e.g., ["EC-Earth3-AerChem", "GISS-E2-1-H", "NorESM2-LM"]
    save_dir="plots_siconc",
    filename=None,               # default auto name if None
    cmap="Blues",
    vmin=0.0, vmax=100.0,        # fixed scale for comparability
    dpi=300,
    smooth=True,
    add_ice_edge=True
):
    """
    Make ONE figure with multiple models for the same region/experiment, sharing ONE colorbar.
    Side-by-side layout, consistent with your single-map style.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Determine which models to include
    all_models = list(siconc_dict.keys()) if models is None else list(models)
    selected = []
    for m in all_models:
        da = (
            siconc_dict.get(m, {})
                      .get(exp, {})
                      .get(region, None)
        )
        if (da is not None) and not (np.isnan(da).all()):
            selected.append((m, da))
    if len(selected) == 0:
        print(f"⚠️ No valid SICONC found for region={region}, exp={exp}.")
        return

    ncols = len(selected)
    proj = _region_proj(region)

    fig = plt.figure(figsize=(4.6 * ncols, 5.6), dpi=dpi)
    axes = []
    meshes = []

    shade_kw = dict(shading="gouraud") if smooth else dict(shading="auto")

    for i, (model, da) in enumerate(selected, start=1):
        ax = fig.add_subplot(1, ncols, i, projection=proj)
        axes.append(ax)

        # Time-mean for the map
        da_mean = da.mean("time") if "time" in da.dims else da

        lon2d, lat2d = _coords_2d(da_mean)
        arr = np.ma.masked_invalid(_to_percent(da_mean).values)

        if region.lower() == "global":
            ax.set_global()
        # Plot field
        mesh = ax.pcolormesh(
            lon2d, lat2d, arr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), **shade_kw
        )
        meshes.append(mesh)

        # 15% ice edge
        if add_ice_edge:
            try:
                ax.contour(
                    lon2d, lat2d, arr, levels=[15.0],
                    transform=ccrs.PlateCarree(), linewidths=0.8
                )
            except Exception:
                pass

        # Styling
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, lw=0.3)
        ax.set_title(f"{model} ({exp}) – {region}")

    # ONE shared colorbar across all subplots
    cbar = fig.colorbar(meshes[0], ax=axes, orientation="horizontal", pad=0.06,fraction=0.03, shrink=0.9)
    cbar.set_label("Sea-ice concentration (%)")

    # Save
    if filename is None:
        filename = f"SICONC_panel_{region}_{exp}.png".replace(" ", "_")
    outpath = os.path.join(save_dir, filename)
    _savefig(fig, outpath)
    print(f"✅ Saved panel: {outpath}")
