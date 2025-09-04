# --- NEW/UPDATED CODE ---
from Common_import2 import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ----------------------------- helpers -----------------------------

def _prepare_lon_lat(da):
    """Return lon2d, lat2d for both regular & curvilinear grids."""
    dims = set(da.dims)

    if {"lat", "lon"}.issubset(dims):  # 1D
        lon2d, lat2d = np.meshgrid(da.lon, da.lat)
    elif {"j", "i"}.issubset(dims) and {"latitude", "longitude"}.issubset(da.coords):  # 2D curvilinear
        lat2d, lon2d = da.latitude, da.longitude
    else:
        raise ValueError("Cannot find latitude/longitude in DataArray")
    return lon2d, lat2d


from matplotlib.path import Path
import numpy as np

def _circular_polar_boundary(ax, radius=0.50):
    """
    Give polar stereographic maps a clean circular frame.
    The boundary must be centred at (0.5, 0.5) in axes coordinates,
    otherwise the map is clipped incorrectly.
    """
    theta = np.linspace(0, 2 * np.pi, 361)
    # shift the circle to the centre (0.5,0.5) of the axes
    center_x, center_y = 0.5, 0.5
    verts = np.vstack([
        center_x + radius * np.sin(theta),
        center_y + radius * np.cos(theta)
    ]).T
    ax.set_boundary(Path(verts), transform=ax.transAxes)


def _add_region_labels(ax, region, transform,
                       base_fontsize=5, marker_size=0.5, marker_color="black"):
    """
    Add scientific area names with markers. Automatically scales font size so
    longer names are plotted smaller. Markers anchor each label on the map.
    
    Parameters
    ----------
    region : str
        "arctic", "antarctic" or "global".
    transform : cartopy coordinate transform
        The coordinate system (normally PlateCarree) for lon/lat.
    base_fontsize : int or float, optional
        Base font size. Short names use this; long names scale down.
    marker_size : int or float, optional
        Size of the dots indicating feature locations.
    marker_color : str, optional
        Colour of the dots.
    """
    region = region.lower()

    # Define label locations (lon, lat). Feel free to add more names here.
    if region == "arctic":
        labels = {
            "Bea":      (-150, 73),
            "Chu":       (-168, 69),
            "Eas": (155,  73),
            "Lap":        (125,  76),
            "Kar":          (75,   73),
            "Bar":       (40,   74),
            "Gre":     (-15,  75),
            "Baf":        (-65,  70),
            "Hud":        (-90,  59),   # additional example
            "Nor":     (0,    71),   # additional example
            "Arctic":      (0,    88),
        }
    elif region == "antarctic":
        labels = {
            "Weddell Sea":        (-45, -73),
            "Ross Sea":           (175, -75),
            "Amundsen Sea":       (-115, -72),
            "Bellingshausen Sea": (-75, -70),
            "Scotia Sea":         (-40, -56),
            "Southern Ocean":     (0,   -62),
        }
    else:  # global oceans
        labels = {
            "Pacific Ocean":  (-140,  0),
            "Atlantic Ocean": (-30,  -10),
            "Indian Ocean":   (85,   -15),
            "Arctic Ocean":   (0,    80),
            "Southern Ocean": (20,   -55),
        }

    for name, (lon, lat) in labels.items():
        # Plot a small marker at the location
        ax.plot(lon, lat, marker="o", markersize=marker_size,
                markeredgecolor=marker_color, markerfacecolor=marker_color,
                transform=transform, zorder=5)

        # Adjust font size: shrink text if the name is long
        length = len(name)
        if length <= 12:
            fontsize = base_fontsize
        elif length <= 20:
            fontsize = base_fontsize - 1
        else:
            fontsize = base_fontsize - 2

        # Add text slightly offset from the marker (customise ha, va for neatness)
        ax.text(lon + 3, lat + 3, name,
                transform=transform, fontsize=fontsize,
                ha="left", va="bottom", zorder=10)


def _add_land_ocean(ax, land_color="white", ocean_color="#D6ECFF"):
    """
    Paint ocean first (as a facecolor & feature), then draw land on top so
    land is clean white regardless of data coverage.
    """
    # ocean background
    ax.set_facecolor(ocean_color)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=ocean_color, zorder=0)

    # land overlay (drawn *after* data mesh in plot function via zorder)
    land = cfeature.LAND.with_scale("50m")
    ax.add_feature(land, facecolor=land_color, edgecolor="0.5", linewidth=0.3, zorder=3)

    # coastlines on top
    ax.coastlines(resolution="50m", linewidth=0.4, color="0.3", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.35", zorder=4)


# ----------------------------- main plot -----------------------------

def plot_data_map(data, model, region,
                  variable_name="Variable", save_dir="pic",
                  cmap="turbo", unit="", experiment="",
                  smooth=False, dpi=300,
                  land_color="white", ocean_color="#D6ECFF",
                  add_area_names=True,  # <-- new
                  annotate_source=False # add a tiny footer if you want
                  ):
    """Draw one map (time averaged outside if you like).

    New features:
      - land drawn as pure white, ocean soft blue
      - region/place labels (Arctic/Antarctic seas, global oceans)
      - circular polar frame
      - improved gridlines/ticks
    """
    if isinstance(data, dict):
        raise ValueError("`data` must be an xarray.DataArray")

    if "time" in data.dims:
        data = data.mean("time")

    lon2d, lat2d = _prepare_lon_lat(data)
    masked_data = np.ma.masked_invalid(data.compute())
    if masked_data.mask.all():
        print(f"⚠ {model}/{experiment}/{region}: only NaNs → skip plot")
        return

    # ── projection
    reg = region.lower()
    projection = {"arctic": ccrs.NorthPolarStereo(),
                  "antarctic": ccrs.SouthPolarStereo()}.get(reg, ccrs.PlateCarree())

    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=projection)

    if reg == "global":
        ax.set_global()
    elif reg == "arctic":
        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        _circular_polar_boundary(ax)
    elif reg == "antarctic":
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        _circular_polar_boundary(ax)

    # ── base map styling
    _add_land_ocean(ax, land_color=land_color, ocean_color=ocean_color)

    # ── data colormap/norm
    shade_kw = dict(shading="gouraud") if smooth else dict(shading="auto")
    data_zorder = 2  # under coastlines/land borders (zorder 3–4)

    if np.nanmin(masked_data) >= 0:
        mesh = ax.pcolormesh(lon2d, lat2d, masked_data,
                             cmap=cmap, transform=ccrs.PlateCarree(),
                             zorder=data_zorder, **shade_kw)
        cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal",
                            pad=0.07, shrink=0.55)
    else:
        vmax = np.nanpercentile(np.abs(masked_data), 99)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        mesh = ax.pcolormesh(lon2d, lat2d, masked_data,
                             cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree(),
                             zorder=data_zorder, **shade_kw)
        cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal",
                            pad=0.07, shrink=0.55, extend="both")

    cbar.set_label(f"{variable_name} ({unit})")

    # ── gridlines / labels
    if reg == "global":
        gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                          color="gray", alpha=0.4, linestyle="--")
        gl.right_labels = False
        gl.top_labels = False
    else:
        ax.gridlines(draw_labels=False, linewidth=0.4,
                     color="gray", alpha=0.35, linestyle="--")

    # ── area names
    if add_area_names:
        _add_region_labels(ax, reg, transform=ccrs.PlateCarree())

    # ── title & optional footer
    title = f"{model} – {region} – {variable_name}"
    if experiment:
        title += f" ({experiment})"
    plt.title(title)

    if annotate_source:
        plt.annotate("Projection: Cartopy; Natural Earth 1:50m",
                     xy=(0.01, 0.01), xycoords="figure fraction",
                     fontsize=6, alpha=0.7)

    plt.tight_layout()

    # ── save
    os.makedirs(save_dir, exist_ok=True)
    safe_exp = str(experiment)[:30]
    fname = f"{model}_{region}_{variable_name}_{safe_exp}.png"
    plt.savefig(os.path.join(save_dir, fname.replace(" ", "_")), dpi=dpi)
    plt.close()


# unchanged, but now benefits from the nicer single-plot:
def plot_all_data_maps(data_dict, variable_name="ERF", save_dir="images", **plot_kwargs):
    """
    Walk results dict and call plot_data_map().
    Supports:
      {model: {experiment: {region: DataArray}}}
      {model: {region: DataArray}}
    """
    if isinstance(data_dict, xr.DataArray):
        plot_data_map(data_dict, model="unknown", region="global",
                      variable_name=variable_name, save_dir=save_dir, **plot_kwargs)
        return

    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dict or a DataArray")

    for model, sub in data_dict.items():
        if not sub:
            continue

        nested = any(isinstance(v, dict) for v in sub.values())

        if nested:  # {model→exp→region}
            for exp, reg_dict in sub.items():
                for region, da in reg_dict.items():
                    if not isinstance(da, xr.DataArray):      continue
                    if da.ndim < 2 or np.isnan(da).all():     continue
                    print(f"Plotting {model} / {exp} / {region}")
                    plot_data_map(da, model, region,
                                  variable_name, save_dir,
                                  experiment=exp, **plot_kwargs)
        else:  # {model→region}
            for region, da in sub.items():
                if not isinstance(da, xr.DataArray):          continue
                if da.ndim < 2 or np.isnan(da).all():         continue
                print(f"Plotting {model} / {region}")
                plot_data_map(da, model, region,
                              variable_name, save_dir, **plot_kwargs)
