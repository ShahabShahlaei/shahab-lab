def plot_emidms_panel_sep_cbar(
    data,
    variable="emidms",
    unit="kg m⁻² s⁻¹",
    cmap="turbo",
    save_root="panel_plots",
    dpi=300,
    add_area_names=True,
    label_base_fontsize=8,
    label_marker_size=2.0,
    label_marker_color="black",
    label_offset=(1.6, 1.4)  # (Δlon°, Δlat°) offset of text from dot
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pathlib
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    import xarray as xr
    from matplotlib.path import Path
    from matplotlib.ticker import ScalarFormatter

    # ---------- global style ----------
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["figure.dpi"]  = dpi

    # ---------- helpers ----------
    def _prepare_lon_lat(da: xr.DataArray):
        dims = set(da.dims)
        if {"lat", "lon"}.issubset(dims):
            lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
        elif {"j","i"}.issubset(dims) and {"latitude","longitude"}.issubset(da.coords):
            lat2d, lon2d = da.latitude.values, da.longitude.values
        else:
            raise ValueError("Cannot find latitude/longitude in DataArray")
        return lon2d, lat2d

    def _circular_polar_boundary(ax, radius=0.50):
        theta = np.linspace(0, 2*np.pi, 361)
        cx, cy = 0.5, 0.5
        verts = np.vstack([cx + radius*np.sin(theta),
                           cy + radius*np.cos(theta)]).T
        ax.set_boundary(Path(verts), transform=ax.transAxes)

    def _add_land_ocean(ax, land_color="white", ocean_color="#D6ECFF"):
        ax.set_facecolor(ocean_color)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor=ocean_color, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"),
                       facecolor=land_color, edgecolor="0.5",
                       linewidth=0.3, zorder=3)
        ax.coastlines(resolution="50m", linewidth=0.4, color="0.3", zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       linewidth=0.3, edgecolor="0.35", zorder=5)

    def _add_region_labels(ax, region, transform,
                           base_fontsize=5, marker_size=0.5, marker_color="black",
                           offset=(3, 3)):
        region = region.lower()
        if region == "arctic":
            labels = {
                "Bea":      (-150, 73),
                "Chu":       (-168, 69),
                "Eas": (155,  73),
                "Lap":        (125,  76),
                "Kar":          (75,   73),
                "Bar":       (40,   74),
                "Gre":     (-15,  75),
                "Nor":     (0,    71),
                "Baf":        (-65,  70),
                "Hud":        (-90,  59),
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
        else:
            labels = {
                "Pacific Ocean":  (-140,  0),
                "Atlantic Ocean": (-30,  -10),
                "Indian Ocean":   (85,   -15),
                "Arctic Ocean":   (0,    80),
                "Southern Ocean": (20,   -55),
            }

        dlon, dlat = offset
        for name, (lon, lat) in labels.items():
            ax.plot(lon, lat, marker="o", markersize=marker_size,
                    markeredgecolor=marker_color, markerfacecolor=marker_color,
                    transform=transform, zorder=6)
            L = len(name)
            fs = base_fontsize if L <= 13 else (base_fontsize-1 if L <= 20 else base_fontsize-2)
            ax.text(lon + dlon, lat + dlat, name, fontsize=fs,
                    ha="left", va="bottom", transform=transform, zorder=6)

    def _mask_region(da2d: xr.DataArray, region: str):
        lon2d, lat2d = _prepare_lon_lat(da2d)
        if region == "Arctic":
            return da2d.where(lat2d >= 66.0)
        elif region == "Antarctic":
            return da2d.where(lat2d <= -66.0)
        return da2d

    # ---------- layout ----------
    REGIONS        = ("Arctic",)           # keep as in your version
    scenario_names = ["control", "2xDMS"]
    NROWS, NCOLS   = 2, len(REGIONS)
    REGION_PROJ = {
        "Global": ccrs.PlateCarree(),
        "Arctic": ccrs.NorthPolarStereo(central_longitude=0.0),
        "Antarctic": ccrs.SouthPolarStereo(central_longitude=0.0),
    }

    # ---------- NEW: compute a COMMON norm per scenario across ALL models ----------
    row_norms_common = {}
    for scen in scenario_names:
        vals = []
        for _model, scenarios in data.items():
            if scen not in scenarios or variable not in scenarios[scen]:
                continue
            da_all = scenarios[scen][variable]
            da_base = da_all.mean("time") if "time" in da_all.dims else da_all
            for reg in REGIONS:
                try:
                    da = _mask_region(da_base, reg)
                    arr = da.values
                    if arr is None: 
                        continue
                    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
                    vals.append(arr)
                except Exception:
                    pass
        if vals:
            allv = np.ma.masked_invalid(np.concatenate([v.ravel() for v in vals]))
            if allv.count() > 0:
                vmax = np.nanpercentile(allv, 99.0)
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = float(np.nanmax(allv)) if np.isfinite(np.nanmax(allv)) else 1.0
                row_norms_common[scen] = mcolors.Normalize(vmin=0.0, vmax=vmax)
            else:
                row_norms_common[scen] = None
        else:
            row_norms_common[scen] = None

    # ---------- draw per-model figures (now using the COMMON row norms) ----------
    for model, scenarios in data.items():
        fig = plt.figure(figsize=(NCOLS * 5.0, NROWS * 7.0))
        fig.subplots_adjust(hspace=0.38, wspace=0.12)

        axes   = np.empty((NROWS, NCOLS), dtype=object)
        meshes = np.empty((NROWS, NCOLS), dtype=object)

        for r, scen in enumerate(scenario_names):
            for c, reg in enumerate(REGIONS):
                proj = REGION_PROJ[reg]
                ax = fig.add_subplot(NROWS, NCOLS, r*NCOLS + c + 1, projection=proj)
                axes[r, c] = ax

                if reg == "Global":
                    ax.set_global()
                elif reg == "Arctic":
                    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
                    _circular_polar_boundary(ax)
                elif reg == "Antarctic":
                    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
                    _circular_polar_boundary(ax)

                _add_land_ocean(ax, land_color="white", ocean_color="#D6ECFF")

                if reg == "Global":
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                                      color="gray", alpha=0.4, linestyle="--")
                    gl.right_labels = False; gl.top_labels = False
                else:
                    ax.gridlines(draw_labels=False, linewidth=0.4,
                                 color="gray", alpha=0.45, linestyle="--")

                ax.set_title(f"{reg}", fontsize=11)

                try:
                    da = scenarios[scen][variable]
                    da = da.mean("time") if "time" in da.dims else da
                    da = _mask_region(da, reg)
                    lon2d, lat2d = _prepare_lon_lat(da)
                    arr = np.where(np.isfinite(da.values), np.maximum(da.values, 0.0), np.nan)
                    mesh = ax.pcolormesh(
                        lon2d, lat2d, arr,
                        cmap=cmap,
                        norm=row_norms_common.get(scen),   # <-- shared scale per row
                        transform=ccrs.PlateCarree(),
                        shading="auto"
                    )
                    meshes[r, c] = mesh
                except Exception:
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_facecolor("lightgray")
                    meshes[r, c] = None

                if add_area_names:
                    _add_region_labels(ax, reg, transform=ccrs.PlateCarree(),
                                       base_fontsize=label_base_fontsize,
                                       marker_size=label_marker_size,
                                       marker_color=label_marker_color,
                                       offset=label_offset)

            # left-side row label
            try:
                axes[r, 0].text(-0.18, 0.5, scen, va="center", ha="right",
                                rotation=90, fontsize=12, fontweight="bold",
                                transform=axes[r, 0].transAxes)
            except Exception:
                pass

        # ---------- thin colorbars placed just below each row ----------
        plt.tight_layout(rect=[0.02, 0.12, 1.00, 0.95])
        fig.canvas.draw()

        CBAR_HEIGHT = 0.010
        GAP_TOP     = 0.020
        GAP_BOT     = 0.012
        LEFT_PAD    = 0.01

        for r, scen in enumerate(scenario_names):
            norm = row_norms_common.get(scen)
            if norm is None:
                continue
            row_axes = [axes[r, c] for c in range(NCOLS) if axes[r, c] is not None]
            xs0 = [ax.get_position().x0 for ax in row_axes]
            xs1 = [ax.get_position().x1 for ax in row_axes]
            y0s = [ax.get_position().y0 for ax in row_axes]
            row_left, row_right, row_bottom = min(xs0), max(xs1), min(y0s)

            gap = GAP_TOP if r == 0 else GAP_BOT
            left  = row_left + LEFT_PAD
            width = (row_right - row_left) - 2*LEFT_PAD
            bottom = max(row_bottom - gap - CBAR_HEIGHT, 0.055)

            cax = fig.add_axes([left, bottom, width, CBAR_HEIGHT])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

            # Consistent scientific notation formatting across figures
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-3, 3))   # force x10^n for very small/large values
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", format=fmt)
            cbar.formatter = fmt
            cbar.ax.tick_params(labelsize=8, pad=2)
            try:
                cbar.ax.xaxis.get_offset_text().set_size(8)  # the ×10^n text
            except Exception:
                pass
            cbar.update_ticks()
            cbar.set_label(f"{variable} ({unit})" if unit else f"{variable}",
                           fontsize=9, labelpad=2)

        # ---------- title & save ----------
        plt.suptitle(f"{model} – {variable}", fontsize=14, fontweight="bold")
        out_dir = pathlib.Path(save_root, model); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model}_{variable}_panel_sepCBAR.png".replace(" ", "_")
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)

        try:
            rel_path = out_path.relative_to(pathlib.Path.cwd())
        except Exception:
            rel_path = out_path.resolve()
        print(f"✓  {rel_path}")
