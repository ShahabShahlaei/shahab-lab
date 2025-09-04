import matplotlib.colors as mcolors
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from watermark import watermark
import os
import glob 
import matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import gc
import os, numpy as np, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import math
from load_data2 import open_emidms
from load_data2 import build_emidms_dict
from analyze_emidms_only import analyze_emidms, print_emidms
from plot_helpers import plot_all_data_maps, _prepare_lon_lat, _add_region_labels, _add_land_ocean, plot_data_map



# from erf_calculator import calculate_mean_erf
# from load_data import load_all_data
# from plot_data_maps import plot_all_data_maps
# from analyze_erf_emidms_siconc import analyze_erf_emidms_siconc, print_results
# from plot_siconc import plot_siconc_maps
# from plot_siconc import _draw_single_siconc
# from diagnose_model_inputs import diagnose_model_inputs