import matplotlib.colors as mcolors
import xarray as xr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from watermark import watermark
from erf_calculator import calculate_mean_erf
from load_data import load_all_data
#from plot_data_maps import plot_all_data_maps
from panel_plot import plot_emidms_panel_sep_cbar
from analyze_erf_emidms_siconc import analyze_erf_emidms_siconc, print_results
import os
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from diagnose_model_inputs import diagnose_model_inputs
import cartopy.crs as ccrs, cartopy.feature as cfeature
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import gc
from plot_siconc import plot_siconc_maps
from plot_siconc import _draw_single_siconc
import os, numpy as np, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import math