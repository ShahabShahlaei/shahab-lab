from common_imports import *
def load_all_data():
    """Load all model data and return as dictionaries."""
    data = {
        'UKESM': {
            '2xDMS': {
                'rlut': xr.open_dataset('rlut_Amon_UKESM1-0-LL_piClim-2xDMS_r1i1p1f4_gn_185001-189412.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_UKESM1-0-LL_piClim-2xDMS_r1i1p1f4_gn_185001-189412.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_UKESM1-0-LL_piClim-2xDMS_r1i1p1f4_gn_185001-189412.nc')['rsdt'],
                'mmrso4': xr.open_dataset('mmrso4_AERmon_UKESM1-0-LL_piClim-2xDMS_r1i1p1f4_gn_185001-189412.nc')['mmrso4']
            },
            'control': {
                'rlut': xr.open_dataset('rlut_Amon_UKESM1-0-LL_piClim-control_r1i1p1f4_gn_185001-189412.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_UKESM1-0-LL_piClim-control_r1i1p1f4_gn_185001-189412.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_UKESM1-0-LL_piClim-control_r1i1p1f4_gn_185001-189412.nc')['rsdt'],
                 'siconc': xr.open_dataset('siconc_SImon_UKESM1-0-LL_piControl_r1i1p1f2_gn_196001-204912.nc')['siconc'],
                'mmrso4': xr.open_dataset('mmrso4_AERmon_UKESM1-0-LL_piClim-control_r1i1p1f4_gn_185001-189412.nc')['mmrso4']
            }
        },
        'NorESM': {
            '2xDMS': {
                'rlut': xr.open_dataset('rlut_Amon_NorESM2-LM_piClim-2xDMS_r1i1p1f1_gn_000101-001012.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_NorESM2-LM_piClim-2xDMS_r1i1p1f1_gn_000101-001012.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_NorESM2-LM_piClim-2xDMS_r1i1p1f1_gn_000101-001012.nc')['rsdt'], 
                'emidms': xr.open_dataset('emidms_AERmon_NorESM2-LM_piClim-2xDMS_r1i1p1f1_gn_000101-001012.nc')['emidms'],
                'mmrso4': xr.open_dataset('mmrso4_AERmon_NorESM2-LM_piClim-2xDMS_r1i1p1f1_gn_000101-001012.nc')['mmrso4']
            },
            'control': {
                'rlut': xr.open_dataset('rlut_Amon_NorESM2-LM_piClim-control_r1i1p1f1_gn_000101-001012.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_NorESM2-LM_piClim-control_r1i1p1f1_gn_000101-001012.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_NorESM2-LM_piClim-control_r1i1p1f1_gn_000101-001012.nc')['rsdt'],
                'emidms': xr.open_dataset('emidms_AERmon_NorESM2-LM_piClim-control_r1i1p1f1_gn_000101-001012.nc')['emidms'],
                'siconc': xr.open_dataset('siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_160001-160912.nc')['siconc'],
                'mmrso4': xr.open_dataset('mmrso4_AERmon_NorESM2-LM_piClim-control_r1i1p1f1_gn_000101-001012.nc')['mmrso4']
            }
        },
        'GISS': {
            '2xDMS': {
                'rlut': xr.open_dataset('rlut_Amon_GISS-E2-1-G_piClim-2xDMS_r1i1p3f1_gn_195001-199012.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_GISS-E2-1-G_piClim-2xDMS_r1i1p3f1_gn_195001-199012.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_GISS-E2-1-G_piClim-2xDMS_r1i1p3f1_gn_195001-199012.nc')['rsdt'], 
                'emidms': xr.open_dataset('emidms_AERmon_GISS-E2-1-G_piClim-2xDMS_r1i1p3f1_gn_195001-199012.nc')['emidms'],
                'mmrso4': xr.open_dataset('mmrso4_AERmon_GISS-E2-1-G_piClim-2xDMS_r1i1p3f1_gn_195001-199012.nc')['mmrso4']
            },
            'control': {
                'rlut': xr.open_dataset('rlut_Amon_GISS-E2-1-G_piClim-control_r1i1p3f1_gn_195001-199012.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_GISS-E2-1-G_piClim-control_r1i1p3f1_gn_195001-199012.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_GISS-E2-1-G_piClim-control_r1i1p3f1_gn_195001-199012.nc')['rsdt'],
                'emidms': xr.open_dataset('emidms_AERmon_GISS-E2-1-G_piClim-control_r1i1p3f1_gn_195001-199012.nc')['emidms'],
                'siconc': xr.open_dataset('siconc_SImon_GISS-E2-1-H_piControl_r1i1p3f1_gr_200001-201912.nc')['siconc'],
                'mmrso4': xr.open_dataset('mmrso4_AERmon_GISS-E2-1-G_piClim-control_r1i1p3f1_gn_195001-199012.nc')['mmrso4']
            }
        },
        'EC-AEREarth3': {
            '2xDMS': {
                'rlut': xr.open_dataset('rlut_Amon_EC-Earth3-AerChem_piClim-2xDMS_r1i1p1f1_gr_185001-185012.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_EC-Earth3-AerChem_piClim-2xDMS_r1i1p1f1_gr_185001-185012.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_EC-Earth3-AerChem_piClim-2xDMS_r1i1p1f1_gr_185001-185012.nc')['rsdt'],
                'emidms': xr.open_dataset('emidms_AERmon_EC-Earth3-AerChem_piClim-2xDMS_r1i1p1f1_gn_185001-185012.nc')['emidms']
            },
            'control': {
                'rlut': xr.open_dataset('rlut_Amon_EC-Earth3-AerChem_piClim-control_r1i1p1f1_gr_185001-185012.nc')['rlut'],
                'rsut': xr.open_dataset('rsut_Amon_EC-Earth3-AerChem_piClim-control_r1i1p1f1_gr_185001-185012.nc')['rsut'],
                'rsdt': xr.open_dataset('rsdt_Amon_EC-Earth3-AerChem_piClim-control_r1i1p1f1_gr_185001-185012.nc')['rsdt'],
                'emidms': xr.open_dataset('emidms_AERmon_EC-Earth3-AerChem_piClim-control_r1i1p1f1_gn_185001-185012.nc')['emidms'],
                'siconc': xr.open_dataset('siconc_SImon_EC-Earth3-AerChem_piControl_r1i1p4f1_gn_185001-185012.nc')['siconc']
            }
        }
    }
    return data