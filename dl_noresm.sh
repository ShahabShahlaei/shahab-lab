#!/bin/bash

var="DMS_SURF_FLUX"
type="singlelevel" #"singlelevel" "modellevels"


for y in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
#for y in 30 31 32 33 34 35 36 37 38 39
do
    wget "http://webdav.swestore.se:2080/snic/bolinc/NorESM2/AnnicaEkman/x_tuona/CRiceS/baseline_new/"$var"/"$var"_NorESM_baseline_monthly_"$type"_00"$y"-01-01_00"$y"-12-31.nc"
    #wget "http://webdav.swestore.se:2080/snic/bolinc/NorESM2/AnnicaEkman/x_tuona/CRiceS/future_ssp585_new/"$var"/"$var"_NorESM_future_ssp585_monthly_"$type"_00"$y"-01-01_00"$y"-12-31.nc"
  # wget "http://webdav.swestore.se:2080/snic/bolinc/NorESM2/AnnicaEkman/x_tuona/CRiceS/seaice_contrib_ssp585_new/"$var"/"$var"_NorESM_seaice_contrib_ssp585_monthly_"$type"_00"$y"-01-01_00"$y"-12-31.nc"
  # wget "http://webdav.swestore.se:2080/snic/bolinc/NorESM2/AnnicaEkman/x_tuona/CRiceS/SST_contrib_ssp585_new/"$var"/"$var"_NorESM_SST_contrib_ssp585_monthly_"$type"_00"$y"-01-01_00"$y"-12-31.nc"
done
