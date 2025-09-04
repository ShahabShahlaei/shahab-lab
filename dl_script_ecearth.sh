
#
# A script for retrieving files from lake.fmi.fi with wget
#**********************************************************************
#
#
INPUT_ROOT="https://crices-task33-output-ecearth.lake.fmi.fi"
#
#
# List of experiments for which data are retrieved
#
#EXPLIST="baseline future_ssp585 SST_contrib_ssp585 seaice_contrib_ssp585
#         future_ssp126 SST_contrib_ssp126 seaice_contrib_ssp126"
#
#EXPLIST="baseline seaice_contrib_ssp585 SST_contrib_ssp585"
EXPLIST="baseline future_ssp585 seaice_contrib_ssp585 SST_contrib_ssp585"

OUTPUT_ROOT="./"
#OUTPUT_ROOT="./"$EXPLIST"/"
#if [ ! -d $OUTPUT_ROOT ]; then
#   mkdir -p $OUTPUT_ROOT
#fi

#
#
# List of years for which data are retrieved
#
YEARLIST="0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013
          0014 0015 0016 0017 0018 0019 0020 0021 0022 0023 0024 0025 0026 0027
          0028 0029 0030 0031 0032 0033 0034 0035 0036 0037 0038 0039" 

#YEARLIST="0000"
#
MONTHLIST="01 02 03 04 05 06 07 08 09 10 11 12"
MODEL="EC-Earth"

#**************************************************************************
# List of variables to be retrieved. If/when you need only some of these,
# leave out the corresponding variables
# ************************************************************************
# 3D, 6-HOURLY variables from IFS
#VARLIST_3D6HR="TA UA VA WAP HUS CLW CLI CL"
#
# 2D, 6-HOURLY variables from IFS
#VARLIST_2D6HR="CLHIGH CLLOW CLMID CLT CLVI LNPS LWP PRW PS PSL PV330K PV375K PV400K PV450K RV850 TAS TDPS TPOT2PVU TS UAS VAS ZG250 ZG500 ZG850 ZMLA"
#
# 2D, DAILY variables from IFS
#VARLIST_2D1DAY="EVSPSBL HFLS HFSS PR PRLSNS PRLSPROF PRRC PRSNC RLDS RLNSCS RLUS RLUTCS RLUT RSDS RSDT RSNSCS RSUS RSUTCS RSUT SICONC SNC SNW TAUU TAUV"
#
# 2D MONTHLY variables from TM5
#VARLIST_2DMONTHLY="OD550AER ABS550AER DRYBC WETBC DRYDUST WETDUST EMIBC EMIDMS EMIDUST EMIOC EMISO2 EMISO4 EMISS"
#
# 3D MONTHLY variables from TM5
#VARLIST_3DMONTHLY="MMRSS MMRDUST MMRBC MMRSO4 MMROC"
#
# Constant fieds
#VARLIST_CONST="OROG LSM"
#
#****************************************************************
# Just for testing, retrieve only some of the variables
#******************************************************************
#
#VARLIST_3D6HR="TA"
#VARLIST_2D6HR="CLHIGH CLLOW CLMID"
#VARLIST_2D1DAY="HFLS HFSS PR" 
#VARLIST_2DMONTHLY="OD550AER ABS550AER"
#VARLIST_3DMONTHLY="MMRBC MMRSO4"
#VARLIST_CONST="OROG LSM"
#
VARLIST="EMIDMS"
#
#**************************************************
# 1) Retrieve 3D, 6-hourly variables from IFS
#**************************************************
# for VAR in ${VARLIST_3D6HR}; do
# ##for VAR in ${VARLIST_EMPTY}; do
#    for EXP in ${EXPLIST}; do
#       INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
#       OUTPUTDIR=${OUTPUT_ROOT} #/${VAR}/${EXP}
# #      if [ ! -d ${OUTPUTDIR} ]; then
# #         mkdir -p ${OUTPUTDIR} 
# #      fi   
# ## Loop over years
#       for YEAR in $YEARLIST; do     
# ## Number of days per month depends if this is a leap year or not
#          RES=$((10#${YEAR}%4))
#          if [[ 10#$RES -eq "0" ]]; then
# ## Leap year
#             DAYSPERMONTH=( 31 29 31 30 31 30 31 31 30 31 30 31 )
#          else
# ## Normal year
#             DAYSPERMONTH=( 31 28 31 30 31 30 31 31 30 31 30 31 )
#          fi
# ## Loop over months
#          for MONTH in $MONTHLIST; do
#             IND=$((10#${MONTH}-1))
#             START=${YEAR}-${MONTH}-01
#             END=${YEAR}-${MONTH}-${DAYSPERMONTH[@]:$IND:1}
#             TIMESTAMP=${START}_${END}
#             FILENAME=${VAR}_${MODEL}_${EXP}_6hourly_modellevels_${TIMESTAMP}.nc           
#             wget -O ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME 
#          done
#       done
#    done   
# done

#**************************************************
# 2) Retrieve 2D, 6-hourly variables from IFS
#**************************************************
#for VAR in ${VARLIST_2D6HR}; do
##for VAR in ${VARLIST_EMPTY}; do
#   for EXP in ${EXPLIST}; do
#      INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
#      OUTPUTDIR=${OUTPUT_ROOT}/${VAR}/${EXP}
#      if [ ! -d ${OUTPUTDIR} ]; then
#         mkdir -p ${OUTPUTDIR} 
#      fi   
## Loop over years
#      for YEAR in $YEARLIST; do     
#         START=${YEAR}-01-01
#         END=${YEAR}-12-31
#         TIMESTAMP=${START}_${END}
#         FILENAME=${VAR}_${MODEL}_${EXP}_6hourly_singlelevel_${TIMESTAMP}.nc
#         wget -O ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME 
#      done
#   done   
#done

#**************************************************
# 3) Retrieve 2D, daily variables from IFS
#**************************************************
#for VAR in ${VARLIST_2D1DAY}; do
##for VAR in ${VARLIST_EMPTY}; do
#   for EXP in ${EXPLIST}; do
#      INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
#      OUTPUTDIR=${OUTPUT_ROOT}/${VAR}/${EXP}
#      if [ ! -d ${OUTPUTDIR} ]; then
#         mkdir -p ${OUTPUTDIR} 
#      fi   
## Loop over years
#      for YEAR in $YEARLIST; do     
#         START=${YEAR}-01-01
#         END=${YEAR}-12-31
#         TIMESTAMP=${START}_${END}
#         FILENAME=${VAR}_${MODEL}_${EXP}_daily_singlelevel_${TIMESTAMP}.nc
#         wget -O ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME 
#      done
#   done   
#done


#**************************************************
# 4) Retrieve 2D monthly variables from TM5
#**************************************************
for VAR in ${VARLIST}; do
   for EXP in ${EXPLIST}; do
     echo $VAR $EXP
     INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
     OUTPUTDIR=${OUTPUT_ROOT}/${EXP}
     for YEAR in $YEARLIST; do     
        START=${YEAR}-01-01
        END=${YEAR}-12-31
        TIMESTAMP=${START}_${END}
        FILENAME=${VAR}_${MODEL}_${EXP}_monthly_singlelevel_${TIMESTAMP}.nc
        wget ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME 
     done
  done   
done


# for VAR in ${VARLIST}; do
#     for EXP in ${EXPLIST}; do
#       echo $VAR $EXP
#       INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
#       OUTPUTDIR=${OUTPUT_ROOT}/${EXP}
#       for YEAR in $YEARLIST; do
#          START=${YEAR}-01-01
#          END=${YEAR}-12-31
#          TIMESTAMP=${START}_${END}
#          FILENAME=${VAR}_${MODEL}_${EXP}_monthly_modellevels_${TIMESTAMP}.nc
#          wget ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME
#       done
#    done
# done

#**************************************************
# 5) Retrieve 3D monthly variables from TM5
#**************************************************
#for VAR in ${VARLIST_3DMONTHLY}; do
##for VAR in ${VARLIST_EMPTY}; do
#   for EXP in ${EXPLIST}; do
#      INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
#      OUTPUTDIR=${OUTPUT_ROOT}/${VAR}/${EXP}
#      if [ ! -d ${OUTPUTDIR} ]; then
#         mkdir -p ${OUTPUTDIR} 
#      fi   
## Loop over years
#      for YEAR in $YEARLIST; do     
#         START=${YEAR}-01-01
#         END=${YEAR}-12-31
#         TIMESTAMP=${START}_${END}
#         FILENAME=${VAR}_${MODEL}_${EXP}_monthly_modellevels_${TIMESTAMP}.nc
#         wget -O ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME 
#      done
#   done   
#done


#**************************************************
# 5) Retrieve constant (time-invariant) fields
#**************************************************
#for VAR in ${VARLIST_CONST}; do
##for VAR in ${VARLIST_EMPTY}; do
#   for EXP in ${EXPLIST}; do
#      INPUTDIR=${INPUT_ROOT}/${VAR}/${EXP}
#      OUTPUTDIR=${OUTPUT_ROOT}/${VAR}/${EXP}
#      if [ ! -d ${OUTPUTDIR} ]; then
#         mkdir -p ${OUTPUTDIR} 
#      fi   
#      FILENAME=${VAR}_${MODEL}_${EXP}.nc
#      wget -O ${OUTPUTDIR}/$FILENAME ${INPUTDIR}/$FILENAME 
#   done   
#done
