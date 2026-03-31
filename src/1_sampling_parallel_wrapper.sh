#!/bin/bash

# =========================
# Parameter / Variablen
# =========================

TILELIST="/data/ahsoka/eocp/forestpulse/02_scripts/DWD/DC_tilelist.txt"
#TILELIST="/data/ahsoka/eocp/forestpulse/02_scripts/DWD/RLP_tilelist.txt"


START=1
#Germany
END=484
# RLP
#END=39
YEAR=2021
JOBS=30
echo "Processing tiles from line $START to $END for year $YEAR with $JOBS parallel jobs."

start_time=$(date +%s)
echo "Start: $(date)"

#for YEAR in 2022 2023 2024 2025 ; do
#    echo "Processing year $YEAR..."
#    sed -n "${START},${END}p" "$TILELIST" | tr -d '\r' | parallel -j $JOBS /data/ahsoka/eocp/forestpulse/02_scripts/dwd-git/src/3_calculate_coefs.sh {} $YEAR
#done
sed -n "${START},${END}p" "$TILELIST" | tr -d '\r' | parallel -j $JOBS python /data/ahsoka/eocp/forestpulse/02_scripts/tree_mask/with_spline_coefs/1_extract.py --tile {} --year $YEAR
end_time=$(date +%s)
echo "End:   $(date)"

runtime=$((end_time - start_time))
echo "Duration: ${runtime} seconds"
