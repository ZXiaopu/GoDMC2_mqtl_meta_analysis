#!/bin/bash

exec > >(tee -a logs/script_00.log) 2>&1

source ../config

echo "Start running script: $(date)"
Rscript 00.missingSNP_summary.R ${GoDMC_02_result} ${hrc_ref} ${out_01}
echo "Script 00 run successfully"
echo "Finish running script: $(date)"
