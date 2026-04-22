#!/bin/bash

exec > >(tee -a logs/script_01_$1.log) 2>&1

source ../config

echo "Start running script: $(date)"
cohort=$1
Rscript 01.observe_genotype_encode.R ${GoDMC_02_result} ${cohort} ${hrc_ref}
echo "Script 01 for $1 run successfully"
echo "Finish running script: $(date)"
