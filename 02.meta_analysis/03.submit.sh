#!/bin/bash

cpg_chunk=$1

for i in $(seq 1 100);
do
sbatch 03.run_convert_hase_light_meta_results.sh ${cpg_chunk} $i
done
