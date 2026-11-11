#!/bin/bash

# in total: 100 genetic chunks; 210 CpG chunks

cpg_chunk=$1

for i in $(seq 1 100);
do
sbatch 01.run_meta_classic_26cohorts.sh ${cpg_chunk} $i
done
