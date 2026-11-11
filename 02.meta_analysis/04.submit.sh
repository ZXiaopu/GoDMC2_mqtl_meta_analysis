#!/bin/bash

cpg_chunk=$1
genetic_chunk=$2

for sub in $(seq 1 5);
do
sbatch 04.run_convert_parquet_to_snpid.sh ${cpg_chunk} $sub ${genetic_chunk}
done
