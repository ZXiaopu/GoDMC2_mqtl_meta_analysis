#!/bin/bash

cpg_chunk=$1
genetic_chunk=$2

sbatch 04.run_convert_parquet_to_snpid.sh ${cpg_chunk} ${genetic_chunk}
