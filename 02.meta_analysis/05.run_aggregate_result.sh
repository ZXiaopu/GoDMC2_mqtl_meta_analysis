#!/bin/bash
#SBATCH --job-name=script05
#SBATCH --time=1:00:0
#SBATCH --mem=8G
#SBATCH --output=script05_%j.log

cpg_chunk=$1

exec > >(tee logs/script05/script05_cpg${cpg_chunk}.log) 2>&1

Rscript 05.aggregate_res.R ${cpg_chunk}
echo "Script 05 on cpg chunk ${cpg_chunk} has been done"
