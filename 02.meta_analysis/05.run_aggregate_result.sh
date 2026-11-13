#!/bin/bash
#SBATCH --job-name=script05
#SBATCH --time=2:00:0
#SBATCH --mem=16G
#SBATCH --output=script05_%j.log

cpg_chunk=$1

Rscript 05.aggregate_res.R ${cpg_chunk}
echo "Script 05 on cpg chunk ${cpg_chunk} has been done"
