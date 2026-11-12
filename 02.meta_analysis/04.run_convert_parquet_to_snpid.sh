#!/bin/bash
#SBATCH --job-name=script04
#SBATCH --time=2:00:0
#SBATCH --mem=16G
#SBATCH --output=script04_%j.log

source ../config

cpg_chunk=$1
genetic_chunk=$2

exec > >(tee logs/script04/script04_cpg${cpg_chunk}_genetic${genetic_chunk}.log) 2>&1

echo "Script 04 on cpg chunk ${1} genetic chunk ${genetic_chunk} starting at $(date +"%Y-%m-%d %H:%M:%S")"

meta_sub_folder=./results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts/meta

mkdir -p ${meta_sub_folder}/snpid_output

python 04.convert_parquet_to_snpid.py \
        --meta_sub_folder ${meta_sub_folder} \
	--cpg_chunk ${cpg_chunk} \
	--genetic_chunk ${genetic_chunk} \
        --ref ${hrc_ref} \
        --out_dir ${meta_sub_folder}/snpid_output/

echo "Script 04 on cpg chunk ${1} genetic chunk ${genetic_chunk} has been done at $(date +"%Y-%m-%d %H:%M:%S")"
