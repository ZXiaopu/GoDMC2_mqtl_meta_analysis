#!/bin/bash
#SBATCH --job-name=script04
#SBATCH --time=6:00:0
#SBATCH --mem=32G
#SBATCH --output=script04_%j.log

source ../config

cpg_chunk=$1
genetic_chunk=$2

exec > >(tee -a logs/script04/script04_cpg${cpg_chunk}_genetic${genetic_chunk}.log) 2>&1

echo "Script 04 on cpg chunk ${1} genetic_chunk ${2} starting at $(date +"%Y-%m-%d %H:%M:%S")"

meta_sub_folder=./results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts/meta

mkdir -p ${meta_sub_folder}/snpid_output

for cpg in `ls ${meta_sub_folder}`;
do
	echo $cpg
	python 04.convert_parquet_to_snpid.py \
        --hase_output_file ${meta_sub_folder}/${cpg}/var_chunk_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts.parquet \
        --ref ${hrc_ref} \
        --out ${meta_sub_folder}/snpid_output/${cpg}_genetic_chunk${genetic_chunk}_26cohorts.parquet
done

echo "Script 04 on cpg_chunk${1} genetic_chunk${2} has been done at $(date +"%Y-%m-%d %H:%M:%S")"
