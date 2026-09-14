#!/bin/bash

#SBATCH --job-name=meta_analysis
#SBATCH --time=12:00:0
#SBATCH --mem=128G
#SBATCH --output=meta_analysis_%j.log

#exec > >(tee -a logs/meta_cpg_chunk${1}_snp_chunk${2}.log) 2>&1

source ../config

genotype_suffix="_04/04/meta_inputs/use_data"
phenotype_suffix="_04/04/meta_inputs/use_data/phenotypes"
derivatives_suffix="_04/04/meta_inputs/part_dev"
cpg_chunk=$1
genetic_chunk=$2

genotype_input=$(awk -v p="${GoDMC_04_result}" -v s="$genotype_suffix" '{printf "%s/%s%s ", p, $0, s}' module_04_cohort_list | sed 's/ $//')
study_name_input=$(cat module_04_cohort_list_28cohorts | tr '\n' ' ')
phenotype_input=$(awk -v p="${GoDMC_04_result}" -v s="$phenotype_suffix" '{printf "%s/%s%s ", p, $0, s}' module_04_cohort_list | sed 's/ $//')
derivatives_input=$(awk -v p="${GoDMC_04_result}" -v s="$derivatives_suffix" '{printf "%s/%s%s ", p, $0, s}' module_04_cohort_list | sed 's/ $//')
snp_inc_input=$(printf "/projects/b35cx/XiaopuZ/GoDMC2_mqtl_meta_analysis/02.meta_analysis/InputData/Genotype/hrc_autosomal_snps_${genetic_chunk} %.0s" {1..28})
cpg_inc_input=$(printf "/projects/b35cx/XiaopuZ/GoDMC2_mqtl_meta_analysis/02.meta_analysis/InputData/DNAm/allCpGs_chunk${cpg_chunk} %.0s" {1..28})
encoded_input=$(printf '1 %.0s' {1..28})
covariate_selection="./InputData/covariates_selection_all.txt"
node="1 1"

mkdir -p ${out_02}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}

echo "Start running script: $(date), cpg chunk ${cpg_chunk}, snp chunk ${genetic_chunk}"
srun 01.meta_classic_light.sh \
	"${genotype_input}" \
	"${study_name_input}" \
	"${phenotype_input}" \
	"${derivatives_input}" \
	"${snp_inc_input}" \
	"${cpg_inc_input}" \
	"${encoded_input}" \
	"${covariate_selection}" \
	"${node}" \
	"${out_02}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}"

echo "job on cpg chunk ${cpg_chunk}, snp chunk ${genetic_chunk} has been done"
echo "Finish running script: $(date)"
