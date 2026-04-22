#!/bin/bash

#SBATCH --job-name=meta_pos_cont
#SBATCH --time=12:00:0
#SBATCH --mem=128G

exec > >(tee -a logs/Positive_control_meta_chr${1}.log) 2>&1

source ../config

genotype_suffix="_04/04/meta_inputs/use_data"
phenotype_suffix="_04/04/meta_inputs/use_data/phenotypes"
derivatives_suffix="_04/04/meta_inputs/part_dev"
chr=$1

genotype_input=$(awk -v p="${GoDMC_04_result}" -v s="$genotype_suffix" '{printf "%s/%s%s ", p, $0, s}' module_04_cohort_list | sed 's/ $//')
study_name_input=$(cat module_04_cohort_list | tr '\n' ' ')
phenotype_input=$(awk -v p="${GoDMC_04_result}" -v s="$phenotype_suffix" '{printf "%s/%s%s ", p, $0, s}' module_04_cohort_list | sed 's/ $//')
derivatives_input=$(awk -v p="${GoDMC_04_result}" -v s="$derivatives_suffix" '{printf "%s/%s%s ", p, $0, s}' module_04_cohort_list | sed 's/ $//')
snp_inc_input=$(printf "hrc_snps_chr${chr} %.0s" {1..19})
cpg_inc_input=$(printf "positive_control_cpg %.0s" {1..19})
encoded_input=$(printf '1 %.0s' {1..19})
covariate_selection="covariates_selection.txt"
node="1 1"

echo "Start running script: $(date)"

bash 01.meta_classic_light.sh \
	"${genotype_input}" \
	"${study_name_input}" \
	"${phenotype_input}" \
	"${derivatives_input}" \
	"${snp_inc_input}" \
	"${cpg_inc_input}" \
	"${encoded_input}" \
	"${covariate_selection}" \
	"${node}" \
	"${meta_analysis_output_positive_cont}_chr${chr}"

echo "Meta-classic running on positive control CpG is done."
echo "Finish running script: $(date)"
