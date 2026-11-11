#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=64G

source ../config

for cohort in `cat module_04_cohort_list_28cohorts`;
do
Rscript 02b.plot_plinkvshase_singlecohort.R \
        ${GoDMC_03_result} \
        ${out_02}/positive_control_28cohorts \
        $cohort \
        ${hrc_ref} \
        ${metal_input_path}
done
