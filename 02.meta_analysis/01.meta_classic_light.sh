#!/bin/bash -l
#SBATCH --output=meta_classic_%j.out
#SBATCH --job-name=meta-classic
#SBATCH --nodes=2
#SBATCH --time=24:0:0

conda activate hase_py2

source config

genotype_input=$1
study_name_input=$2
phenotype_input=$3
derivatives_input=$4
snp_inc_input=$5
cpg_inc_input=$6
encoded_input=$7
node=$8

cohort1="/scratch/prj/bell/recovered/epigenetics/Analysis/subprojects/xiaopu/GoDMC/GoDMC_TwinsUK_EPIC/godmc_phase2/results/04"
cohort2="/scratch/prj/bell/recovered/epigenetics/Analysis/subprojects/xiaopu/GoDMC/Stage2/meQTL_results/Cluster_Filespace/Marioni_Group/Xiaopu/godmc_unrelated/godmc_phase2/results/04"

python ${hase}/hase.py \
-g ${genotype_input} \
-study_name ${study_name_input} \
-ph ${phenotype_input} \
-derivatives ${derivatives_input} \
-mapper ${Mapper_path} \
-snp_id_inc ${snp_inc_input} \
-ph_id_inc ${cpg_inc_input} \
-thr_full_log 0 \
-ref_name ref-hrc \
-ref_path ${hase}/data \
-o ${meta_analysis_output} \
-encoded ${encoded_input} \
--selected-covariates ${covariate_selection} \
-flip_path ${GoDMC_02_result} \
-mode meta-classic \
-max-missingness-rate 1 \
-cluster "y" \
-node 10 ${node}

echo "Running meta-classic mode has been done"
