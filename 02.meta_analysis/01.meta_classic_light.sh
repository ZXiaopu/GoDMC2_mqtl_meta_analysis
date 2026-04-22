#!/bin/bash -l

source ../config

genotype_input=$1
study_name_input=$2
phenotype_input=$3
derivatives_input=$4
snp_inc_input=$5
cpg_inc_input=$6
encoded_input=$7
covariate_selection=$8
node=$9
meta_analysis_output=${10}

apptainer exec -B /projects ${project_path}/hase_glint_py2_isb3.sif python2 ${hase}/hase.py \
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
-node ${node}

echo "Running meta-classic mode has been done"
