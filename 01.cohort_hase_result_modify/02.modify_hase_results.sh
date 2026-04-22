#!/bin/bash

exec > >(tee -a logs/script_02_$1.log) 2>&1

source ../config

echo "Start running script: $(date)"

cohort_name=$1
cohort_path=${GoDMC_04_result}/${cohort_name}_04/04

cp ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5 ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}_wrong.h5

apptainer exec -B /projects ${project_path}/hase_glint_py2_isb3.sif python2 ${hase}/added/reorder_h5.py \
    -i ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5 \
    -a ${GoDMC_02_result}/${cohort_name}_02/02/data.afreq.gz \
    -o ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}_fixed.h5

mv ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}_fixed.h5 ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5

cp ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}_wrong.npy

apptainer exec -B /projects ${project_path}/hase_glint_py2_isb3.sif python2 ${hase}/added/reindex_value.py \
    ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy \
    ${hrc_ref} \
    ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5 \
    ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}_reindex.npy

mv ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}_reindex.npy ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy
cp ${cohort_path}/meta_inputs/mapping/keys_ref-hrc.npy ${Mapper_path}
cp ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy ${Mapper_path}
cp ${cohort_path}/meta_inputs/mapping/flip_ref-hrc_${cohort_name}.npy ${Mapper_path}

echo "Script 02 for ${cohort_name} is done"
echo "Finish running script: $(date)"
