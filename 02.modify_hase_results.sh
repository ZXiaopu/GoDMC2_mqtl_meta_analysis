#!/bin/bash

source config

conda activate hase_py2

cohort_name=$1
cohort_path=${GoDMC_04_result}/${cohort_name}_04/04

cp ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5 ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}_wrong.h5

python ${hase}/added/reorder_h5.py \
    -i ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5 \
    -a ${result_02} \
    -o ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}_fixed.h5

mv ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}_fixed.h5 ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5

cp ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}_wrong.npy

python ${hase}/added/reindex_value.py \
    ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy \
    ${hrc_ref} \
    ${cohort_path}/meta_inputs/use_data/probes/${cohort_name}.h5 \
    ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}_reindex.npy

cp ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}_reindex.npy ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy
cp ${cohort_path}/meta_inputs/mapping/keys_ref-hrc.npy ${Mapper_path}
cp ${cohort_path}/meta_inputs/mapping/values_ref-hrc_${cohort_name}.npy ${Mapper_path}
cp ${cohort_path}/meta_inputs/mapping/flip_ref-hrc_${cohort_name}.npy ${Mapper_path}

echo "Meta_input reindex is done for ${cohort_name}"
