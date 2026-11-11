#!/bin/bash

#SBATCH --job-name=script03
#SBATCH --time=3:00:0
#SBATCH --mem=32G
#SBATCH --output=script03_%j.log
source ../config

cpg_chunk=$1
genetic_chunk=$2
exec > >(tee -a logs/script03/script03_cpg${cpg_chunk}_genetic${genetic_chunk}.log) 2>&1

bash check_logs.sh ${cpg_chunk}

echo "Script 03 on cpg chunk ${1} genetic_chunk ${2} starting at $(date +"%Y-%m-%d %H:%M:%S")"

cd ./results

meta_folder="meta_analysis_cpg_chunk${cpg_chunk}"
meta_sub_folder="meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts"

mkdir -p ${meta_folder}
if [[ -d ${meta_sub_folder} ]]; 
then
	echo "move ${meta_sub_folder} to ${meta_folder}"
	mv ${meta_sub_folder} ${meta_folder}
elif [[ -d ${meta_folder}/${meta_sub_folder} ]]
then
	echo "${meta_sub_folder} already existed in ${meta_folder}"
else
	echo "${meta_sub_folder} does not exist. Check 01 script"
	exit 1
fi

apptainer exec -B /projects ${project_path}/hase_glint_py2_isb3.sif python2 ../03.convert_hase_light_meta_results.py \
       --path ${meta_folder}/${meta_sub_folder}/meta \
       --out-dir ${meta_folder}/${meta_sub_folder}/meta \
       --out-tag cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts \
       --phenotypes ../InputData/DNAm/allCpGs_chunk${cpg_chunk}

echo "Script 03 on cpg chunk ${1} genetic_chunk ${2} has been done at echo $(date +"%Y-%m-%d %H:%M:%S")"
