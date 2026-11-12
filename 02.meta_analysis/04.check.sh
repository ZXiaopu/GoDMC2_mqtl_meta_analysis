#!/bin/bash

cpg_chunk=$1

folder_out=`ls -dir results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk*_26cohorts/meta/snpid_output | wc -l`

if [[ ${folder_out} = 100 ]]
then
	echo "All genetic chunks have the snpid_output folder"
else
	echo "Some genetic chunks miss the snpid_output folder"
	exit
fi

for genetic_chunk in $(seq 1 100);
do
	phen_folder_count=`ls -dir results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts/meta/phenotype* | wc -l`
	out_file_count=`ls results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts/meta/snpid_output | wc -l`
	if [[ ${phen_folder_count} = ${out_file_count} ]]
	then
		echo "cpg_chunk ${cpg_chunk} genetic_chunk ${genetic_chunk} passed check"
	else
		echo "cpg_chunk ${cpg_chunk} genetic_chunk ${genetic_chunk} faileded check"
		exit
	fi
done
