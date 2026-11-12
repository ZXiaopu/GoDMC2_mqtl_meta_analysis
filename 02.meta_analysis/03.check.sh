#!/bin/bash

cpg_chunk=$1

count=`grep "All jobs passed successfully" logs/script03/script03_cpg${cpg_chunk}_genetic* | wc -l`

if [[ $count -eq 100 ]];
then
	echo "All jobs passed script 01 check"
fi

for genetic_chunk in $(seq 1 100);
do
	log_count=`grep "Writing dataset" logs/script03/script03_cpg${cpg_chunk}_genetic${genetic_chunk}.log | wc -l`
	res_count=`ls -dir results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts/meta/phenotype* |wc -l`
	if [[ $log_count = $res_count ]];then 
		echo "cpg_chunk $1 and genetic_chunk ${genetic_chunk} passed check"
	else
		echo "cpg_chunk $1 and genetic_chunk ${genetic_chunk} failed check"
	fi
done
