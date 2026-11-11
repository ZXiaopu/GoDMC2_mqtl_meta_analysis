#!/bin/bash

cpg_chunk=$1

for genetic_chunk in $(seq 1 100);
do
	log_count=`grep "Writing dataset" logs/script03/script03_cpg${cpg_chunk}_genetic${genetic_chunk}.log | wc -l`
	res_count=`ls results/meta_analysis_cpg_chunk${cpg_chunk}/meta_analysis_cpg_chunk${cpg_chunk}_genetic_chunk${genetic_chunk}_26cohorts/meta |wc -l`
	if [[ $log_count = $res_count ]];then 
		echo "cpg_chunk $1 and genetic_chunk ${genetic_chunk} passed check"
	else
		echo "cpg_chunk $1 and genetic_chunk ${genetic_chunk} failed check"
	fi
done
