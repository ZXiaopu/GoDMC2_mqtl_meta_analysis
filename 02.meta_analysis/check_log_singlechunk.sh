#!/bin/bash

cd logs/

cpg_chunk=$1
genetic_chunk=$2

tail -n6 script01_cpg${1}_genetic${2}.log > script01_cpg${1}_genetic${2}.log.tmp

find script01_cpg${1}_genetic${2}.log.tmp -type f -exec grep -L "Running meta-classic mode has been done" {} + | sort >> ../failedfile_cpg${1}_genetic${2}
find script01_cpg${1}_genetic${2}.log.tmp -type f -exec grep -L "Closing remaining open files" {} + | sort >> ../failedfile_cpg${1}_genetic${2}

failed_jobs=$(sort ../failedfile_cpg${1}_genetic${2} | uniq | sed "s/script01_cpg${1}_genetic//g" | sed 's/.log.tmp//g' | tr '\n' ',')
if [[ -n "${failed_jobs}" ]]; then
    echo "Failed jobs found: ${failed_jobs}"
else
    echo "All jobs passed successfully for cpg chunk $1 genetic chunk $2!"
fi

rm script01_cpg${1}_genetic${2}.log.tmp
rm ../failedfile_cpg${1}_genetic${2}
