#!/bin/bash

cd logs/
for i in $(seq 1 100);
do
tail -n6 script01_cpg${1}_genetic${i}.log > script01_cpg${1}_genetic${i}.log.tmp
done

find script01_cpg${1}_genetic*.log.tmp -type f -exec grep -L "Running meta-classic mode has been done" {} + | sort >> ../failedfile_cpg${1}
find script01_cpg${1}_genetic*.log.tmp -type f -exec grep -L "Closing remaining open files" {} + | sort >> ../failedfile_cpg${1}

failed_jobs=$(sort ../failedfile_cpg${1} | uniq | sed "s/script01_cpg${1}_genetic//g" | sed 's/.log.tmp//g' | tr '\n' ',')
if [[ -n "${failed_jobs}" ]]; then
    echo "Failed jobs found: ${failed_jobs}"
else
    echo "All jobs passed successfully for cpg chunk $1!"
fi

