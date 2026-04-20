#!/bin/bash

source ./config

cohort=$1
Rscript 02.observe_genotype_encode.R ${GoDMC_02_result} ${cohort} ${hrc_ref}
