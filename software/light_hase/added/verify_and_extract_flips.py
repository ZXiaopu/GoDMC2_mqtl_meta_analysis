#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys
import re
import os

def verify_flips(ref_path, afreq_path, log_path, out_dir, project_name):
    try:
        # 1. Load Reference (HRC)
        print "Loading_REF"
        # Using columns str_allele1 and str_allele2 as requested
        ref = pd.read_csv(ref_path, sep=r'\s+', compression='gzip', usecols=['ID', 'str_allele1', 'str_allele2'])
        ref.columns = ['ID', 'ref_a1', 'ref_a2']
        ref['ID'] = ref['ID'].astype(str)

        # 2. Load afreq (Cohort)
        print "Loading_AFREQ"
        # Columns in data.afreq.gz: ID, REF, ALT
        afreq = pd.read_csv(afreq_path, sep=r'\s+', compression='gzip', usecols=['ID', 'REF', 'ALT'])
        afreq.columns = ['ID', 'cq_a1', 'cq_a2']
        afreq['ID'] = afreq['ID'].astype(str)

        # 3. Merge and Count Flips
        print "Calculating_Flips"
        merged = pd.merge(ref, afreq, on='ID', how='inner')
        
        # A flip occurs if the cohort allele1 (cq_a1) does not match reference allele1 (ref_a1)
        is_flipped = (merged['ref_a1'].astype(str) != merged['cq_a1'].astype(str))
        flipped_df = merged[is_flipped]
        calculated_count = len(flipped_df)
        print "Calc_Count:" + str(calculated_count)

        # 4. Extract count from HASE log
        print "Reading_Log"
        hase_count = None
        if not os.path.exists(log_path):
            print "Error_Log_Not_Found"
            sys.exit(1)

        with open(log_path, 'r') as f:
            for line in f:
                if "Flipped Alleles Detected" in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        hase_count = int(match.group(1))
                        break
        
        if hase_count is None:
            print "Error_Log_Number_Not_Found"
            sys.exit(1)
        print "HASE_Count:" + str(hase_count)

        # 5. Verification
        if calculated_count != hase_count:
            print "CRITICAL_ERROR_COUNT_MISMATCH"
            msg = "Expected:" + str(hase_count) + "_but_got:" + str(calculated_count)
            print msg
            sys.exit(1)
        
        # 6. Save flipped IDs with Project Name
        # Resulting filename: PROJECT_NAME_flipped_snps.txt
        output_filename = "{}_flipped_snps.txt".format(project_name)
        full_out_path = os.path.join(out_dir, output_filename)
        
        print "Saving_List_to_" + output_filename
        flipped_df['ID'].to_csv(full_out_path, index=False, header=False)
        print "Success_Verified"

    except Exception as e:
        print "Error_General"
        print str(e)
        sys.exit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-r', '--ref', required=True, help='ref-hrc.ref.gz')
    p.add_argument('-a', '--afreq', required=True, help='data.afreq.gz')
    p.add_argument('-l', '--log', required=True, help='HASE log file')
    p.add_argument('-d', '--out_dir', required=True, help='Directory to save the list')
    p.add_argument('-n', '--project_name', required=True, help='Project name for filename')
    
    args = p.parse_args()
    verify_flips(args.ref, args.afreq, args.log, args.out_dir, args.project_name)