#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys

def reorder_h5(h5_in, afreq_in, h5_out):
    try:
        # 1. Load correct ID order
        print "Loading_afreq"
        ref_df = pd.read_csv(afreq_in, sep=r'\s+', compression='gzip', usecols=['ID'])
        correct_order = [str(x) for x in ref_df['ID'].tolist()]
        len_afreq = len(correct_order)

        # 2. Load H5 data
        print "Loading_H5"
        # Try to find the correct key
        store = pd.HDFStore(h5_in, mode='r')
        available_keys = store.keys()
        store.close()
        
        target_key = 'probes'
        if '/probes' not in available_keys and 'probes' not in available_keys:
            if not available_keys:
                print "Error_No_Keys"
                sys.exit(1)
            target_key = available_keys[0].strip('/')

        df = pd.read_hdf(h5_in, key=target_key)
        df['ID'] = df['ID'].astype(str)
        len_h5 = len(df)

        # 3. Length Verification
        if len_afreq != len_h5:
            print "Error_Length_Mismatch"
            print "afreq:" + str(len_afreq) + "_vs_H5:" + str(len_h5)
            sys.exit(1)

        # 4. Reorder and NA check
        print "Reordering"
        df.set_index('ID', inplace=True)
        
        # Check if all IDs in afreq exist in H5 before reindexing
        missing_count = 0
        for snp_id in correct_order:
            if snp_id not in df.index:
                missing_count += 1
        
        if missing_count > 0:
            print "Error_Missing_IDs_In_H5"
            print "Missing_count:" + str(missing_count)
            sys.exit(1)

        # Reindex
        df_new = df.reindex(correct_order).reset_index()

        # Final NA Check on critical column
        if df_new['CHR'].isnull().any():
            print "Error_NA_Detected_After_Reorder"
            sys.exit(1)

        # 5. Save
        print "Saving"
        df_new.to_hdf(
            h5_out, 
            key='probes',
            mode='w',
            format='table',
            data_columns=True,
            complib='zlib',
            complevel=9,
            min_itemsize={'ID': 45}
        )
        print "Success"

    except Exception as e:
        print "Error_General"
        print str(e)
        sys.exit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-a', '--afreq', required=True)
    p.add_argument('-o', '--output', required=True)
    args = p.parse_args()
    reorder_h5(args.input, args.afreq, args.output)