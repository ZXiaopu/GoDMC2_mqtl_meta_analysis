#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import argparse
import gzip
import os
import sys

def load_ref_ids(ref_path):
    """
    Python 2: Skip the first line regardless of content, 
    then load all subsequent non-empty lines.
    """
    ref_ids = []
    with gzip.open(ref_path, 'rb') as f:
        f.readline()
        
        for line in f:
            line = line.strip()

            if not line:
                continue

            fields = line.split()
            if fields:
                ref_ids.append(fields[0])
                
    return ref_ids

def main():
    parser = argparse.ArgumentParser(description='Reindex value file based on h5 and ref order (Python 2 version)')
    parser.add_argument('value_npy', help='Input existing npy file for comparison')
    parser.add_argument('ref_path', help='Reference .ref.gz file')
    parser.add_argument('h5_path', help='Cohort h5 file')
    parser.add_argument('output_npy', help='Output reindexed npy file')
    args = parser.parse_args()

    print "Loading reference IDs from", args.ref_path
    ref_ids = load_ref_ids(args.ref_path)

    print "Loading H5 IDs from", args.h5_path
    h5_df = pd.read_hdf(args.h5_path, key='probes')
    
    h5_ids = h5_df['ID'].values
    
    h5_id_to_index = {snp_id: i for i, snp_id in enumerate(h5_ids)}

    print "Performing reindexing..."
    new_value_list = [h5_id_to_index.get(ref_id, -1) for ref_id in ref_ids]
    new_value = np.array(new_value_list, dtype=np.int32)

    print "New mapping shape:", new_value.shape

    if os.path.exists(args.value_npy):
        old_value = np.load(args.value_npy)
        print "Old mapping shape:", old_value.shape
        print "Saving new mapping to", args.output_npy
        np.save(args.output_npy, new_value)

if __name__ == '__main__':
    main()