import numpy as np
import pandas as pd
import sys
import os
import argparse

def modify_flip(reencode_file, output):
    df = pd.read_csv(reencode_file, sep='\t')

    df['flip'] = -df['ENCODE_CONSISTENCE']

    flip_array = df['flip'].values

    np.save(output, flip_array)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    args = p.parse_args()
    modify_flip(args.input, args.output)

