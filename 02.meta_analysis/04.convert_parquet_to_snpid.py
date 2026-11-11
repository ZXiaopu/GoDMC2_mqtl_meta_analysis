import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hase_output_file", help="parquet format file, single file each time")
parser.add_argument("--ref")
parser.add_argument("--out", help="output file name")

args = parser.parse_args()
df = pd.read_parquet(args.hase_output_file)
ref = pd.read_csv(args.ref,sep=" ")

subset_ref = ref[['ID','str_allele1','str_allele2']]
df['ID'] = df['variant_index'].map(subset_ref['ID'])
df['effect_allele'] = df['variant_index'].map(subset_ref['str_allele1'])
df['non_effect_allele'] = df['variant_index'].map(subset_ref['str_allele2'])
df.columns = df.columns.astype(str)
df.to_parquet(args.out, index=False, compression='gzip')
