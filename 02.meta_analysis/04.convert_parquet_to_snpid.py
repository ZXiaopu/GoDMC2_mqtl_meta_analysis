import pandas as pd
import argparse
import glob
import os
import numpy as np
from scipy.stats import t

parser = argparse.ArgumentParser()
parser.add_argument("--meta_sub_folder", help="Directory containing meta-analysis folders")
parser.add_argument("--cpg_chunk", help="CpG chunk ID")
parser.add_argument("--genetic_chunk", help="Genetic chunk ID")
parser.add_argument("--ref")
parser.add_argument("--out_dir", help="output file name")

args = parser.parse_args()

ref = pd.read_csv(args.ref,sep=" ")
subset_ref = ref[['ID','str_allele1','str_allele2']]

search_pattern = os.path.join(
    args.meta_sub_folder, 
    "phenotype*", 
    f"var_chunk_cpg_chunk{args.cpg_chunk}_genetic_chunk{args.genetic_chunk}_26cohorts.parquet"
)
parquet_files = glob.glob(search_pattern)
print(f"Found {len(parquet_files)} parquet files to process.")

for file_path in parquet_files:
	print(f"Processing: {file_path}")
	df = pd.read_parquet(file_path)
	df['ID'] = df['variant_index'].map(subset_ref['ID'])
	df['effect_allele'] = df['variant_index'].map(subset_ref['str_allele1'])
	df['non_effect_allele'] = df['variant_index'].map(subset_ref['str_allele2'])
	
	t_stats = np.abs(df['beta'].to_numpy() / df['standard_error'].to_numpy())
	degrees_freedom = df['sample_size'].to_numpy() - 2
	df['pvalue'] = 2 * t.sf(t_stats, df=degrees_freedom)

	df.columns = df.columns.astype(str)

	pheno_dir = os.path.dirname(file_path)
	pheno_name = os.path.basename(pheno_dir)
	out = os.path.join(args.out_dir, f"{pheno_name}_genetic_chunk{args.genetic_chunk}_26cohorts.parquet")

	df.to_parquet(out, index=False, compression='gzip')
