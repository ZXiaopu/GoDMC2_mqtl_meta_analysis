import numpy as np
import pandas as pd
import os

reference_ids = np.load('/projects/b35cx/bib_godmc2/results/04/meta_inputs/mapping/keys_ref-hrc.npy', allow_pickle=True)

cohort_h5_ids = pd.read_hdf('/projects/b35cx/bib_godmc2/results/04/meta_inputs/use_data/probes/bib_eur_mother_wrong.h5', key='probes')['ID'].tolist()

index_to_ref = np.load('/projects/b35cx/bib_godmc2/results/04/meta_inputs/mapping/values_ref-hrc_bib_eur_mother_right.npy', allow_pickle=True)

id_to_pos = {snp_id: i for i, snp_id in enumerate(cohort_h5_ids)}

new_index_to_ref = np.array([id_to_pos.get(ref_id, -1) for ref_id in reference_ids], dtype=np.int32)

output_path = '/projects/b35cx/light_hase_test/allMapper'

np.save(os.path.join(output_path, 'values_ref-hrc_bib_eur_mother.npy'), new_index_to_ref)


# new npy
new_npy = np.load("/projects/b35cx/light_hase_test/allMapper/values_ref-hrc_bib_eur_mother.npy", allow_pickle=True)
old_npy = np.load("/projects/b35cx/bib_godmc2/results/04/meta_inputs/mapping/values_ref-hrc_bib_eur_mother.npy", allow_pickle=True)

are_equal = np.array_equal(new_index_to_ref, old_npy)
are_equal
# True


new = np.load("/projects/b35cx/light_hase_test/allMapper/values_ref-hrc_bib_eur_mother.npy", allow_pickle=True)
old = np.load("/projects/b35cx/bib_godmc2/results/04/meta_inputs/mapping/values_ref-hrc_bib_eur_mother.npy", allow_pickle=True)
inter = np.load("/projects/b35cx/bib_godmc2/results/04/meta_inputs/mapping/values_ref-hrc_bib_eur_mother_reindex.npy", allow_pickle=True)

new.shape, old.shape, inter.shape