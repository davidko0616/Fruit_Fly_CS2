import os
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp

class ConnectomeGraph:
    def __init__(self, raw_dir, meta_dir):
        self.raw_dir = raw_dir
        self.meta_dir = meta_dir
        
        print("Loading metadata...")
        meta_path = os.path.join(self.meta_dir, "neuron_annotations.tsv")
        self.df_meta = pd.read_csv(meta_path, sep='\t', low_memory=False)
        # Handle duplicate root_ids by taking the first one
        self.df_meta = self.df_meta.drop_duplicates(subset=['root_id']).set_index('root_id')
        
        print("Loading connections...")
        conn_path = os.path.join(self.raw_dir, "proofread_connections_783.feather")
        self.df_conn = pd.read_feather(conn_path, columns=[
            'pre_pt_root_id', 'post_pt_root_id', 'neuropil', 'syn_count'
        ])
        
    def extract_subgraph_by_neuropil(self, neuropil_name, max_neurons=None):
        if max_neurons is not None and (not isinstance(max_neurons, int) or max_neurons < 2):
            raise ValueError('max_neurons must be an integer of at least 2')
        print(f"Extracting subgraph for neuropil: {neuropil_name}...")
        # Filter connections happening in this neuropil
        sub_conn = self.df_conn[self.df_conn['neuropil'] == neuropil_name].copy()
        
        # Identify all unique neurons involved in this neuropil
        unique_neurons = set(sub_conn['pre_pt_root_id']).union(set(sub_conn['post_pt_root_id']))
        unique_neurons = sorted(unique_neurons)
        if not unique_neurons:
            raise ValueError(f'No connections found for neuropil {neuropil_name!r}')
        
        if max_neurons and len(unique_neurons) > max_neurons:
            print(f"Limiting to top {max_neurons} neurons by degree...")
            degree_counts = pd.concat([sub_conn['pre_pt_root_id'], sub_conn['post_pt_root_id']]).value_counts()
            ranked = degree_counts.rename('degree').rename_axis('root_id').reset_index()
            ranked = ranked.sort_values(['degree', 'root_id'], ascending=[False, True])
            unique_neurons = ranked.head(max_neurons)['root_id'].tolist()
            
            # Filter connections again to only include these neurons
            sub_conn = sub_conn[
                sub_conn['pre_pt_root_id'].isin(unique_neurons) & 
                sub_conn['post_pt_root_id'].isin(unique_neurons)
            ]
        
        # Map root_id -> local matrix index (0 to N-1)
        id_to_idx = {root_id: i for i, root_id in enumerate(unique_neurons)}
        
        # Build sparse adjacency (CSR matrix)
        row_idx = sub_conn['pre_pt_root_id'].map(id_to_idx).values
        col_idx = sub_conn['post_pt_root_id'].map(id_to_idx).values
        weights = sub_conn['syn_count'].values
        
        N = len(unique_neurons)
        adjacency = sp.csr_matrix((weights, (row_idx, col_idx)), shape=(N, N))
        
        # Extract metadata for these specific neurons
        meta_subset = self.df_meta.reindex(unique_neurons)
        
        # Modeling assumption: predicted neurotransmitters set source weight signs.
        # This does not capture receptor-specific or modulatory biological effects.
        # GABA / Glutamate -> Inhibitory (-1)
        # Acetylcholine / others -> Excitatory (+1)
        def get_nt_sign(nt):
            if pd.isna(nt): return 1.0
            nt = str(nt).upper()
            if 'GABA' in nt or 'GLUT' in nt:
                return -1.0
            return 1.0
            
        nt_signs = np.array([get_nt_sign(nt) for nt in meta_subset['top_nt']])
        
        return {
            'adjacency': adjacency,
            'root_ids': unique_neurons,
            'metadata': meta_subset,
            'nt_signs': nt_signs
        }
        
    def assign_io_neurons(self, metadata, allow_fallback=True):
        """Assign disjoint IO groups, with explicit computational fallbacks if needed."""
        if len(metadata) < 2:
            raise ValueError('At least two neurons are needed for disjoint inputs and outputs')
        flows = metadata['flow'].fillna('intrinsic').astype(str).str.lower()
        
        inputs = np.where(flows == 'afferent')[0]
        outputs = np.where(flows == 'efferent')[0]
        if len(inputs) == 0 or len(outputs) == 0:
            if not allow_fallback:
                raise ValueError('Selected graph lacks afferent or efferent neurons')
            warnings.warn('Missing afferent/efferent neurons: using computational IO '
                          'assignments, not a biological sensory/motor mapping.', UserWarning)
        available = np.setdiff1d(np.arange(len(metadata)), np.r_[inputs, outputs])
        if len(inputs) == 0:
            count = min(max(1, len(metadata) // 10), len(available) - int(len(outputs) == 0))
            if count < 1:
                raise ValueError('Not enough unassigned neurons for input fallback')
            inputs = available[:count]
            available = available[count:]
        if len(outputs) == 0:
            if len(available) == 0:
                raise ValueError('Not enough unassigned neurons for output fallback')
            outputs = available[-max(1, len(metadata) // 10):]
        hidden = np.setdiff1d(np.arange(len(metadata)), np.r_[inputs, outputs])
        return inputs, outputs, hidden

if __name__ == "__main__":
    # Simple test of the graph extraction
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph = ConnectomeGraph(
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "metadata")
    )
    
    # Extract a tiny 100-neuron subset of the right Antennal Lobe (AL_R)
    subgraph = graph.extract_subgraph_by_neuropil("AL_R", max_neurons=100)
    
    print(f"\nExtracted subgraph matrix shape: {subgraph['adjacency'].shape}")
    print(f"Non-zero connections: {subgraph['adjacency'].nnz}")
    
    inputs, outputs, hidden = graph.assign_io_neurons(subgraph['metadata'])
    print(f"Input neurons: {len(inputs)}")
    print(f"Output neurons: {len(outputs)}")
    print(f"Hidden neurons: {len(hidden)}")
    
    print("\nNeurotransmitter balance:")
    signs = subgraph['nt_signs']
    print(f"Excitatory (+1): {(signs == 1).sum()}")
    print(f"Inhibitory (-1): {(signs == -1).sum()}")
