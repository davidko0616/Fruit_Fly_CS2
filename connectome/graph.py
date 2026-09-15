import os
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
        self.df_conn = pd.read_feather(conn_path)
        
    def extract_subgraph_by_neuropil(self, neuropil_name, max_neurons=None):
        print(f"Extracting subgraph for neuropil: {neuropil_name}...")
        # Filter connections happening in this neuropil
        sub_conn = self.df_conn[self.df_conn['neuropil'] == neuropil_name].copy()
        
        # Identify all unique neurons involved in this neuropil
        unique_neurons = set(sub_conn['pre_pt_root_id']).union(set(sub_conn['post_pt_root_id']))
        unique_neurons = list(unique_neurons)
        
        if max_neurons and len(unique_neurons) > max_neurons:
            print(f"Limiting to top {max_neurons} neurons by degree...")
            degree_counts = pd.concat([sub_conn['pre_pt_root_id'], sub_conn['post_pt_root_id']]).value_counts()
            unique_neurons = degree_counts.head(max_neurons).index.tolist()
            
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
        
        # Map neurotransmitters to weight signs
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
        
    def assign_io_neurons(self, metadata):
        """Assign matrix indices based on biological flow (sensory, motor, interneuron)"""
        flows = metadata['flow'].fillna('intrinsic').astype(str).str.lower()
        
        inputs = np.where(flows.str.contains('afferent'))[0]
        outputs = np.where(flows.str.contains('efferent'))[0]
        hidden = np.where(~flows.str.contains('afferent|efferent'))[0]
        
        # Fallback if no afferent/efferent found (happens in small isolated subgraphs)
        if len(inputs) == 0:
            inputs = hidden[:max(1, len(hidden)//10)]
        if len(outputs) == 0:
            outputs = hidden[-max(1, len(hidden)//10):]
            
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
