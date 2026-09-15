import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
META_DIR = os.path.join(BASE_DIR, "data", "metadata")

def main():
    print("--- Inspecting FlyWire Data ---")
    
    # 1. Root IDs
    root_ids_path = os.path.join(RAW_DIR, "proofread_root_ids_783.npy")
    if os.path.exists(root_ids_path):
        root_ids = np.load(root_ids_path)
        print(f"\nTotal proofread neurons: {len(root_ids):,}")
    else:
        print("\n[!] Root IDs file missing.")

    # 2. Annotations
    meta_path = os.path.join(META_DIR, "neuron_annotations.tsv")
    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path, sep='\t')
        print(f"\nMetadata rows: {len(df_meta):,}")
        print("Columns:", df_meta.columns.tolist())
        
        if 'top_nt' in df_meta.columns:
            print("\nNeurotransmitter distribution:")
            print(df_meta['top_nt'].value_counts(dropna=False))
        
        if 'flow' in df_meta.columns:
            print("\nFlow distribution:")
            print(df_meta['flow'].value_counts(dropna=False))
    else:
        print("\n[!] Annotations file missing.")
        
    # 3. Connections
    conn_path = os.path.join(RAW_DIR, "proofread_connections_783.feather")
    if os.path.exists(conn_path):
        print("\nLoading connections (this may take a moment)...")
        df_conn = pd.read_feather(conn_path)
        print(f"Total connection records: {len(df_conn):,}")
        print("Columns:", df_conn.columns.tolist())
        print("\nSample rows:")
        print(df_conn.head())
    else:
        print("\n[!] Connections file missing.")

if __name__ == "__main__":
    main()

