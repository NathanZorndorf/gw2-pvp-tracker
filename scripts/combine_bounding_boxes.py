import pandas as pd
import os

def combine_bounding_boxes():
    files = [
        'data/ranked-start-bounding-boxes.csv',
        'data/unranked-start-bounding-boxes.csv', 
        'data/ranked-end-bounding-boxes.csv',
        'data/ranked-icon-bounding-boxes-rectangle.csv'
    ]
    
    dfs = []
    for f in files:
        if os.path.exists(f):
            print(f"Reading {f}")
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        else:
            print(f"Warning: File not found: {f}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Reorder columns to put source_file first or last? Let's keep it last.
        
        output_path = 'data/all_bounding_boxes.csv'
        combined.to_csv(output_path, index=False)
        print(f"Successfully created {output_path} with {len(combined)} rows.")
        print("First few rows:")
        print(combined.head())
    else:
        print("No CSV files found to combine.")

if __name__ == "__main__":
    combine_bounding_boxes()
