import pandas as pd
import glob

def combine_csvs(pattern, output_file):
    all_files = glob.glob(pattern)
    df_list = [pd.read_csv(f) for f in all_files]
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Saved {output_file} ({len(combined)} rows)")
    
n = 1000000

combine_csvs(f"output/*n_{n}_full_orb.csv", f"n_{n}_full_orb.csv")
combine_csvs(f"output/*n_{n}_partitions_orb.csv", f"n_{n}_partitions_orb.csv")
combine_csvs(f"output/*n_{n}_full_mattersim.csv", f"n_{n}_full_mattersim.csv")
combine_csvs(f"output/*n_{n}_partitions_mattersim.csv", f"n_{n}_partitions_mattersim.csv")