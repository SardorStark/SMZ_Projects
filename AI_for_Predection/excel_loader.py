"""
Excel loader: load multiple Excel files from a folder and merge into a single DataFrame.
Assumes typical columns: timestamp/date, roll_id, weight_kg, length_m, defects_count, defect_positions (optional)
"""
from pathlib import Path
import pandas as pd

def load_excel_file(path, date_col_candidates=['date','timestamp','time']):
    df = pd.read_excel(path)
    # try parse date
    for c in date_col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            df.rename(columns={c:'timestamp'}, inplace=True)
            break
    return df

def load_excel_folder(folder):
    folder = Path(folder)
    files = sorted(folder.glob('*.xlsx')) + sorted(folder.glob('*.xls'))
    dfs = []
    for f in files:
        try:
            df = load_excel_file(f)
            df['source_file'] = str(f.name)
            dfs.append(df)
            print(f"Loaded {f.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    if not dfs:
        return None
    big = pd.concat(dfs, ignore_index=True)
    # basic cleaning
    if 'timestamp' in big.columns:
        big = big.sort_values('timestamp')
    return big

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data_excel', help='Folder with excel files')
    p.add_argument('--output', default='AI_for_Predection/merged.csv')
    args = p.parse_args()
    df = load_excel_folder(args.input)
    if df is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print('Saved merged to', args.output)
