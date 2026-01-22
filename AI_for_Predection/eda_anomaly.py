"""
EDA and anomaly detection script.
Performs: summary stats, missing values, IQR outliers and IsolationForest anomalies; saves diagnostic plots.
"""
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

sns.set()

def summary(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    desc = df.describe(include='all')
    desc.to_csv(outdir / 'summary_describe.csv')
    df.isnull().sum().to_csv(outdir / 'missing_counts.csv')

def iqr_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return (df[col] < low) | (df[col] > high)

def isolation_anomalies(df, features, contamination=0.01):
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    X = df[features].fillna(0).values
    iso.fit(X)
    preds = iso.predict(X)
    # -1 anomaly, 1 normal
    return preds == -1

def plot_distributions(df, cols, outdir):
    for c in cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[c].dropna(), kde=True)
        plt.title(f'Distribution: {c}')
        plt.savefig(outdir / f'dist_{c}.png')
        plt.close()

def run(input_csv, outdir):
    df = pd.read_csv(input_csv)
    outdir = Path(outdir)
    summary(df, outdir)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_distributions(df, numeric, outdir)

    # IQR per numeric
    iqr_flags = pd.DataFrame(False, index=df.index, columns=['iqr_outlier'])
    for c in numeric:
        mask = iqr_outliers(df, c)
        iqr_flags.loc[mask, 'iqr_outlier'] = True
    # IsolationForest on selected features
    iso_feats = numeric if len(numeric) > 0 else []
    if iso_feats:
        iso_mask = isolation_anomalies(df, iso_feats, contamination=0.01)
    else:
        iso_mask = np.array([False]*len(df))

    df['iqr_outlier'] = iqr_flags['iqr_outlier']
    df['isolation_anomaly'] = iso_mask
    df['anomaly_any'] = df['iqr_outlier'] | df['isolation_anomaly']

    df[df['anomaly_any']].to_csv(outdir / 'anomalies.csv', index=False)
    print('Anomalies:', df['anomaly_any'].sum())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input merged CSV')
    parser.add_argument('--outdir', default='AI_for_Predection/eda_outputs')
    args = parser.parse_args()
    run(args.input, args.outdir)
