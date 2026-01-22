"""
Baseline modeling: feature engineering and LightGBM regression for predicting next roll weight/length.
Sane defaults provided; later you will supply exact formulas to compute 'norms'.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def create_features(df):
    df = df.copy()
    # ensure timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp')
    # basic features
    if 'weight_kg' in df.columns:
        df['weight_prev'] = df['weight_kg'].shift(1)
        df['weight_roll_mean_3'] = df['weight_kg'].rolling(3).mean()
    if 'length_m' in df.columns:
        df['length_prev'] = df['length_m'].shift(1)
        df['length_roll_mean_3'] = df['length_m'].rolling(3).mean()
    # fill na
    df = df.fillna(0)
    return df


def train_model(input_csv, target_col='weight_kg', output_model='AI_for_Predection/model.joblib'):
    df = pd.read_csv(input_csv)
    df = create_features(df)
    features = [c for c in df.columns if c not in ['timestamp','source_file','defect_image_path','full_frame_path'] and c != target_col]
    X = df[features]
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if lgb is None:
        raise RuntimeError('lightgbm not installed. pip install lightgbm')

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=100, early_stopping_rounds=10)
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    print('MAE:', mean_absolute_error(y_val, preds))
    print('RMSE:', mean_squared_error(y_val, preds, squared=False))

    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)
    print('Model saved to', output_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--target', default='weight_kg')
    parser.add_argument('--output', default='AI_for_Predection/model.joblib')
    args = parser.parse_args()
    train_model(args.input, target_col=args.target, output_model=args.output)
