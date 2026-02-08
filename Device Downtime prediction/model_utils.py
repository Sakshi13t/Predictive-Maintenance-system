import pandas as pd
import numpy as np
import joblib
import json

# Load models
reg_model_dict = joblib.load("downtime_predictor_xgbmodel.pkl")
clf_model_dict = joblib.load("xgb_failure_classifier.pkl")

reg_model = reg_model_dict.get("model")
clf_model = clf_model_dict.get("model")
threshold = clf_model_dict.get("threshold", 0.8)

# Load feature names
with open("regression_model_features.json") as f:
    reg_features = json.load(f)
with open("classification_model_features.json") as f:
    clf_features = json.load(f)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df['time_down'] = pd.to_datetime(df['time_down'])
    df['time_up'] = pd.to_datetime(df['time_up'])
    df['last_active_time'] = pd.to_datetime(df['last_active_time'])

    df['downtime_minutes'] = (df['time_up'] - df['time_down']).dt.total_seconds() / 60
    df['downtime_log'] = np.log1p(df['downtime_minutes'])

    df['down_hour'] = df['time_down'].dt.hour
    df['down_day'] = df['time_down'].dt.day
    df['down_minute'] = df['time_down'].dt.minute
    df['month'] = df['time_down'].dt.month
    df['dayofweek'] = df['time_down'].dt.dayofweek
    df['dayofmonth'] = df['time_down'].dt.day
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    df['idel_time'] = pd.to_numeric(df['idel_time'], errors='coerce').fillna(0)

    df['bin_long'] = (df['idel_time'] > 30).astype(int)
    df['bin_medium'] = ((df['idel_time'] > 10) & (df['idel_time'] <= 30)).astype(int)
    df['bin_short'] = (df['idel_time'] <= 10).astype(int)
    df['is_4_916667'] = (df['idel_time'].round(6) == 4.916667).astype(int)

    df.sort_values(['device_name', 'time_down'], inplace=True)

    df['time_since_last_down'] = df.groupby('device_name')['time_down'].diff().dt.total_seconds().fillna(0)
    df['time_since_last_up'] = df.groupby('device_name')['time_up'].diff().dt.total_seconds().fillna(0)

    device_map = {name: i for i, name in enumerate(df['device_name'].fillna('unknown').unique())}
    df['device_name_te'] = df['device_name'].map(device_map)

    for col in ['downtime_minutes', 'idel_time', 'time_since_last_down', 'time_since_last_up']:
        grp = df.groupby('device_name')[col]
        df[f'{col}_mean'] = grp.transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df[f'{col}_median'] = grp.transform(lambda x: x.rolling(window=3, min_periods=1).median())
        df[f'{col}_std'] = grp.transform(lambda x: x.rolling(window=3, min_periods=1).std()).fillna(0)
        df[f'{col}_max'] = grp.transform(lambda x: x.rolling(window=3, min_periods=1).max())
        df[f'{col}_min'] = grp.transform(lambda x: x.rolling(window=3, min_periods=1).min())
        df[f'{col}_count'] = grp.transform(lambda x: x.rolling(window=3, min_periods=1).count())

    df['recent_failures_1d'] = 0
    for device in df['device_name'].unique():
        device_df = df[df['device_name'] == device].copy()
        times = device_df['time_down'].values
        counts = []
        for i, current_time in enumerate(times):
            past_24h = pd.to_datetime(current_time) - pd.Timedelta(days=1)
            count = device_df[(device_df['time_down'] >= past_24h) &
                              (device_df['time_down'] < current_time)].shape[0]
            counts.append(count)
        df.loc[df['device_name'] == device, 'recent_failures_1d'] = counts
    return df

def _encode_categoricals(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Label encode any object or category columns used in features list."""
    for col in features:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

def preprocess_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    # Save original values
    original_cols = df[['plant_name', 'location_name', 'device_name', 'device_ip']].copy()

    # Preprocess for model input
    df = preprocess_features(df)

    missing_reg = [f for f in reg_features if f not in df.columns]
    missing_clf = [f for f in clf_features if f not in df.columns]
    if missing_reg or missing_clf:
        raise ValueError(f"Missing features - Regression: {missing_reg}, Classification: {missing_clf}")

    # Encode for prediction
    df = _encode_categoricals(df, reg_features)
    df = _encode_categoricals(df, clf_features)

    # Predict
    df['predicted_downtime_minutes'] = reg_model.predict(df[reg_features])
    # df['will_fail_soon'] = (clf_model.predict_proba(df[clf_features])[:, 1] >= threshold).astype(int)
    df['is_failure_prob'] = clf_model.predict_proba(df[clf_features])[:, 1]
    df['will_fail_soon'] = (df['is_failure_prob'] >= threshold).astype(int)


    # Add back original names (replace encoded)
    # df[['plant_name', 'location_name', 'device_name', 'device_ip']] = original_cols
    for col in ['plant_name', 'location_name', 'device_name', 'device_ip']:
        df[col] = original_cols[col].values


    # Return required output
    # return df[['plant_name', 'location_name', 'device_name', 'device_ip', 'is_failure_prob','predicted_downtime_minutes', 'will_fail_soon']]
    return df[['plant_name', 'location_name', 'device_name', 'device_ip', 'is_failure_prob', 'predicted_downtime_minutes', 'will_fail_soon','time_down']].copy().assign(plant_code=df['plant_name'])


