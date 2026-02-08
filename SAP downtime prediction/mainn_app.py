from flask import Flask, jsonify
import pandas as pd
import joblib
from sqlalchemy import create_engine
from datetime import datetime
import os
import json

# Import preprocessing functions
from preprocess_utils import preprocess_for_stage1, preprocess_for_stage2

# ================================
# Config
# ================================
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "sapdb")

# ================================
# Load Models & Features
# ================================
stage1_model = joblib.load("stage1_xgb_model.pkl")
with open("model_features.txt") as f:
    stage1_features = [line.strip() for line in f]

stage2a_model = joblib.load("xgb_stage2a_504_vs_rest.pkl")
with open("stage2a_model_features.txt") as f:
    stage2a_features = [line.strip() for line in f]

stage2b_model = joblib.load("xgb_stage2b_500_503_noresp.pkl")
with open("stage2b_model_features.txt") as f:
    stage2b_features = [line.strip() for line in f]

# Load Stage-1 best threshold
with open("stage1_best_threshold.json") as f:
    stage1_best_threshold = json.load(f).get("best_threshold", 0.5)

# Stage-2 thresholds
stage2a_threshold = 0.639  # tuned from Stage-2A F1/Recall
stage2b_threshold = 0.6    # minimum probability for Stage-2B alerts

# Stage-2B mapping
error_type_mapping = {
    0: "No Response",
    1: "500",
    2: "503"
}

# Friendly stage names
stage_code_map = {
    "AUTO_PDF": "Auto-Invoice",
    "CCI": "Cancel Check-In",
    "CGI": "Cancel Gate-In",
    "CI": "Check-In",
    "CPI": "Cancel Packing-In",
    "CPO": "Cancel Packing-Out",
    "GI": "Gate-In",
    "GO": "Gate-Out",
    "GW": "Gross Weight",
    "PI": "Packing-In",
    "PO": "Packing-Out",
    "TW": "Tare Weight",
    "WI": "Weigh-In",
    "WO": "Weigh-Out",
    "YARD-IN": "Yard-In"
}

error_description_mapping = {
    "500": "Internal Server Error",
    "503": "Service Unavailable",
    "504": "Gateway Timeout / Error",
    "No Response": "No response received from the system"
}

# DB Utils
def fetch_latest_data(limit=50):
    try:
        query = f"""
            SELECT *
            FROM interfaceplms.outbound_stagedetails
            ORDER BY created_date DESC
            LIMIT {limit};
        """
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return pd.DataFrame()

# Prediction Pipeline
def run_predictions():
    df_raw = fetch_latest_data(limit=100)
    if df_raw.empty:
        return []

    alerts = []

    try:
        # Stage-1 preprocessing
        X_stage1 = preprocess_for_stage1(df_raw)
        probs = stage1_model.predict_proba(X_stage1)[:, 1] if hasattr(stage1_model, "predict_proba") else None
        preds = stage1_model.predict(X_stage1)

        for idx, pred in enumerate(preds):
            prob_error = probs[idx] if probs is not None else None

            # Apply Stage-1 threshold
            if prob_error is not None and prob_error < stage1_best_threshold:
                continue

            if pred == 1 or (prob_error is not None and prob_error >= stage1_best_threshold):
                record = df_raw.iloc[idx]
                stage_code = record.get("stage_code", "")
                stage_name = stage_code_map.get(stage_code, stage_code)

                # Stage-2A: 504 vs Rest
                if stage_code == "504":
                    X_2a = preprocess_for_stage2(pd.DataFrame([record]), stage="2a")
                    err_type_pred = stage2a_model.predict(X_2a)[0]
                    prob_stage2 = stage2a_model.predict_proba(X_2a).max() if hasattr(stage2a_model, "predict_proba") else None
                    err_type_name = str(err_type_pred)

                    # Apply Stage-2A threshold
                    if prob_stage2 is not None and prob_stage2 < stage2a_threshold:
                        continue

                # Stage-2B: 500 vs 503 vs No Response
                else:
                    X_2b = preprocess_for_stage2(pd.DataFrame([record]), stage="2b")
                    err_type_pred = stage2b_model.predict(X_2b)[0]
                    prob_stage2 = stage2b_model.predict_proba(X_2b).max() if hasattr(stage2b_model, "predict_proba") else None
                    err_type_name = error_type_mapping.get(int(err_type_pred), str(err_type_pred))

                    # Apply Stage-2B threshold
                    if prob_stage2 is not None and prob_stage2 < stage2b_threshold:
                        continue
                
                alerts.append({
                    "plantCode": record.get("plant_code", ""),
                    "stageCode": stage_name,
                    "errorType": err_type_name,
                    "errorDescription": error_description_mapping.get(err_type_name, "Unknown Error"),
                    "probError": round(float(prob_error), 4) if prob_error is not None else None,
                    "timeToError": "Immediate attention required",  # placeholder
                    "alertTimestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                })


    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")

    # Deduplicate alerts by plant + stage + error type only
    unique_alerts = {}
    for alert in alerts:
        key = (alert["plantCode"], alert["stageCode"], alert["errorType"])
        if key not in unique_alerts:
            unique_alerts[key] = alert

    return list(unique_alerts.values())

# Flask App
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    alerts = run_predictions()
    return jsonify({"predictions": alerts, "count": len(alerts)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
