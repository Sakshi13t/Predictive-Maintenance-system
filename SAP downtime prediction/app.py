from flask import Flask, jsonify
import pandas as pd
import joblib
import os
import json
import requests
from sqlalchemy import create_engine
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from collections import defaultdict
import logging
from logging.handlers import RotatingFileHandler
from preprocess_utils import preprocess_for_stage1, preprocess_for_stage2

# ---------------------- Configuration -----------------------
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
LAST_PROCESSED_FILE = "last_processed.txt"
SENT_ALERTS_FILE = "sent_alerts.json"
ALERT_EXPIRY_MINUTES = 30  # prevent resending within 30 mins
N8N_EMAIL_WEBHOOK = os.environ.get("N8N_EMAIL_WEBHOOK")

IST = timezone(timedelta(hours=5, minutes=30))

# ----------- Logging Setup (file + console, rotation) ---------
LOG_FILE = "sap_downtime.log"
logger = logging.getLogger("SAPDowntime")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------- Model Loading --------------------------
try:
    stage1_model = joblib.load("stage1_xgb_model.pkl")
    logger.info("Loaded stage1_xgb_model.pkl")
    with open("model_features.txt") as f:
        stage1_features = [line.strip() for line in f]
    stage2a_model = joblib.load("xgb_stage2a_504_vs_rest.pkl")
    logger.info("Loaded xgb_stage2a_504_vs_rest.pkl")
    stage2b_model = joblib.load("xgb_stage2b_500_503_noresp.pkl")
    logger.info("Loaded xgb_stage2b_500_503_noresp.pkl")
    with open("stage1_best_threshold.json") as f:
        stage1_best_threshold = json.load(f).get("best_threshold", 0.5)
    stage2a_threshold = 0.639
    stage2b_threshold = 0.6
except Exception as e:
    logger.error(f"Model loading error: {e}", exc_info=True)
    raise

# -------- Error, Stage, and Recommendation Mapping ----------
error_type_mapping = {0: "No Response", 1: "500", 2: "503"}
stage_code_map = {
    "AUTO_PDF": "Auto-Invoice", "CCI": "Cancel Check-In", "CGI": "Cancel Gate-In",
    "CI": "Check-In", "CPI": "Cancel Packing-In", "CPO": "Cancel Packing-Out",
    "GI": "Gate-In", "GO": "Gate-Out", "GW": "Gross Weight", "PI": "Packing-In",
    "PO": "Packing-Out", "TW": "Tare Weight", "WI": "Weigh-In", "WO": "Weigh-Out",
    "YARD-IN": "Yard-In"
}
error_description_mapping = {
    "500": "Internal Server Error",
    "503": "Service Unavailable",
    "504": "Gateway Timeout / Error",
    "No Response": "System did not return a valid response; potential network or service issue"
}
ERROR_RECOMMENDATIONS = {
    "No Response": "Investigate {stage} process for connectivity or server issues immediately.",
    "500": "Check SAP system logs for 500 error in {stage}.",
    "503": "Check SAP service availability for {stage}.",
    "504": "Check network connectivity or server performance for {stage}."
}

STAGE_RECOMMENDATIONS = {
    "Auto-Invoice": {"500": "Check SAP logs for Auto-Invoice module for any internal server errors.",
                     "No Response": "Investigate Auto-Invoice process connectivity or service issues immediately."},
    "Cancel Check-In": {"504": "Verify network and gateway response for Cancel Check-In process.",
                        "No Response": "Check Cancel Check-In process for unresponsive service issues."},
    "Cancel Gate-In": {"503": "Ensure Cancel Gate-In service is available and running properly.",
                       "No Response": "Investigate Cancel Gate-In process connectivity or server issues."},
    "Check-In": {"500": "Investigate Check-In service logs for server errors.",
                 "No Response": "Check Check-In process for connectivity or server issues immediately."},
    "Cancel Packing-In": {"504": "Check network connectivity and server response for Cancel Packing-In process."},
    "Cancel Packing-Out": {"503": "Verify Cancel Packing-Out service availability."},
    "Gate-In": {"500": "Check Gate-In module logs for server errors.",
                "No Response": "Investigate Gate-In process connectivity or service failures."},
    "Gate-Out": {"504": "Check Gateway timeout issues for Gate-Out process."},
    "Gross Weight": {"503": "Ensure Gross Weight service is available."},
    "Packing-In": {"500": "Investigate Packing-In process logs for server errors."},
    "Packing-Out": {"504": "Check Packing-Out process for timeout or network issues."},
    "Tare Weight": {"503": "Ensure Tare Weight service availability."},
    "Weigh-In": {"500": "Investigate Weigh-In service logs for internal errors."},
    "Weigh-Out": {"504": "Check Weigh-Out process for gateway or network issues."},
    "Yard-In": {"No Response": "Investigate Yard-In process connectivity or service issues immediately."}
}

# -------- Atomic JSON Alert Tracking Utilities ----------
def load_sent_alerts():
    if os.path.exists(SENT_ALERTS_FILE):
        with open(SENT_ALERTS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Empty or corrupt sent_alerts.json")
                return {}
    return {}

def save_sent_alerts(sent_alerts):
    # Atomic write for file persistence
    tmp_file = SENT_ALERTS_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(sent_alerts, f)
    os.replace(tmp_file, SENT_ALERTS_FILE)

def should_send_alert(plant, stage, error):
    sent_alerts = load_sent_alerts()
    key = f"{plant}_{stage}_{error}"
    last_sent_str = sent_alerts.get(key)
    if last_sent_str:
        try:
            last_sent = datetime.strptime(last_sent_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.warning(f"Alert timestamp parse error: {e}")
            return True
        if (datetime.utcnow() - last_sent).total_seconds() < ALERT_EXPIRY_MINUTES * 60:
            return False
    return True

def mark_alert_sent(plant, stage, error):
    sent_alerts = load_sent_alerts()
    key = f"{plant}_{stage}_{error}"
    sent_alerts[key] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    save_sent_alerts(sent_alerts)

def update_last_processed_timestamp(df):
    if not df.empty:
        max_ts = df['created_date'].max()
        with open(LAST_PROCESSED_FILE, "w") as f:
            f.write(str(max_ts))

# ----------- DB Fetch (Error Handling) ----------------
def fetch_latest_data(limit=100):
    try:
        last_ts = "1970-01-01 00:00:00"
        if os.path.exists(LAST_PROCESSED_FILE):
            with open(LAST_PROCESSED_FILE, "r") as f:
                last_ts = f.read().strip()
        query = f"""
            SELECT *
            FROM interfaceplms.outbound_stagedetails
            WHERE created_date > '{last_ts}'
            ORDER BY created_date ASC
            LIMIT {limit};
        """
        db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
        engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
        df = pd.read_sql(query, engine)
        logger.info(f"Fetched {len(df)} new records from DB")
        return df
    except Exception as e:
        logger.error("DB error", exc_info=True)
        return pd.DataFrame()

# ------------ Email Map Fetch (DB logging) --------------
def fetch_plant_email_map():
    try:
        db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
        engine = create_engine(db_url)
        client_df = pd.read_sql("""
            SELECT plant_code, email_to 
            FROM masterplms.automailer_mail
            WHERE email_type = 'client_mail';
        """, engine)
        engineer_df = pd.read_sql("""
            SELECT plant_code, email_to 
            FROM masterplms.automailer_mail
            WHERE email_type = 'plant_engineer';
        """, engine)
        client_map = client_df.set_index('plant_code')['email_to'].to_dict()
        engineer_map = engineer_df.set_index('plant_code')['email_to'].to_dict()
        logger.info("Fetched plant-client and plant-engineer email mappings")
        return {
            "client": client_map,
            "engineer": engineer_map,
            "_default_client": "sakshi.tandon@amzbizsol.in",
            "_default_engineer": "plant.support@amzbizsol.in"
        }
    except Exception as e:
        logger.error("Could not fetch plant emails", exc_info=True)
        return {
            "client": {"_default_": "sakshi.tandon@amzbizsol.in"},
            "engineer": {"_default_": "plant.support@amzbizsol.in"}
        }

# ------------ Email Sending (logging & error) ------------
def send_alerts_to_n8n(alerts):
    email_maps = fetch_plant_email_map()
    grouped_by_plant = defaultdict(list)
    for alert in alerts:
        grouped_by_plant[alert['plantCode']].append(alert)

    for plant, plant_alerts in grouped_by_plant.items():
        for alert in plant_alerts:
            # Route by probability
            recipient = (email_maps["client"].get(plant, email_maps["_default_client"])
                         if alert['probError'] is not None and alert['probError'] >= 0.8
                         else email_maps["engineer"].get(plant, email_maps["_default_engineer"]))
            if not should_send_alert(alert['plantCode'], alert['stageCode'], alert['errorType']):
                logger.info(f"Skipping alert for {plant}, {alert['stageCode']}, {alert['errorType']} due to expiry window")
                continue
            mark_alert_sent(alert['plantCode'], alert['stageCode'], alert['errorType'])

            stage_display = stage_code_map.get(alert['stageCode'], alert['stageCode'])
            error_code = alert['errorType']
            error_display = error_description_mapping.get(error_code, error_code)
            prob_display = f"{alert['probError']*100:.1f}%" if alert.get('probError') else "N/A"
            alert_type = "Critical" if alert['timeToError'] == "Immediate attention required" else "Warning"
            recommendation = STAGE_RECOMMENDATIONS.get(stage_display, {}).get(
                error_code,
                ERROR_RECOMMENDATIONS.get(error_code, "No specific action defined.")
            )
            recommendation_text = recommendation.format(stage=stage_display)
            critical_style = "color:#d9534f; font-weight:bold;" if alert_type == "Critical" else ""
            critical_icon = "⚠" if alert_type == "Critical" else ""

            table_row = f"""
                <tr>
                    <td>{stage_display}</td>
                    <td>{error_code}</td>
                    <td>{error_display}</td>
                    <td style="{critical_style}">{critical_icon}{alert['timeToError']}</td>
                    <td>{prob_display}</td>
                    <td>{alert_type}</td>
                    <td>{recommendation_text}</td>
                    <td>{alert['alertTimestamp']}</td>
                </tr>
            """

            html_body = f"""
            <html>
            <body>
                <h2>SAP Monitoring System Alert</h2>
                <p>Dear Team,<br>
                   This is an automated alert for <strong>Plant {plant}</strong> regarding predicted SAP errors. Please review and take action immediately to avoid downtime.</p>
                <table border="1" cellpadding="10" style="border-collapse:collapse;">
                    <tr>
                        <th>Stage</th>
                        <th>HTTP Code</th>
                        <th>Error Description</th>
                        <th>Predicted Downtime</th>
                        <th>Prediction Probability</th>
                        <th>Alert Type</th>
                        <th>Recommended Action</th>
                        <th>Predicted At</th>
                    </tr>
                    {table_row}
                </table>
                <hr />
                <small>SAP Monitoring System | Version 1.0 | Contact: support@amzbizsol.in</small>
            </body>
            </html>
            """

            payload = {"to": recipient, "subject": f"SAP Downtime Alert - Plant {plant}", "body": html_body, "isHtml": True}
            try:
                response = requests.post(N8N_EMAIL_WEBHOOK, json=payload, timeout=10)
                logger.info(f"Email sent to {recipient}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error sending email to {recipient}: {e}", exc_info=True)

# --------------- Prediction Pipeline (with logging) ----------------
def run_predictions():
    IST = timezone(timedelta(hours=5, minutes=30))
    try:
        df_raw = fetch_latest_data(limit=100)
        if df_raw.empty:
            logger.info("No new data for prediction")
            return []
        alerts = []
        X_stage1 = preprocess_for_stage1(df_raw)
        probs = stage1_model.predict_proba(X_stage1)[:, 1] if hasattr(stage1_model, "predict_proba") else None
        preds = stage1_model.predict(X_stage1)
        for idx, pred in enumerate(preds):
            prob_error = probs[idx] if probs is not None else None
            if prob_error is not None and prob_error < stage1_best_threshold:
                continue
            if pred == 1 or (prob_error is not None and prob_error >= stage1_best_threshold):
                record = df_raw.iloc[idx]
                stage_code = record.get("stage_code", "")
                stage_name = stage_code_map.get(stage_code, stage_code)
                if stage_code == "504":
                    X_2a = preprocess_for_stage2(pd.DataFrame([record]), stage="2a")
                    err_type_pred = stage2a_model.predict(X_2a)[0]
                    prob_stage2 = stage2a_model.predict_proba(X_2a).max() if hasattr(stage2a_model, "predict_proba") else None
                    err_type_name = str(err_type_pred)
                    if prob_stage2 is not None and prob_stage2 < stage2a_threshold:
                        continue
                else:
                    X_2b = preprocess_for_stage2(pd.DataFrame([record]), stage="2b")
                    err_type_pred = stage2b_model.predict(X_2b)[0]
                    prob_stage2 = stage2b_model.predict_proba(X_2b).max() if hasattr(stage2b_model, "predict_proba") else None
                    err_type_name = error_type_mapping.get(int(err_type_pred), str(err_type_pred))
                    if prob_stage2 is not None and prob_stage2 < stage2b_threshold:
                        continue
                alert = {
                    "plantCode": record.get("plant_code", ""),
                    "stageCode": stage_name,
                    "errorType": err_type_name,
                    "errorDescription": error_description_mapping.get(err_type_name, "Unknown Error"),
                    "probError": round(float(prob_error), 4) if prob_error is not None else None,
                    "timeToError": "Immediate attention required",
                    "alertTimestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                }
                alerts.append(alert)
        update_last_processed_timestamp(df_raw)
        if alerts:
            send_alerts_to_n8n(alerts)
        logger.info(f"Prediction batch processed, {len(alerts)} alerts generated")
        return alerts
    except Exception as e:
        logger.error(f"[PREDICTION ERROR] {e}", exc_info=True)
        return []

# -------- Flask App and Scheduler setup (with logging) ---------
app = Flask(__name__)

scheduler = BackgroundScheduler()
scheduler.add_job(func=run_predictions, trigger="interval", minutes=10)
scheduler.start()
logger.info("Scheduler started, interval 10 min")

import atexit
atexit.register(lambda: (logger.info("Shutting down scheduler"), scheduler.shutdown()))

@app.route("/predict", methods=["GET"])
def predict():
    logger.info("API /predict called")
    alerts = run_predictions()
    logger.info(f"API /predict returned {len(alerts)} alerts")
    return jsonify({"predictions": alerts, "count": len(alerts)})

if __name__ == "__main__":
    logger.info("Starting SAP Downtime Prediction Service (Flask app)")
    app.run(host="0.0.0.0", port=5001, debug=False)
