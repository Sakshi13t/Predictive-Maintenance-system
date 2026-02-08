from flask import Flask, jsonify
import pandas as pd
import joblib
from sqlalchemy import create_engine
from datetime import datetime
import os
import json
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from collections import defaultdict
from preprocess_utils import preprocess_for_stage1, preprocess_for_stage2
import logging 
from datetime import datetime, timedelta, timezone

# Config
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
LAST_PROCESSED_FILE = "last_processed.txt"
SENT_ALERTS_FILE = "sent_alerts.json"
ALERT_EXPIRY_MINUTES = 30  # prevent resending within 30 mins
N8N_EMAIL_WEBHOOK = os.environ.get("N8N_EMAIL_WEBHOOK")

# For production-based  
# from sqlalchemy.orm import sessionmaker
# engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}", pool_pre_ping=True, pool_recycle=3600)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Load Models & Thresholds
stage1_model = joblib.load("stage1_xgb_model.pkl")
with open("model_features.txt") as f:
    stage1_features = [line.strip() for line in f]

stage2a_model = joblib.load("xgb_stage2a_504_vs_rest.pkl")
stage2b_model = joblib.load("xgb_stage2b_500_503_noresp.pkl")

with open("stage1_best_threshold.json") as f:
    stage1_best_threshold = json.load(f).get("best_threshold", 0.5)

stage2a_threshold = 0.639
stage2b_threshold = 0.6

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

# --- JSON Alert Tracking Utils ---
def load_sent_alerts():
    if os.path.exists(SENT_ALERTS_FILE):
        with open(SENT_ALERTS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_sent_alerts(sent_alerts):
    with open(SENT_ALERTS_FILE, "w") as f:
        json.dump(sent_alerts, f)

def should_send_alert(plant, stage, error):
    sent_alerts = load_sent_alerts()
    key = f"{plant}_{stage}_{error}"
    last_sent_str = sent_alerts.get(key)
    if last_sent_str:
        last_sent = datetime.strptime(last_sent_str, "%Y-%m-%d %H:%M:%S")
        if (datetime.utcnow() - last_sent).total_seconds() < ALERT_EXPIRY_MINUTES * 60:
            return False
    return True

def mark_alert_sent(plant, stage, error):
    sent_alerts = load_sent_alerts()
    key = f"{plant}_{stage}_{error}"
    sent_alerts[key] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    save_sent_alerts(sent_alerts)

# DB Utils
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

        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        return pd.read_sql(query, engine)
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.error("DB error", exc_info=True)
        # print(f"[DB ERROR] {e}")
        return pd.DataFrame()

# def fetch_plant_email_map():
#     try:
#         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
#         query = """
#             SELECT plant_code, email_to 
#             FROM masterplms.automailer_mail
#             WHERE email_type = 'mail_alert_test';
#         """
#         df = pd.read_sql(query, engine)
#         plant_email_map = df.set_index('plant_code')['email_to'].to_dict()
#         plant_email_map["_default_"] = "support@amzbizsol.in"
#         return plant_email_map
#     except Exception as e:
#         print(f"[DB ERROR] Could not fetch plant emails: {e}")
#         return {"_default_": "support@amzbizsol.in"}
    
def fetch_plant_email_map():
    try:
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        
        # Fetch client emails
        client_df = pd.read_sql("""
            SELECT plant_code, email_to 
            FROM masterplms.automailer_mail
            WHERE email_type = 'client_mail';
        """, engine)

        # Fetch plant engineer emails
        engineer_df = pd.read_sql("""
            SELECT plant_code, email_to 
            FROM masterplms.automailer_mail
            WHERE email_type = 'plant_engineer';
        """, engine)

        # Build dicts
        client_map = client_df.set_index('plant_code')['email_to'].to_dict()
        engineer_map = engineer_df.set_index('plant_code')['email_to'].to_dict()

        return {
            "client": client_map,
            "engineer": engineer_map,
            "_default_client": "sakshi.tandon@amzbizsol.in",
            "_default_engineer": "plant.support@amzbizsol.in"
        }
    except Exception as e:
        print(f"[DB ERROR] Could not fetch plant emails: {e}")
        return {
            "client": {"_default_": "sakshi.tandon@amzbizsol.in"},
            "engineer": {"_default_": "plant.support@amzbizsol.in"}
        }

def update_last_processed_timestamp(df):
    if not df.empty:
        max_ts = df['created_date'].max()
        with open(LAST_PROCESSED_FILE, "w") as f:
            f.write(str(max_ts))

# Recommendations
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

# --- Email Sending Function ---
def send_alerts_to_n8n(alerts):
    email_maps = fetch_plant_email_map()
    grouped_by_plant = defaultdict(list)
    for alert in alerts:
        grouped_by_plant[alert['plantCode']].append(alert)

    for plant, plant_alerts in grouped_by_plant.items():
        if alert['probError'] is not None and alert['probError'] >= 0.8:
                # Send to client
            recipient = email_maps["client"].get(plant, email_maps["_default_client"])
        else:
                # Send to plant engineer
            recipient = email_maps["engineer"].get(plant, email_maps["_default_engineer"])

        # Filter out alerts already sent
        plant_alerts_to_send = []
        for alert in plant_alerts:
            if should_send_alert(alert['plantCode'], alert['stageCode'], alert['errorType']):
                plant_alerts_to_send.append(alert)
                mark_alert_sent(alert['plantCode'], alert['stageCode'], alert['errorType'])

        if not plant_alerts_to_send:
            continue

        # Prepare email HTML
        table_rows = ""
        row_colors = ["#ffffff", "#f7f7f7"]
        for idx, alert in enumerate(plant_alerts_to_send):
            stage_display = stage_code_map.get(alert['stageCode'], alert['stageCode'])
            error_code = alert['errorType']
            error_display = error_description_mapping.get(alert['errorType'], alert['errorType'])
            prob_display = f"{alert['probError']*100:.1f}%" if alert.get('probError') else "N/A"
            alert_type = "Critical" if alert['timeToError'] == "Immediate attention required" else "Warning"
            recommendation = STAGE_RECOMMENDATIONS.get(stage_display, {}).get(
                error_code,
                ERROR_RECOMMENDATIONS.get(error_code, "No specific action defined.")
            )
            recommendation_text = recommendation.format(stage=stage_display)
            critical_style = "color:#d9534f; font-weight:bold;" if alert_type == "Critical" else ""
            critical_icon = "⚠" if alert_type == "Critical" else ""

            table_rows += f"""
                <tr style="background-color:{row_colors[idx % 2]};">
                    <td style="border:1px solid #ddd; padding:8px;">{stage_display}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{error_code}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{error_display}</td>
                    <td style="border:1px solid #ddd; padding:8px; {critical_style}">{critical_icon}{alert['timeToError']}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{prob_display}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{alert_type}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{recommendation_text}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{alert['alertTimestamp']}</td>
                </tr>
            """

        critical_count = sum(1 for a in plant_alerts_to_send if a['timeToError'] == "Immediate attention required")
        summary_text = f"Total Alerts: {len(plant_alerts_to_send)} | Critical Alerts: {critical_count}"

        subject = f"SAP Downtime Alert - Plant {plant}"
        html_body = f"""
        <div style="font-family: Arial, sans-serif; font-size:14px; color:#333; line-height:1.5;">
            <h2 style="color:#2E86C1; margin-bottom:0;">SAP Monitoring System Alert</h2>
            <p style="margin-top:5px;">Dear Team,</p>
            <p>This is an automated alert for <strong>Plant {plant}</strong> regarding predicted SAP errors. Please review and take action immediately to avoid downtime.</p>
            <p><strong>Summary:</strong> {summary_text}</p>
            <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; width:100%; text-align:center; border:1px solid #ccc;">
                <thead style="background-color:#007bff; font-weight:bold; color:white;">
                    <tr>
                        <th style="border:1px solid #ccc; padding:10px;">Stage</th>
                        <th style="border:1px solid #ccc; padding:10px;">HTTP Code</th>
                        <th style="border:1px solid #ccc; padding:10px;">Error Description</th>
                        <th style="border:1px solid #ccc; padding:10px;">Predicted Downtime</th>
                        <th style="border:1px solid #ccc; padding:10px;">Prediction Probability</th>
                        <th style="border:1px solid #ccc; padding:10px;">Alert Type</th>
                        <th style="border:1px solid #ccc; padding:10px;">Recommended Action</th>
                        <th style="border:1px solid #ccc; padding:10px;">Predicted At</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            <hr style="margin-top:20px; margin-bottom:5px; border:none; border-top:1px solid #ccc;">
            <p style="font-size:12px; color:#555;">We request that you investigate the cause of this issue and take necessary steps to resolve it promptly to avoid future disruptions. </p>
            <br>
            <p style="font-size:12px; color:#555;">Best regards,</p>
            <br>
            <p style="font-size:12px; color:#555;"> PLMS Monitoring Team</p>
        </div>
        """
        payload = {"to": recipient, "subject": subject, "body": html_body, "isHtml": True}
        try:
            response = requests.post(N8N_EMAIL_WEBHOOK, json=payload)
            print(f"Email sent to {recipient}: {response.status_code}")
        except Exception as e:
            print(f"Error sending email to {recipient}: {e}")

# --- Prediction Pipeline ---
def run_predictions():
    df_raw = fetch_latest_data(limit=100)
    if df_raw.empty:
        return []

    alerts = []
    IST = timezone(timedelta(hours=5, minutes=30))
    try:
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

    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")

    update_last_processed_timestamp(df_raw)
    if alerts:
        send_alerts_to_n8n(alerts)

    return alerts

# --- Flask App + Scheduler ---
app = Flask(__name__)
scheduler = BackgroundScheduler()
scheduler.add_job(func=run_predictions, trigger="interval", minutes=10)
scheduler.start()

@app.route("/predict", methods=["GET"])
def predict():
    alerts = run_predictions()
    return jsonify({"predictions": alerts, "count": len(alerts)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


# from flask import Flask, jsonify
# import pandas as pd
# import joblib
# from sqlalchemy import create_engine
# from datetime import datetime
# import os
# import json
# import requests
# from apscheduler.schedulers.background import BackgroundScheduler
# from collections import defaultdict
# from preprocess_utils import preprocess_for_stage1, preprocess_for_stage2
# import logging 
# from datetime import datetime, timedelta, timezone

# # Config
# DB_USER = os.getenv("DB_USER")
# DB_PASS = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_NAME = os.getenv("DB_NAME")
# LAST_PROCESSED_FILE = "last_processed.txt"
# SENT_ALERTS_FILE = "sent_alerts.json"
# ALERT_EXPIRY_MINUTES = 30  # prevent resending within 30 mins
# N8N_EMAIL_WEBHOOK = os.environ.get("N8N_EMAIL_WEBHOOK")

# # For production-based  
# # from sqlalchemy.orm import sessionmaker
# # engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}", pool_pre_ping=True, pool_recycle=3600)
# # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Load Models & Thresholds
# stage1_model = joblib.load("stage1_xgb_model.pkl")
# with open("model_features.txt") as f:
#     stage1_features = [line.strip() for line in f]

# stage2a_model = joblib.load("xgb_stage2a_504_vs_rest.pkl")
# stage2b_model = joblib.load("xgb_stage2b_500_503_noresp.pkl")

# with open("stage1_best_threshold.json") as f:
#     stage1_best_threshold = json.load(f).get("best_threshold", 0.5)

# stage2a_threshold = 0.639
# stage2b_threshold = 0.6

# error_type_mapping = {0: "No Response", 1: "500", 2: "503"}
# stage_code_map = {
#     "AUTO_PDF": "Auto-Invoice", "CCI": "Cancel Check-In", "CGI": "Cancel Gate-In",
#     "CI": "Check-In", "CPI": "Cancel Packing-In", "CPO": "Cancel Packing-Out",
#     "GI": "Gate-In", "GO": "Gate-Out", "GW": "Gross Weight", "PI": "Packing-In",
#     "PO": "Packing-Out", "TW": "Tare Weight", "WI": "Weigh-In", "WO": "Weigh-Out",
#     "YARD-IN": "Yard-In"
# }

# error_description_mapping = {
#     "500": "Internal Server Error",
#     "503": "Service Unavailable",
#     "504": "Gateway Timeout / Error",
#     "No Response": "System did not return a valid response; potential network or service issue"
# }

# # --- JSON Alert Tracking Utils ---
# def load_sent_alerts():
#     if os.path.exists(SENT_ALERTS_FILE):
#         with open(SENT_ALERTS_FILE, "r") as f:
#             try:
#                 return json.load(f)
#             except json.JSONDecodeError:
#                 return {}
#     return {}

# def save_sent_alerts(sent_alerts):
#     with open(SENT_ALERTS_FILE, "w") as f:
#         json.dump(sent_alerts, f)

# def should_send_alert(plant, stage, error):
#     sent_alerts = load_sent_alerts()
#     key = f"{plant}_{stage}_{error}"
#     last_sent_str = sent_alerts.get(key)
#     if last_sent_str:
#         last_sent = datetime.strptime(last_sent_str, "%Y-%m-%d %H:%M:%S")
#         if (datetime.utcnow() - last_sent).total_seconds() < ALERT_EXPIRY_MINUTES * 60:
#             return False
#     return True

# def mark_alert_sent(plant, stage, error):
#     sent_alerts = load_sent_alerts()
#     key = f"{plant}_{stage}_{error}"
#     sent_alerts[key] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
#     save_sent_alerts(sent_alerts)

# # DB Utils
# def fetch_latest_data(limit=100):
#     try:
#         last_ts = "1970-01-01 00:00:00"
#         if os.path.exists(LAST_PROCESSED_FILE):
#             with open(LAST_PROCESSED_FILE, "r") as f:
#                 last_ts = f.read().strip()
#         query = f"""
#             SELECT *
#             FROM interfaceplms.outbound_stagedetails
#             WHERE created_date > '{last_ts}'
#             ORDER BY created_date ASC
#             LIMIT {limit};
#         """

#         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
#         return pd.read_sql(query, engine)
#     except Exception as e:
#         logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
#         logging.error("DB error", exc_info=True)
#         # print(f"[DB ERROR] {e}")
#         return pd.DataFrame()

# # def fetch_plant_email_map():
# #     try:
# #         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
# #         query = """
# #             SELECT plant_code, email_to 
# #             FROM masterplms.automailer_mail
# #             WHERE email_type = 'mail_alert_test';
# #         """
# #         df = pd.read_sql(query, engine)
# #         plant_email_map = df.set_index('plant_code')['email_to'].to_dict()
# #         plant_email_map["_default_"] = "support@amzbizsol.in"
# #         return plant_email_map
# #     except Exception as e:
# #         print(f"[DB ERROR] Could not fetch plant emails: {e}")
# #         return {"_default_": "support@amzbizsol.in"}
    
# def fetch_plant_email_map():
#     try:
#         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        
#         # Fetch client emails
#         client_df = pd.read_sql("""
#             SELECT plant_code, email_to 
#             FROM masterplms.automailer_mail
#             WHERE email_type = 'mail_alert_test';
#         """, engine)

#         # Fetch plant engineer emails
#         engineer_df = pd.read_sql("""
#             SELECT plant_code, email_to 
#             FROM masterplms.automailer_mail
#             WHERE email_type = 'plant_engineer';
#         """, engine)

#         # Build dicts
#         client_map = client_df.set_index('plant_code')['email_to'].to_dict()
#         engineer_map = engineer_df.set_index('plant_code')['email_to'].to_dict()

#         return {
#             "client": client_map,
#             "engineer": engineer_map,
#             "_default_client": "support@amzbizsol.in",
#             "_default_engineer": "plant.support@amzbizsol.in"
#         }
#     except Exception as e:
#         print(f"[DB ERROR] Could not fetch plant emails: {e}")
#         return {
#             "client": {"_default_": "support@amzbizsol.in"},
#             "engineer": {"_default_": "plant.support@amzbizsol.in"}
#         }

# def update_last_processed_timestamp(df):
#     if not df.empty:
#         max_ts = df['created_date'].max()
#         with open(LAST_PROCESSED_FILE, "w") as f:
#             f.write(str(max_ts))

# # Recommendations
# ERROR_RECOMMENDATIONS = {
#     "No Response": "Investigate {stage} process for connectivity or server issues immediately.",
#     "500": "Check SAP system logs for 500 error in {stage}.",
#     "503": "Check SAP service availability for {stage}.",
#     "504": "Check network connectivity or server performance for {stage}."
# }

# STAGE_RECOMMENDATIONS = {
#     "Auto-Invoice": {"500": "Check SAP logs for Auto-Invoice module for any internal server errors.",
#                      "No Response": "Investigate Auto-Invoice process connectivity or service issues immediately."},
#     "Cancel Check-In": {"504": "Verify network and gateway response for Cancel Check-In process.",
#                         "No Response": "Check Cancel Check-In process for unresponsive service issues."},
#     "Cancel Gate-In": {"503": "Ensure Cancel Gate-In service is available and running properly.",
#                        "No Response": "Investigate Cancel Gate-In process connectivity or server issues."},
#     "Check-In": {"500": "Investigate Check-In service logs for server errors.",
#                  "No Response": "Check Check-In process for connectivity or server issues immediately."},
#     "Cancel Packing-In": {"504": "Check network connectivity and server response for Cancel Packing-In process."},
#     "Cancel Packing-Out": {"503": "Verify Cancel Packing-Out service availability."},
#     "Gate-In": {"500": "Check Gate-In module logs for server errors.",
#                 "No Response": "Investigate Gate-In process connectivity or service failures."},
#     "Gate-Out": {"504": "Check Gateway timeout issues for Gate-Out process."},
#     "Gross Weight": {"503": "Ensure Gross Weight service is available."},
#     "Packing-In": {"500": "Investigate Packing-In process logs for server errors."},
#     "Packing-Out": {"504": "Check Packing-Out process for timeout or network issues."},
#     "Tare Weight": {"503": "Ensure Tare Weight service availability."},
#     "Weigh-In": {"500": "Investigate Weigh-In service logs for internal errors."},
#     "Weigh-Out": {"504": "Check Weigh-Out process for gateway or network issues."},
#     "Yard-In": {"No Response": "Investigate Yard-In process connectivity or service issues immediately."}
# }

# # --- Email Sending Function ---
# def send_alerts_to_n8n(alerts):
#     email_maps = fetch_plant_email_map()
#     grouped_by_plant = defaultdict(list)
#     for alert in alerts:
#         grouped_by_plant[alert['plantCode']].append(alert)

#     for plant, plant_alerts in grouped_by_plant.items():
#         if alert['probError'] is not None and alert['probError'] >= 0.8:
#                 # Send to client
#             recipient = email_maps["client"].get(plant, email_maps["_default_client"])
#         else:
#                 # Send to plant engineer
#             recipient = email_maps["engineer"].get(plant, email_maps["_default_engineer"])

#         # Filter out alerts already sent
#         plant_alerts_to_send = []
#         for alert in plant_alerts:
#             if should_send_alert(alert['plantCode'], alert['stageCode'], alert['errorType']):
#                 plant_alerts_to_send.append(alert)
#                 mark_alert_sent(alert['plantCode'], alert['stageCode'], alert['errorType'])

#         if not plant_alerts_to_send:
#             continue

#         # Prepare email HTML
#         table_rows = ""
#         row_colors = ["#ffffff", "#f7f7f7"]
#         for idx, alert in enumerate(plant_alerts_to_send):
#             stage_display = stage_code_map.get(alert['stageCode'], alert['stageCode'])
#             error_code = alert['errorType']
#             error_display = error_description_mapping.get(alert['errorType'], alert['errorType'])
#             prob_display = f"{alert['probError']*100:.1f}%" if alert.get('probError') else "N/A"
#             alert_type = "Critical" if alert['timeToError'] == "Immediate attention required" else "Warning"
#             recommendation = STAGE_RECOMMENDATIONS.get(stage_display, {}).get(
#                 error_code,
#                 ERROR_RECOMMENDATIONS.get(error_code, "No specific action defined.")
#             )
#             recommendation_text = recommendation.format(stage=stage_display)
#             critical_style = "color:#d9534f; font-weight:bold;" if alert_type == "Critical" else ""
#             critical_icon = "⚠" if alert_type == "Critical" else ""

#             table_rows += f"""
#                 <tr style="background-color:{row_colors[idx % 2]};">
#                     <td style="border:1px solid #ddd; padding:8px;">{stage_display}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{error_code}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{error_display}</td>
#                     <td style="border:1px solid #ddd; padding:8px; {critical_style}">{critical_icon}{alert['timeToError']}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{prob_display}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{alert_type}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{recommendation_text}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{alert['alertTimestamp']}</td>
#                 </tr>
#             """

#         critical_count = sum(1 for a in plant_alerts_to_send if a['timeToError'] == "Immediate attention required")
#         summary_text = f"Total Alerts: {len(plant_alerts_to_send)} | Critical Alerts: {critical_count}"

#         subject = f"SAP Downtime Alert - Plant {plant}"
#         html_body = f"""
#         <div style="font-family: Arial, sans-serif; font-size:14px; color:#333; line-height:1.5;">
#             <h2 style="color:#2E86C1; margin-bottom:0;">SAP Monitoring System Alert</h2>
#             <p style="margin-top:5px;">Dear Team,</p>
#             <p>This is an automated alert for <strong>Plant {plant}</strong> regarding predicted SAP errors. Please review and take action immediately to avoid downtime.</p>
#             <p><strong>Summary:</strong> {summary_text}</p>
#             <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; width:100%; text-align:left; border:1px solid #ccc;">
#                 <thead style="background-color:#f0f0f0; font-weight:bold; color:#333;">
#                     <tr>
#                         <th style="border:1px solid #ccc; padding:10px;">Stage</th>
#                         <th style="border:1px solid #ccc; padding:10px;">HTTP Code</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Error Description</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Predicted Downtime</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Prediction Probability</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Alert Type</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Recommended Action</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Predicted At</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#                     {table_rows}
#                 </tbody>
#             </table>
#             <hr style="margin-top:20px; margin-bottom:5px; border:none; border-top:1px solid #ccc;">
#             <p style="font-size:12px; color:#555;">SAP Monitoring System | Version 1.0 | Contact: support@amzbizsol.in</p>
#         </div>
#         """
#         payload = {"to": recipient, "subject": subject, "body": html_body, "isHtml": True}
#         try:
#             response = requests.post(N8N_EMAIL_WEBHOOK, json=payload)
#             print(f"Email sent to {recipient}: {response.status_code}")
#         except Exception as e:
#             print(f"Error sending email to {recipient}: {e}")

# # --- Prediction Pipeline ---
# def run_predictions():
#     df_raw = fetch_latest_data(limit=100)
#     if df_raw.empty:
#         return []

#     alerts = []
#     IST = timezone(timedelta(hours=5, minutes=30))
#     try:
#         X_stage1 = preprocess_for_stage1(df_raw)
#         probs = stage1_model.predict_proba(X_stage1)[:, 1] if hasattr(stage1_model, "predict_proba") else None
#         preds = stage1_model.predict(X_stage1)

#         for idx, pred in enumerate(preds):
#             prob_error = probs[idx] if probs is not None else None
#             if prob_error is not None and prob_error < stage1_best_threshold:
#                 continue

#             if pred == 1 or (prob_error is not None and prob_error >= stage1_best_threshold):
#                 record = df_raw.iloc[idx]
#                 stage_code = record.get("stage_code", "")
#                 stage_name = stage_code_map.get(stage_code, stage_code)

#                 if stage_code == "504":
#                     X_2a = preprocess_for_stage2(pd.DataFrame([record]), stage="2a")
#                     err_type_pred = stage2a_model.predict(X_2a)[0]
#                     prob_stage2 = stage2a_model.predict_proba(X_2a).max() if hasattr(stage2a_model, "predict_proba") else None
#                     err_type_name = str(err_type_pred)
#                     if prob_stage2 is not None and prob_stage2 < stage2a_threshold:
#                         continue
#                 else:
#                     X_2b = preprocess_for_stage2(pd.DataFrame([record]), stage="2b")
#                     err_type_pred = stage2b_model.predict(X_2b)[0]
#                     prob_stage2 = stage2b_model.predict_proba(X_2b).max() if hasattr(stage2b_model, "predict_proba") else None
#                     err_type_name = error_type_mapping.get(int(err_type_pred), str(err_type_pred))
#                     if prob_stage2 is not None and prob_stage2 < stage2b_threshold:
#                         continue

#                 alert = {
#                     "plantCode": record.get("plant_code", ""),
#                     "stageCode": stage_name,
#                     "errorType": err_type_name,
#                     "errorDescription": error_description_mapping.get(err_type_name, "Unknown Error"),
#                     "probError": round(float(prob_error), 4) if prob_error is not None else None,
#                     "timeToError": "Immediate attention required",
#                     "alertTimestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
#                 }
#                 alerts.append(alert)

#     except Exception as e:
#         print(f"[PREDICTION ERROR] {e}")

#     update_last_processed_timestamp(df_raw)
#     if alerts:
#         send_alerts_to_n8n(alerts)

#     return alerts

# # --- Flask App + Scheduler ---
# app = Flask(__name__)
# scheduler = BackgroundScheduler()
# scheduler.add_job(func=run_predictions, trigger="interval", minutes=10)
# scheduler.start()

# @app.route("/predict", methods=["GET"])
# def predict():
#     alerts = run_predictions()
#     return jsonify({"predictions": alerts, "count": len(alerts)})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)

# from flask import Flask, jsonify
# import pandas as pd
# import joblib
# from sqlalchemy import create_engine
# from datetime import datetime
# import os
# import json
# import requests
# from apscheduler.schedulers.background import BackgroundScheduler
# from collections import defaultdict
# from preprocess_utils import preprocess_for_stage1, preprocess_for_stage2


# # Config
# DB_USER = os.getenv("DB_USER", "root")
# DB_PASS = os.getenv("DB_PASSWORD", "password")
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_NAME = os.getenv("DB_NAME", "sapdb")
# LAST_PROCESSED_FILE = "last_processed.txt"
# DOWNTIME_LOG_FILE = "downtime_records.csv"
# N8N_EMAIL_WEBHOOK = os.environ.get("N8N_EMAIL_WEBHOOK",
#                                    "https://sakshi395.app.n8n.cloud/webhook/sap-alert-email")

# PLANT_EMAIL_MAP = {
#     "N202": "sakshi.tandon@amzbizsol.in",
#     "N205": "sakshi.tandon@amzbizsol.in",
#     "N212": "sakshi.tandon@amzbizsol.in",
#     "N225": "sakshi.tandon@amzbizsol.in",
#     "N239": "sakshi.tandon@amzbizsol.in",
#     "N622": "sakshi.tandon@amzbizsol.in",
#     "NE03": "sakshi.tandon@amzbizsol.in",
#     "NE25": "sakshi.tandon@amzbizsol.in",
#     "NE29": "sakshi.tandon@amzbizsol.in",
#     "NE30": "sakshi.tandon@amzbizsol.in",
#     "NT45": "sakshi.tandon@amzbizsol.in",
#     "NT60": "sakshi.tandon@amzbizsol.in",
#     "_default_": "sakshi.tandon@amzbizsol.in"
# }

# # Load Models & Thresholds
# stage1_model = joblib.load("stage1_xgb_model.pkl")
# with open("model_features.txt") as f:
#     stage1_features = [line.strip() for line in f]

# stage2a_model = joblib.load("xgb_stage2a_504_vs_rest.pkl")
# stage2b_model = joblib.load("xgb_stage2b_500_503_noresp.pkl")

# with open("stage1_best_threshold.json") as f:
#     stage1_best_threshold = json.load(f).get("best_threshold", 0.5)

# stage2a_threshold = 0.639
# stage2b_threshold = 0.6

# error_type_mapping = {0: "No Response", 1: "500", 2: "503"}
# stage_code_map = {
#     "AUTO_PDF": "Auto-Invoice", "CCI": "Cancel Check-In", "CGI": "Cancel Gate-In",
#     "CI": "Check-In", "CPI": "Cancel Packing-In", "CPO": "Cancel Packing-Out",
#     "GI": "Gate-In", "GO": "Gate-Out", "GW": "Gross Weight", "PI": "Packing-In",
#     "PO": "Packing-Out", "TW": "Tare Weight", "WI": "Weigh-In", "WO": "Weigh-Out",
#     "YARD-IN": "Yard-In"
# }
# error_description_mapping = {
#     "500": "Internal Server Error",
#     "503": "Service Unavailable",
#     "504": "Gateway Timeout / Error",
#     "No Response": "System did not return a valid response; potential network or service issue"
# }

# # DB Utils
# def fetch_latest_data(limit=100):
#     try:
#         last_ts = "1970-01-01 00:00:00"
#         if os.path.exists(LAST_PROCESSED_FILE):
#             with open(LAST_PROCESSED_FILE, "r") as f:
#                 last_ts = f.read().strip()
#         query = f"""
#             SELECT *
#             FROM interfaceplms.outbound_stagedetails
#             WHERE created_date > '{last_ts}'
#             ORDER BY created_date ASC
#             LIMIT {limit};
#         """
#         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
#         return pd.read_sql(query, engine)
#     except Exception as e:
#         print(f"[DB ERROR] {e}")
#         return pd.DataFrame()


# def update_last_processed_timestamp(df):
#     if not df.empty:
#         max_ts = df['created_date'].max()
#         with open(LAST_PROCESSED_FILE, "w") as f:
#             f.write(str(max_ts))


# def log_downtime_data(record, predicted_type):
#     data = {
#         "created_date": record.get("created_date"),
#         "sending_date": record.get("sending_date"),
#         "stage_code": record.get("stage_code"),
#         "plant_code": record.get("plant_code"),
#         "error_type": predicted_type
#     }
#     df_new = pd.DataFrame([data])
#     if os.path.exists(DOWNTIME_LOG_FILE):
#         df_existing = pd.read_csv(DOWNTIME_LOG_FILE)
#         df_combined = pd.concat([df_existing, df_new], ignore_index=True)
#         df_combined.to_csv(DOWNTIME_LOG_FILE, index=False)
#     else:
#         df_new.to_csv(DOWNTIME_LOG_FILE, index=False)


# # Email Alert Logic

# # Define default recommendations for error types
# ERROR_RECOMMENDATIONS = {
#     "No Response": "Investigate {stage} process for connectivity or server issues immediately.",
#     "500": "Check SAP system logs for 500 error in {stage}.",
#     "503": "Check SAP service availability for {stage}.",
#     "504": "Check network connectivity or server performance for {stage}."
# }

# # Optional stage-specific overrides
# STAGE_RECOMMENDATIONS = {
#     "Auto-Invoice": {
#         "500": "Check SAP logs for Auto-Invoice module for any internal server errors.",
#         "No Response": "Investigate Auto-Invoice process connectivity or service issues immediately."
#     },
#     "Cancel Check-In": {
#         "504": "Verify network and gateway response for Cancel Check-In process.",
#         "No Response": "Check Cancel Check-In process for unresponsive service issues."
#     },
#     "Cancel Gate-In": {
#         "503": "Ensure Cancel Gate-In service is available and running properly.",
#         "No Response": "Investigate Cancel Gate-In process connectivity or server issues."
#     },
#     "Check-In": {
#         "500": "Investigate Check-In service logs for server errors.",
#         "No Response": "Check Check-In process for connectivity or server issues immediately."
#     },
#     "Cancel Packing-In": {
#         "504": "Check network connectivity and server response for Cancel Packing-In process."
#     },
#     "Cancel Packing-Out": {
#         "503": "Verify Cancel Packing-Out service availability."
#     },
#     "Gate-In": {
#         "500": "Check Gate-In module logs for server errors.",
#         "No Response": "Investigate Gate-In process connectivity or service failures."
#     },
#     "Gate-Out": {
#         "504": "Check Gateway timeout issues for Gate-Out process."
#     },
#     "Gross Weight": {
#         "503": "Ensure Gross Weight service is available."
#     },
#     "Packing-In": {
#         "500": "Investigate Packing-In process logs for server errors."
#     },
#     "Packing-Out": {
#         "504": "Check Packing-Out process for timeout or network issues."
#     },
#     "Tare Weight": {
#         "503": "Ensure Tare Weight service availability."
#     },
#     "Weigh-In": {
#         "500": "Investigate Weigh-In service logs for internal errors."
#     },
#     "Weigh-Out": {
#         "504": "Check Weigh-Out process for gateway or network issues."
#     },
#     "Yard-In": {
#         "No Response": "Investigate Yard-In process connectivity or service issues immediately."
#     }
# }


# def send_alerts_to_n8n(alerts):
#     grouped_by_plant = defaultdict(list)
#     for alert in alerts:
#         grouped_by_plant[alert['plantCode']].append(alert)

#     for plant, plant_alerts in grouped_by_plant.items():
#         recipient = PLANT_EMAIL_MAP.get(plant, PLANT_EMAIL_MAP["_default_"])

#         # Count critical alerts
#         critical_count = sum(1 for a in plant_alerts if a['timeToError'] == "Immediate attention required")
        
#         # Build table rows with alternating colors and critical highlighting
#         table_rows = ""
#         row_colors = ["#ffffff", "#f7f7f7"]
#         for idx, alert in enumerate(plant_alerts):
#             stage_display = stage_code_map.get(alert['stageCode'], alert['stageCode'])
#             error_code = alert['errorType']
#             error_display = error_description_mapping.get(alert['errorType'], alert['errorType'])
#             prob_display = f"{alert['probError']*100:.1f}%" if alert.get('probError') else "N/A"
#             alert_type = "Critical" if alert['timeToError'] == "Immediate attention required" else "Warning"

#             # Determine recommendation
#             recommendation = STAGE_RECOMMENDATIONS.get(stage_display, {}).get(
#                 error_code,
#                 ERROR_RECOMMENDATIONS.get(error_code, "No specific action defined.")
#             )
#             recommendation_text = recommendation.format(stage=stage_display)

#             # Highlight critical alerts
#             critical_style = "color:#d9534f; font-weight:bold;" if alert_type == "Critical" else ""
#             critical_icon = "⚠ " if alert_type == "Critical" else ""

#             table_rows += f"""
#                 <tr style="background-color:{row_colors[idx % 2]};">
#                     <td style="border:1px solid #ddd; padding:8px;">{stage_display}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{error_code}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{error_display}</td>
#                     <td style="border:1px solid #ddd; padding:8px; {critical_style}">{critical_icon}{alert['timeToError']}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{prob_display}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{alert_type}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{recommendation_text}</td>
#                     <td style="border:1px solid #ddd; padding:8px;">{alert['alertTimestamp']}</td>
#                 </tr>
#             """

#         # Build summary
#         summary_text = f"Total Alerts: {len(plant_alerts)} | Critical Alerts: {critical_count} | All alerts require immediate investigation to prevent downtime"

#         subject = f"SAP Downtime Alert - Plant {plant}"
#         html_body = f"""
#         <div style="font-family: Arial, sans-serif; font-size:14px; color:#333; line-height:1.5;">
#             <h2 style="color:#d9534f; margin-bottom:0;">SAP Monitoring System Alert</h2>
#             <p style="margin-top:5px;">Dear Team,</p>
#             <p>This is an automated alert for <strong>Plant {plant}</strong> regarding predicted SAP errors. Please review and take action immediately to avoid downtime.</p>
            
#             <p><strong>Summary:</strong> {summary_text}</p>
            
#             <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; width:100%; text-align:left; border:1px solid #ccc;">
#                 <thead style="background-color:#f0f0f0; font-weight:bold; color:#333;">
#                     <tr>
#                         <th style="border:1px solid #ccc; padding:10px;">Stage</th>
#                         <th style="border:1px solid #ccc; padding:10px;">HTTP Code</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Error Description</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Predicted Downtime</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Prediction Probability</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Alert Type</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Recommended Action</th>
#                         <th style="border:1px solid #ccc; padding:10px;">Predicted At</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#                     {table_rows}
#                 </tbody>
#             </table>
            
#             <hr style="margin-top:20px; margin-bottom:5px; border:none; border-top:1px solid #ccc;">
#             <p style="font-size:12px; color:#555;">SAP Monitoring System | Version 1.0 | Contact: support@example.com</p>
#         </div>
#         """

#         payload = {"to": recipient, "subject": subject, "body": html_body, "isHtml": True}
#         try:
#             response = requests.post(N8N_EMAIL_WEBHOOK, json=payload)
#             print(f"Email sent to {recipient}: {response.status_code}")
#         except Exception as e:
#             print(f"Error sending email to {recipient}: {e}")

# # Prediction Pipeline
# def run_predictions():
#     df_raw = fetch_latest_data(limit=100)
#     if df_raw.empty:
#         return []

#     alerts = []
#     try:
#         X_stage1 = preprocess_for_stage1(df_raw)
#         probs = stage1_model.predict_proba(X_stage1)[:, 1] if hasattr(stage1_model, "predict_proba") else None
#         preds = stage1_model.predict(X_stage1)

#         for idx, pred in enumerate(preds):
#             prob_error = probs[idx] if probs is not None else None
#             if prob_error is not None and prob_error < stage1_best_threshold:
#                 continue

#             if pred == 1 or (prob_error is not None and prob_error >= stage1_best_threshold):
#                 record = df_raw.iloc[idx]
#                 stage_code = record.get("stage_code", "")
#                 stage_name = stage_code_map.get(stage_code, stage_code)

#                 if stage_code == "504":
#                     X_2a = preprocess_for_stage2(pd.DataFrame([record]), stage="2a")
#                     err_type_pred = stage2a_model.predict(X_2a)[0]
#                     prob_stage2 = stage2a_model.predict_proba(X_2a).max() if hasattr(stage2a_model, "predict_proba") else None
#                     err_type_name = str(err_type_pred)
#                     if prob_stage2 is not None and prob_stage2 < stage2a_threshold:
#                         continue
#                 else:
#                     X_2b = preprocess_for_stage2(pd.DataFrame([record]), stage="2b")
#                     err_type_pred = stage2b_model.predict(X_2b)[0]
#                     prob_stage2 = stage2b_model.predict_proba(X_2b).max() if hasattr(stage2b_model, "predict_proba") else None
#                     err_type_name = error_type_mapping.get(int(err_type_pred), str(err_type_pred))
#                     if prob_stage2 is not None and prob_stage2 < stage2b_threshold:
#                         continue

#                 alert = {
#                     "plantCode": record.get("plant_code", ""),
#                     "stageCode": stage_name,
#                     "errorType": err_type_name,
#                     "errorDescription": error_description_mapping.get(err_type_name, "Unknown Error"),
#                     "probError": round(float(prob_error), 4) if prob_error is not None else None,
#                     "timeToError": "Immediate attention required",
#                     "alertTimestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
#                 }
#                 alerts.append(alert)
#                 log_downtime_data(record, err_type_name)
#     except Exception as e:
#         print(f"[PREDICTION ERROR] {e}")

#     # Deduplicate alerts
#     unique_alerts = {}
#     for alert in alerts:
#         key = (alert["plantCode"], alert["stageCode"], alert["errorType"])
#         if key not in unique_alerts:
#             unique_alerts[key] = alert

#     update_last_processed_timestamp(df_raw)

#     alerts_to_send = list(unique_alerts.values())
#     if alerts_to_send:
#         send_alerts_to_n8n(alerts_to_send)

#     return alerts_to_send

# # Flask App + Scheduler
# app = Flask(__name__)
# scheduler = BackgroundScheduler()
# scheduler.add_job(func=run_predictions, trigger="interval", minutes=5)
# scheduler.start()

# @app.route("/predict", methods=["GET"])
# def predict():
#     alerts = run_predictions()
#     return jsonify({"predictions": alerts, "count": len(alerts)})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)

