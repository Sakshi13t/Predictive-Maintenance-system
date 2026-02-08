# ===== NEW MAIL TEMPLATE AS PER REQUIREMENT ===== ##
from flask import Flask, jsonify
import pandas as pd
import requests
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import os
import json  
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from model_utils import preprocess_and_predict
from config import DB_CONFIG, DATABASE_URL, N8N_DEVICE_WEBHOOK, get_plant_email_map
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

LAST_SEEN_FILE = "last_seen_device_time.txt"
ALERT_LOG_FILE = "alerted_devices.json"  
ALERT_COOLDOWN_MINUTES = 30 

# Logging Setup
LOG_FILE = "downtime_prediction.log"
logger = logging.getLogger("DowntimePredictor")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler with rotation (5MB per file, keep 5 backups)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Utilities for alert logging
def load_alert_log():
    if os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_alert_log(log_data): 
    with open(ALERT_LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)

def get_last_seen_time():
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, 'r') as f:
            return f.read().strip()
    else:
        return (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

def save_last_seen_time(ts):
    with open(LAST_SEEN_FILE, 'w') as f:
        f.write(str(ts))

# Location Prefix Mapping Logic
PREFIX_MAPPING = {
    "WB": "Weighbridge",
    "GATE-IN": "Main Gate - Entry",
    "GATE-OUT": "Main Gate - Exit",
    "PACKING-IN": "Packing Section - Inward",
    "PACKING-OUT": "Packing Section - Outward",
    "Packing OUT": "Packing Section - Outward",  
    "YARD-IN": "Yard Entry",
    "YARD-OUT": "Yard Exit",
    "YARD-HIGHWAY": "Yard - Highway Exit",
    "YARD-IN-BLK": "Yard Entry - Block",
    "INNER-YARD-IN": "Inner Yard Entry",
    "RM Unloading Point IN": "Raw Material Unloading - Entry",
    "RM Unloading Point OUT": "Raw Material Unloading - Exit",
    "Invoice": "Invoice Section",
    "Invoice Kiosk": "Invoice Kiosk",
    "KIOSK GATE": "Gate Kiosk",
    "TRIP-CLOSSER": "Trip Closer"
}

def resolve_location_name(code: str) -> str:
    """Map location code to readable name dynamically."""
    if not code:
        return code

    if code in PREFIX_MAPPING:
        return PREFIX_MAPPING[code]

    for prefix, label in PREFIX_MAPPING.items():
        if code.startswith(prefix):
            suffix = code.replace(prefix, "").strip("-_ ")
            return f"{label} {suffix}" if suffix else label

    return code

def fetch_device_predictions(last_seen_time=None):
    session = Session()
    try:
        if last_seen_time:
            query = text("""
                SELECT * FROM powerbiplms.vw_hardware_downtime
                WHERE time_down > :last_seen_time
                ORDER BY time_down ASC
                LIMIT 100
            """)
            result = session.execute(query, {"last_seen_time": last_seen_time})
        else:
            query = text("""
                SELECT * FROM powerbiplms.vw_hardware_downtime
                ORDER BY time_down ASC
                LIMIT 100
            """)
            result = session.execute(query)

        rows = result.fetchall()
        columns = result.keys()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        session.close()

def build_device_email(plant, alerts, now):
    """Builds styled HTML email for device downtime alerts (email-safe with inline CSS)."""
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; font-size: 14px;">
        <h2 style="color:#2E86C1;">Device Downtime Prediction Alerts</h2>
        <p>The following potential downtimes have been detected for <b>Plant: {plant}</b>:</p>
        
        <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
            <tr>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Stage Code</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Stage Name</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Device IP</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Device Name</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Predicted Downtime (min)</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Failure Probability</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:center; font-weight:bold; color:white; background-color:#007bff;">Predicted At</th>
            </tr>
    """

    for idx, device in enumerate(alerts):
        row_bg = "#f9f9f9da" if idx % 2 == 1 else "#ffffff"  
        html_body += f"""
            <tr style="background-color:{row_bg};">
                <td style="border:1px solid #ddd; padding:8px;">{device.get('location_code', '')}</td>
                <td style="border:1px solid #ddd; padding:8px;">{device.get('location_name', '')}</td>
                <td style="border:1px solid #ddd; padding:8px;">{device.get('device_ip', '')}</td>
                <td style="border:1px solid #ddd; padding:8px;">{device.get('device_name', '')}</td>
                <td style="border:1px solid #ddd; padding:8px;">{round(device.get('predicted_downtime_minutes', 0), 2)}</td>
                <td style="border:1px solid #ddd; padding:8px;">{round(device.get('is_failure_prob', 0) * 100, 1)}%</td>
                <td style="border:1px solid #ddd; padding:8px;">{now.strftime("%Y-%m-%d %H:%M:%S")}</td>
            </tr>
        """

    html_body += """
        </table>
        <p>This is an automated alert.</p>
        <p>We request that you investigate the cause of this issue and take necessary steps to resolve it promptly to avoid future disruptions.</p>
        <p style="font-weight:600;">Best regards,</p>
        <p style="font-weight:600;">PLMS Monitoring Team</p>
    </body>
    </html>
    """
    return html_body

def group_and_send_device_alerts(alerts, recipient_type="client"):
    # fetch mappings
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

    client_map = dict(zip(client_df['plant_code'], client_df['email_to']))
    engineer_map = dict(zip(engineer_df['plant_code'], engineer_df['email_to']))

    grouped = defaultdict(list)
    alert_log = load_alert_log()  
    now = datetime.now()  

    for alert in alerts:
        device_ip = alert.get('device_ip')
        last_alert_time_str = alert_log.get(device_ip)

        if last_alert_time_str:
            try:
                last_alert_time = datetime.fromisoformat(last_alert_time_str)
                if now - last_alert_time < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
                    logger.warning(f"Skipping alert for {device_ip} (recently alerted)")
                    continue
            except Exception as e:
                logger.error(f"Error parsing timestamp for {device_ip}: {e}")

        plant_code = alert.get('plant_code') or alert.get('plant_name') or 'UNKNOWN'
        grouped[plant_code].append(alert)

    for plant, device_alerts in grouped.items():
        if recipient_type == "client":
            recipient = client_map.get(plant)
        else:
            recipient = engineer_map.get(plant)

        if not recipient or not device_alerts:
            continue

        # Build styled email
        html_body = build_device_email(plant, device_alerts, now)

        # Update log
        for device in device_alerts:
            alert_log[device.get('device_ip')] = now.isoformat()

        payload = {
            "to": recipient,
            "subject": f"Device Downtime Alert - Plant {plant}",
            "body": html_body,
            "isHtml": True
        }

        try:
            response = requests.post(N8N_DEVICE_WEBHOOK, json=payload)
            logger.info(f"Email sent to {recipient}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending email to {recipient}: {e}", exc_info=True)

    save_alert_log(alert_log)

# def group_and_send_device_alerts(alerts):
#     plant_email_map = get_plant_email_map()
#     grouped = defaultdict(list)

#     alert_log = load_alert_log()  
#     now = datetime.now()  

#     for alert in alerts:
#         device_ip = alert.get('device_ip')
#         last_alert_time_str = alert_log.get(device_ip)

#         if last_alert_time_str:
#             try:
#                 last_alert_time = datetime.fromisoformat(last_alert_time_str)
#                 if now - last_alert_time < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
#                     logger.warning(f"Skipping alert for {device_ip} (recently alerted)")
#                     continue
#             except Exception as e:
#                 logger.error(f"Error parsing timestamp for {device_ip}: {e}")

#         plant_code = alert.get('plant_code') or alert.get('plant_name') or 'UNKNOWN'
#         grouped[plant_code].append(alert)

#     for plant, device_alerts in grouped.items():
#         recipient = plant_email_map.get(plant, plant_email_map.get('_default_', 'sakshi.tandon@amzbizsol.in'))
#         if not recipient or not device_alerts:
#             continue

#         # Build styled email
#         html_body = build_device_email(plant, device_alerts, now)

#         # Update log
#         for device in device_alerts:
#             alert_log[device.get('device_ip')] = now.isoformat()

#         payload = {
#             "to": recipient,
#             "subject": f"Device Downtime Alert - Plant {plant}",
#             "body": html_body,
#             "isHtml": True
#         }

#         try:
#             response = requests.post(N8N_DEVICE_WEBHOOK, json=payload)
#             logger.info(f"Email sent to {recipient}: {response.status_code}")
#         except Exception as e:
#             logger.error(f"Error sending email to {recipient}: {e}", exc_info=True)

#     save_alert_log(alert_log) 
    
def process_device_predictions():
    last_seen = get_last_seen_time()
    all_data = fetch_device_predictions(last_seen)

    if not all_data:
        logger.info("[Scheduler] No new data found")
        return

    df = pd.DataFrame(all_data)
    df_pred = preprocess_and_predict(df)

    if df_pred is None or df_pred.empty:
        logger.info("[Scheduler] No predictions available")
        return

    df_pred['is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)

    # keep both <80% and >=80%
    df_pred = df_pred[df_pred['will_fail_soon'] == 1]

    if df_pred.empty:
        logger.info("[Scheduler] No alerts to send")
        return

    df_pred.sort_values(['plant_code', 'is_failure_prob'], ascending=[True, False], inplace=True)
    df_pred = df_pred.groupby('plant_code').head(5)
    df_pred = df_pred.drop_duplicates(subset=["device_ip", "device_name"])

    alerts = df_pred.to_dict(orient="records")

    # enrich with location names
    for alert in alerts:
        raw_code = alert.get("location_name") or alert.get("location_code")
        alert["location_code"] = raw_code
        alert["location_name"] = resolve_location_name(raw_code)

    # split alerts based on probability threshold
    engineer_alerts = [a for a in alerts if a['is_failure_prob'] < 0.80]
    client_alerts   = [a for a in alerts if a['is_failure_prob'] >= 0.80]

    latest_seen = df_pred["time_down"].max()

    # Send to respective groups
    if engineer_alerts:
        group_and_send_device_alerts(engineer_alerts, recipient_type="engineer")

    if client_alerts:
        group_and_send_device_alerts(client_alerts, recipient_type="client")

    if latest_seen:
        save_last_seen_time(latest_seen)

    logger.info(f"[Scheduler] Alerts sent: {len(alerts)}, updated last_seen to {latest_seen}")

# def process_device_predictions():
#     last_seen = get_last_seen_time()
#     all_data = fetch_device_predictions(last_seen)

#     if not all_data:
#         logger.info("[Scheduler] No new data found")
#         return

#     df = pd.DataFrame(all_data)
#     df_pred = preprocess_and_predict(df)

#     if df_pred is None or df_pred.empty:
#         logger.info("[Scheduler] No predictions available")
#         return

#     df_pred['is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)

#     df_pred = df_pred[
#         (df_pred['will_fail_soon'] == 1) &
#         (df_pred['is_failure_prob'] >= 0.80)
#     ]

#     if not df_pred.empty:
#         df_pred.sort_values(['plant_code', 'is_failure_prob'], ascending=[True, False], inplace=True)
#         df_pred = df_pred.groupby('plant_code').head(5)
#         df_pred = df_pred.sort_values("is_failure_prob", ascending=False)
#         df_pred = df_pred.drop_duplicates(subset=["device_ip", "device_name"])

#     if df_pred.empty:
#         logger.info("[Scheduler] No alerts to send")
#         return

#     alerts = df_pred.to_dict(orient="records")

#     for alert in alerts:
#         raw_code = alert.get("location_name") or alert.get("location_code")
#         alert["location_code"] = raw_code   # keep the raw code explicitly
#         alert["location_name"] = resolve_location_name(raw_code)  # mapped description
        

#     latest_seen = df_pred["time_down"].max()

#     group_and_send_device_alerts(alerts)

#     if latest_seen:
#         save_last_seen_time(latest_seen)

#     logger.info(f"[Scheduler] Alerts sent: {len(alerts)}, updated last_seen to {latest_seen}")


@app.route('/predict/devices', methods=['GET'])

def predict_devices():
    process_device_predictions()
    return jsonify({"status": "triggered_by_api"})

if __name__ == '__main__':
    print("Starting Device Downtime Prediction Service...")

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=process_device_predictions, trigger="interval", minutes=10)  
    scheduler.start()

    try:
        app.run(debug=True, port=5002, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()    

## =====WORKING + MY DEFAULT MAIL TEMPLATE===== ##

# from flask import Flask, jsonify
# import pandas as pd
# import requests
# from collections import defaultdict
# from datetime import datetime, timezone, timedelta
# import os
# import json  
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
# from model_utils import preprocess_and_predict
# from config import DB_CONFIG, DATABASE_URL, N8N_DEVICE_WEBHOOK, get_plant_email_map

# app = Flask(__name__)
# engine = create_engine(DATABASE_URL)
# Session = sessionmaker(bind=engine)

# LAST_SEEN_FILE = "last_seen_device_time.txt"
# ALERT_LOG_FILE = "alerted_devices.json"  
# ALERT_COOLDOWN_MINUTES = 30 

# # Utilities for alert logging
# def load_alert_log():
#     if os.path.exists(ALERT_LOG_FILE):
#         with open(ALERT_LOG_FILE, 'r') as f:
#             return json.load(f)
#     return {}

# def save_alert_log(log_data):
#     with open(ALERT_LOG_FILE, 'w') as f:
#         json.dump(log_data, f, indent=2)

# def get_last_seen_time():
#     if os.path.exists(LAST_SEEN_FILE):
#         with open(LAST_SEEN_FILE, 'r') as f:
#             return f.read().strip()
#     else:
#         return (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

# def save_last_seen_time(ts):
#     with open(LAST_SEEN_FILE, 'w') as f:
#         f.write(str(ts))

# def fetch_device_predictions(last_seen_time=None):
#     session = Session()
#     try:
#         if last_seen_time:
#             query = text("""
#                 SELECT * FROM powerbiplms.vw_hardware_downtime
#                 WHERE time_down > :last_seen_time
#                 ORDER BY time_down ASC
#                 LIMIT 100
#             """)
#             result = session.execute(query, {"last_seen_time": last_seen_time})
#         else:
#             query = text("""
#                 SELECT * FROM powerbiplms.vw_hardware_downtime
#                 ORDER BY time_down ASC
#                 LIMIT 100
#             """)
#             result = session.execute(query)

#         rows = result.fetchall()
#         columns = result.keys()
#         return [dict(zip(columns, row)) for row in rows]
#     finally:
#         session.close()

# def group_and_send_device_alerts(alerts):
#     plant_email_map = get_plant_email_map()
#     grouped = defaultdict(list)

#     alert_log = load_alert_log()  
#     now = datetime.now()  

#     for alert in alerts:
#         device_ip = alert.get('device_ip')
#         last_alert_time_str = alert_log.get(device_ip)

#         if last_alert_time_str:
#             try:
#                 last_alert_time = datetime.fromisoformat(last_alert_time_str)
#                 if now - last_alert_time < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
#                     print(f"Skipping alert for {device_ip} (recently alerted)")
#                     continue
#             except Exception as e:
#                 print(f"Error parsing timestamp for {device_ip}: {e}")

#         plant_code = alert.get('plant_code') or alert.get('plant_name') or 'UNKNOWN'
#         grouped[plant_code].append(alert)

#     for plant, device_alerts in grouped.items():
#         recipient = plant_email_map.get(plant, plant_email_map.get('_default_', 'sakshi.tandon@amzbizsol.in'))
#         if not recipient or not device_alerts:
#             continue

#         rows = ""
#         for device in device_alerts:
#             device_ip = device.get('device_ip')
#             rows += f"""
#                 <tr>
#                     <td>{device.get('location_name', '')}</td>
#                     <td>{device_ip}</td>
#                     <td>{device.get('device_name', '')}</td>
#                     <td>{round(device.get('predicted_downtime_minutes', 0), 2)}</td>
#                     <td>{round(device.get('is_failure_prob', 0) * 100, 1)}%</td>
#                     <td>{now.strftime("%Y-%m-%d %H:%M:%S")}</td>
#                 </tr>
#             """
#             alert_log[device_ip] = now.isoformat()  

#         html_body = f"""
#         <div style='font-family:Arial, sans-serif; font-size:14px; color:#333;'>
#             <p>Dear Team,</p>
#             <p>This is an automated alert from your <strong>Device Downtime Monitoring System</strong>.</p>
#             <p><strong>Plant:</strong> {plant}</p>
#             <p><strong>Predicted Device Downtimes:</strong></p>
#             <table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; text-align: left;'>
#                 <thead style='background-color: #f2f2f2;'>
#                     <tr>
#                         <th>Location</th>
#                         <th>Device IP</th>
#                         <th>Device Name</th>
#                         <th>Predicted Downtime (min)</th>
#                         <th>Failure Probability</th>
#                         <th>Predicted At</th>
#                     </tr>
#                 </thead>
#                 <tbody>{rows}</tbody>
#             </table>
#             <p>Please take necessary action.</p>
#             <p>Regards,<br>Device Monitoring System</p>
#         </div>
#         """

#         payload = {
#             "to": recipient,
#             "subject": f"Device Downtime Alert - Plant {plant}",
#             "body": html_body,
#             "isHtml": True
#         }

#         try:
#             response = requests.post(N8N_DEVICE_WEBHOOK, json=payload)
#             print(f"Email sent to {recipient}: {response.status_code}")
#         except Exception as e:
#             print(f"Error sending email to {recipient}: {e}")

#     save_alert_log(alert_log) 

# @app.route('/predict/devices', methods=['GET'])
# def predict_devices():
#     last_seen = get_last_seen_time()
#     all_data = fetch_device_predictions(last_seen)

#     if not all_data:
#         return jsonify({
#             "status": "no_data",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     df = pd.DataFrame(all_data)
#     df_pred = preprocess_and_predict(df)

#     if df_pred is None or df_pred.empty:
#         return jsonify({
#             "status": "no_predictions",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     df_pred['is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)

#     df_pred = df_pred[
#         (df_pred['will_fail_soon'] == 1) &
#         (df_pred['is_failure_prob'] >= 0.40)
#     ]

#     # if not df_pred.empty:
#     #     df_pred.sort_values(['plant_code', 'is_failure_prob'], ascending=[True, False], inplace=True)
#     #     df_pred = df_pred.groupby('plant_code').head(3)

#     if not df_pred.empty:
#         df_pred.sort_values(['plant_code', 'is_failure_prob'], ascending=[True, False], inplace=True)
#         df_pred = df_pred.groupby('plant_code').head(5) 

#         # Deduplicate by Device IP + Name, keeping highest failure prob
#         df_pred = df_pred.sort_values("is_failure_prob", ascending=False)
#         df_pred = df_pred.drop_duplicates(subset=["device_ip", "device_name"])

#     if df_pred.empty:
#         return jsonify({
#             "status": "no_alerts",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     alerts = df_pred.to_dict(orient="records")
#     latest_seen = df_pred["time_down"].max()

#     group_and_send_device_alerts(alerts)

#     if latest_seen:
#         save_last_seen_time(latest_seen)

#     return jsonify({
#         "status": "processed",
#         "alerts_sent": len(alerts),
#         "last_seen_updated_to": latest_seen
#     })

# if __name__ == '__main__':
#     print("Starting Device Downtime Prediction Service...")
#     app.run(debug=True, port=5002)
    
    

## CORRECT MAIL SENDING - BY GROUPING 

# from flask import Flask, jsonify
# import pandas as pd
# import requests
# from collections import defaultdict
# from datetime import datetime, timezone, timedelta
# import os
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
# from model_utils import preprocess_and_predict
# from config import DB_CONFIG, DATABASE_URL, N8N_DEVICE_WEBHOOK, get_plant_email_map

# app = Flask(__name__)
# engine = create_engine(DATABASE_URL)
# Session = sessionmaker(bind=engine)
# LAST_SEEN_FILE = "last_seen_device_time.txt"

# def get_last_seen_time():
#     if os.path.exists(LAST_SEEN_FILE):
#         with open(LAST_SEEN_FILE, 'r') as f:
#             return f.read().strip()
#     else:
#         return (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

# def save_last_seen_time(ts):
#     with open(LAST_SEEN_FILE, 'w') as f:
#         f.write(str(ts)) 

# def fetch_device_predictions(last_seen_time=None):
#     session = Session()
#     try:
#         if last_seen_time:
#             query = text("""
#                 SELECT * FROM powerbiplms.vw_hardware_downtime
#                 WHERE time_down > :last_seen_time
#                 ORDER BY time_down ASC
#                 LIMIT 100
#             """)
#             result = session.execute(query, {"last_seen_time": last_seen_time})
#         else:
#             query = text("""
#                 SELECT * FROM powerbiplms.vw_hardware_downtime
#                 ORDER BY time_down ASC
#                 LIMIT 100
#             """)
#             result = session.execute(query)

#         rows = result.fetchall()
#         columns = result.keys()
#         return [dict(zip(columns, row)) for row in rows]
#     finally:
#         session.close()

# def group_and_send_device_alerts(alerts):
#     plant_email_map = get_plant_email_map()
#     grouped = defaultdict(list)

#     for alert in alerts:
#         plant_code = alert.get('plant_code') or alert.get('plant_name') or 'UNKNOWN'
#         grouped[plant_code].append(alert)

#     for plant, device_alerts in grouped.items():
#         recipient = plant_email_map.get(plant, plant_email_map.get('_default_', 'sakshi.tandon@amzbizsol.in'))
#         if not recipient:
#             continue

#         rows = ""
#         for device in device_alerts:
#             rows += f"""
#                 <tr>
#                     <td>{device.get('location_name', '')}</td>
#                     <td>{device.get('device_ip', '')}</td>
#                     <td>{device.get('device_name', '')}</td>
#                     <td>{round(device.get('predicted_downtime_minutes', 0), 2)}</td>
#                     <td>{round(device.get('is_failure_prob', 0) * 100, 1)}%</td>
#                     <td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
#                 </tr>
#             """

#         html_body = f"""
#         <div style='font-family:Arial, sans-serif; font-size:14px; color:#333;'>
#             <p>Dear Team,</p>
#             <p>This is an automated alert from your <strong>Device Downtime Monitoring System</strong>.</p>
#             <p><strong>Plant:</strong> {plant}</p>
#             <p><strong>Predicted Device Downtimes:</strong></p>
#             <table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; text-align: left;'>
#                 <thead style='background-color: #f2f2f2;'>
#                     <tr>
#                         <th>Location</th>
#                         <th>Device IP</th>
#                         <th>Device Name</th>
#                         <th>Predicted Downtime (min)</th>
#                         <th>Failure Probability</th>
#                         <th>Predicted At</th>
#                     </tr>
#                 </thead>
#                 <tbody>{rows}</tbody>
#             </table>
#             <p>Please take necessary action.</p>
#             <p>Regards,<br>Device Monitoring System</p>
#         </div>
#         """

#         payload = {
#             "to": recipient,
#             "subject": f"Device Downtime Alert - Plant {plant}",
#             "body": html_body,
#             "isHtml": True
#         }

#         try:
#             response = requests.post(N8N_DEVICE_WEBHOOK, json=payload)
#             print(f"Email sent to {recipient}: {response.status_code}")
#         except Exception as e:
#             print(f"Error sending email to {recipient}: {e}")


# @app.route('/predict/devices', methods=['GET'])
# def predict_devices():
#     last_seen = get_last_seen_time()
#     all_data = fetch_device_predictions(last_seen)

#     if not all_data:
#         return jsonify({
#             "status": "no_data",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     df = pd.DataFrame(all_data)
#     df_pred = preprocess_and_predict(df)

#     if df_pred is None or df_pred.empty:
#         return jsonify({
#             "status": "no_predictions",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     # Ensure probability is valid
#     df_pred['is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)

#     # 🔹 Step 1: Keep only devices flagged to fail and above minimum threshold
#     df_pred = df_pred[
#         (df_pred['will_fail_soon'] == 1) &
#         (df_pred['is_failure_prob'] >= 0.40)  # Set your minimum confidence here
#     ]

#     # 🔹 Step 2: For each plant, pick top 3 devices by failure probability
#     if not df_pred.empty:
#         df_pred.sort_values(['plant_code', 'is_failure_prob'], ascending=[True, False], inplace=True)
#         df_pred = df_pred.groupby('plant_code').head(3)  # only top 3 risky devices per plant

#     if df_pred.empty:
#         return jsonify({
#             "status": "no_alerts",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     alerts = df_pred.to_dict(orient="records")
#     # latest_seen = df["time_down"].max()
#     latest_seen = df_pred["time_down"].max()

 
#     group_and_send_device_alerts(alerts)

#     if latest_seen:
#         save_last_seen_time(latest_seen)

#     return jsonify({
#         "status": "processed",
#         "alerts_sent": len(alerts),
#         "last_seen_updated_to": latest_seen
#     })

# if __name__ == '__main__':
#     print("Starting Device Downtime Prediction Service...")
#     app.run(debug=True, port=5002)


# from flask import Flask, jsonify
# import pymysql
# import json
# import requests
# from collections import defaultdict
# from datetime import datetime, timezone, timedelta
# import os
# from sqlalchemy import create_engine
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
# import pandas as pd
# from model_utils import preprocess_and_predict


# app = Flask(__name__)

# # --- DB Config ---
# DB_CONFIG = {
#     'host': "10.0.4.23",
#     'user': "ReadOnly",
#     'password': "Amazin%40123",
#     'port': 3306,
#     'database': 'powerbiplms'
# }

# DATABASE_URL = (
#     f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# )

# engine = create_engine(DATABASE_URL)

# # --- Email Webhook ---
# N8N_DEVICE_WEBHOOK = "https://sakshi395.app.n8n.cloud/webhook-test/device-alert-email"

# # --- Plant Email Mapping ---
# PLANT_EMAIL_MAP = {
#     "N202": "randhir.kumar@amzbizsol.in",
#     "N205": "randhir.kumar@amzbizsol.in",
#     "N212": "randhir.kumar@amzbizsol.in",
#     "N225": "randhir.kumar@amzbizsol.in",
#     "N239": "mansi.rawat@amzbizsol.in",
#     "N622": "mansi.rawat@amzbizsol.in",
#     "NE03": "mansi.rawat@amzbizsol.in",
#     "NE25": "mansi.rawat@amzbizsol.in",
#     "NE29": "sakshi.tandon@amzbizsol.in",
#     "NE30": "sakshi.tandon@amzbizsol.in",
#     "NT45": "sakshi.tandon@amzbizsol.in",
#     "NT60": "sakshi.tandon@amzbizsol.in",
#     "_default_": "sakshi.tandon@amzbizsol.in"
# }

# # --- File to store last seen timestamp ---
# LAST_SEEN_FILE = "last_seen_device_time.txt"


# def get_last_seen_time():
#     if os.path.exists(LAST_SEEN_FILE):
#         with open(LAST_SEEN_FILE, 'r') as f:
#             return f.read().strip()
#     else:
#         return (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

        
# def save_last_seen_time(ts):
#     with open(LAST_SEEN_FILE, 'w') as f:
#         f.write(str(ts)) 

# Session = sessionmaker(bind=engine)

# def fetch_device_predictions(last_seen_time=None):
#     session = Session()
#     try:
#         if last_seen_time:
#             query = text("""
#                 SELECT * FROM powerbiplms.vw_hardware_downtime
#                 WHERE time_down > :last_seen_time
#                 ORDER BY time_down ASC
#                 LIMIT 100
#             """)
#             result = session.execute(query, {"last_seen_time": last_seen_time})
#         else:
#             query = text("""
#                 SELECT * FROM powerbiplms.vw_hardware_downtime
#                 ORDER BY time_down ASC
#                 LIMIT 100
#             """)
#             result = session.execute(query)

#         rows = result.fetchall()
#         columns = result.keys()
#         return [dict(zip(columns, row)) for row in rows]
#     finally:
#         session.close()
# def group_and_send_device_alerts(alerts):
#     grouped = defaultdict(list)

#     for alert in alerts:
#         plant_code = alert.get('plant_code') or alert.get('plant_name') or 'UNKNOWN'
#         grouped[plant_code].append(alert)

#     for plant, device_alerts in grouped.items():
#         recipient = PLANT_EMAIL_MAP.get(plant, PLANT_EMAIL_MAP['_default_'])
#         if not recipient:
#             continue

#         rows = ""
#         for device in device_alerts:
#             rows += f"""
#                 <tr>
#                     <td>{device.get('location_name', '')}</td>
#                     <td>{device.get('device_ip', '')}</td>
#                     <td>{device.get('device_name', '')}</td>
#                     <td>{round(device.get('predicted_downtime_minutes', 0), 2)}</td>
#                     <td>{round(device.get('is_failure_prob', 0) * 100, 1)}%</td>
#                     <td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
#                 </tr>
#             """

#         html_body = f"""
#         <div style='font-family:Arial, sans-serif; font-size:14px; color:#333;'>
#             <p>Dear Team,</p>
#             <p>This is an automated alert from your <strong>Device Downtime Monitoring System</strong>.</p>
#             <p><strong>Plant:</strong> {plant}</p>
#             <p><strong>Predicted Device Downtimes:</strong></p>
#             <table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; text-align: left;'>
#                 <thead style='background-color: #f2f2f2;'>
#                     <tr>
#                         <th>Location</th>
#                         <th>Device IP</th>
#                         <th>Device Name</th>
#                         <th>Predicted Downtime (min)</th>
#                         <th>Failure Probability</th>
#                         <th>Predicted At</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#                     {rows}
#                 </tbody>
#             </table>
#             <p>Please take necessary action.</p>
#             <p>Regards,<br>Device Monitoring System</p>
#         </div>
#         """

#         payload = {
#             "to": recipient,
#             "subject": f"Device Downtime Alert - Plant {plant}",
#             "body": html_body,
#             "isHtml": True
#         }

#         try:
#             response = requests.post(N8N_DEVICE_WEBHOOK, json=payload)
#             print(f"Email sent to {recipient}: {response.status_code}")
#         except Exception as e:
#             print(f"Error sending email to {recipient}: {e}")

# @app.route('/predict/devices', methods=['GET'])
# def predict_devices():
#     last_seen = get_last_seen_time()
#     all_data = fetch_device_predictions(last_seen)

#     if not all_data:
#         return jsonify({
#             "status": "no_data",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     df = pd.DataFrame(all_data)
#     df_pred = preprocess_and_predict(df)

#     if df_pred is None or df_pred.empty:
#         return jsonify({
#             "status": "no_predictions",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     df_pred.loc[:, 'is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)
#     # df_pred['is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)
#     df_pred = df_pred[
#         (df_pred['will_fail_soon'] == 1) &
#         (df_pred['is_failure_prob'] >= 0.65)
#     ]

#     if df_pred.empty:
#         return jsonify({
#             "status": "no_alerts",
#             "alerts_sent": 0,
#             "last_seen_updated_to": last_seen
#         })

#     alerts = df_pred.to_dict(orient="records")
#     latest_seen = df["time_down"].max()
#     group_and_send_device_alerts(alerts)

#     if latest_seen:
#         save_last_seen_time(latest_seen)

#     return jsonify({
#         "status": "processed",
#         "alerts_sent": len(alerts),
#         "last_seen_updated_to": latest_seen
#     })

# if __name__ == '__main__':
#     print("Starting Device Downtime Prediction Service...")
#     app.run(debug=True, port=5002)


### LATEST WORKING CODE - SENDING MAILS BUT SEPARATELY 

# from flask import Flask, jsonify
# import pandas as pd
# import mysql.connector
# from datetime import datetime, timezone
# import os
# import requests
# import threading
# import time

# from model_utils import preprocess_and_predict

# app = Flask(__name__)

# # --- DB Config ---
# DB_CONFIG = {
#     "host": "10.6.1.11",
#     "user": "dbadmin",
#     "password": "Amazin@1234",
#     "database": "powerbiplms",
#     "port": 3306
# }

# # --- Webhook URL for n8n ---
# N8N_WEBHOOK_URL = os.environ.get(
#     "N8N_DEVICE_WEBHOOK_URL",
#     "https://sakshi395.app.n8n.cloud/webhook/6ca0c506-04e7-4f8d-b8e0-1f1aecf6d6e2"
# )

# # --- Last Seen Tracking ---
# LAST_SEEN_FILE = "last_seen_device_time.txt"
# DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# def get_last_seen_time():
#     if os.path.exists(LAST_SEEN_FILE):
#         with open(LAST_SEEN_FILE, "r") as f:
#             return f.read().strip()
#     else:
#         return (datetime.now(timezone.utc) - pd.Timedelta(hours=24)).strftime(DATETIME_FORMAT)

# def update_last_seen_time(new_time):
#     with open(LAST_SEEN_FILE, "w") as f:
#         f.write(new_time)

# # --- Core Prediction Logic ---
# def check_device_failures():
#     print("Running device prediction cycle...")
#     last_seen_time = get_last_seen_time()

#     conn = mysql.connector.connect(**DB_CONFIG)
#     query = f"""
        # SELECT * FROM powerbiplms.vw_hardware_downtime
        # WHERE time_down > '{last_seen_time}'
        # ORDER BY time_down ASC
        # LIMIT 100;
#     """

#     df = pd.read_sql(query, conn)
#     conn.close()

#     if df.empty:
#         print("No new device records.")
#         return

#     df_pred = preprocess_and_predict(df)
#     df_pred['is_failure_prob'] = df_pred['is_failure_prob'].fillna(0.0)
#     df_pred = df_pred[(df_pred['will_fail_soon'] == 1) & (df_pred['is_failure_prob'] >= 0.65)]

#     if df_pred.empty:
#         print("No failures predicted.")
#         return

#     for _, row in df_pred.iterrows():
#         try:
#             payload = {
#                 "device_name": row.get("device_name", "N/A"),
#                 "device_ip": row.get("device_ip", "N/A"),
#                 "location_name": row.get("location_name", "N/A"),
#                 "plant_name": row.get("plant_name", "N/A"),
#                 "predicted_downtime_minutes": round(float(row.get("predicted_downtime_minutes", 0)), 2),
#                 "is_failure_prob": round(float(row.get("is_failure_prob", 0)), 4),
#                 "alertTimestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }

#             print("Payload to be sent:", payload)  # <-- ADD THIS LINE

#             response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
#             response.raise_for_status()
#             print(f"Alert sent for device {payload['device_name']}")
#         except Exception as e:
#             print(f"Failed to send alert: {e}")


#     max_time_down = pd.to_datetime(df["time_down"]).max()
#     update_last_seen_time(max_time_down.strftime(DATETIME_FORMAT))

# # --- Background Scheduler ---
# def run_scheduler(interval_secs=600):
#     def loop():
#         time.sleep(interval_secs)  
#         while True:
#             check_device_failures()
#             time.sleep(interval_secs)
#     thread = threading.Thread(target=loop, daemon=True)
#     thread.start()

# # --- Optional manual trigger endpoint ---
# @app.route("/predict/devices", methods=["GET"])
# def manual_device_trigger():
#     check_device_failures()
#     return jsonify({"message": "Device prediction triggered."})

# if __name__ == "__main__":
#     print("🚀 Starting Device Downtime Prediction Service...")
#     run_scheduler(interval_secs=600)  # Run every 5 mins
#     app.run(debug=False, port=5000)


## INTEGRATED LOGIC OF FAILED PROB. AND LAST FETCH TIME(WORKING)
# from flask import Flask, jsonify
# import pandas as pd
# import mysql.connector
# from datetime import datetime, timezone
# import os
# from model_utils import preprocess_and_predict  

# app = Flask(__name__)

# DB_CONFIG = {
#     "host": "10.0.4.23",
#     "user": "ReadOnly",
#     "password": "Amazin@123",
#     "database": "powerbiplms",
#     "port": 3306
# }

# LAST_SEEN_FILE = "last_seen_device_time.txt"
# DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# def get_last_seen_time():
#     if os.path.exists(LAST_SEEN_FILE):
#         with open(LAST_SEEN_FILE, "r") as f:
#             return f.read().strip()
#     else:
#         # First time run - set to 24 hours ago
#         return (datetime.now(timezone.utc) - pd.Timedelta(hours=24)).strftime(DATETIME_FORMAT)

# def update_last_seen_time(new_time):
#     with open(LAST_SEEN_FILE, "w") as f:
#         f.write(new_time)


# @app.route("/predict/devices", methods=["GET"])
# def predict_devices():
#     try:
#         last_seen_time = get_last_seen_time()

#         conn = mysql.connector.connect(**DB_CONFIG)
#         query = f"""
#             SELECT * FROM powerbiplms.vw_hardware_downtime
#             WHERE time_down > '{last_seen_time}'
#             ORDER BY time_down ASC
#             LIMIT 100;
#         """
#         df = pd.read_sql(query, conn)
#         conn.close()

#         if df.empty:
#             return jsonify({"status": "success", "data": [], "message": "No new records found."})

#         # Predict and filter using model
#         result_df = preprocess_and_predict(df)  # Should add 'is_failure_prob' and 'will_fail_soon'

#         # Filter on both probability and will_fail_soon
#         result_df = result_df[
#             (result_df["will_fail_soon"] == 1) &
#             (result_df["is_failure_prob"] >= 0.7)
#         ]

#         if result_df.empty:
#             return jsonify({"status": "success", "data": [], "message": "No failures predicted."})

#         # Convert to JSON
#         result_json = result_df.to_dict(orient="records")

#         # Update the last seen time to max of new records
#         max_time_down = pd.to_datetime(df["time_down"]).max()
#         update_last_seen_time(max_time_down.strftime(DATETIME_FORMAT))

#         return jsonify({"status": "success", "data": result_json})

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})


# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, jsonify
# import pandas as pd
# import mysql.connector
# from model_utils import preprocess_and_predict

# app = Flask(__name__)

# DB_CONFIG = {
#     "host": "10.0.4.23",
#     "user": "ReadOnly",
#     "password": "Amazin@123",
#     "database": "powerbiplms",
#     "port": 3306
# }

# @app.route("/predict/devices", methods=["GET"])
# def predict_devices():
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         query = "SELECT * FROM powerbiplms.vw_hardware_downtime LIMIT 100;"
#         df = pd.read_sql(query, conn) 
#         conn.close()

#         result_df = preprocess_and_predict(df)

#         # Only include devices predicted to fail soon
#         if 'will_fail_soon' in result_df.columns:
#             result_df = result_df[result_df['will_fail_soon'] == 1]

#         result_json = result_df.to_dict(orient="records")
#         return jsonify({"status": "success", "data": result_json})

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
