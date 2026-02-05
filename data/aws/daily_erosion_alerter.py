# daily_erosion_alerter.py
#
# AWS Lambda function to monitor rainfall erosivity from GeoSphere Austria,
# calculate erosion risk based on the RUSLE2 model, and send email alerts
# via AWS SES for significant events.
#
# This script is designed to be triggered by Amazon EventBridge on a schedule (e.g., once a day).

import boto3
import urllib.parse
import urllib.request
import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from botocore.exceptions import ClientError
from html import escape

# ==============================================================================
# --- Lambda Environment Variable Configuration ---
#
# These variables must be configured in your AWS Lambda function's settings.
#
# ==============================================================================

# --- API & Location Configuration ---
# GeoSphere API endpoint for the desired dataset.
API_RESOURCE_ID = os.environ.get("API_RESOURCE_ID", "tawes-v1-10min")
# Central point for station discovery.
CENTER_LAT = float(os.environ.get("CENTER_LAT", "47.22517226258929"))
CENTER_LON = float(os.environ.get("CENTER_LON", "15.911707948071635"))
# Radius in km from the central point to search for stations.
BUFFER_KM = float(os.environ.get("BUFFER_KM", "25"))
# Time window in hours to analyse for recent events.
TIME_WINDOW_HOURS = int(os.environ.get("TIME_WINDOW_HOURS", "24"))

# --- Email & Alerting Configuration ---
# The verified "From" email address in your AWS SES account.
EMAIL_FROM = os.environ.get("EMAIL_FROM")
# Comma-separated list of recipient email addresses.
EMAIL_TO = os.environ.get("EMAIL_TO")
# The AWS region where your SES service is configured.
SES_REGION = os.environ.get("SES_REGION", "eu-north-1")
# URL to your Streamlit dashboard for a "view more" link in the email.
STREAMLIT_APP_URL = os.environ.get("STREAMLIT_APP_URL", "")

# --- Debug/Test Mode ---
# Set to "true" to force an email to be sent even if no alerts are triggered.
# The email will contain the top 5 stations by rainfall as an example.
FORCE_SEND_EMAIL = os.environ.get("FORCE_SEND_EMAIL", "false").lower() == "true"


# ==============================================================================
# --- Core Scientific & Application Logic ---
#
# This logic is adapted from the `weather_app.py` to ensure consistency
# between the dashboard and the alerts.
#
# ==============================================================================

# --- Constants ---
API_HOST = "https://dataset.api.hub.geosphere.at"
API_VERSION = "v1"

# Storm delineation parameters (Wischmeier & Smith, 1978)
STORM_GAP_HOURS = 6
STORM_GAP_RAIN_MM = 1.27
STORM_MIN_RAIN_MM = 12.7
STORM_INTENSE_15MIN_MM = 6.0

# Alert thresholds for EI30
ALERT_THRESHOLDS = [
    (200, "High", "#e74c3c"),
    (100, "Moderate", "#e67e22"),
    (25, "Low", "#f1c40f"),
    (0, "No risk", "#2ecc71"),
]

# SCS/NRCS Antecedent Moisture Condition (AMC) classification
AMC_CLASSES = [
    (28.0, "AMC III", "Wet—high runoff potential"),
    (13.0, "AMC II", "Moderate"),
    (0.0, "AMC I", "Dry—low runoff potential"),
]

# --- Utility Functions ---

def get_alert_level(ei30):
    for threshold, label, color in ALERT_THRESHOLDS:
        if ei30 >= threshold:
            return label, color
    return "No risk", "#2ecc71"

def classify_amc(antecedent_mm):
    for threshold, cls, description in AMC_CLASSES:
        if antecedent_mm >= threshold:
            return cls, description
    return "AMC I", "Dry—low runoff potential"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# --- Data Fetching & Parsing ---

def _http_get_json(url):
    """Makes an HTTP GET request and returns the JSON response."""
    print(f"Fetching URL: {url}")
    try:
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception as e:
        print(f"ERROR: HTTP request failed for {url}. Reason: {e}")
        raise

def fetch_station_metadata(buffer_km):
    """Fetches station metadata and filters by distance from the central point."""
    url = f"{API_HOST}/{API_VERSION}/station/current/{API_RESOURCE_ID}/metadata"
    data = _http_get_json(url)
    stations = {}
    for s in data.get("stations", []):
        dist = haversine_km(CENTER_LAT, CENTER_LON, s["lat"], s["lon"])
        if dist <= buffer_km:
            stations[str(s["id"])] = {
                "id": str(s["id"]),
                "name": s.get("name", str(s["id"])),
                "lat": s["lat"],
                "lon": s["lon"],
                "distance_km": round(dist, 1),
            }
    return stations

def fetch_data_for_window(station_ids, start_dt, end_dt, parameters="RR"):
    """Fetches data for a given set of stations and a specific time window."""
    params = {
        "parameters": parameters,
        "station_ids": ",".join(station_ids),
        "start": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M"),
        "output_format": "geojson",
    }
    url = f"{API_HOST}/{API_VERSION}/station/historical/{API_RESOURCE_ID}?{urllib.parse.urlencode(params)}"
    return _http_get_json(url)

def parse_geojson_to_df(geojson, stations_meta):
    """Parses GeoSphere GeoJSON response into a Pandas DataFrame."""
    # This function is simplified for the Lambda context, assuming only RR is requested.
    timestamps = geojson.get("timestamps", [])
    features = geojson.get("features", [])
    rows = []
    for feature in features:
        props = feature.get("properties", {})
        station_id = str(props.get("station", ""))
        if station_id not in stations_meta:
            continue
        
        rr_data = props.get("parameters", {}).get("RR", {}).get("data", [])
        
        for i in range(min(len(rr_data), len(timestamps))):
            raw_val = rr_data[i]
            rows.append({
                "timestamp": timestamps[i],
                "station_id": station_id,
                "RR": float(raw_val) if raw_val is not None else 0.0,
            })

    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

# --- Erosivity Calculation ---

def _compute_energy(intensity, rr):
    """Computes kinetic energy using the RUSLE2 equation."""
    # RUSLE2: McGregor et al. (1995), Eq. 7 in Nearing et al. (2017)
    e_unit = 0.29 * (1.0 - 0.72 * np.exp(-0.082 * intensity))
    return e_unit * rr

def _compute_i30(rr, interval_minutes):
    """Computes maximum 30-min intensity from rainfall array."""
    if interval_minutes > 30: # Cannot be computed for hourly data
        return np.max(rr) * (60 / interval_minutes)

    n_30 = max(1, int(30 / interval_minutes))
    if len(rr) >= n_30:
        return np.max(np.convolve(rr, np.ones(n_30), mode="valid")) * 2.0
    
    return np.sum(rr) * (60 / (len(rr) * interval_minutes)) if len(rr) > 0 else 0.0

def delineate_storms(rr, interval_minutes):
    """Splits a rainfall array into individual storm events."""
    n = len(rr)
    if n == 0:
        return []
    gap_intervals = max(1, int(STORM_GAP_HOURS * 60 / interval_minutes))
    if n <= gap_intervals:
        return [(0, n)] if np.sum(rr) > 0 else []
    
    cumsum = np.concatenate([[0.0], np.cumsum(rr)])
    rolling_sum = cumsum[gap_intervals:] - cumsum[:-gap_intervals]
    
    in_gap = np.zeros(n, dtype=bool)
    for i in range(len(rolling_sum)):
        if rolling_sum[i] < STORM_GAP_RAIN_MM:
            in_gap[i : i + gap_intervals] = True

    storms, start = [], None
    for i in range(n):
        if not in_gap[i] and start is None:
            start = i
        elif in_gap[i] and start is not None:
            storms.append((start, i))
            start = None
    if start is not None:
        storms.append((start, n))
    return storms

def compute_station_erosivity(df_station, interval_minutes=10):
    """Computes storm-delineated EI30 for one station."""
    if df_station.empty or "RR" not in df_station.columns:
        return []
    
    df = df_station.sort_values("timestamp").reset_index()
    rr_all = df["RR"].to_numpy(dtype=float)
    storm_slices = delineate_storms(rr_all, interval_minutes)
    
    storm_results = []
    for start, end in storm_slices:
        rr_storm = rr_all[start:end]
        total_rain = np.sum(rr_storm)
        if total_rain <= 0:
            continue
        
        n_15 = max(1, math.ceil(15 / interval_minutes))
        max_15_min_rain = np.max(np.convolve(rr_storm, np.ones(n_15), mode="valid")) if len(rr_storm) >= n_15 else total_rain
        
        is_erosive = (total_rain >= STORM_MIN_RAIN_MM) or (max_15_min_rain >= STORM_INTENSE_15MIN_MM)
        
        intensity = rr_storm * (60.0 / interval_minutes)
        e_total = np.sum(_compute_energy(intensity, rr_storm))
        i30 = _compute_i30(rr_storm, interval_minutes)
        ei30 = e_total * i30
        
        alert_label, alert_color = get_alert_level(ei30)
        
        storm_results.append({
            "total_rain": round(total_rain, 2),
            "I30": round(i30, 2),
            "EI30": round(ei30, 2),
            "is_erosive": is_erosive,
            "alert_label": alert_label,
            "alert_color": alert_color,
        })
        
    return storm_results

# --- Email Generation and Sending ---

def build_html_email_body(alert_stations, antecedent_totals, non_alert_stations):
    """Builds an HTML-formatted email body."""
    
    # CSS for styling
    header_bg = "#2c3e50"
    header_text = "#ecf0f1"
    border_color = "#bdc3c7"
    bg_color = "#ffffff"
    body_text = "#34495e"
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px; }}
            .container {{ background-color: {bg_color}; border: 1px solid #dddddd; max-width: 800px; margin: auto; }}
            .header {{ background-color: {header_bg}; color: {header_text}; padding: 20px; text-align: center; }}
            .header h1 {{ margin: 0; }}
            .content {{ padding: 20px; color: {body_text}; }}
            .summary-table, .alert-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .summary-table th, .summary-table td, .alert-table th, .alert-table td {{ text-align: left; padding: 12px; border-bottom: 1px solid {border_color}; }}
            .summary-table th {{ background-color: #f2f2f2; }}
            .footer {{ font-size: 0.8em; text-align: center; color: #7f8c8d; padding: 20px; }}
            .risk-High {{ color: {ALERT_THRESHOLDS[0][2]}; font-weight: bold; }}
            .risk-Moderate {{ color: {ALERT_THRESHOLDS[1][2]}; font-weight: bold; }}
            .risk-Low {{ color: {ALERT_THRESHOLDS[2][2]}; }}
            a {{ color: #3498db; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>Erosion Risk Alert</h1></div>
            <div class="content">
                <h2>Analysis for the Last {TIME_WINDOW_HOURS} Hours</h2>
                <table class="summary-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Analysis Center</td><td>{CENTER_LAT:.5f}, {CENTER_LON:.5f}</td></tr>
                    <tr><td>Buffer Radius</td><td>{BUFFER_KM} km</td></tr>
                    <tr><td>Stations with Risk</td><td>{len(alert_stations)}</td></tr>
                </table>
    """

    if alert_stations:
        html += "<h3>Stations with Significant Erosion Risk</h3>"
        html += """
                <table class="alert-table">
                    <tr>
                        <th>Station</th>
                        <th>Risk Level</th>
                        <th>EI30</th>
                        <th>Rainfall (mm)</th>
                        <th>Antecedent (5d)</th>
                        <th>AMC</th>
                    </tr>
        """
        for alert in alert_stations:
            station_id = alert['station_id']
            ant_rain = antecedent_totals.get(station_id, 0.0)
            amc_class, _ = classify_amc(ant_rain)
            html += f"""
                    <tr>
                        <td>{escape(alert['station_name'])}</td>
                        <td class="risk-{escape(alert['alert_label'])}">{escape(alert['alert_label'])}</td>
                        <td>{alert['EI30']}</td>
                        <td>{alert['total_rain']}</td>
                        <td>{ant_rain} mm</td>
                        <td>{amc_class}</td>
                    </tr>
            """
        html += "</table>"
    elif FORCE_SEND_EMAIL:
         html += "<h3>No Significant Events Detected (Test Mode)</h3>"
         html += "<p>This is a test email. The following stations had the most rainfall in the period:</p>"
         html += """
                <table class="alert-table">
                    <tr><th>Station</th><th>Rainfall (mm)</th><th>EI30</th></tr>
        """
         for station in non_alert_stations:
             html += f"""
                    <tr>
                        <td>{escape(station['station_name'])}</td>
                        <td>{station['total_rain']}</td>
                        <td>{station['EI30']}</td>
                    </tr>
            """
         html += "</table>"
    else:
        html += "<h3>No Significant Erosion Events Detected</h3>"


    if STREAMLIT_APP_URL:
        html += f"""
                <p style="text-align:center; margin-top:30px;">
                    <a href="{STREAMLIT_APP_URL}">View Full Analysis Dashboard</a>
                </p>
        """

    html += f"""
            </div>
            <div class="footer">
                This is an automated alert. Analysis based on GeoSphere Austria data.
            </div>
        </div>
    </body>
    </html>
    """
    return html

def send_alert_email(subject, html_body):
    """Sends an email using AWS SES."""
    if not EMAIL_FROM or not EMAIL_TO:
        print("ERROR: EMAIL_FROM and EMAIL_TO environment variables must be set.")
        return False, "Email configuration is missing."

    ses = boto3.client("ses", region_name=SES_REGION)
    
    to_addresses = [e.strip() for e in EMAIL_TO.split(",") if e.strip()]
    
    if not to_addresses:
        print("ERROR: No recipient email addresses specified in EMAIL_TO.")
        return False, "No recipients specified."

    # Check verification status for each recipient
    try:
        verification = ses.get_identity_verification_attributes(Identities=to_addresses)
        verified_recipients = []
        skipped_unverified = []
        for addr in to_addresses:
            if verification["VerificationAttributes"].get(addr, {}).get("VerificationStatus") == "Success":
                verified_recipients.append(addr)
            else:
                skipped_unverified.append(addr)
        
        if skipped_unverified:
            print(f"WARNING: Skipping unverified email addresses: {', '.join(skipped_unverified)}")

        if not verified_recipients:
            print("ERROR: No verified recipient email addresses. Email not sent.")
            return False, "No verified recipients."

    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        print(f"ERROR: Failed to check email verification status: {error_message}")
        return False, f"SES verification error: {error_message}"

    sent_count = 0
    failed_sends = {}
    for recipient_addr in verified_recipients:
        try:
            ses.send_email(
                Source=EMAIL_FROM,
                Destination={"ToAddresses": [recipient_addr]}, # Send individually
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
                },
            )
            print(f"Email sent successfully to: {recipient_addr}")
            sent_count += 1
        except ClientError as e:
            error_message = e.response["Error"]["Message"]
            print(f"ERROR: Failed to send email to {recipient_addr}. Reason: {error_message}")
            failed_sends[recipient_addr] = error_message
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while sending email to {recipient_addr}: {e}")
            failed_sends[recipient_addr] = str(e)

    if sent_count > 0:
        if failed_sends:
            return True, f"Email sent to {sent_count} recipients, but failed for {len(failed_sends)}."
        return True, "Email sent successfully to all verified recipients."
    else:
        return False, f"No emails were sent. Failed for all {len(failed_sends)} recipients."

# ==============================================================================
# --- Lambda Handler ---
# ==============================================================================

def lambda_handler(event, context):
    """
    Main entry point for the AWS Lambda function.
    """
    print(f"Starting erosion alert check at {datetime.now(timezone.utc).isoformat()}")
    
    # 1. Fetch station metadata
    try:
        stations_meta = fetch_station_metadata(BUFFER_KM)
        if not stations_meta:
            print("No stations found in buffer. Exiting.")
            return {"statusCode": 200, "body": "No stations found."}
        print(f"Found {len(stations_meta)} stations.")
    except Exception as e:
        return {"statusCode": 500, "body": f"Failed to fetch station metadata: {e}"}

    station_ids = list(stations_meta.keys())
    now = datetime.now(timezone.utc)
    
    # 2. Fetch data for main window and antecedent window
    try:
        # Main window (e.g., last 24 hours)
        main_window_start = now - timedelta(hours=TIME_WINDOW_HOURS)
        main_geojson = fetch_data_for_window(station_ids, main_window_start, now)
        
        # Antecedent window (5 days before main window)
        ant_window_end = main_window_start
        ant_window_start = ant_window_end - timedelta(days=5)
        ant_geojson = fetch_data_for_window(station_ids, ant_window_start, ant_window_end)
        
    except Exception as e:
        return {"statusCode": 500, "body": f"Failed to fetch GeoSphere data: {e}"}

    # 3. Parse data and compute erosivity & antecedent totals
    df_main = parse_geojson_to_df(main_geojson, stations_meta)
    df_ant = parse_geojson_to_df(ant_geojson, stations_meta)

    antecedent_totals = df_ant.groupby('station_id')['RR'].sum().to_dict() if not df_ant.empty else {}

    all_station_results = []
    for station_id, meta in stations_meta.items():
        df_station = df_main[df_main["station_id"] == station_id]
        storm_results = compute_station_erosivity(df_station)
        
        # Find the worst erosive storm in the window
        erosive_storms = [s for s in storm_results if s["is_erosive"]]
        worst_storm = max(erosive_storms, key=lambda s: s["EI30"]) if erosive_storms else None

        result = {
            "station_id": station_id,
            "station_name": meta["name"],
        }
        if worst_storm:
             result.update(worst_storm)
        else: # Add placeholder values if no erosive storm
            result.update({
                "total_rain": df_station['RR'].sum() if not df_station.empty else 0,
                "I30": 0, "EI30": 0, "alert_label": "No risk"
            })
        all_station_results.append(result)
        
    # 4. Filter for significant alerts
    alert_stations = [s for s in all_station_results if s["alert_label"] != "No risk"]
    alert_stations.sort(key=lambda x: x["EI30"], reverse=True)

    # 5. Build and send email if necessary
    highest_risk_label = alert_stations[0]['alert_label'] if alert_stations else "No Risk"
    
    if not alert_stations and not FORCE_SEND_EMAIL:
        print("No significant erosion events detected. No email will be sent.")
        return {"statusCode": 200, "body": "No significant events."}

    # For debug mode, get top 5 stations by rain if no alerts
    non_alert_stations_for_debug = []
    if FORCE_SEND_EMAIL and not alert_stations:
        all_station_results.sort(key=lambda x: x["total_rain"], reverse=True)
        non_alert_stations_for_debug = all_station_results[:5]

    # Build and send email
    subject = f"Erosion Alert: {highest_risk_label} ({len(alert_stations)} Station(s))"
    if FORCE_SEND_EMAIL:
        subject = f"[TEST] {subject}"
        
    html_body = build_html_email_body(alert_stations, antecedent_totals, non_alert_stations_for_debug)
    
    success, message = send_alert_email(subject, html_body)

    if success:
        return {"statusCode": 200, "body": message}
    else:
        return {"statusCode": 500, "body": message}
