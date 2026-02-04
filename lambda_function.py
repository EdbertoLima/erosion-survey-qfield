import boto3
import urllib.parse
import urllib.request
import os
import json
import math
from datetime import datetime, timedelta, timezone
from botocore.exceptions import ClientError

# --- Env vars (Lambda Environment variables) ---
# GeoSphere TAWES API
API_BASE = os.environ.get(
    "API_BASE",
    "https://dataset.api.hub.geosphere.at/v1/station"
)

# Central analysis point and buffer radius
CENTER_LAT = float(os.environ.get("CENTER_LAT", "47.22517226258929"))
CENTER_LON = float(os.environ.get("CENTER_LON", "15.911707948071635"))
BUFFER_KM = float(os.environ.get("BUFFER_KM", "50"))

# Time window in hours to analyse
TIME_WINDOW_HOURS = int(os.environ.get("TIME_WINDOW_HOURS", "72"))

# Email (SES)
EMAIL_FROM = os.environ["EMAIL_FROM"]
EMAIL_TO = os.environ["EMAIL_TO"]
SES_REGION = os.environ.get("SES_REGION", "eu-north-1")

# Set to "true" to always send the email regardless of erosion risk (for testing)
FORCE_SEND_EMAIL = os.environ.get("FORCE_SEND_EMAIL", "false").lower() == "true"

STREAMLIT_APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://bodenerosionkartieranleitung.streamlit.app"
)

# --- Alert thresholds (same as weather_app.py) ---
ALERT_THRESHOLDS = [
    (400, "red",    "High erosion risk"),
    (100, "orange", "Moderate erosion risk"),
    (25,  "yellow", "Low erosion risk"),
    (0,   "green",  "No risk"),
]


def get_alert_level(ei30):
    for threshold, level, label in ALERT_THRESHOLDS:
        if ei30 >= threshold:
            return level, label
    return "green", "No risk"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _http_get_json(url):
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def fetch_station_metadata():
    url = f"{API_BASE}/current/tawes-v1-10min/metadata"
    data = _http_get_json(url)
    stations = {}
    for s in data.get("stations", []):
        dist = haversine_km(CENTER_LAT, CENTER_LON, s["lat"], s["lon"])
        if dist <= BUFFER_KM:
            stations[s["id"]] = {
                "id": s["id"],
                "name": s.get("name", str(s["id"])),
                "lat": s["lat"],
                "lon": s["lon"],
                "distance_km": round(dist, 1),
            }
    return stations


def fetch_rainfall_data(station_ids, hours):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    # Round down to midnight (00:00) of that day
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    start_str = start.strftime("%Y-%m-%dT%H:%M")
    end_str = now.strftime("%Y-%m-%dT%H:%M")

    params = {
        "parameters": "RR",
        "station_ids": ",".join(station_ids),
        "start": start_str,
        "end": end_str,
        "output_format": "geojson",
    }
    url = f"{API_BASE}/historical/tawes-v1-10min?{urllib.parse.urlencode(params)}"
    return _http_get_json(url)


def parse_rainfall(geojson, stations_meta):
    """Parse GeoJSON into {station_id: [(timestamp_str, rr_value), ...]}."""
    timestamps = geojson.get("timestamps", [])
    features = geojson.get("features", [])
    result = {}

    for feature in features:
        props = feature.get("properties", {})
        station_id = props.get("station", "")
        if station_id not in stations_meta:
            continue

        rr_param = props.get("parameters", {}).get("RR", {})
        data = rr_param.get("data", [])

        series = []
        for i in range(min(len(data), len(timestamps))):
            val = data[i]
            rr = float(val) if val is not None else 0.0
            series.append((timestamps[i], rr))

        result[station_id] = series

    return result


def compute_ei30(rr_values):
    """Compute EI30 erosivity index from a list of 10-min RR values (mm).

    Returns dict with total_rainfall, I30, E_total, EI30, alert_level, alert_label.
    """
    if not rr_values:
        return {
            "total_rainfall": 0.0, "I30": 0.0, "E_total": 0.0, "EI30": 0.0,
            "alert_level": "green", "alert_label": "No risk",
        }

    # Intensity in mm/h (10-min interval -> multiply by 6)
    intensities = [rr * 6.0 for rr in rr_values]

    # Unit kinetic energy (Brown & Foster 1987)
    e_intervals = []
    for rr, intensity in zip(rr_values, intensities):
        e_unit = 0.29 * (1.0 - 0.72 * math.exp(-0.05 * intensity))
        e_intervals.append(e_unit * rr)

    E_total = sum(e_intervals)
    total_rainfall = sum(rr_values)

    # I30: max rolling 30-min intensity (3 consecutive 10-min intervals)
    if len(rr_values) >= 3:
        rolling_sums = [
            sum(rr_values[i:i + 3])
            for i in range(len(rr_values) - 2)
        ]
        I30 = max(rolling_sums) * 2.0  # sum of 3 intervals x 2 -> mm/h
    elif rr_values:
        I30 = sum(rr_values) * (6.0 / len(rr_values))
    else:
        I30 = 0.0

    EI30 = E_total * I30

    level, label = get_alert_level(EI30)

    return {
        "total_rainfall": round(total_rainfall, 2),
        "I30": round(I30, 2),
        "E_total": round(E_total, 4),
        "EI30": round(EI30, 2),
        "alert_level": level,
        "alert_label": label,
    }


def build_email_body(alerts, window_hours):
    """Build the email body text from a list of station alerts."""
    lines = [
        "Bodenerosion Erosion Risk Alert",
        "=" * 40,
        "",
        f"Analysis window: last {window_hours} hours",
        f"Analysis center: {CENTER_LAT:.5f}, {CENTER_LON:.5f}",
        f"Buffer radius: {BUFFER_KM} km",
        "",
        "Stations with erosion risk:",
        "-" * 40,
    ]

    for alert in alerts:
        lines.append("")
        lines.append(f"  Station:   {alert['station_name']}")
        lines.append(f"  Risk:      {alert['alert_label']}")
        lines.append(f"  EI30:      {alert['EI30']} MJ*mm*ha-1*h-1")
        lines.append(f"  I30:       {alert['I30']} mm/h")
        lines.append(f"  Rainfall:  {alert['total_rainfall']} mm")
        lines.append(f"  Distance:  {alert['distance_km']} km from center")

    lines.extend([
        "",
        "-" * 40,
        "",
        "View the full analysis on the Streamlit dashboard:",
        STREAMLIT_APP_URL,
        "",
        "Alert thresholds:",
        "  EI30 >= 400  ->  High erosion risk",
        "  EI30 >= 100  ->  Moderate erosion risk",
        "  EI30 >= 25   ->  Low erosion risk",
        "  EI30 <  25   ->  No risk",
    ])

    return "\n".join(lines)


def lambda_handler(event, context):
    # 1. Fetch station metadata
    print("Fetching station metadata...")
    try:
        stations_meta = fetch_station_metadata()
    except Exception as e:
        print(f"Error fetching station metadata: {e}")
        return {"status": "failed", "error": str(e)}

    if not stations_meta:
        print("No stations found within buffer radius.")
        return {"status": "ok", "message": "No stations found", "email_sent": False}

    station_ids = list(stations_meta.keys())
    print(f"Found {len(station_ids)} stations within {BUFFER_KM} km")

    # 2. Fetch rainfall data
    print("Fetching rainfall data...")
    try:
        geojson = fetch_rainfall_data(station_ids, TIME_WINDOW_HOURS)
    except Exception as e:
        print(f"Error fetching rainfall data: {e}")
        return {"status": "failed", "error": str(e)}

    # 3. Parse and compute EI30 per station
    station_series = parse_rainfall(geojson, stations_meta)

    alerts = []
    all_results = []
    for sid, meta in stations_meta.items():
        series = station_series.get(sid, [])
        rr_values = [rr for _, rr in series]
        result = compute_ei30(rr_values)
        result["station_id"] = sid
        result["station_name"] = meta["name"]
        result["distance_km"] = meta["distance_km"]
        all_results.append(result)

        if result["alert_level"] != "green":
            alerts.append(result)

    print(f"EI30 computed for {len(all_results)} stations, {len(alerts)} with risk")

    # 4. Only send email if there is erosion risk (or FORCE_SEND_EMAIL is on)
    if not alerts and not FORCE_SEND_EMAIL:
        print("No erosion risk detected. No email sent.")
        return {
            "status": "ok",
            "stations_checked": len(all_results),
            "alerts": 0,
            "email_sent": False,
        }

    if FORCE_SEND_EMAIL and not alerts:
        print("FORCE_SEND_EMAIL is enabled. Sending email with all station results.")
        # Use all results so the test email has content
        email_stations = all_results
    else:
        email_stations = alerts

    # Sort by EI30 descending
    email_stations.sort(key=lambda x: x["EI30"], reverse=True)

    # Determine highest risk level for subject line
    if alerts:
        highest_label = email_stations[0]["alert_label"]
    else:
        highest_label = "No risk (test mode)"

    subject = f"Erosion Alert: {highest_label} ({len(email_stations)} station(s))"
    if FORCE_SEND_EMAIL:
        subject = f"[TEST] {subject}"
    body = build_email_body(email_stations, TIME_WINDOW_HOURS)

    print(f"Sending alert email: {subject}")
    ses = boto3.client("ses", region_name=SES_REGION)
    try:
        to_addresses = [e.strip() for e in EMAIL_TO.split(",") if e.strip()]

        # Filter out unverified email addresses
        verification = ses.get_identity_verification_attributes(
            Identities=to_addresses
        )
        verified = [
            addr for addr in to_addresses
            if verification["VerificationAttributes"].get(addr, {}).get("VerificationStatus") == "Success"
        ]
        skipped = set(to_addresses) - set(verified)
        if skipped:
            print(f"Skipping unverified addresses: {', '.join(skipped)}")
        if not verified:
            print("No verified recipients. Email not sent.")
            return {
                "status": "ok",
                "stations_checked": len(all_results),
                "alerts": len(alerts),
                "email_sent": False,
                "reason": "No verified recipients",
                "skipped_addresses": list(skipped),
            }

        sent = []
        failed = {}
        for addr in verified:
            try:
                ses.send_email(
                    Source=EMAIL_FROM,
                    Destination={"ToAddresses": [addr]},
                    Message={
                        "Subject": {"Data": subject},
                        "Body": {"Text": {"Data": body}},
                    },
                )
                print(f"Email sent to: {addr}")
                sent.append(addr)
            except ClientError as e:
                print(f"Failed to send to {addr}: {e}")
                failed[addr] = str(e)

        return {
            "status": "ok",
            "stations_checked": len(all_results),
            "alerts": len(alerts),
            "highest_risk": highest_label,
            "email_sent": len(sent) > 0,
            "sent_to": sent,
            "failed": failed,
            "skipped_unverified": list(skipped),
        }
    except ClientError as e:
        print(f"SES Error: {e}")
        return {"status": "failed", "error": str(e), "email_sent": False}
