import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta, timezone

import math

API_BASE = "https://dataset.api.hub.geosphere.at/v1/station"

# Central analysis point and buffer radius
CENTER_LAT = 48.34335611029373
CENTER_LON = 15.854961267324692
BUFFER_KM = 50


def haversine_km(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

AGGREGATION_OPTIONS = {
    "10 min": "10min",
    "30 min": "30min",
    "1 hour": "1h",
    "24 hours": "24h",
}

TIME_RANGE_OPTIONS = {
    "6 h": 6,
    "12 h": 12,
    "24 h": 24,
    "48 h": 48,
    "72 h": 72,
}

ALERT_THRESHOLDS = [
    (400, "red", "High erosion risk", "#e74c3c"),
    (100, "orange", "Moderate erosion risk", "#e67e22"),
    (25, "yellow", "Low erosion risk", "#f1c40f"),
    (0, "green", "No risk", "#2ecc71"),
]


def get_alert_level(ei30):
    """Return (level_name, label, color_hex) for a given EI30 value."""
    for threshold, level, label, color in ALERT_THRESHOLDS:
        if ei30 >= threshold:
            return level, label, color
    return "green", "No risk", "#2ecc71"


@st.cache_data(ttl=600)
def fetch_station_metadata():
    """Fetch all TAWES stations and keep those within BUFFER_KM of the center point."""
    url = f"{API_BASE}/current/tawes-v1-10min/metadata"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    stations = {}
    for s in data.get("stations", []):
        dist = haversine_km(CENTER_LAT, CENTER_LON, s["lat"], s["lon"])
        if dist <= BUFFER_KM:
            stations[s["id"]] = {
                "id": s["id"],
                "name": s.get("name", str(s["id"])),
                "lat": s["lat"],
                "lon": s["lon"],
                "altitude": s.get("altitude", None),
                "state": s.get("state", ""),
                "distance_km": round(dist, 1),
            }
    return stations


@st.cache_data(ttl=300)
def fetch_rainfall_data(station_ids_tuple, hours=72, parameters="RR"):
    """Fetch historical TAWES data for given parameters and time window."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M")
    end_str = now.strftime("%Y-%m-%dT%H:%M")

    url = f"{API_BASE}/historical/tawes-v1-10min"
    params = {
        "parameters": parameters,
        "station_ids": ",".join(station_ids_tuple),
        "start": start_str,
        "end": end_str,
        "output_format": "geojson",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_antecedent_rainfall(station_ids_tuple, window_hours=72):
    """Fetch 5-day (120h) antecedent rainfall ending at the start of the main window."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=window_hours)
    antecedent_start = window_start - timedelta(hours=120)
    start_str = antecedent_start.strftime("%Y-%m-%dT%H:%M")
    end_str = window_start.strftime("%Y-%m-%dT%H:%M")

    url = f"{API_BASE}/historical/tawes-v1-10min"
    params = {
        "parameters": "RR",
        "station_ids": ",".join(station_ids_tuple),
        "start": start_str,
        "end": end_str,
        "output_format": "geojson",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_geojson_to_df(geojson, stations_meta, parameter_defaults=None):
    """Parse GeoSphere GeoJSON response into a pandas DataFrame.

    parameter_defaults: dict mapping parameter name to default value for missing data.
                        e.g. {"RR": 0.0, "TL": np.nan, "SCHNEE": np.nan}
                        If None, defaults to {"RR": 0.0}.
    """
    if parameter_defaults is None:
        parameter_defaults = {"RR": 0.0}

    timestamps = geojson.get("timestamps", [])
    features = geojson.get("features", [])

    rows = []
    for feature in features:
        props = feature.get("properties", {})
        station_id = props.get("station", "")
        parameters_data = props.get("parameters", {})
        station_name = stations_meta.get(station_id, {}).get("name", station_id)

        # Determine which parameters are present
        param_names = list(parameters_data.keys())
        if not param_names:
            continue

        # Use the first parameter's data length to determine row count
        first_param = param_names[0]
        data_length = len(parameters_data[first_param].get("data", []))

        for i in range(min(data_length, len(timestamps))):
            row = {
                "timestamp": timestamps[i],
                "station_id": station_id,
                "station_name": station_name,
            }
            for pname in param_names:
                raw_val = parameters_data[pname].get("data", [])[i] if i < len(parameters_data[pname].get("data", [])) else None
                default = parameter_defaults.get(pname, np.nan)
                row[pname] = raw_val if raw_val is not None else default
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def compute_ei30(df_station):
    """Compute EI30 erosivity index for a single station's 10-min RR data.

    Returns dict with: total_rainfall, I30, E_total, EI30, alert_level, alert_label, alert_color
    """
    if df_station.empty or "RR" not in df_station.columns:
        return {
            "total_rainfall": 0.0, "I30": 0.0, "E_total": 0.0, "EI30": 0.0,
            "alert_level": "green", "alert_label": "No risk", "alert_color": "#2ecc71",
        }

    rr = df_station["RR"].values.astype(float)
    # Intensity in mm/h (10-min interval → multiply by 6)
    intensity = rr * 6.0
    # Unit kinetic energy (Brown & Foster 1987)
    e_unit = 0.29 * (1.0 - 0.72 * np.exp(-0.05 * intensity))
    # Energy per interval
    e_interval = e_unit * rr
    E_total = float(np.sum(e_interval))
    total_rainfall = float(np.sum(rr))

    # I30: max rolling 30-min intensity (3 consecutive 10-min intervals)
    if len(rr) >= 3:
        rolling_30 = np.convolve(rr, np.ones(3), mode="valid")
        I30 = float(np.max(rolling_30) * 2.0)  # sum of 3 intervals × 2 → mm/h
    else:
        I30 = float(np.sum(rr) * (6.0 / len(rr)) if len(rr) > 0 else 0.0)

    EI30 = E_total * I30

    level, label, color = get_alert_level(EI30)

    return {
        "total_rainfall": round(total_rainfall, 2),
        "I30": round(I30, 2),
        "E_total": round(E_total, 4),
        "EI30": round(EI30, 2),
        "alert_level": level,
        "alert_label": label,
        "alert_color": color,
    }


def compute_all_stations_ei30(df_raw, stations_meta):
    """Compute EI30 for all stations. Returns dict station_id → result dict."""
    results = {}
    for station_id in stations_meta:
        df_station = df_raw[df_raw["station_id"] == station_id].sort_values("timestamp")
        results[station_id] = compute_ei30(df_station)
    return results


def compute_antecedent_totals(geojson, stations_meta):
    """Compute 5-day antecedent rainfall totals per station."""
    df = parse_geojson_to_df(geojson, stations_meta, parameter_defaults={"RR": 0.0})
    totals = {}
    if df.empty:
        return totals
    for station_id in stations_meta:
        station_df = df[df["station_id"] == station_id]
        totals[station_id] = round(float(station_df["RR"].sum()), 2) if not station_df.empty else 0.0
    return totals


def compute_snowmelt_risk(df_context, stations_meta):
    """Determine snowmelt risk per station from TL and SCHNEE data.

    Returns dict station_id → {snowmelt_risk: bool, latest_temp: float|None, latest_snow_depth: float|None}
    """
    results = {}
    for station_id in stations_meta:
        sdf = df_context[df_context["station_id"] == station_id].sort_values("timestamp")
        risk = False
        latest_temp = None
        latest_snow = None

        if not sdf.empty:
            has_tl = "TL" in sdf.columns
            has_schnee = "SCHNEE" in sdf.columns

            if has_tl:
                tl_valid = sdf["TL"].dropna()
                if not tl_valid.empty:
                    latest_temp = round(float(tl_valid.iloc[-1]), 1)

            if has_schnee:
                schnee_valid = sdf["SCHNEE"].dropna()
                if not schnee_valid.empty:
                    latest_snow = round(float(schnee_valid.iloc[-1]), 1)

            if has_tl and has_schnee:
                # Snowmelt risk: any interval with TL > 0 AND SCHNEE > 0
                mask = (sdf["TL"] > 0) & (sdf["SCHNEE"] > 0)
                risk = bool(mask.any())

        results[station_id] = {
            "snowmelt_risk": risk,
            "latest_temp": latest_temp,
            "latest_snow_depth": latest_snow,
        }
    return results


def aggregate_rainfall(df, freq):
    """Aggregate 10-min rainfall data to the chosen frequency by summing."""
    if df.empty:
        return df
    agg = (
        df.groupby([pd.Grouper(key="timestamp", freq=freq), "station_id", "station_name"])
        ["RR"]
        .sum()
        .reset_index()
    )
    return agg


def build_map(stations_meta, ei30_results=None):
    """Build a folium map with CircleMarkers colored/sized by alert level."""
    if not stations_meta:
        return folium.Map(location=[47.5, 13.5], zoom_start=7)

    # Center on the analysis point, zoom to show the 50 km buffer
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=9)

    # Draw the 50 km buffer circle
    folium.Circle(
        location=[CENTER_LAT, CENTER_LON],
        radius=BUFFER_KM * 1000,
        color="#3388ff",
        fill=True,
        fill_opacity=0.05,
        weight=2,
        dash_array="5",
        tooltip=f"{BUFFER_KM} km buffer",
    ).add_to(m)

    # Mark the center point
    folium.Marker(
        location=[CENTER_LAT, CENTER_LON],
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        tooltip="Kaindorf",
    ).add_to(m)

    # Determine max rainfall for radius scaling
    if ei30_results:
        max_rain = max((r["total_rainfall"] for r in ei30_results.values()), default=1.0)
    else:
        max_rain = 1.0
    if max_rain <= 0:
        max_rain = 1.0

    for sid, s in stations_meta.items():
        alt_text = f"{s['altitude']} m" if s["altitude"] is not None else "N/A"
        dist_text = f"{s['distance_km']} km" if "distance_km" in s else ""

        if ei30_results and sid in ei30_results:
            res = ei30_results[sid]
            color = res["alert_color"]
            # Scale radius: min 5, max 20, linear by total rainfall
            radius = 5 + 15 * min(res["total_rainfall"] / max_rain, 1.0)
            popup_text = (
                f"<b>{s['name']}</b><br>"
                f"Altitude: {alt_text}<br>"
                f"Distance: {dist_text}<br>"
                f"Total rainfall: {res['total_rainfall']} mm<br>"
                f"I30: {res['I30']} mm/h<br>"
                f"EI30: {res['EI30']} MJ\u00b7mm\u00b7ha\u207b\u00b9\u00b7h\u207b\u00b9<br>"
                f"Alert: {res['alert_label']}"
            )
        else:
            color = "#2ecc71"
            radius = 5
            popup_text = f"<b>{s['name']}</b><br>ID: {s['id']}<br>Altitude: {alt_text}<br>Distance: {dist_text}"

        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=s["name"],
        ).add_to(m)

    return m


def main():
    st.set_page_config(page_title="Rainfall – Erosion Alert", layout="wide")
    st.title("Rainfall Monitor – Erosion Alert")

    # Fetch station metadata
    with st.spinner("Loading station metadata..."):
        try:
            stations_meta = fetch_station_metadata()
        except Exception as e:
            st.error(f"Failed to fetch station metadata: {e}")
            return

    if not stations_meta:
        st.warning("No matching stations found in API metadata.")
        return

    # ── Sidebar controls ──
    st.sidebar.header("Settings")

    # Time range
    time_label = st.sidebar.radio(
        "Time range",
        options=list(TIME_RANGE_OPTIONS.keys()),
        index=4,  # default 72h
    )
    time_hours = TIME_RANGE_OPTIONS[time_label]

    # Aggregation
    agg_label = st.sidebar.radio(
        "Aggregation interval",
        options=list(AGGREGATION_OPTIONS.keys()),
        index=0,
    )
    agg_freq = AGGREGATION_OPTIONS[agg_label]

    # Station selection
    name_to_id = {s["name"]: sid for sid, s in stations_meta.items()}
    sorted_names = sorted(name_to_id.keys())
    selected_names = st.sidebar.multiselect(
        "Stations",
        options=sorted_names,
        default=sorted_names,
    )
    selected_station_ids = [name_to_id[n] for n in selected_names]

    if not selected_station_ids:
        st.info("Select at least one station in the sidebar.")
        return

    st.caption(f"Last {time_label} of precipitation from GeoSphere Austria TAWES stations")

    # ── Fetch main data (RR, TL, SCHNEE) ──
    station_tuple = tuple(selected_station_ids)
    with st.spinner("Fetching weather data..."):
        try:
            geojson = fetch_rainfall_data(station_tuple, hours=time_hours, parameters="RR,TL,SCHNEE")
        except Exception as e:
            st.error(f"Failed to fetch weather data: {e}")
            return

    df_raw = parse_geojson_to_df(
        geojson, stations_meta,
        parameter_defaults={"RR": 0.0, "TL": np.nan, "SCHNEE": np.nan},
    )

    if df_raw.empty:
        st.warning("No data returned for the selected stations and time range.")
        return

    # ── Compute EI30 (always from 10-min RR data) ──
    selected_meta = {sid: stations_meta[sid] for sid in selected_station_ids if sid in stations_meta}
    ei30_results = compute_all_stations_ei30(df_raw, selected_meta)

    # ── Fetch antecedent rainfall ──
    antecedent_totals = {}
    try:
        ant_geojson = fetch_antecedent_rainfall(station_tuple, window_hours=time_hours)
        antecedent_totals = compute_antecedent_totals(ant_geojson, selected_meta)
    except Exception:
        pass  # non-critical

    # ── Compute snowmelt risk ──
    snowmelt_results = compute_snowmelt_risk(df_raw, selected_meta)

    # ── Alert Banner ──
    alert_stations = {
        sid: res for sid, res in ei30_results.items()
        if res["alert_level"] != "green"
    }
    if alert_stations:
        sorted_alerts = sorted(alert_stations.items(), key=lambda x: x[1]["EI30"], reverse=True)
        for sid, res in sorted_alerts:
            name = selected_meta.get(sid, {}).get("name", sid)
            level = res["alert_level"]
            if level == "red":
                st.error(f"**{name}** — {res['alert_label']} (EI30 = {res['EI30']})")
            elif level == "orange":
                st.warning(f"**{name}** — {res['alert_label']} (EI30 = {res['EI30']})")
            elif level == "yellow":
                st.warning(f"**{name}** — {res['alert_label']} (EI30 = {res['EI30']})")
    else:
        st.success("All stations: No erosion risk (EI30 < 25)")

    # ── Map (after EI30 computation) ──
    st.subheader("Station Map")
    station_map = build_map(selected_meta, ei30_results)
    folium_static(station_map, width=900, height=450)

    # ── Aggregate for display ──
    df_agg = aggregate_rainfall(df_raw, agg_freq)

    # ── Time-series chart ──
    st.subheader(f"Precipitation ({agg_label} totals, last {time_label})")
    fig = px.bar(
        df_agg,
        x="timestamp",
        y="RR",
        color="station_name",
        barmode="group",
        labels={"RR": f"Rainfall ({agg_label}) [mm]", "timestamp": "Time"},
    )
    fig.update_layout(legend_title_text="Station", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative chart ──
    st.subheader(f"Cumulative Precipitation ({time_label})")
    df_agg_sorted = df_agg.sort_values("timestamp")
    df_agg_sorted["cumulative"] = df_agg_sorted.groupby("station_name")["RR"].cumsum()
    fig_cum = px.line(
        df_agg_sorted,
        x="timestamp",
        y="cumulative",
        color="station_name",
        labels={"cumulative": "Cumulative Rainfall [mm]", "timestamp": "Time"},
    )
    fig_cum.update_layout(legend_title_text="Station", hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Enhanced Summary Table ──
    st.subheader("Summary Statistics")
    summary_rows = []
    for sid in selected_station_ids:
        if sid not in selected_meta:
            continue
        name = selected_meta[sid]["name"]
        res = ei30_results.get(sid, {})
        ant = antecedent_totals.get(sid, 0.0)
        snow = snowmelt_results.get(sid, {})
        summary_rows.append({
            "Station": name,
            "Total (mm)": res.get("total_rainfall", 0.0),
            "I30 (mm/h)": res.get("I30", 0.0),
            "EI30": res.get("EI30", 0.0),
            "Alert": res.get("alert_label", "No risk"),
            "Antecedent 5d (mm)": ant,
            "Snowmelt risk": "Yes" if snow.get("snowmelt_risk") else "No",
            "Latest Temp (\u00b0C)": snow.get("latest_temp") if snow.get("latest_temp") is not None else "N/A",
            "Snow Depth (cm)": snow.get("latest_snow_depth") if snow.get("latest_snow_depth") is not None else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("EI30", ascending=False)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Context Data Section ──
    st.subheader("Context Data")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Antecedent Rainfall (5 days before window)**")
        ant_rows = []
        for sid in selected_station_ids:
            if sid not in selected_meta:
                continue
            name = selected_meta[sid]["name"]
            total = antecedent_totals.get(sid, 0.0)
            ant_rows.append({"Station": name, "5-day Antecedent (mm)": total})
        ant_df = pd.DataFrame(ant_rows)
        if not ant_df.empty:
            ant_df = ant_df.sort_values("5-day Antecedent (mm)", ascending=False)
        st.dataframe(ant_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Snowmelt Risk Assessment**")
        snow_rows = []
        for sid in selected_station_ids:
            if sid not in selected_meta:
                continue
            name = selected_meta[sid]["name"]
            snow = snowmelt_results.get(sid, {})
            temp_str = f"{snow['latest_temp']} \u00b0C" if snow.get("latest_temp") is not None else "N/A"
            snow_str = f"{snow['latest_snow_depth']} cm" if snow.get("latest_snow_depth") is not None else "N/A"
            snow_rows.append({
                "Station": name,
                "Snowmelt Risk": "Yes" if snow.get("snowmelt_risk") else "No",
                "Latest Temp": temp_str,
                "Snow Depth": snow_str,
            })
        snow_df = pd.DataFrame(snow_rows)
        if not snow_df.empty:
            snow_df = snow_df.sort_values("Snowmelt Risk", ascending=False)
        st.dataframe(snow_df, use_container_width=True, hide_index=True)

    # ── Raw data (expandable) ──
    with st.expander("Raw Aggregated Data"):
        st.dataframe(
            df_agg.sort_values(["station_name", "timestamp"]),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
