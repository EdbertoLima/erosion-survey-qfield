import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta, timezone
from pyproj import Transformer
import math
from io import BytesIO
import base64
from scipy.interpolate import Rbf
from PIL import Image, ImageColor


# ── API Configuration ──
API_HOST = "https://dataset.api.hub.geosphere.at"
API_VERSION = "v1"

DATA_SOURCES = {
    "TAWES Weather Station": {
        "resource_id": "tawes-v1-10min",
        "api_type": "station",
        "mode_data": "historical",
        "mode_meta": "current",
        "interval_minutes": 10,
        "request_params": "RR,TL,SCHNEE",
        "rain_param": "RR",
        "param_map": {"RR": "RR", "TL": "TL", "SCHNEE": "SCHNEE"},
        "description": "TAWES 10-min station data",
        "i30_correction": 1.0,  # 10-min intervals span exactly 30 min (3 intervals)
    },
    "Weather Station 10 min": {
        "resource_id": "klima-v2-10min",
        "api_type": "station",
        "mode_data": "historical",
        "mode_meta": "historical",
        "interval_minutes": 10,
        "request_params": "rr,tl,sh",
        "rain_param": "rr",
        "param_map": {"rr": "RR", "tl": "TL", "sh": "SCHNEE"},
        "description": "Klima v2 10-min station data",
        "i30_correction": 1.0,
    },
    "INCA Hourly": {
        "resource_id": "inca-v1-1h-1km",
        "api_type": "timeseries",
        "mode_data": "historical",
        "mode_meta": "historical",
        "interval_minutes": 60,
        "request_params": "RR,T2M",
        "rain_param": "RR",
        "param_map": {"RR": "RR", "T2M": "TL"},
        "description": "INCA 1h 1km grid (virtual stations from CSV)",
        "i30_correction": 1.0,  # already an approximation for hourly data
    },
}

# Storm delineation parameters (Wischmeier & Smith, 1978; Nearing et al., 2017)
STORM_GAP_HOURS = 6        # minimum dry gap separating storms
STORM_GAP_RAIN_MM = 1.27   # max rainfall during gap to count as "dry"
STORM_MIN_RAIN_MM = 12.7   # minimum total rain for an erosive storm
STORM_INTENSE_15MIN_MM = 6.0  # storms with >= this in ~15 min are included regardless

# Central analysis point and buffer radius
CENTER_LAT = 47.22517226258929
CENTER_LON = 15.911707948071635

CSV_PATH = "data/raw/messstellen_nlv.csv"

# CRS transformer: EPSG:31287 (MGI / Austria Lambert) → EPSG:4326 (WGS84)
_transformer = Transformer.from_crs("EPSG:31287", "EPSG:4326", always_xy=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


AGGREGATION_OPTIONS_10MIN = {
    "10 min": "10min",
    "30 min": "30min",
    "1 hour": "1h",
    "24 hours": "24h",
}

AGGREGATION_OPTIONS_HOURLY = {
    "1 hour": "1h",
    "3 hours": "3h",
    "6 hours": "6h",
    "24 hours": "24h",
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


# SCS/NRCS Antecedent Moisture Condition (5-day antecedent rainfall)
AMC_CLASSES = [
    (28.0, "AMC III", "Wet \u2014 high runoff potential"),
    (13.0, "AMC II", "Moderate"),
    (0.0, "AMC I", "Dry \u2014 low runoff potential"),
]


def classify_amc(antecedent_mm):
    """Classify 5-day antecedent rainfall into AMC I/II/III (SCS/NRCS)."""
    for threshold, cls, description in AMC_CLASSES:
        if antecedent_mm >= threshold:
            return cls, description
    return "AMC I", "Dry \u2014 low runoff potential"


def build_api_url(api_type, mode, resource_id):
    """Construct a GeoSphere dataset API URL."""
    return f"{API_HOST}/{API_VERSION}/{api_type}/{mode}/{resource_id}"


# ── Station Metadata ──

@st.cache_data(ttl=600)
def load_csv_stations(buffer_km):
    """Load virtual stations from CSV, convert EPSG:31287 → WGS84, filter by buffer."""
    df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1")
    stations = {}
    for _, row in df.iterrows():
        try:
            x = float(str(row["xrkko08"]).replace(",", "."))
            y = float(str(row["yhkko09"]).replace(",", "."))
        except (ValueError, TypeError):
            continue
        lon, lat = _transformer.transform(x, y)
        dist = haversine_km(CENTER_LAT, CENTER_LON, lat, lon)
        if dist <= buffer_km:
            sid = str(row["dbmsnr"])
            stations[sid] = {
                "id": sid,
                "name": str(row.get("mstnam02", sid)),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "altitude": int(float(str(row["mpmua04"]).replace(",", "."))) if pd.notna(row.get("mpmua04")) else None,
                "state": str(row.get("gew03", "")),
                "distance_km": round(dist, 1),
            }
    return stations


@st.cache_data(ttl=600)
def fetch_station_metadata_api(resource_id, api_type, mode_meta, buffer_km):
    """Fetch station metadata from API and filter by buffer distance."""
    url = build_api_url(api_type, mode_meta, resource_id) + "/metadata"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    stations = {}
    for s in data.get("stations", []):
        dist = haversine_km(CENTER_LAT, CENTER_LON, s["lat"], s["lon"])
        if dist <= buffer_km:
            sid = str(s["id"])
            stations[sid] = {
                "id": sid,
                "name": s.get("name", sid),
                "lat": s["lat"],
                "lon": s["lon"],
                "altitude": s.get("altitude", None),
                "state": s.get("state", ""),
                "distance_km": round(dist, 1),
            }
    return stations


def get_stations(source_name, buffer_km):
    """Get station metadata for the selected data source."""
    cfg = DATA_SOURCES[source_name]
    if cfg["api_type"] == "timeseries":
        return load_csv_stations(buffer_km)
    return fetch_station_metadata_api(
        cfg["resource_id"], cfg["api_type"], cfg["mode_meta"], buffer_km
    )


# ── Data Fetching ──

@st.cache_data(ttl=300)
def _fetch_station_data(resource_id, mode, station_ids_tuple, start_str, end_str, parameters):
    """Fetch data from a station endpoint."""
    url = build_api_url("station", mode, resource_id)
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
def _fetch_inca_data(lat_lon_tuple, station_ids_tuple, start_str, end_str, parameters):
    """Fetch data from INCA timeseries endpoint using lat/lon coordinates."""
    url = build_api_url("timeseries", "historical", "inca-v1-1h-1km")
    # Use list of tuples to support repeated lat_lon parameter
    param_list = [
        ("parameters", parameters),
        ("start", start_str),
        ("end", end_str),
        ("output_format", "geojson"),
    ]
    for ll in lat_lon_tuple:
        param_list.append(("lat_lon", ll))

    resp = requests.get(url, params=param_list, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Inject virtual station IDs into features (order matches lat_lon order)
    for i, feature in enumerate(data.get("features", [])):
        if i < len(station_ids_tuple):
            feature.setdefault("properties", {})["station"] = station_ids_tuple[i]
    return data


def fetch_weather_data(source_name, station_ids, stations_meta, hours=None,
                       parameters=None, start_str=None, end_str=None):
    """Unified data-fetch dispatcher."""
    cfg = DATA_SOURCES[source_name]
    if parameters is None:
        parameters = cfg["request_params"]

    if start_str is None or end_str is None:
        start_str, end_str = _time_window(hours)
    station_tuple = tuple(station_ids)

    if cfg["api_type"] == "timeseries":
        lat_lon_tuple = tuple(
            f"{stations_meta[sid]['lat']},{stations_meta[sid]['lon']}"
            for sid in station_ids
        )
        return _fetch_inca_data(lat_lon_tuple, station_tuple, start_str, end_str, parameters)
    return _fetch_station_data(
        cfg["resource_id"], cfg["mode_data"], station_tuple, start_str, end_str, parameters,
    )


def fetch_antecedent_data(source_name, station_ids, stations_meta, window_hours=None,
                          window_start_dt=None):
    """Fetch 5-day (120 h) antecedent rainfall ending at the start of the main window."""
    cfg = DATA_SOURCES[source_name]
    if window_start_dt is not None:
        window_start = window_start_dt
    else:
        now = datetime.now(timezone.utc)
        window_start = (now - timedelta(hours=window_hours)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
    ant_start = (window_start - timedelta(hours=120)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    start_str = ant_start.strftime("%Y-%m-%dT%H:%M")
    end_str = window_start.strftime("%Y-%m-%dT%H:%M")
    station_tuple = tuple(station_ids)
    rain_param = cfg["rain_param"]

    if cfg["api_type"] == "timeseries":
        lat_lon_tuple = tuple(
            f"{stations_meta[sid]['lat']},{stations_meta[sid]['lon']}"
            for sid in station_ids
        )
        return _fetch_inca_data(lat_lon_tuple, station_tuple, start_str, end_str, rain_param)
    return _fetch_station_data(
        cfg["resource_id"], cfg["mode_data"], station_tuple, start_str, end_str, rain_param,
    )


# ── Parsing ──

def parse_geojson_to_df(geojson, stations_meta, param_map, parameter_defaults=None):
    """Parse GeoSphere GeoJSON response into a DataFrame with normalized column names.

    param_map: dict mapping source-specific parameter names to standard names,
               e.g. {"rr": "RR", "tl": "TL", "sh": "SCHNEE"}.
    parameter_defaults: dict mapping *standard* names to default values for nulls.
    """
    if parameter_defaults is None:
        parameter_defaults = {"RR": 0.0}

    timestamps = geojson.get("timestamps", [])
    features = geojson.get("features", [])
    rows = []

    for feature in features:
        props = feature.get("properties", {})
        station_id = str(props.get("station", ""))
        parameters_data = props.get("parameters", {})
        station_name = stations_meta.get(station_id, {}).get("name", station_id)

        param_names = list(parameters_data.keys())
        if not param_names:
            continue

        data_length = len(parameters_data[param_names[0]].get("data", []))

        for i in range(min(data_length, len(timestamps))):
            row = {
                "timestamp": timestamps[i],
                "station_id": station_id,
                "station_name": station_name,
            }
            for pname in param_names:
                std_name = param_map.get(pname, pname)
                raw_val = (
                    parameters_data[pname]["data"][i]
                    if i < len(parameters_data[pname].get("data", []))
                    else None
                )
                default = parameter_defaults.get(std_name, np.nan)
                row[std_name] = raw_val if raw_val is not None else default
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ── Computation ──

def _compute_energy(intensity, rr, energy_equation):
    """Compute unit energy and interval energy arrays."""
    if energy_equation == "USLE":
        # Wischmeier & Smith (1978), Eq. 1 in Nearing et al. (2017)
        e_unit = np.where(
            intensity > 0,
            np.minimum(0.119 + 0.0873 * np.log10(np.maximum(intensity, 1e-6)), 0.283),
            0.0,
        )
    else:
        # RUSLE2: McGregor et al. (1995), Eq. 7 in Nearing et al. (2017)
        e_unit = 0.29 * (1.0 - 0.72 * np.exp(-0.082 * intensity))
    return e_unit, e_unit * rr


def _compute_i30(rr, interval_minutes, i30_correction=1.0):
    """Compute maximum 30-min intensity from rainfall array."""
    intervals_per_hour = 60.0 / interval_minutes
    if interval_minutes <= 30:
        n = max(1, int(30 / interval_minutes))
        if len(rr) >= n:
            rolling_30 = np.convolve(rr, np.ones(n), mode="valid")
            I30 = float(np.max(rolling_30) * 2.0)  # 30-min sum → mm/h
        else:
            I30 = float(np.sum(rr) * intervals_per_hour) if len(rr) > 0 else 0.0
    else:
        # Hourly or coarser: approximate I30 as max hourly intensity
        intensity = rr * intervals_per_hour
        I30 = float(np.max(intensity)) if len(rr) > 0 else 0.0
    return I30 * i30_correction


def delineate_storms(rr, interval_minutes):
    """Split a rainfall array into individual storm events (USLE methodology).

    A storm break is a period of >= STORM_GAP_HOURS where < STORM_GAP_RAIN_MM falls.
    Returns list of (start_idx, end_idx) tuples.
    """
    n = len(rr)
    if n == 0:
        return []

    gap_intervals = max(1, int(STORM_GAP_HOURS * 60 / interval_minutes))

    if n <= gap_intervals:
        return [(0, n)] if np.sum(rr) > 0 else []

    # Cumulative sum for fast window calculations
    cumsum = np.concatenate([[0.0], np.cumsum(rr)])
    # rolling_sum[i] = sum(rr[i : i + gap_intervals])
    rolling_sum = cumsum[gap_intervals:] - cumsum[:-gap_intervals]

    # Mark intervals that lie inside any dry-gap window
    in_gap = np.zeros(n, dtype=bool)
    for i in range(len(rolling_sum)):
        if rolling_sum[i] < STORM_GAP_RAIN_MM:
            in_gap[i: i + gap_intervals] = True

    # Group consecutive non-gap intervals into storms
    storms = []
    start = None
    for i in range(n):
        if not in_gap[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                storms.append((start, i))
                start = None
    if start is not None:
        storms.append((start, n))

    return storms


def _storm_is_erosive(rr, interval_minutes):
    """Check if a storm meets the minimum erosive threshold (Wischmeier, 1959)."""
    total = float(np.sum(rr))
    if total >= STORM_MIN_RAIN_MM:
        return True
    # Also include if >= 6 mm fell within ~15 min
    if interval_minutes <= 30:
        n_15 = max(1, math.ceil(15 / interval_minutes))
        if len(rr) >= n_15:
            rolling_15 = np.convolve(rr, np.ones(n_15), mode="valid")
            if float(np.max(rolling_15)) >= STORM_INTENSE_15MIN_MM:
                return True
    return False


def compute_station_erosivity(df_station, interval_minutes, energy_equation, i30_correction):
    """Delineate storms and compute per-storm EI30 for one station.

    Returns dict with aggregate results plus a ``storms`` list.
    """
    empty = {
        "total_rainfall": 0.0, "I30": 0.0, "E_total": 0.0,
        "EI30": 0.0, "R_sum": 0.0,
        "alert_level": "green", "alert_label": "No risk", "alert_color": "#2ecc71",
        "storms": [], "n_erosive_storms": 0,
    }
    if df_station.empty or "RR" not in df_station.columns:
        return empty

    df = df_station.sort_values("timestamp").reset_index(drop=True)
    rr_all = df["RR"].values.astype(float)
    timestamps = df["timestamp"].values
    intervals_per_hour = 60.0 / interval_minutes
    n = len(rr_all)

    # Delineate storms
    storm_slices = delineate_storms(rr_all, interval_minutes)
    if not storm_slices:
        return empty

    # Is the last storm potentially still ongoing?
    last_end = storm_slices[-1][1] if storm_slices else 0

    storm_results = []
    for s_start, s_end in storm_slices:
        rr = rr_all[s_start:s_end]
        if float(np.sum(rr)) <= 0:
            continue

        is_ongoing = (s_end >= n)  # extends to end of data
        is_erosive = _storm_is_erosive(rr, interval_minutes)

        intensity = rr * intervals_per_hour
        _, e_interval = _compute_energy(intensity, rr, energy_equation)
        E_total = float(np.sum(e_interval))
        I30 = _compute_i30(rr, interval_minutes, i30_correction)
        EI30 = E_total * I30

        level, label, color = get_alert_level(EI30) if is_erosive else ("green", "No risk", "#2ecc71")

        storm_results.append({
            "start": pd.Timestamp(timestamps[s_start]),
            "end": pd.Timestamp(timestamps[min(s_end - 1, n - 1)]),
            "duration_hours": round((s_end - s_start) * interval_minutes / 60.0, 1),
            "total_rain": round(float(np.sum(rr)), 2),
            "I30": round(I30, 2),
            "E_total": round(E_total, 4),
            "EI30": round(EI30, 2),
            "is_ongoing": is_ongoing,
            "is_erosive": is_erosive,
            "alert_level": level,
            "alert_label": label,
            "alert_color": color,
        })

    # Aggregate: alert is driven by the worst *erosive* storm
    erosive = [s for s in storm_results if s["is_erosive"]]
    if erosive:
        worst = max(erosive, key=lambda s: s["EI30"])
        alert_level, alert_label, alert_color = worst["alert_level"], worst["alert_label"], worst["alert_color"]
        max_ei30 = worst["EI30"]
    else:
        alert_level, alert_label, alert_color = "green", "No risk", "#2ecc71"
        max_ei30 = 0.0

    return {
        "total_rainfall": round(sum(s["total_rain"] for s in storm_results), 2),
        "I30": round(max((s["I30"] for s in storm_results), default=0.0), 2),
        "E_total": round(sum(s["E_total"] for s in storm_results), 4),
        "EI30": round(max_ei30, 2),
        "R_sum": round(sum(s["EI30"] for s in erosive), 2),
        "alert_level": alert_level,
        "alert_label": alert_label,
        "alert_color": alert_color,
        "storms": storm_results,
        "n_erosive_storms": len(erosive),
    }


def compute_all_stations_ei30(df_raw, stations_meta, interval_minutes=10,
                              energy_equation="RUSLE2", i30_correction=1.0):
    """Compute storm-delineated EI30 for every station."""
    results = {}
    for station_id in stations_meta:
        df_s = df_raw[df_raw["station_id"] == station_id].sort_values("timestamp")
        results[station_id] = compute_station_erosivity(
            df_s, interval_minutes, energy_equation, i30_correction,
        )
    return results


def compute_antecedent_totals(geojson, stations_meta, param_map):
    """Compute 5-day antecedent rainfall totals per station."""
    df = parse_geojson_to_df(geojson, stations_meta, param_map, parameter_defaults={"RR": 0.0})
    totals = {}
    if df.empty:
        return totals
    for station_id in stations_meta:
        sdf = df[df["station_id"] == station_id]
        totals[station_id] = round(float(sdf["RR"].sum()), 2) if not sdf.empty else 0.0
    return totals


def compute_snowmelt_risk(df_context, stations_meta):
    """Determine snowmelt risk per station from TL and SCHNEE data."""
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
                mask = (sdf["TL"] > 0) & (sdf["SCHNEE"] > 0)
                risk = bool(mask.any())

        results[station_id] = {
            "snowmelt_risk": risk,
            "latest_temp": latest_temp,
            "latest_snow_depth": latest_snow,
        }
    return results


def create_interpolation_heatmap(stations_meta, ei30_results, grid_resolution=100):
    """
    Create an IDW-like spatial interpolation heatmap of EI30 values.
    Returns a base64 encoded PNG and the bounds for map overlay.
    """
    points = []
    values = []
    for sid, result in ei30_results.items():
        if sid in stations_meta:
            points.append([stations_meta[sid]['lon'], stations_meta[sid]['lat']])
            values.append(result.get('EI30', 0.0))

    if len(points) < 3:
        return None, None  # Need at least 3 points for interpolation

    points = np.array(points)
    values = np.array(values)

    # Define grid bounds based on station locations
    lon_min, lat_min = points.min(axis=0)
    lon_max, lat_max = points.max(axis=0)
    
    # Add a small buffer to the bounds
    lon_buf = (lon_max - lon_min) * 0.1
    lat_buf = (lat_max - lat_min) * 0.1
    bounds = [[lat_min - lat_buf, lon_min - lon_buf], [lat_max + lat_buf, lon_max + lon_buf]]

    # Create a grid: y (lat) descending, x (lon) ascending
    grid_lat, grid_lon = np.mgrid[lat_max + lat_buf:lat_min - lat_buf:complex(0, grid_resolution), 
                                  lon_min - lon_buf:lon_max + lon_buf:complex(0, grid_resolution)]

    # RBF interpolation (x=lon, y=lat)
    rbf = Rbf(points[:, 0], points[:, 1], values, function='linear')
    interp_values = rbf(grid_lon, grid_lat)
    
    # Normalize values for coloring
    vmin, vmax = np.nanmin(interp_values), np.nanmax(interp_values)
    if vmax <= vmin:
        vmax = vmin + 1.0
    # Use np.nan_to_num to handle potential NaNs from interpolation
    normalized_values = np.nan_to_num((interp_values - vmin) / (vmax - vmin))
    
    # Create an image from the interpolated data
    img = Image.new('RGBA', (grid_resolution, grid_resolution))
    pixels = img.load()
    
    # Colormap (e.g., from green to red)
    cmap = [ImageColor.getrgb(c) for c in ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']]
    
    for i in range(grid_resolution):  # Corresponds to x, or columns (lon)
        for j in range(grid_resolution):  # Corresponds to y, or rows (lat)
            val = normalized_values[j, i] # index as (row, col) -> (lat, lon)
            
            # Simple linear interpolation between colormap colors
            idx = val * (len(cmap) - 1)
            idx_floor = int(idx)
            idx_ceil = min(idx_floor + 1, len(cmap) - 1)
            
            if idx_floor == idx_ceil:
                color = cmap[idx_floor]
            else:
                interp = idx - idx_floor
                color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(cmap[idx_floor], cmap[idx_ceil]))

            # Set opacity based on value (fade out low values)
            opacity = int(255 * max(0, val - 0.1) / 0.9) if val > 0.1 else 0
            pixels[i, j] = color + (opacity,)

    # Convert image to base64 string
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str, bounds



def aggregate_rainfall(df, freq):
    """Aggregate rainfall data to the chosen frequency by summing."""
    if df.empty:
        return df
    return (
        df.groupby([pd.Grouper(key="timestamp", freq=freq), "station_id", "station_name"])
        ["RR"]
        .sum()
        .reset_index()
    )


# ── Map ──

def build_map(stations_meta, ei30_results=None, heatmap_img=None, heatmap_bounds=None, buffer_km=25):
    """Build a folium map with CircleMarkers colored/sized by alert level."""
    if not stations_meta:
        return folium.Map(location=[47.5, 13.5], zoom_start=7)

    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=9)

    if heatmap_img and heatmap_bounds:
        img_overlay = folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{heatmap_img}",
            bounds=heatmap_bounds,
            opacity=0.6,
            name="EI30 Heatmap",
        )
        img_overlay.add_to(m)
        folium.LayerControl().add_to(m)

    # Draw buffer circles
    drawn_radii = []
    # Draw the main, user-selected buffer first
    folium.Circle(
        location=[CENTER_LAT, CENTER_LON],
        radius=buffer_km * 1000,
        color="#e74c3c", fill=True, fill_opacity=0.05, weight=2, dash_array="5",
        tooltip=f"{buffer_km} km buffer (selected)",
    ).add_to(m)
    drawn_radii.append(buffer_km)

    # Draw other reference circles if they are different
    for radius_km_ref, label in [(50, "50 km buffer"), (25, "25 km buffer"), (10, "10 km buffer")]:
        if radius_km_ref not in drawn_radii:
            folium.Circle(
                location=[CENTER_LAT, CENTER_LON],
                radius=radius_km_ref * 1000,
                color="#3388ff", fill=True, fill_opacity=0.05, weight=1, dash_array="5",
                tooltip=label,
            ).add_to(m)

    folium.Marker(
        location=[CENTER_LAT, CENTER_LON],
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        tooltip="Kaindorf",
    ).add_to(m)

    max_rain = 1.0
    if ei30_results:
        max_rain = max((r["total_rainfall"] for r in ei30_results.values()), default=1.0)
    if max_rain <= 0:
        max_rain = 1.0

    for sid, s in stations_meta.items():
        alt_text = f"{s['altitude']} m" if s.get("altitude") is not None else "N/A"
        dist_text = f"{s['distance_km']} km" if "distance_km" in s else ""

        if ei30_results and sid in ei30_results:
            res = ei30_results[sid]
            color = res["alert_color"]
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
            popup_text = (
                f"<b>{s['name']}</b><br>ID: {s['id']}<br>"
                f"Altitude: {alt_text}<br>Distance: {dist_text}"
            )

        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=s["name"],
        ).add_to(m)

    return m


# ── Main ──

def main():
    st.set_page_config(page_title="Rainfall \u2013 Erosion Alert", layout="wide")
    st.title("Rainfall Monitor \u2013 Erosion Alert")

    # ── Sidebar controls ──
    st.sidebar.header("Settings")

    # Data source
    source_name = st.sidebar.selectbox(
        "Data Source",
        options=list(DATA_SOURCES.keys()),
        index=0,
    )
    source_cfg = DATA_SOURCES[source_name]
    interval_minutes = source_cfg["interval_minutes"]

    buffer_km = st.sidebar.slider("Buffer Radius (km)", 5, 100, 25)


    # Time range
    st.sidebar.markdown("##### Custom Date Range")
    today = datetime.now(timezone.utc)
    default_start = today - timedelta(days=3)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=default_start)
    with col2:
        end_date = st.date_input("End date", value=today)

    # Convert to datetime objects for fetching
    start_dt_utc = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt_utc = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    start_str = start_dt_utc.strftime("%Y-%m-%dT%H:%M")
    end_str = end_dt_utc.strftime("%Y-%m-%dT%H:%M")

    # Aggregation (options depend on data resolution)
    agg_options = AGGREGATION_OPTIONS_HOURLY if interval_minutes >= 60 else AGGREGATION_OPTIONS_10MIN
    agg_label = st.sidebar.radio(
        "Aggregation interval",
        options=list(agg_options.keys()),
        index=0,
    )
    agg_freq = agg_options[agg_label]

    # Energy equation for EI30
    energy_equation = st.sidebar.radio(
        "Energy equation",
        options=["RUSLE2", "USLE"],
        index=0,
        help=(
            "**RUSLE2** (recommended): e = 0.29[1 - 0.72 exp(-0.082 i)], "
            "McGregor et al. (1995).\n\n"
            "**USLE**: e = 0.119 + 0.0873 log\u2081\u2080(i), capped at 0.283, "
            "Wischmeier & Smith (1978).\n\n"
            "The former RUSLE equation (b=0.05) is **not recommended** as it "
            "underestimates erosivity by ~14% (Nearing et al., 2017)."
        ),
    )

    show_heatmap = st.sidebar.checkbox("Show EI30 Heatmap", value=False)

    # ── Station metadata ──
    with st.spinner("Loading station metadata..."):
        try:
            stations_meta = get_stations(source_name, buffer_km)
        except Exception as e:
            st.error(f"Failed to load station metadata: {e}")
            return

    if not stations_meta:
        st.warning("No matching stations found within the buffer zone.")
        return

    # Station selection
    name_to_id = {s["name"]: sid for sid, s in stations_meta.items()}
    sorted_names = sorted(name_to_id.keys())
    selected_names = st.sidebar.multiselect("Stations", options=sorted_names, default=sorted_names)
    selected_station_ids = [name_to_id[n] for n in selected_names]

    if not selected_station_ids:
        st.info("Select at least one station in the sidebar.")
        return

    st.caption(
        f"Precipitation from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} from GeoSphere Austria \u2014 {source_cfg['description']}"
    )

    if source_cfg["api_type"] == "timeseries":
        st.info(
            "INCA grid data: Virtual stations created based on hydrographic station locations "
            "(https://ehyd.gv.at/). I30 is approximated from hourly data and may underestimate "
            "sub-hourly peak intensities."
        )

    # ── Fetch main data ──
    selected_meta = {
        sid: stations_meta[sid] for sid in selected_station_ids if sid in stations_meta
    }

    with st.spinner("Fetching weather data..."):
        try:
            geojson = fetch_weather_data(source_name, selected_station_ids, selected_meta, start_str=start_str, end_str=end_str)
        except Exception as e:
            st.error(f"Failed to fetch weather data: {e}")
            return

    param_map = source_cfg["param_map"]
    df_raw = parse_geojson_to_df(
        geojson, selected_meta, param_map,
        parameter_defaults={"RR": 0.0, "TL": np.nan, "SCHNEE": np.nan},
    )

    if df_raw.empty:
        st.warning("No data returned for the selected stations and time range.")
        return

    # ── EI30 (storm-delineated, per USLE methodology) ──
    i30_correction = source_cfg["i30_correction"]
    ei30_results = compute_all_stations_ei30(
        df_raw, selected_meta, interval_minutes, energy_equation, i30_correction,
    )

    # ── Antecedent rainfall ──
    antecedent_totals = {}
    try:
        ant_geojson = fetch_antecedent_data(
            source_name, selected_station_ids, selected_meta, window_start_dt=start_dt_utc,
        )
        antecedent_totals = compute_antecedent_totals(ant_geojson, selected_meta, param_map)
    except Exception:
        pass  # non-critical

    # ── Snowmelt risk ──
    snowmelt_results = compute_snowmelt_risk(df_raw, selected_meta)

    # ── Field Survey Recommendation ──
    now_utc = datetime.now(timezone.utc)
    survey_needed = False
    ongoing_events = []
    recent_events = []  # erosive storms ended within last 24 h

    for sid, res in ei30_results.items():
        name = selected_meta.get(sid, {}).get("name", sid)
        for storm in res.get("storms", []):
            if not storm["is_erosive"]:
                continue
            if storm["is_ongoing"]:
                ongoing_events.append((name, sid, storm))
                survey_needed = True
            else:
                hours_ago = (now_utc - storm["end"].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 3600
                if hours_ago <= 24:
                    recent_events.append((name, sid, storm, round(hours_ago, 1)))
                    survey_needed = True

    st.subheader("Field Survey Recommendation")
    if ongoing_events:
        for name, sid, storm in ongoing_events:
            st.error(
                f"**Erosive event in progress at {name}** \u2014 "
                f"{storm['alert_label']} (EI30 = {storm['EI30']}). "
                f"Total so far: {storm['total_rain']} mm, I30: {storm['I30']} mm/h. "
                f"Monitor conditions and plan survey after event ends."
            )
    if recent_events:
        for name, sid, storm, hours_ago in sorted(recent_events, key=lambda x: x[2]["EI30"], reverse=True):
            st.warning(
                f"**Field survey recommended at {name}** \u2014 "
                f"{storm['alert_label']} (EI30 = {storm['EI30']}). "
                f"Event ended ~{hours_ago:.0f}h ago "
                f"({storm['end'].strftime('%Y-%m-%d %H:%M')} UTC). "
                f"Rain: {storm['total_rain']} mm in {storm['duration_hours']}h, "
                f"I30: {storm['I30']} mm/h."
            )
    if not survey_needed:
        # Check for any alert-level stations (non-green)
        alert_stations = {
            sid: res for sid, res in ei30_results.items() if res["alert_level"] != "green"
        }
        if alert_stations:
            for sid, res in sorted(alert_stations.items(), key=lambda x: x[1]["EI30"], reverse=True):
                name = selected_meta.get(sid, {}).get("name", sid)
                st.info(
                    f"**{name}** \u2014 {res['alert_label']} (EI30 = {res['EI30']}), "
                    f"but erosive events ended > 24h ago."
                )
        else:
            st.success("No erosive storm events detected in the selected time window. No field survey needed.")

    # ── Storm Events Detail ──
    all_storms = []
    for sid, res in ei30_results.items():
        name = selected_meta.get(sid, {}).get("name", sid)
        for storm in res.get("storms", []):
            all_storms.append({
                "Station": name,
                "Start (UTC)": storm["start"].strftime("%Y-%m-%d %H:%M"),
                "End (UTC)": storm["end"].strftime("%Y-%m-%d %H:%M") + (" *" if storm["is_ongoing"] else ""),
                "Duration (h)": storm["duration_hours"],
                "Rain (mm)": storm["total_rain"],
                "I30 (mm/h)": storm["I30"],
                "EI30": storm["EI30"],
                "Erosive": "Yes" if storm["is_erosive"] else "No",
                "Alert": storm["alert_label"],
            })

    if all_storms:
        with st.expander(f"Storm Events ({len(all_storms)} storms detected)", expanded=bool(survey_needed)):
            st.caption(
                f"Storm delineation: {STORM_GAP_HOURS}h gap with < {STORM_GAP_RAIN_MM} mm. "
                f"Erosive threshold: >= {STORM_MIN_RAIN_MM} mm total "
                f"(or >= {STORM_INTENSE_15MIN_MM} mm in 15 min). "
                f"'*' = possibly ongoing."
            )
            storm_df = pd.DataFrame(all_storms)
            storm_df = storm_df.sort_values("EI30", ascending=False)
            st.dataframe(storm_df, use_container_width=True, hide_index=True)

    # ── Interpolation Heatmap ──
    heatmap_img, heatmap_bounds = None, None
    if show_heatmap:
        with st.spinner("Generating interpolation heatmap..."):
            if len(selected_station_ids) < 3:
                st.warning("Heatmap requires at least 3 stations with data to be selected.")
            else:
                heatmap_img, heatmap_bounds = create_interpolation_heatmap(selected_meta, ei30_results)
                if not heatmap_img:
                    st.warning("Could not generate heatmap. Not enough data or an error occurred during interpolation.")

    # ── Map ──
    st.subheader("Station Map")
    station_map = build_map(selected_meta, ei30_results, heatmap_img, heatmap_bounds, buffer_km)
    folium_static(station_map, width=900, height=450)

    # ── Aggregate for display ──
    df_agg = aggregate_rainfall(df_raw, agg_freq)

    # ── Time-series chart ──
    st.subheader(f"Precipitation ({agg_label} totals)")
    fig = px.bar(
        df_agg, x="timestamp", y="RR", color="station_name", barmode="group",
        labels={"RR": f"Rainfall ({agg_label}) [mm]", "timestamp": "Time"},
    )
    fig.update_layout(legend_title_text="Station", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative chart ──
    st.subheader("Cumulative Precipitation")
    df_agg_sorted = df_agg.sort_values("timestamp")
    df_agg_sorted["cumulative"] = df_agg_sorted.groupby("station_name")["RR"].cumsum()
    fig_cum = px.line(
        df_agg_sorted, x="timestamp", y="cumulative", color="station_name",
        labels={"cumulative": "Cumulative Rainfall [mm]", "timestamp": "Time"},
    )
    fig_cum.update_layout(legend_title_text="Station", hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Hyetograph (Intensity Chart) ──
    st.subheader("Rainfall Intensity")
    # Use a copy to avoid Streamlit's "modified" warning
    df_intensity = df_raw.copy()
    df_intensity["intensity_mmh"] = df_intensity["RR"] * (60.0 / interval_minutes)
    fig_hyeto = px.bar(
        df_intensity, x="timestamp", y="intensity_mmh", color="station_name", barmode="group",
        labels={"intensity_mmh": "Intensity [mm/h]", "timestamp": "Time"},
    )
    fig_hyeto.update_layout(legend_title_text="Station", hovermode="x unified")
    st.plotly_chart(fig_hyeto, use_container_width=True)

    # ── Summary Table ──
    st.subheader("Summary Statistics")
    summary_rows = []
    for sid in selected_station_ids:
        if sid not in selected_meta:
            continue
        name = selected_meta[sid]["name"]
        res = ei30_results.get(sid, {})
        ant = antecedent_totals.get(sid, 0.0)
        amc_class, _ = classify_amc(ant)
        snow = snowmelt_results.get(sid, {})
        summary_rows.append({
            "Station": name,
            "Storms": res.get("n_erosive_storms", 0),
            "Total (mm)": res.get("total_rainfall", 0.0),
            "I30 (mm/h)": res.get("I30", 0.0),
            "Max EI30": res.get("EI30", 0.0),
            "R sum": res.get("R_sum", 0.0),
            "Alert": res.get("alert_label", "No risk"),
            "Antecedent 5d (mm)": ant,
            "AMC Class": amc_class,
            "Snowmelt risk": "Yes" if snow.get("snowmelt_risk") else "No",
            "Latest Temp (\u00b0C)": snow.get("latest_temp") if snow.get("latest_temp") is not None else "N/A",
            "Snow Depth (cm)": snow.get("latest_snow_depth") if snow.get("latest_snow_depth") is not None else "N/A",
        })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Max EI30", ascending=False)
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
            ant = antecedent_totals.get(sid, 0.0)
            amc_class, amc_desc = classify_amc(ant)
            ant_rows.append({
                "Station": selected_meta[sid]["name"],
                "5-day Antecedent (mm)": ant,
                "AMC Class": amc_class,
                "Description": amc_desc,
            })
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
            snow = snowmelt_results.get(sid, {})
            temp_str = f"{snow['latest_temp']} \u00b0C" if snow.get("latest_temp") is not None else "N/A"
            snow_str = f"{snow['latest_snow_depth']} cm" if snow.get("latest_snow_depth") is not None else "N/A"
            snow_rows.append({
                "Station": selected_meta[sid]["name"],
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
            use_container_width=True, hide_index=True,
        )

    # ── References ──
    st.divider()
    st.subheader("References")
    st.markdown(
        f"**Energy equation:** {energy_equation} | "
        f"**Storm delineation:** {STORM_GAP_HOURS}h gap / {STORM_GAP_RAIN_MM} mm | "
        f"**Erosive threshold:** {STORM_MIN_RAIN_MM} mm\n\n"
        "- Nearing, M.A., Yin, S., Borrelli, P., Polyakov, V.O. (2017). "
        "Rainfall erosivity: An historical review. "
        "*Catena*, 157, 357\u2013362. "
        "[doi:10.1016/j.catena.2017.06.004]"
        "(https://doi.org/10.1016/j.catena.2017.06.004)\n"
        "- Wischmeier, W.H., Smith, D.D. (1978). "
        "Predicting rainfall erosion losses. "
        "*USDA Agriculture Handbook* No. 537. "
        "(Storm delineation: 6h gap with < 1.27 mm; "
        "erosive threshold: 12.7 mm or 6 mm in 15 min)\n"
        "- McGregor, K.C., Bingner, R.L., Bowie, A.J., Foster, G.R. (1995). "
        "Erosivity index values for northern Mississippi. "
        "*Trans. ASAE*, 38(4), 1039\u20131047.\n"
        "- Brown, L.C., Foster, G.R. (1987). "
        "Storm erosivity using idealized intensity distributions. "
        "*Trans. ASAE*, 30(2), 379\u2013386.\n"
        "- Wischmeier, W.H. (1959). "
        "A rainfall erosion index for a universal soil loss equation. "
        "*Soil Sci. Soc. Am. Proc.*, 23, 322\u2013326.\n"
    )


if __name__ == "__main__":
    main()
