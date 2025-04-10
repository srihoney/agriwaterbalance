import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
import io
import math

# --------------------------------------------------------------------------------
# 1. Large Kc Database
# --------------------------------------------------------------------------------
CROP_DATABASE = {
    "Alfalfa": {"Kc_ini": 0.40, "Kc_mid": 1.20, "Kc_end": 1.15, "Kcb_mid": 1.15, "Kcb_end": 1.10, "total_days_default": 150},
    "Almond": {"Kc_ini": 0.40, "Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60, "total_days_default": 240},
    "Apple/Pear": {"Kc_ini": 0.45, "Kc_mid": 1.15, "Kc_end": 0.70, "Kcb_mid": 1.10, "Kcb_end": 0.65, "total_days_default": 250},
    "Banana": {"Kc_ini": 0.50, "Kc_mid": 1.10, "Kc_end": 1.00, "Kcb_mid": 1.05, "Kcb_end": 0.95, "total_days_default": 330},
    "Barley": {"Kc_ini": 0.30, "Kc_mid": 1.15, "Kc_end": 0.25, "Kcb_mid": 1.10, "Kcb_end": 0.20, "total_days_default": 120},
    "Basil": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.80, "Kcb_mid": 0.95, "Kcb_end": 0.75, "total_days_default": 70},
    "Bean": {"Kc_ini": 0.50, "Kc_mid": 1.15, "Kc_end": 0.90, "Kcb_mid": 1.10, "Kcb_end": 0.85, "total_days_default": 70},
    "Beet": {"Kc_ini": 0.50, "Kc_mid": 1.10, "Kc_end": 0.95, "Kcb_mid": 1.05, "Kcb_end": 0.85, "total_days_default": 80},
    "Broccoli": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 90},
    "Cabbage": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.90, "Kcb_mid": 0.95, "Kcb_end": 0.85, "total_days_default": 100},
    "Carrot": {"Kc_ini": 0.70, "Kc_mid": 1.05, "Kc_end": 0.95, "Kcb_mid": 1.00, "Kcb_end": 0.90, "total_days_default": 70},
    "Cauliflower": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 90},
    "Celery": {"Kc_ini": 0.70, "Kc_mid": 1.05, "Kc_end": 0.90, "Kcb_mid": 1.00, "Kcb_end": 0.85, "total_days_default": 120},
    "Chickpea": {"Kc_ini": 0.40, "Kc_mid": 1.15, "Kc_end": 0.25, "Kcb_mid": 1.10, "Kcb_end": 0.20, "total_days_default": 100},
    "Citrus": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 1.00, "Kcb_mid": 0.95, "Kcb_end": 0.95, "total_days_default": 365},
    "Coriander": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.85, "Kcb_mid": 0.95, "Kcb_end": 0.80, "total_days_default": 60},
    "Cotton": {"Kc_ini": 0.35, "Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60, "total_days_default": 180},
    "Cucumber": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.75, "Kcb_mid": 0.95, "Kcb_end": 0.70, "total_days_default": 60},
    "Eggplant": {"Kc_ini": 0.60, "Kc_mid": 1.10, "Kc_end": 0.90, "Kcb_mid": 1.05, "Kcb_end": 0.85, "total_days_default": 120},
    "Faba bean": {"Kc_ini": 0.50, "Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60, "total_days_default": 120},
    "Garlic": {"Kc_ini": 0.70, "Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65, "total_days_default": 150},
    "Grape": {"Kc_ini": 0.30, "Kc_mid": 0.85, "Kc_end": 0.45, "Kcb_mid": 0.80, "Kcb_end": 0.40, "total_days_default": 165},
    "Green bean": {"Kc_ini": 0.50, "Kc_mid": 1.05, "Kc_end": 0.90, "Kcb_mid": 1.00, "Kcb_end": 0.85, "total_days_default": 60},
    "Lentil": {"Kc_ini": 0.50, "Kc_mid": 1.10, "Kc_end": 0.25, "Kcb_mid": 1.05, "Kcb_end": 0.20, "total_days_default": 100},
    "Lettuce": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 60},
    "Maize": {"Kc_ini": 0.30, "Kc_mid": 1.20, "Kc_end": 0.60, "Kcb_mid": 1.15, "Kcb_end": 0.55, "total_days_default": 120},
    "Melon": {"Kc_ini": 0.50, "Kc_mid": 1.05, "Kc_end": 0.65, "Kcb_mid": 1.00, "Kcb_end": 0.60, "total_days_default": 90},
    "Millet": {"Kc_ini": 0.30, "Kc_mid": 1.10, "Kc_end": 0.40, "Kcb_mid": 1.05, "Kcb_end": 0.35, "total_days_default": 90},
    "Mustard": {"Kc_ini": 0.35, "Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30, "total_days_default": 120},
    "Okra": {"Kc_ini": 0.50, "Kc_mid": 1.15, "Kc_end": 0.75, "Kcb_mid": 1.10, "Kcb_end": 0.70, "total_days_default": 90},
    "Olive": {"Kc_ini": 0.50, "Kc_mid": 0.75, "Kc_end": 0.65, "Kcb_mid": 0.70, "Kcb_end": 0.60, "total_days_default": 240},
    "Onion (dry)": {"Kc_ini": 0.70, "Kc_mid": 1.10, "Kc_end": 0.65, "Kcb_mid": 1.05, "Kcb_end": 0.60, "total_days_default": 150},
    "Onion (fresh)": {"Kc_ini": 0.70, "Kc_mid": 1.10, "Kc_end": 0.80, "Kcb_mid": 1.05, "Kcb_end": 0.75, "total_days_default": 100},
    "Parsley": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.85, "Kcb_mid": 0.95, "Kcb_end": 0.80, "total_days_default": 75},
    "Peanut": {"Kc_ini": 0.40, "Kc_mid": 1.10, "Kc_end": 0.60, "Kcb_mid": 1.05, "Kcb_end": 0.55, "total_days_default": 130},
    "Peas (fresh)": {"Kc_ini": 0.50, "Kc_mid": 1.15, "Kc_end": 0.95, "Kcb_mid": 1.10, "Kcb_end": 0.90, "total_days_default": 95},
    "Pepper": {"Kc_ini": 0.60, "Kc_mid": 1.15, "Kc_end": 0.90, "Kcb_mid": 1.10, "Kcb_end": 0.85, "total_days_default": 120},
    "Pistachio": {"Kc_ini": 0.40, "Kc_mid": 1.15, "Kc_end": 0.55, "Kcb_mid": 1.10, "Kcb_end": 0.50, "total_days_default": 240},
    "Potato": {"Kc_ini": 0.50, "Kc_mid": 1.15, "Kc_end": 0.75, "Kcb_mid": 1.10, "Kcb_end": 0.70, "total_days_default": 120},
    "Pumpkin": {"Kc_ini": 0.50, "Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65, "total_days_default": 100},
    "Quinoa": {"Kc_ini": 0.30, "Kc_mid": 1.05, "Kc_end": 0.45, "Kcb_mid": 1.00, "Kcb_end": 0.40, "total_days_default": 120},
    "Radish": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.80, "Kcb_mid": 0.95, "Kcb_end": 0.75, "total_days_default": 30},
    "Rapeseed": {"Kc_ini": 0.35, "Kc_mid": 1.15, "Kc_end": 0.40, "Kcb_mid": 1.10, "Kcb_end": 0.35, "total_days_default": 150},
    "Rice": {"Kc_ini": 1.05, "Kc_mid": 1.20, "Kc_end": 0.90, "Kcb_mid": 1.15, "Kcb_end": 0.85, "total_days_default": 120},
    "Sorghum": {"Kc_ini": 0.30, "Kc_mid": 1.05, "Kc_end": 0.40, "Kcb_mid": 1.00, "Kcb_end": 0.35, "total_days_default": 120},
    "Soybean": {"Kc_ini": 0.40, "Kc_mid": 1.15, "Kc_end": 0.50, "Kcb_mid": 1.10, "Kcb_end": 0.45, "total_days_default": 120},
    "Spinach": {"Kc_ini": 0.70, "Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 45},
    "Sugarbeet": {"Kc_ini": 0.35, "Kc_mid": 1.20, "Kc_end": 0.60, "Kcb_mid": 1.15, "Kcb_end": 0.55, "total_days_default": 160},
    "Sugarcane": {"Kc_ini": 0.40, "Kc_mid": 1.25, "Kc_end": 1.10, "Kcb_mid": 1.20, "Kcb_end": 1.05, "total_days_default": 330},
    "Sunflower": {"Kc_ini": 0.35, "Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30, "total_days_default": 120},
    "Sweet Potato": {"Kc_ini": 0.50, "Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60, "total_days_default": 135},
    "Tomato (fresh)": {"Kc_ini": 0.60, "Kc_mid": 1.15, "Kc_end": 0.80, "Kcb_mid": 1.10, "Kcb_end": 0.75, "total_days_default": 120},
    "Tomato (proc)": {"Kc_ini": 0.60, "Kc_mid": 1.15, "Kc_end": 0.70, "Kcb_mid": 1.10, "Kcb_end": 0.65, "total_days_default": 130},
    "Turnip": {"Kc_ini": 0.70, "Kc_mid": 1.05, "Kc_end": 0.80, "Kcb_mid": 1.00, "Kcb_end": 0.75, "total_days_default": 60},
    "Walnut": {"Kc_ini": 0.40, "Kc_mid": 1.20, "Kc_end": 0.70, "Kcb_mid": 1.15, "Kcb_end": 0.65, "total_days_default": 240},
    "Watermelon": {"Kc_ini": 0.40, "Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65, "total_days_default": 90},
    "Wheat": {"Kc_ini": 0.70, "Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30, "total_days_default": 150},
    "Winter Squash": {"Kc_ini": 0.50, "Kc_mid": 1.00, "Kc_end": 0.70, "Kcb_mid": 0.95, "Kcb_end": 0.65, "total_days_default": 105},
    "Zucchini": {"Kc_ini": 0.70, "Kc_mid": 1.05, "Kc_end": 0.80, "Kcb_mid": 1.00, "Kcb_end": 0.75, "total_days_default": 50}
}

# --------------------------------------------------------------------------------
# 2. Configure Requests Session with Retries
# --------------------------------------------------------------------------------
session = requests.Session()
retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
session.mount('https://', HTTPAdapter(max_retries=retries))

# --------------------------------------------------------------------------------
# 3. Streamlit Page Configuration & (Optional) Logo
# --------------------------------------------------------------------------------
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

try:
    with open("logo.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    logo_url = f"data:image/png;base64,{encoded_string}"
except FileNotFoundError:
    logo_url = ""

st.markdown(
    """
    <style>
    body {
        margin: 0;
        padding: 0;
    }
    .header-container {
        position: relative;
        background-color: #1E3A8A;
        padding: 20px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header-logo {
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
    }
    .header-logo img {
        width: 100px;
        height: auto;
    }
    .header-title {
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .sub-header {
        color: #1E3A8A;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .footer {
        background-color: #1E3A8A;
        color: white;
        padding: 10px;
        text-align: center;
        position: fixed;
        bottom: 0;
        width: 100%;
        border-radius: 5px 5px 0 0;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    .stFileUploader {
        border: 2px dashed #1E3A8A;
        border-radius: 5px;
        padding: 10px;
    }
    .stSelectbox {
        background-color: #F1F5F9;
        border-radius: 5px;
    }
    .stNumberInput input {
        background-color: #F1F5F9;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    f"""
    <div class="header-container">
        <div class="header-logo">
            <img src="{logo_url}" alt="Logo">
        </div>
        <div class="header-title">AgriWaterBalance</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("**A Tool for Crop Water Management**", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 4. TABS: Only "Setup Simulation" & "Results" - NO Irrigation Calendar
# --------------------------------------------------------------------------------
setup_tab, results_tab = st.tabs(["Setup Simulation", "Results"])

# --------------------------------------------------------------------------------
# 5. Session State
# --------------------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'soil_profile' not in st.session_state:
    st.session_state.soil_profile = None
if 'weather_cache' not in st.session_state:
    st.session_state.weather_cache = {}
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.now().date()
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False

# We'll store the next 5 days forecast in a separate DF
if 'forecast_5days_df' not in st.session_state:
    st.session_state.forecast_5days_df = pd.DataFrame()

current_date = datetime.now().date()
if st.session_state.last_reset_date != current_date:
    st.session_state.api_calls = 0
    st.session_state.last_reset_date = current_date

# --------------------------------------------------------------------------------
# 6. Functions
# --------------------------------------------------------------------------------
def compute_Ks(Dr, RAW, TAW):
    if Dr <= RAW:
        return 1.0
    elif Dr >= TAW:
        return 0.0
    else:
        return max(0.0, (TAW - Dr) / (TAW - RAW))

def compute_Kr(TEW, REW, Ew):
    if Ew <= REW:
        return 1.0
    elif Ew >= TEW:
        return 0.0
    else:
        return (TEW - Ew) / (TEW - REW)

# ----------------------------
# New: FAO Penman-Monteith ET0 Calculator
# ----------------------------
def eto_penman_monteith(tmax, tmin, tmean, rh_max, rh_min, u2, latitude, elevation, doy):
    # Constants
    G_sc = 0.0820  # solar constant [MJ m^-2 min^-1]
    sigma = 4.903e-9  # Stefan-Boltzmann constant [MJ m^-2 day^-1 K^-4]
    albedo = 0.23  # albedo for reference crop
    
    e_s_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    e_s_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    e_s = (e_s_tmax + e_s_tmin) / 2.0
    e_a = (e_s_tmin * (rh_max / 100.0) + e_s_tmax * (rh_min / 100.0)) / 2.0
    delta = 4098 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    P = 101.3 * (((293 - 0.0065 * elevation) / 293) ** 5.26)
    gamma = 0.000665 * P
    phi = math.radians(latitude)
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    delta_sun = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    omega_s = math.acos(-math.tan(phi) * math.tan(delta_sun))
    Ra = (24 * 60 / math.pi) * G_sc * dr * (omega_s * math.sin(phi) * math.sin(delta_sun) + math.cos(phi) * math.cos(delta_sun) * math.sin(omega_s))
    k_rs = 0.16
    Rs = k_rs * math.sqrt(max(0, tmax - tmin)) * Ra
    Rso = (0.75 + 2e-5 * elevation) * Ra
    Rns = (1 - albedo) * Rs
    tmax_K = tmax + 273.16
    tmin_K = tmin + 273.16
    Rnl = sigma * ((tmax_K ** 4 + tmin_K ** 4) / 2) * (0.34 - 0.14 * math.sqrt(e_a)) * (1.35 * (Rs / Rso) - 0.35)
    Rn = Rns - Rnl
    G = 0
    ET0 = (0.408 * delta * (Rn - G) + gamma * (900 / (tmean + 273)) * u2 * (e_s - e_a)) / (delta + gamma * (1 + 0.34 * u2))
    return ET0

# ----------------------------
# Modified: 5-Day Forecast Fetch with Penman-Monteith ET0 Calculation
# ----------------------------
def fetch_5day_forecast(lat, lon):
    cache_key = f"{lat}_{lon}_5day_forecast"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    if lat == 0 and lon == 0:
        st.warning("Invalid lat/lon => no forecast.")
        return pd.DataFrame()
    
    api_key = "fe2d869569674a4afbfca57707bdf691"
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        st.session_state.api_calls += 1
        data = resp.json()
        
        dd = {}
        now_ = datetime.now().date()
        cutoff = now_ + timedelta(days=4)
        for entry in data['list']:
            dt_obj = datetime.fromtimestamp(entry['dt'])
            dt_date = dt_obj.date()
            if dt_date > cutoff:
                continue
            ds = dt_date.strftime("%Y-%m-%d")
            tmax_ = entry['main']['temp_max']
            tmin_ = entry['main']['temp_min']
            humidity = entry['main']['humidity']
            wind_speed = entry['wind']['speed'] if 'wind' in entry and 'speed' in entry['wind'] else 0
            if ds not in dd:
                dd[ds] = {
                    "tmax": tmax_,
                    "tmin": tmin_,
                    "rh_values": [humidity],
                    "wind_values": [wind_speed]
                }
            else:
                dd[ds]["tmax"] = max(dd[ds]["tmax"], tmax_)
                dd[ds]["tmin"] = min(dd[ds]["tmin"], tmin_)
                dd[ds]["rh_values"].append(humidity)
                dd[ds]["wind_values"].append(wind_speed)
        
        rows = []
        for dsi in sorted(dd.keys()):
            d_ = pd.to_datetime(dsi)
            tmax_day = dd[dsi]["tmax"]
            tmin_day = dd[dsi]["tmin"]
            Tmean = (tmax_day + tmin_day) / 2.0
            rh_max = max(dd[dsi]["rh_values"])
            rh_min = min(dd[dsi]["rh_values"])
            wind_avg_10 = np.mean(dd[dsi]["wind_values"])
            u2 = wind_avg_10 * (4.87 / math.log(67.8 * 10 - 5.42))
            doy = d_.timetuple().tm_yday
            elevation = 0
            eto_val = eto_penman_monteith(tmax_day, tmin_day, Tmean, rh_max, rh_min, u2, lat, elevation, doy)
            
            rows.append({
                "Date": d_,
                "Tmax": round(tmax_day, 2),
                "Tmin": round(tmin_day, 2),
                "ETo": round(eto_val, 2)
            })
        
        fdf = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        st.session_state.weather_cache[cache_key] = fdf
        st.session_state.forecast_5days_df = fdf.copy()
        return fdf
    except Exception as e:
        st.error(f"Forecast data fetch error: {e}")
        return pd.DataFrame()

def fetch_historical_nasa(lat, lon, start_date, end_date):
    cache_key = f"{lat}_{lon}_nasa_{start_date}_{end_date}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    try:
        start_str = start_date.strftime("%Y%m%d")
        end_str   = end_date.strftime("%Y%m%d")
        url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
               f"parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG"
               f"&longitude={lon}&latitude={lat}"
               f"&start={start_str}&end={end_str}&format=JSON")
        r = session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        dts = pd.date_range(start_date, end_date)
        ET0_, PP_ = [], []
        for dt_ in dts:
            ds = dt_.strftime("%Y%m%d")
            rad_ = data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(ds, 0)
            eto_v = rad_ * 0.2
            ET0_.append(eto_v)
            prec = data['properties']['parameter']['PRECTOTCORR'].get(ds, 0)
            PP_.append(prec)
        
        wdf = pd.DataFrame({
            "Date": dts,
            "ET0": ET0_,
            "Precipitation": PP_,
            "Irrigation": [0]*len(dts)
        })
        st.session_state.weather_cache[cache_key] = wdf
        return wdf
    except:
        st.warning("Unable to fetch historical NASA-POWER data.")
        return pd.DataFrame()

def simulate_SIMdualKc(weather_df, crop_stages_df, soil_df,
                       track_drainage=True, enable_yield=False, Ym=0, Ky=0,
                       use_transp=False, WP_yield=0,
                       enable_leaching=False, nitrate_conc=10.0,
                       total_N_input=100.0, leaching_fraction=0.1,
                       dynamic_root=False, init_root=300, max_root=800, days_to_max=60):
    if weather_df.empty:
        st.error("Weather data is empty.")
        return None
    
    NDAYS = len(weather_df)
    crop_stages_df = crop_stages_df.sort_values("Start_Day").reset_index(drop=True)
    
    day_kcb = np.zeros(NDAYS)
    day_p   = np.zeros(NDAYS)
    day_ke  = np.zeros(NDAYS)
    day_root= np.zeros(NDAYS)
    
    for i in range(len(crop_stages_df) - 1):
        st_i = int(crop_stages_df.iloc[i]['Start_Day']) - 1
        en_i = int(crop_stages_df.iloc[i+1]['End_Day']) - 1
        if en_i < 0: 
            continue
        en_i = min(en_i, NDAYS-1)
        st_i = max(st_i, 0)
        if st_i > en_i:
            continue
        
        kcb_s = crop_stages_df.iloc[i]['Kcb']
        kcb_e = crop_stages_df.iloc[i+1]['Kcb']
        p_s   = crop_stages_df.iloc[i]['p']
        p_e   = crop_stages_df.iloc[i+1]['p']
        ke_s  = crop_stages_df.iloc[i]['Ke']
        ke_e  = crop_stages_df.iloc[i+1]['Ke']
        rd_s  = crop_stages_df.iloc[i]['Root_Depth_mm']
        rd_e  = crop_stages_df.iloc[i+1]['Root_Depth_mm']
        
        L = en_i - st_i + 1
        day_kcb[st_i:en_i+1]  = np.linspace(kcb_s, kcb_e, L)
        day_p[st_i:en_i+1]    = np.linspace(p_s,   p_e,   L)
        day_ke[st_i:en_i+1]   = np.linspace(ke_s,  ke_e,  L)
        day_root[st_i:en_i+1] = np.linspace(rd_s,  rd_e,  L)
    
    i_last = len(crop_stages_df) - 1
    st_l = int(crop_stages_df.iloc[i_last]['Start_Day']) - 1
    en_l = int(crop_stages_df.iloc[i_last]['End_Day']) - 1
    if st_l < 0: st_l = 0
    if en_l < 0: en_l = 0
    if en_l > NDAYS-1: en_l = NDAYS-1
    if st_l <= en_l:
        day_kcb[st_l:en_l+1]  = crop_stages_df.iloc[i_last]['Kcb']
        day_p[st_l:en_l+1]    = crop_stages_df.iloc[i_last]['p']
        day_ke[st_l:en_l+1]   = crop_stages_df.iloc[i_last]['Ke']
        day_root[st_l:en_l+1] = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    if en_l < NDAYS-1:
        day_kcb[en_l+1:]  = crop_stages_df.iloc[i_last]['Kcb']
        day_p[en_l+1:]    = crop_stages_df.iloc[i_last]['p']
        day_ke[en_l+1:]   = crop_stages_df.iloc[i_last]['Ke']
        day_root[en_l+1:] = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    
    if dynamic_root:
        root_lin = np.linspace(init_root, max_root, min(days_to_max, NDAYS)).tolist()
        if NDAYS > days_to_max:
            root_lin += [max_root]*(NDAYS - days_to_max)
        day_root = np.array(root_lin[:NDAYS])
    
    TEW = soil_df['TEW'].sum()
    REW = soil_df['REW'].sum()
    E_count = REW
    
    SW_layers = []
    for j in range(len(soil_df)):
        fc_j = soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']
        SW_layers.append(fc_j)
    
    results = []
    cumIrr, cumPrec = 0, 0
    
    for d_i in range(NDAYS):
        date_i = weather_df.iloc[d_i]['Date']
        ET0_i  = max(0, weather_df.iloc[d_i]['ET0'])
        PR_i   = max(0, weather_df.iloc[d_i]['Precipitation'])
        IR_i   = max(0, weather_df.iloc[d_i]['Irrigation'])
        
        cumIrr += IR_i
        cumPrec += PR_i
        
        Kcb_i = day_kcb[d_i]
        p_i   = day_p[d_i]
        ke0_i = day_ke[d_i]
        rd_i  = max(1, day_root[d_i])
        
        tot_depth = 0
        sum_FC, sum_WP, SW_root = 0, 0, 0
        for j in range(len(soil_df)):
            layer_d = soil_df.iloc[j]['Depth_mm']
            WP_j = soil_df.iloc[j]['WP']
            FC_j = soil_df.iloc[j]['FC']
            
            new_d = tot_depth + layer_d
            fraction = 0
            if new_d <= rd_i:
                fraction = 1.0
            elif tot_depth < rd_i < new_d:
                fraction = (rd_i - tot_depth) / layer_d
            if fraction > 0:
                sum_FC += FC_j * layer_d * fraction
                sum_WP += WP_j * layer_d * fraction
                SW_root += SW_layers[j] * fraction
            tot_depth = new_d
        
        TAW_ = sum_FC - sum_WP
        RAW_ = p_i * TAW_
        Dr_  = sum_FC - SW_root
        
        Ks_ = compute_Ks(Dr_, RAW_, TAW_)
        Kr_ = compute_Kr(TEW, REW, E_count)
        Ke_ = Kr_ * ke0_i
        
        ETc_ = (Kcb_i * Ks_ + Ke_) * ET0_i
        Etc_trans = Kcb_i * Ks_ * ET0_i
        Etc_evap  = Ke_ * ET0_i
        
        infiltration = PR_i + IR_i
        excess = infiltration - ETc_
        drainage = 0
        runoff = 0
        
        if track_drainage:
            for j in range(len(soil_df)):
                layer_fc = soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']
                gap_j = layer_fc - SW_layers[j]
                if gap_j > 0 and excess > 0:
                    added = min(excess, gap_j)
                    SW_layers[j] += added
                    excess -= added
            drainage = max(0, excess)
            for j in range(len(soil_df)):
                layer_wp = soil_df.iloc[j]['WP'] * soil_df.iloc[j]['Depth_mm']
                layer_fc = soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']
                SW_layers[j] = max(layer_wp, min(layer_fc, SW_layers[j]))
        
        tr_rem = Etc_trans
        if tr_rem > 0:
            tot_depth = 0
            for j in range(len(soil_df)):
                layer_d = soil_df.iloc[j]['Depth_mm']
                new_d = tot_depth + layer_d
                fraction = 0
                if new_d <= rd_i:
                    fraction = 1.0
                elif tot_depth < rd_i < new_d:
                    fraction = (rd_i - tot_depth) / layer_d
                if fraction > 0:
                    wp_j = soil_df.iloc[j]['WP'] * layer_d
                    available_j = SW_layers[j] - wp_j
                    portion = tr_rem * fraction
                    actual_remove = min(portion, available_j)
                    SW_layers[j] -= actual_remove
                    tr_rem -= actual_remove
                    if tr_rem <= 0:
                        break
                tot_depth = new_d
        
        ev_rem = Etc_evap
        if ev_rem > 0 and len(soil_df) > 0:
            fc_0 = soil_df.iloc[0]['FC'] * soil_df.iloc[0]['Depth_mm']
            wp_0 = soil_df.iloc[0]['WP'] * soil_df.iloc[0]['Depth_mm']
            available_0 = SW_layers[0] - wp_0
            actual_rm = min(ev_rem, available_0)
            SW_layers[0] -= actual_rm
        
        if infiltration >= 4.0:
            E_count = 0
        else:
            E_count += Etc_evap
        E_count = max(0, min(E_count, TEW))
        
        new_SWroot = 0
        tot_depth = 0
        sum_FC2 = 0
        for j in range(len(soil_df)):
            layer_d = soil_df.iloc[j]['Depth_mm']
            new_d = tot_depth + layer_d
            fraction = 0
            if new_d <= rd_i:
                fraction = 1.0
            elif tot_depth < rd_i < new_d:
                fraction = (rd_i - tot_depth) / layer_d
            new_SWroot += SW_layers[j] * fraction
            sum_FC2 += soil_df.iloc[j]['FC'] * layer_d * fraction
            tot_depth = new_d
        Dr_end = sum_FC2 - new_SWroot
        
        yield_ = None
        if enable_yield:
            if (Ym > 0 and Ky > 0 and ETc_ > 0):
                y_ = Ym * (1 - Ky * (1 - (ETc_ / (Kcb_i * ET0_i + 1e-9))))
                yield_ = max(0, y_)
            if use_transp and WP_yield > 0:
                if yield_ is None:
                    yield_ = 0
                yield_ += WP_yield * Etc_trans
        
        leaching_ = 0
        if enable_leaching:
            leaching_ = drainage * 10 * (nitrate_conc * 1e-6) * 1000
        
        day_out = {
            "Date": date_i,
            "ET0 (mm)": ET0_i,
            "Precip (mm)": PR_i,
            "Irrigation (mm)": IR_i,
            "Runoff (mm)": runoff,
            "Infiltration (mm)": infiltration,
            "Ks": Ks_,
            "Kr": Kr_,
            "Ke": Ke_,
            "ETa (mm)": ETc_,
            "ETa_transp (mm)": Etc_trans,
            "ETa_evap (mm)": Etc_evap,
            "Drainage (mm)": drainage,
            "Dr_start (mm)": Dr_,
            "Dr_end (mm)": Dr_end,
            "TAW (mm)": TAW_,
            "RAW (mm)": RAW_,
            "SW_root_start (mm)": SW_root,
            "SW_root_end (mm)": new_SWroot,
            "Cumulative_Irrig (mm)": cumIrr,
            "Cumulative_Precip (mm)": cumPrec
        }
        if yield_ is not None:
            day_out["Yield (ton/ha)"] = yield_
        if enable_leaching:
            day_out["Leaching (kg/ha)"] = leaching_
        
        for j in range(len(soil_df)):
            fc_layer = soil_df.iloc[j]["FC"] * soil_df.iloc[j]["Depth_mm"]
            sw_percent = (SW_layers[j] / fc_layer) * 100 if fc_layer > 0 else None
            day_out[f"Layer{j}_SW (%)"] = round(sw_percent, 2) if sw_percent is not None else None
        
        results.append(day_out)
    
    return pd.DataFrame(results)

# ----------------------------
# Modified: Auto-Stages Generation using Kc_ini, Kc_mid, Kc_end
# ----------------------------
def create_auto_stages_for_crop(crop_name):
    total_d = CROP_DATABASE[crop_name]["total_days_default"]
    init_d = int(total_d * 0.2)
    dev_d  = int(total_d * 0.3)
    mid_d  = int(total_d * 0.3)
    late_d = total_d - (init_d + dev_d + mid_d)
    
    Kc_ini = CROP_DATABASE[crop_name]["Kc_ini"]
    Kc_mid = CROP_DATABASE[crop_name]["Kc_mid"]
    Kc_end = CROP_DATABASE[crop_name]["Kc_end"]
    
    stg = [
        {"Start_Day": 1,
         "End_Day": init_d,
         "Kcb": Kc_ini,
         "Root_Depth_mm": 100,
         "p": 0.5,
         "Ke": 1.0},
        {"Start_Day": init_d + 1,
         "End_Day": init_d + dev_d,
         "Kcb": Kc_mid,
         "Root_Depth_mm": 400,
         "p": 0.5,
         "Ke": 0.5},
        {"Start_Day": init_d + dev_d + 1,
         "End_Day": init_d + dev_d + mid_d,
         "Kcb": Kc_mid,
         "Root_Depth_mm": 600,
         "p": 0.5,
         "Ke": 0.2},
        {"Start_Day": init_d + dev_d + mid_d + 1,
         "End_Day": total_d,
         "Kcb": Kc_end,
         "Root_Depth_mm": 600,
         "p": 0.5,
         "Ke": 0.1}
    ]
    return pd.DataFrame(stg)

# --------------------------------------------------------------------------------
# 7. SETUP TAB
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown('<p style="font-size:16px;">1. Select Crop</p>', unsafe_allow_html=True)
    crop_list = list(CROP_DATABASE.keys())
    selected_crop = st.selectbox("Choose your crop", crop_list)
    st.write(f"**Selected Crop**: {selected_crop}")
    st.write(f"- Kc_mid={CROP_DATABASE[selected_crop]['Kc_mid']}, Kc_end={CROP_DATABASE[selected_crop]['Kc_end']}")
    st.write(f"- Kcb_mid={CROP_DATABASE[selected_crop]['Kcb_mid']}, Kcb_end={CROP_DATABASE[selected_crop]['Kcb_end']}")

    st.markdown('<p style="font-size:16px;">2. Weather Data</p>', unsafe_allow_html=True)
    weather_file = st.file_uploader("Upload CSV with [Date,ET0,Precipitation,Irrigation], or rely on forecast", type=["csv","txt"])

    st.markdown('<p style="font-size:16px;">3. Crop Stage Data</p>', unsafe_allow_html=True)
    use_custom_stage = st.checkbox("Upload custom Crop Stage Data?", value=False)
    st.write("*Otherwise, auto from known stage durations.*")
    custom_crop_file = None
    if use_custom_stage:
        custom_crop_file = st.file_uploader("Upload Crop Stage CSV (Start_Day, End_Day, Kcb, Root_Depth_mm, p, Ke)", type=["csv","txt"])

    st.markdown('<p style="font-size:16px;">4. Soil Layers Data</p>', unsafe_allow_html=True)
    soil_file = st.file_uploader("Upload soil data (Depth_mm, FC, WP, TEW, REW) or use default", type=["csv","txt"])

    st.markdown('<p style="font-size:16px;">5. Additional Options</p>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        track_drainage = st.checkbox("Track Drainage?", value=True)
        enable_yield   = st.checkbox("Enable Yield Estimation?", value=False)
        ym_, ky_, use_transp, wp_yield = 0, 0, False, 0
        if enable_yield:
            st.write("**Yield Options**")
            ym_ = st.number_input("Max Yield (ton/ha)?", min_value=0.0, value=10.0)
            ky_ = st.number_input("Ky (yield response factor)?", min_value=0.0, value=1.0)
            use_transp = st.checkbox("Use Transp-based approach (WP_yield)?", value=False)
            if use_transp:
                wp_yield = st.number_input("WP_yield (ton/ha per mm)?", min_value=0.0, value=0.012, step=0.001)

    with colB:
        enable_leaching = st.checkbox("Enable Leaching?", value=False)
        nitrate_conc, totalN, lf = 10.0, 100.0, 0.1
        if enable_leaching:
            nitrate_conc = st.number_input("Nitrate mg/L", min_value=0.0, value=10.0)
            totalN       = st.number_input("Total N input (kg/ha)?", min_value=0.0, value=100.0)
            lf           = st.number_input("Leaching Fraction (0-1)?", min_value=0.0, max_value=1.0, value=0.1)

    st.markdown('<p style="font-size:16px;">6. 5-Day Forecast Options</p>', unsafe_allow_html=True)
    enable_forecast = st.checkbox("Enable 5-Day Forecast?", value=True)
    lat_, lon_ = 0.0, 0.0
    current_day = 1  # default day of season
    if enable_forecast:
        lat_ = st.number_input("Latitude", value=35.0)
        lon_ = st.number_input("Longitude", value=-80.0)
        current_day = st.number_input("Current Day of Season", value=1, step=1, min_value=1)

    st.markdown('<p style="font-size:16px;">7. Dynamic Root Growth?</p>', unsafe_allow_html=True)
    dynamic_root = st.checkbox("Enable dynamic root growth?", value=False)
    init_rd, max_rd, days_mx = 300, 800, 60
    if dynamic_root:
        init_rd = st.number_input("Initial Root Depth (mm)", min_value=50, value=300)
        max_rd  = st.number_input("Max Root Depth (mm)", min_value=50, value=800)
        days_mx = st.number_input("Days to reach max root depth?", min_value=1, value=60)

    st.markdown('<p style="font-size:16px;">8. Run Simulation</p>', unsafe_allow_html=True)
    run_button = st.button("Run Simulation")

    if run_button:
        st.session_state["simulation_done"] = True
        
        # Step 1: Possibly fetch next 5-day forecast
        if enable_forecast:
            f5df = fetch_5day_forecast(lat_, lon_)
            if f5df.empty:
                st.warning("No 5-day forecast found or lat/lon invalid.")
            else:
                # Retrieve crop coefficients for forecast computations
                total_d = CROP_DATABASE[selected_crop]["total_days_default"]
                Kc_ini = CROP_DATABASE[selected_crop]["Kc_ini"]
                Kc_mid = CROP_DATABASE[selected_crop]["Kc_mid"]
                Kc_end = CROP_DATABASE[selected_crop]["Kc_end"]
                
                # Define a function for computing crop coefficient based on season day
                def get_crop_coef(day, total_d, Kc_ini, Kc_mid, Kc_end):
                    if day <= total_d * 0.2:
                        return Kc_ini
                    elif day <= total_d * 0.5:
                        return Kc_ini + (Kc_mid - Kc_ini) * ((day - total_d * 0.2) / (total_d * 0.3))
                    elif day <= total_d * 0.8:
                        return Kc_mid
                    else:
                        return Kc_mid + (Kc_end - Kc_mid) * ((day - total_d * 0.8) / (total_d * 0.2))
                
                f5df["Forecast_Day"] = range(int(current_day), int(current_day) + len(f5df))
                f5df["Kc"] = f5df["Forecast_Day"].apply(lambda d: get_crop_coef(d, total_d, Kc_ini, Kc_mid, Kc_end))
                f5df["ETa"] = f5df["ETo"] * f5df["Kc"]
                st.session_state.forecast_5days_df = f5df
        
        # Step 2: Build or load main weather df
        if weather_file is not None:
            try:
                wdf = pd.read_csv(weather_file)
                if "Date" not in wdf.columns:
                    st.error("Weather file missing 'Date' column. Stopping.")
                    st.stop()
                if pd.api.types.is_string_dtype(wdf["Date"]):
                    wdf["Date"] = pd.to_datetime(wdf["Date"])
                wdf = wdf.sort_values("Date").reset_index(drop=True)
            except:
                st.warning("Could not parse weather file. Attempting fallback if forecast is on.")
                if enable_forecast:
                    today_ = datetime.now().date()
                    hist_start = today_ - timedelta(days=5)
                    hist_end   = today_ - timedelta(days=1)
                    hist_wdf   = fetch_historical_nasa(lat_, lon_, hist_start, hist_end)
                    if hist_wdf.empty:
                        wdf = pd.DataFrame()
                    else:
                        if "forecast_5days_df" in st.session_state and not st.session_state.forecast_5days_df.empty:
                            fore_wdf = st.session_state.forecast_5days_df.copy().rename(columns={"ETo": "ET0"})
                            fore_wdf["Precipitation"] = 0.0
                            fore_wdf["Irrigation"] = 0.0
                            wdf = pd.concat([hist_wdf, fore_wdf]).sort_values("Date").reset_index(drop=True)
                        else:
                            wdf = hist_wdf
                else:
                    wdf = pd.DataFrame()
        else:
            if enable_forecast:
                today_ = datetime.now().date()
                hist_start = today_ - timedelta(days=5)
                hist_end   = today_ - timedelta(days=1)
                hist_wdf   = fetch_historical_nasa(lat_, lon_, hist_start, hist_end)
                if hist_wdf.empty:
                    wdf = pd.DataFrame()
                else:
                    if "forecast_5days_df" in st.session_state and not st.session_state.forecast_5days_df.empty:
                        fore_wdf = st.session_state.forecast_5days_df.copy().rename(columns={"ETo": "ET0"})
                        fore_wdf["Precipitation"] = 0.0
                        fore_wdf["Irrigation"] = 0.0
                        wdf = pd.concat([hist_wdf, fore_wdf]).sort_values("Date").reset_index(drop=True)
                    else:
                        wdf = hist_wdf
            else:
                st.warning("No weather file & forecast disabled => no data. Stopping.")
                wdf = pd.DataFrame()

        if wdf is None or wdf.empty:
            st.error("No valid weather data => cannot run water-balance.")
            st.stop()

        if use_custom_stage and custom_crop_file is not None:
            try:
                stage_df = pd.read_csv(custom_crop_file)
            except:
                st.error("Could not parse stage file => using auto.")
                stage_df = create_auto_stages_for_crop(selected_crop)
        elif use_custom_stage and custom_crop_file is None:
            st.error("Checked custom stage but no file => using auto.")
            stage_df = create_auto_stages_for_crop(selected_crop)
        else:
            stage_df = create_auto_stages_for_crop(selected_crop)

        if soil_file is not None:
            try:
                soil_df = pd.read_csv(soil_file)
            except:
                st.error("Could not parse soil file => default.")
                soil_df = pd.DataFrame({
                    "Depth_mm": [200, 100],
                    "FC": [0.30, 0.30],
                    "WP": [0.15, 0.15],
                    "TEW": [20, 0],
                    "REW": [5, 0]
                })
        else:
            soil_df = pd.DataFrame({
                "Depth_mm": [200, 100],
                "FC": [0.30, 0.30],
                "WP": [0.15, 0.15],
                "TEW": [20, 0],
                "REW": [5, 0]
            })

        res_df = simulate_SIMdualKc(
            weather_df=wdf,
            crop_stages_df=stage_df,
            soil_df=soil_df,
            track_drainage=track_drainage,
            enable_yield=enable_yield,
            Ym=ym_,
            Ky=ky_,
            use_transp=use_transp,
            WP_yield=wp_yield,
            enable_leaching=enable_leaching,
            nitrate_conc=nitrate_conc,
            total_N_input=totalN,
            leaching_fraction=lf,
            dynamic_root=dynamic_root,
            init_root=init_rd,
            max_root=max_rd,
            days_to_max=days_mx
        )

        st.session_state.results_df = res_df
        st.session_state.soil_profile = soil_df
        st.success("Simulation complete! Check 'Results' tab.")

# --------------------------------------------------------------------------------
# 8. RESULTS TAB
# --------------------------------------------------------------------------------
with results_tab:
    if not st.session_state.get("simulation_done", False):
        st.info("Please run the simulation in the 'Setup Simulation' tab first.")
    else:
        if st.session_state.results_df is None or st.session_state.results_df.empty:
            st.warning("No results found. Please re-run the simulation.")
        else:
            results_df = st.session_state.results_df
            st.markdown("## Simulation Results")
            st.dataframe(results_df)
            st.download_button("Download Results (.csv)",
                               results_df.to_csv(index=False),
                               file_name="results.csv",
                               mime="text/csv")

            st.markdown("## Charts")
            plot_options = ["Daily ET Components", "Root Zone Depletion", "Daily Drainage", "Soil Water in Root Zone"]
            if "Yield (ton/ha)" in results_df.columns:
                plot_options.append("Daily Estimated Yield")
            if "Leaching (kg/ha)" in results_df.columns:
                plot_options.append("Leaching (kg/ha)")

            selected_plot = st.selectbox("Select a plot to view", plot_options)

            if selected_plot == "Daily ET Components":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["ETa (mm)"], label="ETa total")
                ax.plot(results_df["Date"], results_df["ETa_transp (mm)"], label="ETa transp")
                ax.plot(results_df["Date"], results_df["ETa_evap (mm)"], label="ETa evap")
                ax.set_xlabel("Date")
                ax.set_ylabel("mm")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="et_components.png", mime="image/png")

            elif selected_plot == "Root Zone Depletion":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Dr_start (mm)"], label="Dr start")
                ax.plot(results_df["Date"], results_df["Dr_end (mm)"], label="Dr end")
                ax.set_xlabel("Date")
                ax.set_ylabel("Depletion (mm)")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="root_zone_depletion.png", mime="image/png")

            elif selected_plot == "Daily Drainage":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Drainage (mm)"], label="Drainage")
                ax.set_xlabel("Date")
                ax.set_ylabel("mm")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="drainage.png", mime="image/png")

            elif selected_plot == "Soil Water in Root Zone":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["SW_root_start (mm)"], label="RootZ Start")
                ax.plot(results_df["Date"], results_df["SW_root_end (mm)"], label="RootZ End")
                ax.set_xlabel("Date")
                ax.set_ylabel("mm water")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="soil_water.png", mime="image/png")

            elif selected_plot == "Daily Estimated Yield":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Yield (ton/ha)"], label="Yield (ton/ha)")
                ax.set_xlabel("Date")
                ax.set_ylabel("ton/ha")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="yield.png", mime="image/png")

            elif selected_plot == "Leaching (kg/ha)":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Leaching (kg/ha)"], label="Leaching")
                ax.set_xlabel("Date")
                ax.set_ylabel("kg/ha")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="leaching.png", mime="image/png")

            st.markdown("## Next 5 Days Forecast (ETo & ETa)")
            f5_df = st.session_state.forecast_5days_df.copy()
            if not f5_df.empty:
                columns_to_show = ["Date", "Tmax", "Tmin", "ETo", "ETa"]
                f5_df_display = f5_df[columns_to_show].copy()
                st.dataframe(f5_df_display)
                st.download_button("Download Next 5 Days Forecast (ETo, ETa)",
                                   f5_df_display.to_csv(index=False),
                                   file_name="next_5days_forecast.csv",
                                   mime="text/csv")
            else:
                st.write("No separate 5-day forecast found (or forecast disabled).")

# Footer
st.markdown(
    '<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>',
    unsafe_allow_html=True
)
