import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import math
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
import io
import calendar

# --------------------------------------------------------------------------------
# 1. Large Kc Database
# --------------------------------------------------------------------------------
CROP_DATABASE = {
    "Carrot": {"Kc_mid":1.05, "Kc_end":0.95, "Kcb_mid":1.00, "Kcb_end":0.90, "total_days_default":90},
    "Beet": {"Kc_mid":1.10, "Kc_end":0.95, "Kcb_mid":1.05, "Kcb_end":0.85, "total_days_default":100},
    "Garlic": {"Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65, "total_days_default":120},
    "Onion (fresh)": {"Kc_mid":1.10, "Kc_end":0.80, "Kcb_mid":1.05, "Kcb_end":0.75, "total_days_default":110},
    "Onion (dry)": {"Kc_mid":1.10, "Kc_end":0.65, "Kcb_mid":1.05, "Kcb_end":0.60, "total_days_default":120},
    "Cabbage": {"Kc_mid":1.00, "Kc_end":0.90, "Kcb_mid":0.95, "Kcb_end":0.85, "total_days_default":90},
    "Lettuce": {"Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90, "total_days_default":65},
    "Spinach": {"Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90, "total_days_default":55},
    "Broccoli": {"Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90, "total_days_default":80},
    "Cauliflower": {"Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90, "total_days_default":95},
    "Green bean": {"Kc_mid":1.05, "Kc_end":0.90, "Kcb_mid":1.00, "Kcb_end":0.85, "total_days_default":75},
    "Tomato (fresh)": {"Kc_mid":1.15, "Kc_end":0.80, "Kcb_mid":1.10, "Kcb_end":0.75, "total_days_default":120},
    "Tomato (proc)": {"Kc_mid":1.15, "Kc_end":0.70, "Kcb_mid":1.10, "Kcb_end":0.65, "total_days_default":110},
    "Pepper": {"Kc_mid":1.15, "Kc_end":0.90, "Kcb_mid":1.10, "Kcb_end":0.85, "total_days_default":130},
    "Eggplant": {"Kc_mid":1.10, "Kc_end":0.90, "Kcb_mid":1.05, "Kcb_end":0.85, "total_days_default":130},
    "Zucchini": {"Kc_mid":1.05, "Kc_end":0.80, "Kcb_mid":1.00, "Kcb_end":0.75, "total_days_default":60},
    "Cucumber": {"Kc_mid":1.00, "Kc_end":0.75, "Kcb_mid":0.95, "Kcb_end":0.70, "total_days_default":70},
    "Melon": {"Kc_mid":1.05, "Kc_end":0.65, "Kcb_mid":1.00, "Kcb_end":0.60, "total_days_default":85},
    "Watermelon": {"Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65, "total_days_default":90},
    "Pumpkin": {"Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65, "total_days_default":100},
    "Okra": {"Kc_mid":1.15, "Kc_end":0.75, "Kcb_mid":1.10, "Kcb_end":0.70, "total_days_default":100},
    "Basil": {"Kc_mid":1.00, "Kc_end":0.80, "Kcb_mid":0.95, "Kcb_end":0.75, "total_days_default":60},
    "Parsley": {"Kc_mid":1.00, "Kc_end":0.85, "Kcb_mid":0.95, "Kcb_end":0.80, "total_days_default":70},
    "Coriander": {"Kc_mid":1.00, "Kc_end":0.85, "Kcb_mid":0.95, "Kcb_end":0.80, "total_days_default":65},
    "Celery": {"Kc_mid":1.05, "Kc_end":0.90, "Kcb_mid":1.00, "Kcb_end":0.85, "total_days_default":120},
    "Turnip": {"Kc_mid":1.05, "Kc_end":0.80, "Kcb_mid":1.00, "Kcb_end":0.75, "total_days_default":85},
    "Radish": {"Kc_mid":1.00, "Kc_end":0.80, "Kcb_mid":0.95, "Kcb_end":0.75, "total_days_default":45},
    "Wheat": {"Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30, "total_days_default":150},
    "Barley": {"Kc_mid":1.15, "Kc_end":0.25, "Kcb_mid":1.10, "Kcb_end":0.20, "total_days_default":130},
    "Maize": {"Kc_mid":1.20, "Kc_end":0.60, "Kcb_mid":1.15, "Kcb_end":0.55, "total_days_default":140},
    "Rice": {"Kc_mid":1.20, "Kc_end":0.90, "Kcb_mid":1.15, "Kcb_end":0.85, "total_days_default":160},
    "Sorghum": {"Kc_mid":1.05, "Kc_end":0.40, "Kcb_mid":1.00, "Kcb_end":0.35, "total_days_default":120},
    "Soybean": {"Kc_mid":1.15, "Kc_end":0.50, "Kcb_mid":1.10, "Kcb_end":0.45, "total_days_default":130},
    "Bean": {"Kc_mid":1.15, "Kc_end":0.90, "Kcb_mid":1.10, "Kcb_end":0.85, "total_days_default":95},
    "Peanut": {"Kc_mid":1.10, "Kc_end":0.60, "Kcb_mid":1.05, "Kcb_end":0.55, "total_days_default":135},
    "Cotton": {"Kc_mid":1.15, "Kc_end":0.65, "Kcb_mid":1.10, "Kcb_end":0.60, "total_days_default":160},
    "Sugarbeet": {"Kc_mid":1.20, "Kc_end":0.60, "Kcb_mid":1.15, "Kcb_end":0.55, "total_days_default":180},
    "Sugarcane": {"Kc_mid":1.25, "Kc_end":1.10, "Kcb_mid":1.20, "Kcb_end":1.05, "total_days_default":300},
    "Sunflower": {"Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30, "total_days_default":120},
    "Rapeseed": {"Kc_mid":1.15, "Kc_end":0.40, "Kcb_mid":1.10, "Kcb_end":0.35, "total_days_default":150},
    "Mustard": {"Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30, "total_days_default":120},
    "Faba bean": {"Kc_mid":1.15, "Kc_end":0.65, "Kcb_mid":1.10, "Kcb_end":0.60, "total_days_default":130},
    "Chickpea": {"Kc_mid":1.15, "Kc_end":0.25, "Kcb_mid":1.10, "Kcb_end":0.20, "total_days_default":120},
    "Millet": {"Kc_mid":1.10, "Kc_end":0.40, "Kcb_mid":1.05, "Kcb_end":0.35, "total_days_default":100},
    "Quinoa": {"Kc_mid":1.05, "Kc_end":0.45, "Kcb_mid":1.00, "Kcb_end":0.40, "total_days_default":120},
    "Lentil": {"Kc_mid":1.10, "Kc_end":0.25, "Kcb_mid":1.05, "Kcb_end":0.20, "total_days_default":110},
    "Potato": {"Kc_mid":1.15, "Kc_end":0.75, "Kcb_mid":1.10, "Kcb_end":0.70, "total_days_default":110}
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
st.set_page_config(page_title="Advanced AgriWaterBalance", layout="wide")

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
        <div class="header-title">Advanced AgriWaterBalance</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("**A Professional Tool for Crop Water Management**", unsafe_allow_html=True)

# Tabs
setup_tab, results_tab, irrig_calendar_tab = st.tabs(["Setup Simulation", "Results", "Irrigation Calendar"])

# --------------------------------------------------------------------------------
# 4. Session State
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

# We'll also store the next 5 days forecast ETo & ETa in a separate DF
if 'forecast_5days_df' not in st.session_state:
    st.session_state.forecast_5days_df = pd.DataFrame()

current_date = datetime.now().date()
if st.session_state.last_reset_date != current_date:
    st.session_state.api_calls = 0
    st.session_state.last_reset_date = current_date

# --------------------------------------------------------------------------------
# 5. Water Balance & Weather Data
# --------------------------------------------------------------------------------
def compute_Ks(Dr, RAW, TAW):
    """
    Ks = stress coefficient for transpiration
    """
    if Dr <= RAW:
        return 1.0
    elif Dr >= TAW:
        return 0.0
    else:
        return max(0.0, (TAW - Dr) / (TAW - RAW))

def compute_Kr(TEW, REW, Ew):
    """
    Kr = stress coefficient for evaporation
    """
    if Ew <= REW:
        return 1.0
    elif Ew >= TEW:
        return 0.0
    else:
        return (TEW - Ew) / (TEW - REW)

def fetch_weather_data(lat, lon, start_date, end_date, forecast=True):
    """
    Fetches weather data (historical or 5-day forecast).
    We'll store results in a cache to reduce repeated calls.
    Also *separately* we handle the next 5 days' ETo & ETa for the "calendar."
    """
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}_{forecast}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    # Forecast path (OpenWeatherMap)
    if forecast:
        if st.session_state.api_calls >= 1000:
            st.warning("Daily API call limit reached.")
            return None
        if lat == 0 and lon == 0:
            st.warning("Invalid lat/lon.")
            return None
        
        # Replace with your own OWM API key:
        api_key = "YOUR_OPENWEATHER_API_KEY_HERE"
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            st.session_state.api_calls += 1
            data = resp.json()
            
            # We'll accumulate daily tmax/tmin
            dd = {}
            for entry in data['list']:
                dt_obj = datetime.fromtimestamp(entry['dt'])
                dt_ = dt_obj.date()
                # Only up to 5 days from now:
                if dt_ > (datetime.now().date() + timedelta(days=4)):
                    continue
                ds = dt_.strftime("%Y-%m-%d")
                tmax_ = entry['main']['temp_max']
                tmin_ = entry['main']['temp_min']
                if ds not in dd:
                    dd[ds] = {
                        'tmax': tmax_,
                        'tmin': tmin_
                    }
                else:
                    dd[ds]['tmax'] = max(dd[ds]['tmax'], tmax_)
                    dd[ds]['tmin'] = min(dd[ds]['tmin'], tmin_)
            
            # Build daily DataFrame (only for next 5 days)
            five_dates = sorted(dd.keys())
            daily_rows = []
            for dsi in five_dates:
                dat_ = pd.to_datetime(dsi)
                tmax_ = dd[dsi]['tmax']
                tmin_ = dd[dsi]['tmin']
                if tmax_ < tmin_:
                    tmax_, tmin_ = tmin_, tmax_
                # Simple Hargreaves for daily ETo
                Ra = 10
                Tmean = (tmax_ + tmin_) / 2
                eto_ = 0.0023 * Ra * (Tmean + 17.8) * ((tmax_ - tmin_) ** 0.5)
                # We'll store ETo in mm
                # For now, precipitation from OWM we can sum up from 3h intervals,
                # but let's keep it simple and set 0 if not found
                daily_rows.append({
                    "Date": dat_,
                    "ET0": round(max(0, eto_), 2),
                    "Precipitation": 0.0,  # simplified
                    "Irrigation": 0.0
                })
            
            # We'll store this 5-day forecast separately for ETo & ETa
            f5_df = pd.DataFrame(daily_rows)
            # Save to session for the calendar & results table
            st.session_state.forecast_5days_df = f5_df.copy()
            
            # For the main "weather_df" usage in water-balance, 
            # we assume the same structure: Date, ET0, Precip, Irrigation
            # If user wants historical + forecast combined, 
            # we have fewer days here, but let's just return these few days
            wdf = f5_df.rename(columns={"ET0": "ET0", 
                                        "Precipitation": "Precipitation"})
            wdf = wdf.sort_values("Date").reset_index(drop=True)
            
            st.session_state.weather_cache[cache_key] = wdf
            return wdf
        
        except Exception as e:
            st.error(f"Unable to fetch forecast data: {e}")
            return None
    
    # Historical path (NASA POWER)
    else:
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
            r = session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            dts = pd.date_range(start_date, end_date)
            ET0_, PP_ = [], []
            for dt_ in dts:
                ds = dt_.strftime("%Y%m%d")
                rad_ = data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(ds, 0)
                # simplified ET0
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
            return None


def simulate_SIMdualKc(weather_df, crop_stages_df, soil_df,
                       track_drainage=True, enable_yield=False, Ym=0, Ky=0,
                       use_transp=False, WP_yield=0,
                       enable_leaching=False, nitrate_conc=10.0,
                       total_N_input=100.0, leaching_fraction=0.1,
                       dynamic_root=False, init_root=300, max_root=800, days_to_max=60):
    """
    Full daily water balance using dual Kc approach.
    Produces columns such as Dr_end (mm), RAW (mm), ETa (mm), etc.
    """
    if weather_df.empty:
        st.error("Weather data is empty.")
        return None
    
    NDAYS = len(weather_df)
    crop_stages_df = crop_stages_df.sort_values("Start_Day").reset_index(drop=True)
    
    # Prepare daily arrays
    day_kcb = np.zeros(NDAYS)
    day_p   = np.zeros(NDAYS)
    day_ke  = np.zeros(NDAYS)
    day_root= np.zeros(NDAYS)
    
    # Fill arrays from crop_stages_df
    for i in range(len(crop_stages_df) - 1):
        st_i = int(crop_stages_df.iloc[i]['Start_Day']) - 1
        en_i = int(crop_stages_df.iloc[i+1]['End_Day']) - 1
        if en_i < 0:
            continue
        en_i = min(en_i, NDAYS-1)
        st_i = max(0, st_i)
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
        day_p[st_i:en_i+1]    = np.linspace(p_s, p_e, L)
        day_ke[st_i:en_i+1]   = np.linspace(ke_s, ke_e, L)
        day_root[st_i:en_i+1] = np.linspace(rd_s, rd_e, L)
    
    # Last stage
    i_last = len(crop_stages_df) - 1
    st_l = int(crop_stages_df.iloc[i_last]['Start_Day']) - 1
    en_l = int(crop_stages_df.iloc[i_last]['End_Day']) - 1
    if st_l < 0: st_l = 0
    if en_l < 0: en_l = 0
    if en_l > NDAYS - 1: en_l = NDAYS - 1
    if st_l <= en_l:
        day_kcb[st_l:en_l+1]  = crop_stages_df.iloc[i_last]['Kcb']
        day_p[st_l:en_l+1]    = crop_stages_df.iloc[i_last]['p']
        day_ke[st_l:en_l+1]   = crop_stages_df.iloc[i_last]['Ke']
        day_root[st_l:en_l+1] = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    if en_l < NDAYS-1:
        day_kcb[en_l+1:] = crop_stages_df.iloc[i_last]['Kcb']
        day_p[en_l+1:]   = crop_stages_df.iloc[i_last]['p']
        day_ke[en_l+1:]  = crop_stages_df.iloc[i_last]['Ke']
        day_root[en_l+1:]= crop_stages_df.iloc[i_last]['Root_Depth_mm']
    
    # Dynamic root option
    if dynamic_root:
        root_lin = np.linspace(init_root, max_root, min(days_to_max, NDAYS)).tolist()
        if NDAYS>days_to_max:
            root_lin += [max_root]*(NDAYS - days_to_max)
        day_root = np.array(root_lin[:NDAYS])
    
    # Evap params
    TEW = soil_df['TEW'].sum()
    REW = soil_df['REW'].sum()
    E_count = REW
    
    # Initialize soil water at FC
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
        
        # TAW, RAW in root zone
        tot_depth = 0
        sum_FC, sum_WP, SW_root = 0, 0, 0
        for j in range(len(soil_df)):
            layer_d = soil_df.iloc[j]['Depth_mm']
            WP_j    = soil_df.iloc[j]['WP']
            FC_j    = soil_df.iloc[j]['FC']
            
            new_d = tot_depth + layer_d
            fraction=0
            if new_d <= rd_i:
                fraction=1.0
            elif tot_depth<rd_i<new_d:
                fraction=(rd_i - tot_depth)/layer_d
            if fraction>0:
                sum_FC   += FC_j*layer_d*fraction
                sum_WP   += WP_j*layer_d*fraction
                SW_root  += SW_layers[j]*fraction
            tot_depth=new_d
        
        TAW_ = sum_FC - sum_WP
        RAW_ = p_i*TAW_
        Dr_  = sum_FC - SW_root
        
        Ks_ = compute_Ks(Dr_, RAW_, TAW_)
        Kr_ = compute_Kr(TEW, REW, E_count)
        Ke_ = Kr_*ke0_i
        
        ETc_ = (Kcb_i*Ks_ + Ke_)*ET0_i
        Etc_trans = Kcb_i*Ks_*ET0_i
        Etc_evap  = Ke_*ET0_i
        
        infiltration = PR_i+IR_i
        runoff=0
        excess = infiltration-ETc_
        drainage=0
        
        if track_drainage:
            for j in range(len(soil_df)):
                layer_fc = soil_df.iloc[j]['FC']*soil_df.iloc[j]['Depth_mm']
                gap_j = layer_fc - SW_layers[j]
                if gap_j>0 and excess>0:
                    added = min(excess, gap_j)
                    SW_layers[j]+= added
                    excess-=added
            drainage = max(0, excess)
            for j in range(len(soil_df)):
                layer_wp = soil_df.iloc[j]['WP']*soil_df.iloc[j]['Depth_mm']
                layer_fc = soil_df.iloc[j]['FC']*soil_df.iloc[j]['Depth_mm']
                SW_layers[j] = max(layer_wp, min(layer_fc, SW_layers[j]))
        
        # Remove water by transpiration
        tr_rem = Etc_trans
        if tr_rem>0:
            tot_depth=0
            for j in range(len(soil_df)):
                layer_d = soil_df.iloc[j]['Depth_mm']
                new_d = tot_depth + layer_d
                fraction=0
                if new_d<=rd_i:
                    fraction=1.0
                elif tot_depth<rd_i<new_d:
                    fraction=(rd_i - tot_depth)/layer_d
                if fraction>0:
                    wp_j = soil_df.iloc[j]['WP']*layer_d
                    available_j = SW_layers[j] - wp_j
                    portion = tr_rem*fraction
                    actual_remove = min(portion, available_j)
                    SW_layers[j]-= actual_remove
                    tr_rem-= actual_remove
                    if tr_rem<=0:
                        break
                tot_depth=new_d
        
        # Remove water by evaporation from top layer
        ev_rem = Etc_evap
        if ev_rem>0 and len(soil_df)>0:
            fc_0 = soil_df.iloc[0]['FC']*soil_df.iloc[0]['Depth_mm']
            wp_0 = soil_df.iloc[0]['WP']*soil_df.iloc[0]['Depth_mm']
            available_0 = SW_layers[0] - wp_0
            actual_rm = min(ev_rem, available_0)
            SW_layers[0]-= actual_rm
        
        # E_count
        if infiltration>=4.0:
            E_count=0
        else:
            E_count+=Etc_evap
        E_count = max(0, min(E_count, TEW))
        
        # final root zone SW
        new_SWroot=0
        tot_depth=0
        sum_FC2=0
        for j in range(len(soil_df)):
            layer_d = soil_df.iloc[j]['Depth_mm']
            new_d = tot_depth+layer_d
            fraction=0
            if new_d<=rd_i:
                fraction=1.0
            elif tot_depth<rd_i<new_d:
                fraction=(rd_i - tot_depth)/layer_d
            new_SWroot+= SW_layers[j]*fraction
            sum_FC2  += soil_df.iloc[j]['FC']*layer_d*fraction
            tot_depth = new_d
        Dr_end = sum_FC2-new_SWroot
        
        yield_=None
        if enable_yield:
            if (Ym>0 and Ky>0 and ETc_>0):
                y_ = Ym*(1 - Ky*(1 - (ETc_/(Kcb_i*ET0_i+1e-9))))
                yield_ = max(0, y_)
            if use_transp and WP_yield>0:
                if yield_ is None:
                    yield_=0
                yield_ += WP_yield*Etc_trans
        
        leaching_=0
        if enable_leaching:
            leaching_= drainage*10*(nitrate_conc*1e-6)*1000
        
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
            day_out[f"Layer{j}_SW (mm)"] = SW_layers[j]
        
        results.append(day_out)
    
    outdf = pd.DataFrame(results)
    return outdf

def create_auto_stages_for_crop(crop_name):
    """
    Creates 4 basic stages from total_days_default and Kcb data.
    """
    total_d = CROP_DATABASE[crop_name]["total_days_default"]
    init_d = int(total_d*0.2)
    dev_d  = int(total_d*0.3)
    mid_d  = int(total_d*0.3)
    late_d = total_d - (init_d+dev_d+mid_d)
    
    kcb_mid = CROP_DATABASE[crop_name]["Kcb_mid"]
    kcb_end = CROP_DATABASE[crop_name]["Kcb_end"]
    
    stg = [
        {"Start_Day": 1,                  "End_Day": init_d,               "Kcb": 0.15,     "Root_Depth_mm":100, "p": 0.5, "Ke":1.0},
        {"Start_Day": init_d+1,          "End_Day": init_d+dev_d,         "Kcb": kcb_mid,  "Root_Depth_mm":400, "p": 0.5, "Ke":0.5},
        {"Start_Day": init_d+dev_d+1,    "End_Day": init_d+dev_d+mid_d,    "Kcb": kcb_mid,  "Root_Depth_mm":600, "p": 0.5, "Ke":0.2},
        {"Start_Day": init_d+dev_d+mid_d+1, "End_Day": total_d,            "Kcb": kcb_end,  "Root_Depth_mm":600, "p": 0.5, "Ke":0.1}
    ]
    return pd.DataFrame(stg)

# -------------------------------------------------------------------------
# NEW: Function to build a monthly calendar with only the next 5 days
# showing ETo & ETa from st.session_state["forecast_5days_df"].
# -------------------------------------------------------------------------
def generate_monthly_calendar_html():
    """
    Build an HTML table for the current month, 
    filling only the next 5 days from forecast_5days_df with ETo & ETa.
    """
    fdf = st.session_state.forecast_5days_df
    if fdf.empty:
        return "<p>No next-5-days forecast found.</p>"
    
    # We'll build a dict from 'Date' -> (ET0, ETa)
    # We'll compute ETa as ET0*(some Kc?), or we might already have it if user wants.
    # But in this example, let's assume the advanced simulation isn't used for the calendar,
    # we only show the forecast ETo and a simple assumption for ETa (like ETa=ET0?).
    # If you want to incorporate a dynamic Kc, you can do so. 
    # For simplicity, let's just do ETa = ET0 if we didn't compute it otherwise.
    
    # Check if there's an "ETa" column. If not, let's define ETa=ET0.
    if "ETa" not in fdf.columns:
        fdf["ETa"] = fdf["ET0"]  # fallback
    
    # We want next 5 days from today
    today = datetime.now().date()
    next_5 = [today + timedelta(days=i) for i in range(5)]
    next_5_str = {d.strftime("%Y-%m-%d") for d in next_5}
    
    forecast_map = {}
    for i, row in fdf.iterrows():
        ds = row["Date"].strftime("%Y-%m-%d")
        # If there's a separate column for ETa
        et0_val = row["ET0"]
        eta_val = row["ETa"]
        forecast_map[ds] = (et0_val, eta_val)
    
    # We'll create a calendar for the current month
    year = today.year
    month = today.month
    
    cal = calendar.Calendar(firstweekday=0)
    # We'll store weeks
    weeks = []
    current_week = []
    
    for day in cal.itermonthdates(year, month):
        current_week.append(day)
        if len(current_week) == 7:
            weeks.append(current_week)
            current_week = []
    if current_week:
        weeks.append(current_week)
    
    # Build HTML
    html = """
    <style>
    table.simple-calendar {
        width: 100%;
        border-collapse: collapse;
    }
    table.simple-calendar th, 
    table.simple-calendar td {
        border: 1px solid #ccc;
        width: 14.28%;
        vertical-align: top;
        height: 80px;
        text-align: left;
        padding: 6px;
    }
    table.simple-calendar th {
        background: #5DADE2;
        color: #fff;
    }
    .day-number {
        font-weight: bold;
    }
    .dimmed {
        color: #aaa;
    }
    .forecast-box {
        font-size: 12px;
        margin-top: 4px;
    }
    </style>
    <table class="simple-calendar">
    <tr>
      <th>Mon</th><th>Tue</th><th>Wed</th>
      <th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th>
    </tr>
    """
    
    for week in weeks:
        html += "<tr>"
        for d in week:
            day_str = d.strftime("%Y-%m-%d")
            display_day = d.day
            # Check if day belongs to this month
            in_month = (d.month == month)
            # We'll dim if not in month
            day_class = ""
            if not in_month:
                day_class = "dimmed"
            
            # Check if it's in next_5_str, and if we have forecast
            cell_content = ""
            if day_str in next_5_str and day_str in forecast_map:
                et0_val, eta_val = forecast_map[day_str]
                cell_content = f"""
                <div class="forecast-box">
                  ETo: {et0_val} mm<br>
                  ETa: {eta_val} mm
                </div>
                """
            
            html += f"<td class='{day_class}'>"
            html += f"<div class='day-number'>{display_day}</div>"
            html += cell_content
            html += "</td>"
        html += "</tr>"
    
    html += "</table>"
    return html

# --------------------------------------------------------------------------------
# 6. SETUP TAB
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown('<p style="font-size:16px;">1. Select Crop</p>', unsafe_allow_html=True)
    crop_list = list(CROP_DATABASE.keys())
    selected_crop = st.selectbox("Choose your crop", crop_list)
    st.write(f"**Selected Crop**: {selected_crop}")
    st.write(f"- Kc_mid={CROP_DATABASE[selected_crop]['Kc_mid']}, Kc_end={CROP_DATABASE[selected_crop]['Kc_end']}")
    st.write(f"- Kcb_mid={CROP_DATABASE[selected_crop]['Kcb_mid']}, Kcb_end={CROP_DATABASE[selected_crop]['Kcb_end']}")
    
    st.markdown('<p style="font-size:16px;">2. Weather Data</p>', unsafe_allow_html=True)
    weather_file = st.file_uploader("Upload CSV with columns [Date,ET0,Precipitation,Irrigation], or rely on forecast", 
                                    type=["csv","txt"])

    st.markdown('<p style="font-size:16px;">3. Crop Stage Data</p>', unsafe_allow_html=True)
    use_custom_stage = st.checkbox("Upload custom Crop Stage Data?", value=False)
    st.write("*Otherwise, we compute automatically from known stage durations.*")
    custom_crop_file = None
    if use_custom_stage:
        custom_crop_file = st.file_uploader("Upload Crop Stage CSV (Start_Day, End_Day, Kcb, Root_Depth_mm, p, Ke)",
                                            type=["csv","txt"])

    st.markdown('<p style="font-size:16px;">4. Soil Layers Data</p>', unsafe_allow_html=True)
    soil_file = st.file_uploader("Upload soil data (Depth_mm, FC, WP, TEW, REW) or use default", 
                                 type=["csv","txt"])
    
    st.markdown('<p style="font-size:16px;">5. Additional Options</p>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        track_drainage = st.checkbox("Track Drainage?", value=True)
        enable_yield = st.checkbox("Enable Yield Estimation?", value=False)
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
            totalN = st.number_input("Total N input (kg/ha)?", min_value=0.0, value=100.0)
            lf = st.number_input("Leaching Fraction (0-1)?", min_value=0.0, max_value=1.0, value=0.1)
    
    st.markdown('<p style="font-size:16px;">6. ETA Forecast (5-day) Options</p>', unsafe_allow_html=True)
    enable_forecast = st.checkbox("Enable 5-Day Forecast?", value=True)
    lat_, lon_ = 0.0, 0.0
    if enable_forecast:
        lat_ = st.number_input("Latitude?", value=35.0)
        lon_ = st.number_input("Longitude?", value=-80.0)
    
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
        st.success("Simulation is running...")
        st.session_state["simulation_done"] = True
        
        # Build or load weather df
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
                st.warning("Could not parse weather file. Trying forecast if enabled.")
                if enable_forecast:
                    today = datetime.now().date()
                    hist_start = today - timedelta(days=5)
                    hist_end   = today - timedelta(days=1)
                    fore_start = today
                    fore_end   = today + timedelta(days=4)
                    hist_wdf   = fetch_weather_data(lat_, lon_, hist_start, hist_end, forecast=False)
                    fore_wdf   = fetch_weather_data(lat_, lon_, fore_start, fore_end, forecast=True)
                    if hist_wdf is not None and fore_wdf is not None:
                        wdf = pd.concat([hist_wdf, fore_wdf]).sort_values("Date").reset_index(drop=True)
                    else:
                        wdf = pd.DataFrame()
                else:
                    wdf = pd.DataFrame()
        else:
            # No file => try forecast if enabled
            if enable_forecast:
                today = datetime.now().date()
                hist_start = today - timedelta(days=5)
                hist_end   = today - timedelta(days=1)
                fore_start = today
                fore_end   = today + timedelta(days=4)
                hist_wdf   = fetch_weather_data(lat_, lon_, hist_start, hist_end, forecast=False)
                fore_wdf   = fetch_weather_data(lat_, lon_, fore_start, fore_end, forecast=True)
                if hist_wdf is not None and fore_wdf is not None:
                    wdf = pd.concat([hist_wdf, fore_wdf]).sort_values("Date").reset_index(drop=True)
                else:
                    wdf = pd.DataFrame()
            else:
                st.warning("No weather file & forecast disabled => no data. Stopping.")
                st.stop()
        
        if wdf is None or wdf.empty:
            st.error("No valid weather data. Stopping.")
            st.stop()
        
        # Build or load stage df
        if use_custom_stage and custom_crop_file is not None:
            try:
                stage_df = pd.read_csv(custom_crop_file)
            except:
                st.error("Could not parse stage file => using auto.")
                stage_df = create_auto_stages_for_crop(selected_crop)
        elif use_custom_stage and custom_crop_file is None:
            st.error("You checked custom stage but didn't upload => using auto.")
            stage_df = create_auto_stages_for_crop(selected_crop)
        else:
            stage_df = create_auto_stages_for_crop(selected_crop)
        
        # Soil data
        if soil_file is not None:
            try:
                soil_df = pd.read_csv(soil_file)
            except:
                st.error("Could not parse soil file => using default.")
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
        
        # Run simulation
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
        st.success("Simulation completed! Go to the 'Results' tab to see the output.")

# --------------------------------------------------------------------------------
# 7. RESULTS TAB
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
                # Download
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=600)
                buf.seek(0)
                st.download_button("Download Plot", buf, file_name="et_components.png", mime="image/png")

            elif selected_plot == "Root Zone Depletion":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Dr_start (mm)"], label="Dr start")
                ax.plot(results_df["Date"], results_df["Dr_end (mm)"],   label="Dr end")
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
                ax.plot(results_df["Date"], results_df["SW_root_end (mm)"],   label="RootZ End")
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
            
            # Also show a separate table for the next 5 days ETo & ETa
            st.markdown("## Next 5 Days Forecast (ETo & ETa)")
            f5_df = st.session_state.forecast_5days_df.copy()
            if not f5_df.empty:
                # If we want an ETa column, let's define ETa = ET0 * some Kc or user approach
                # But let's assume the forecast_5days_df might only have "ET0".
                # We'll define ETa = ET0 for demonstration, unless you have a separate approach
                if "ETa" not in f5_df.columns:
                    f5_df["ETa"] = f5_df["ET0"]
                f5_df_display = f5_df[["Date","ET0","ETa"]].copy()
                st.dataframe(f5_df_display)
                st.download_button("Download Next 5 Days Forecast",
                                   f5_df_display.to_csv(index=False),
                                   file_name="next_5days_forecast.csv",
                                   mime="text/csv")
            else:
                st.write("No separate 5-day forecast found or forecast disabled.")

# --------------------------------------------------------------------------------
# 8. IRRIGATION CALENDAR TAB
# --------------------------------------------------------------------------------
with irrig_calendar_tab:
    if not st.session_state.get("simulation_done", False):
        st.info("Please run the simulation first in 'Setup Simulation'.")
    else:
        # Show a monthly calendar. Only the next 5 days from the forecast show ETo & ETa.
        st.markdown("## Irrigation Calendar (Next 5 Days ETo & ETa Only)")
        cal_html = generate_monthly_calendar_html()
        st.markdown(cal_html, unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>',
    unsafe_allow_html=True
)
