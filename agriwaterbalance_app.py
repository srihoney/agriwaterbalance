import os
import io
import calendar
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64

# --------------------------------------------------------------------------------
# 1. Large Kc Database: merging Pereira et al. (2021a,b) plus additional references
# --------------------------------------------------------------------------------
# References:
#    (a) Standard single and basal crop coefficients for vegetable crops, an update of FAO56 (Pereira et al., 2021)
#    (b) Standard single and basal crop coefficients for field crops. Updates to FAO56 (Pereira et al., 2021)
# Additional typical values from broader FAO-56 literature are included.
# For each crop, store Kc_mid, Kc_end, Kcb_mid, Kcb_end and a default total crop cycle length.
CROP_DATABASE = {
    # ---------- Vegetables ----------
    "Carrot": {"Kc_mid": 1.05, "Kc_end": 0.95, "Kcb_mid": 1.00, "Kcb_end": 0.90, "total_days_default": 90},
    "Beet": {"Kc_mid": 1.10, "Kc_end": 0.95, "Kcb_mid": 1.05, "Kcb_end": 0.85, "total_days_default": 100},
    "Garlic": {"Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65, "total_days_default": 120},
    "Onion (fresh)": {"Kc_mid": 1.10, "Kc_end": 0.80, "Kcb_mid": 1.05, "Kcb_end": 0.75, "total_days_default": 110},
    "Onion (dry)": {"Kc_mid": 1.10, "Kc_end": 0.65, "Kcb_mid": 1.05, "Kcb_end": 0.60, "total_days_default": 120},
    "Cabbage": {"Kc_mid": 1.00, "Kc_end": 0.90, "Kcb_mid": 0.95, "Kcb_end": 0.85, "total_days_default": 90},
    "Lettuce": {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 65},
    "Spinach": {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 55},
    "Broccoli": {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 80},
    "Cauliflower": {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90, "total_days_default": 95},
    "Green bean": {"Kc_mid": 1.05, "Kc_end": 0.90, "Kcb_mid": 1.00, "Kcb_end": 0.85, "total_days_default": 75},
    "Tomato (fresh)": {"Kc_mid": 1.15, "Kc_end": 0.80, "Kcb_mid": 1.10, "Kcb_end": 0.75, "total_days_default": 120},
    "Tomato (proc)": {"Kc_mid": 1.15, "Kc_end": 0.70, "Kcb_mid": 1.10, "Kcb_end": 0.65, "total_days_default": 110},
    "Pepper": {"Kc_mid": 1.15, "Kc_end": 0.90, "Kcb_mid": 1.10, "Kcb_end": 0.85, "total_days_default": 130},
    "Eggplant": {"Kc_mid": 1.10, "Kc_end": 0.90, "Kcb_mid": 1.05, "Kcb_end": 0.85, "total_days_default": 130},
    "Zucchini": {"Kc_mid": 1.05, "Kc_end": 0.80, "Kcb_mid": 1.00, "Kcb_end": 0.75, "total_days_default": 60},
    "Cucumber": {"Kc_mid": 1.00, "Kc_end": 0.75, "Kcb_mid": 0.95, "Kcb_end": 0.70, "total_days_default": 70},
    "Melon": {"Kc_mid": 1.05, "Kc_end": 0.65, "Kcb_mid": 1.00, "Kcb_end": 0.60, "total_days_default": 85},
    "Watermelon": {"Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65, "total_days_default": 90},
    "Pumpkin": {"Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65, "total_days_default": 100},
    "Okra": {"Kc_mid": 1.15, "Kc_end": 0.75, "Kcb_mid": 1.10, "Kcb_end": 0.70, "total_days_default": 100},
    "Basil": {"Kc_mid": 1.00, "Kc_end": 0.80, "Kcb_mid": 0.95, "Kcb_end": 0.75, "total_days_default": 60},
    "Parsley": {"Kc_mid": 1.00, "Kc_end": 0.85, "Kcb_mid": 0.95, "Kcb_end": 0.80, "total_days_default": 70},
    "Coriander": {"Kc_mid": 1.00, "Kc_end": 0.85, "Kcb_mid": 0.95, "Kcb_end": 0.80, "total_days_default": 65},
    "Celery": {"Kc_mid": 1.05, "Kc_end": 0.90, "Kcb_mid": 1.00, "Kcb_end": 0.85, "total_days_default": 120},
    "Turnip": {"Kc_mid": 1.05, "Kc_end": 0.80, "Kcb_mid": 1.00, "Kcb_end": 0.75, "total_days_default": 85},
    "Radish": {"Kc_mid": 1.00, "Kc_end": 0.80, "Kcb_mid": 0.95, "Kcb_end": 0.75, "total_days_default": 45},

    # ---------- Field Crops ----------
    "Wheat": {"Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30, "total_days_default": 150},
    "Barley": {"Kc_mid": 1.15, "Kc_end": 0.25, "Kcb_mid": 1.10, "Kcb_end": 0.20, "total_days_default": 130},
    "Maize": {"Kc_mid": 1.20, "Kc_end": 0.60, "Kcb_mid": 1.15, "Kcb_end": 0.55, "total_days_default": 140},
    "Rice": {"Kc_mid": 1.20, "Kc_end": 0.90, "Kcb_mid": 1.15, "Kcb_end": 0.85, "total_days_default": 160},
    "Sorghum": {"Kc_mid": 1.05, "Kc_end": 0.40, "Kcb_mid": 1.00, "Kcb_end": 0.35, "total_days_default": 120},
    "Soybean": {"Kc_mid": 1.15, "Kc_end": 0.50, "Kcb_mid": 1.10, "Kcb_end": 0.45, "total_days_default": 130},
    "Bean": {"Kc_mid": 1.15, "Kc_end": 0.90, "Kcb_mid": 1.10, "Kcb_end": 0.85, "total_days_default": 95},
    "Peanut": {"Kc_mid": 1.10, "Kc_end": 0.60, "Kcb_mid": 1.05, "Kcb_end": 0.55, "total_days_default": 135},
    "Cotton": {"Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60, "total_days_default": 160},
    "Sugarbeet": {"Kc_mid": 1.20, "Kc_end": 0.60, "Kcb_mid": 1.15, "Kcb_end": 0.55, "total_days_default": 180},
    "Sugarcane": {"Kc_mid": 1.25, "Kc_end": 1.10, "Kcb_mid": 1.20, "Kcb_end": 1.05, "total_days_default": 300},
    "Sunflower": {"Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30, "total_days_default": 120},
    "Rapeseed": {"Kc_mid": 1.15, "Kc_end": 0.40, "Kcb_mid": 1.10, "Kcb_end": 0.35, "total_days_default": 150},
    "Mustard": {"Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30, "total_days_default": 120},
    "Faba bean": {"Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60, "total_days_default": 130},
    "Chickpea": {"Kc_mid": 1.15, "Kc_end": 0.25, "Kcb_mid": 1.10, "Kcb_end": 0.20, "total_days_default": 120},
    "Millet": {"Kc_mid": 1.10, "Kc_end": 0.40, "Kcb_mid": 1.05, "Kcb_end": 0.35, "total_days_default": 100},
    "Quinoa": {"Kc_mid": 1.05, "Kc_end": 0.45, "Kcb_mid": 1.00, "Kcb_end": 0.40, "total_days_default": 120},
    "Lentil": {"Kc_mid": 1.10, "Kc_end": 0.25, "Kcb_mid": 1.05, "Kcb_end": 0.20, "total_days_default": 110},
    "Potato": {"Kc_mid": 1.15, "Kc_end": 0.75, "Kcb_mid": 1.10, "Kcb_end": 0.70, "total_days_default": 110}
}

# --------------------------------------------------------------------------------
# 2. Configure Requests Session with Retries
# --------------------------------------------------------------------------------
session = requests.Session()
retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
session.mount('https://', HTTPAdapter(max_retries=retries))

# --------------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Logo
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Advanced AgriWaterBalance", layout="wide")

try:
    with open("logo.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    logo_url = f"data:image/png;base64,{encoded_string}"
except FileNotFoundError:
    logo_url = ""

# Custom CSS for small fonts on Setup headers and for clean high‐resolution graphs
st.markdown(f"""
    <style>
    body {{ margin: 0; padding: 0; }}
    .header-container {{
        position: relative;
        background-color: #1E3A8A;
        padding: 20px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .header-logo {{
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
    }}
    .header-logo img {{ width: 100px; height: auto; }}
    .header-title {{ color: white; font-size: 36px; font-weight: bold; text-align: center; }}
    .sub-header {{ color: #1E3A8A; font-size: 24px; font-weight: bold; margin-top: 20px; }}
    .small-header {{ font-size: 14px; font-weight: bold; color: #1E3A8A; margin-top: 10px; }}
    .footer {{
        background-color: #1E3A8A; color: white; padding: 10px; text-align: center;
        position: fixed; bottom: 0; width: 100%; border-radius: 5px 5px 0 0;
    }}
    .stButton>button {{
        background-color: #2563EB; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px;
    }}
    .stButton>button:hover {{ background-color: #1E40AF; }}
    .stFileUploader {{
        border: 2px dashed #1E3A8A; border-radius: 5px; padding: 10px;
    }}
    .stSelectbox {{ background-color: #F1F5F9; border-radius: 5px; }}
    .stNumberInput input {{ background-color: #F1F5F9; border-radius: 5px; }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-container">
        <div class="header-logo">
            <img src="{logo_url}" alt="Logo">
        </div>
        <div class="header-title">Advanced AgriWaterBalance</div>
    </div>
""", unsafe_allow_html=True)
st.markdown("**A Professional, All-in-One Tool for Crop Water Management**", unsafe_allow_html=True)

# Create 3 main tabs: Setup, Results, Irrigation Calendar
setup_tab, results_tab, irrig_calendar_tab = st.tabs(["Setup Simulation", "Results", "Irrigation Calendar"])

# --------------------------------------------------------------------------------
# 4. Session State
# --------------------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'weather_cache' not in st.session_state:
    st.session_state.weather_cache = {}
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.now().date()

current_date = datetime.now().date()
if st.session_state.last_reset_date != current_date:
    st.session_state.api_calls = 0
    st.session_state.last_reset_date = current_date

# --------------------------------------------------------------------------------
# 5. Water Balance Functions (SIMDualKc-like approach)
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

def fetch_weather_data(lat, lon, start_date, end_date, forecast=True, manual_data=None):
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}_{forecast}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    # Manual input option removed—always use forecast (API)
    if forecast:
        if st.session_state.api_calls >= 1000:
            st.warning("Daily API call limit reached.")
            return None
        if lat == 0 and lon == 0:
            st.warning("Invalid lat/lon.")
            return None
        today = datetime.now().date()
        maxf = today + timedelta(days=5)
        if start_date < today or end_date > maxf:
            st.warning("Adjusting forecast to next 5 days.")
            start_date = today
            end_date = today + timedelta(days=4)
        api_key = "fe2d869569674a4afbfca57707bdf691"  # Replace with your API key
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            st.session_state.api_calls += 1
            data = resp.json()
            dd = {}
            for entry in data['list']:
                dt_ = datetime.fromtimestamp(entry['dt']).date()
                if start_date <= dt_ <= end_date:
                    ds = dt_.strftime("%Y-%m-%d")
                    if ds not in dd:
                        dd[ds] = {
                            'tmax': entry['main']['temp_max'],
                            'tmin': entry['main']['temp_min'],
                            'precip': entry.get('rain', {}).get('3h', 0)
                        }
                    else:
                        dd[ds]['tmax'] = max(dd[ds]['tmax'], entry['main']['temp_max'])
                        dd[ds]['tmin'] = min(dd[ds]['tmin'], entry['main']['temp_min'])
                        dd[ds]['precip'] += entry.get('rain', {}).get('3h', 0)
            ds_sorted = sorted(dd.keys())
            dates, ET0_list, precip_list = [], [], []
            for dsi in ds_sorted:
                d_date = pd.to_datetime(dsi)
                dates.append(d_date)
                tmax = dd[dsi]['tmax']
                tmin = dd[dsi]['tmin']
                if tmax < tmin:
                    tmax, tmin = tmin, tmax
                Ra = 10.0
                Tmean = (tmax + tmin) / 2.0
                eto_val = 0.0023 * Ra * (Tmean + 17.8) * ((tmax - tmin) ** 0.5)
                ET0_list.append(max(0, eto_val))
                precip_list.append(dd[dsi]['precip'])
            wdf = pd.DataFrame({
                "Date": dates,
                "ET0": ET0_list,
                "Precipitation": precip_list,
                "Irrigation": [0] * len(dates)
            }).sort_values("Date").reset_index(drop=True)
            st.session_state.weather_cache[cache_key] = wdf
            return wdf
        except:
            st.error("Unable to fetch forecast data.")
            return None
    else:
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
            r = session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            dts = pd.date_range(start_date, end_date)
            ET0_list, precip_list = [], []
            for dt_ in dts:
                ds = dt_.strftime("%Y%m%d")
                rad_ = data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(ds, 0)
                eto_v = rad_ * 0.2
                ET0_list.append(eto_v)
                prec = data['properties']['parameter']['PRECTOTCORR'].get(ds, 0)
                precip_list.append(prec)
            wdf = pd.DataFrame({
                "Date": dts,
                "ET0": ET0_list,
                "Precipitation": precip_list,
                "Irrigation": [0] * len(dts)
            })
            st.session_state.weather_cache[cache_key] = wdf
            return wdf
        except:
            st.warning("Unable to fetch historical weather data.")
            return None

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
    day_root = np.zeros(NDAYS)
    
    for i in range(len(crop_stages_df) - 1):
        st_i = int(crop_stages_df.iloc[i]['Start_Day']) - 1
        en_i = int(crop_stages_df.iloc[i]['End_Day']) - 1
        if en_i < 0:
            continue
        en_i = min(en_i, NDAYS - 1)
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
    
    i_last = len(crop_stages_df) - 1
    st_l = int(crop_stages_df.iloc[i_last]['Start_Day']) - 1
    en_l = int(crop_stages_df.iloc[i_last]['End_Day']) - 1
    if st_l < 0:
        st_l = 0
    if en_l < 0:
        en_l = 0
    if en_l > NDAYS - 1:
        en_l = NDAYS - 1
    if st_l <= en_l:
        day_kcb[st_l:en_l+1]  = crop_stages_df.iloc[i_last]['Kcb']
        day_p[st_l:en_l+1]    = crop_stages_df.iloc[i_last]['p']
        day_ke[st_l:en_l+1]   = crop_stages_df.iloc[i_last]['Ke']
        day_root[st_l:en_l+1] = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    if en_l < NDAYS - 1:
        day_kcb[en_l+1:] = crop_stages_df.iloc[i_last]['Kcb']
        day_p[en_l+1:]   = crop_stages_df.iloc[i_last]['p']
        day_ke[en_l+1:]  = crop_stages_df.iloc[i_last]['Ke']
        day_root[en_l+1:] = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    
    if dynamic_root:
        root_lin = np.linspace(init_root, max_root, min(days_to_max, NDAYS)).tolist()
        if NDAYS > days_to_max:
            root_lin += [max_root] * (NDAYS - days_to_max)
        day_root = np.array(root_lin[:NDAYS])
    
    TEW = soil_df['TEW'].sum()
    REW = soil_df['REW'].sum()
    E_count = REW
    
    SW_layers = [soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm'] for j in range(len(soil_df))]
    
    results = []
    cumIrr = 0
    cumPrec = 0
    
    for d_i in range(NDAYS):
        date_i = weather_df.iloc[d_i]['Date']
        ET0_i = max(0, weather_df.iloc[d_i]['ET0'])
        PR_i = max(0, weather_df.iloc[d_i]['Precipitation'])
        IR_i = max(0, weather_df.iloc[d_i]['Irrigation'])
        
        cumIrr += IR_i
        cumPrec += PR_i
        
        Kcb_i = day_kcb[d_i]
        p_i = day_p[d_i]
        ke0_i = day_ke[d_i]
        rd_i = max(1, day_root[d_i])
        
        tot_depth = 0
        sum_FC = 0
        sum_WP = 0
        SW_root = 0
        for j in range(len(SW_layers)):
            layer_d = soil_df.iloc[j]['Depth_mm']
            FC_j = soil_df.iloc[j]['FC']
            WP_j = soil_df.iloc[j]['WP']
            new_d = tot_depth + layer_d
            fraction = 0
            if new_d <= rd_i:
                fraction = 1.0
            elif tot_depth < rd_i < new_d:
                fraction = (rd_i - tot_depth) / layer_d
            sw_j = SW_layers[j]
            if fraction > 0:
                sum_FC += FC_j * layer_d * fraction
                sum_WP += WP_j * layer_d * fraction
                SW_root += sw_j * fraction
            tot_depth = new_d
        TAW_ = (sum_FC - sum_WP)
        RAW_ = p_i * TAW_
        Dr_ = (sum_FC - SW_root)
        
        Ks_ = compute_Ks(Dr_, RAW_, TAW_)
        Kr_ = compute_Kr(TEW, REW, E_count)
        Ke_ = Kr_ * ke0_i
        
        ETc_ = (Kcb_i * Ks_ + Ke_) * ET0_i
        Etc_trans = Kcb_i * Ks_ * ET0_i
        Etc_evap = Ke_ * ET0_i
        infiltration = PR_i + IR_i
        runoff = 0
        
        excess = infiltration - ETc_
        drainage = 0
        if track_drainage:
            for j in range(len(SW_layers)):
                layer_fc = soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']
                gap_j = layer_fc - SW_layers[j]
                if gap_j > 0 and excess > 0:
                    added = min(excess, gap_j)
                    SW_layers[j] += added
                    excess -= added
            drainage = max(0, excess)
            for j in range(len(SW_layers)):
                max_sw = soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']
                min_sw = soil_df.iloc[j]['WP'] * soil_df.iloc[j]['Depth_mm']
                SW_layers[j] = max(min_sw, min(max_sw, SW_layers[j]))
        else:
            drainage = 0
        
        tr_rem = Etc_trans
        if tr_rem > 0 and SW_root > 0:
            tot_depth = 0
            for j in range(len(SW_layers)):
                layer_d = soil_df.iloc[j]['Depth_mm']
                new_d = tot_depth + layer_d
                fraction = 0
                if new_d <= rd_i:
                    fraction = 1.0
                elif tot_depth < rd_i < new_d:
                    fraction = (rd_i - tot_depth) / layer_d
                tot_depth = new_d
                if fraction > 0:
                    fc_j = soil_df.iloc[j]['FC'] * layer_d
                    wp_j = soil_df.iloc[j]['WP'] * layer_d
                    available_j = SW_layers[j] - wp_j
                    portion = tr_rem * fraction
                    actual_remove = min(portion, available_j)
                    SW_layers[j] -= actual_remove
                    tr_rem -= actual_remove
                    if tr_rem <= 0:
                        break
        
        ev_rem = Etc_evap
        if ev_rem > 0:
            fc_0 = soil_df.iloc[0]['FC'] * soil_df.iloc[0]['Depth_mm']
            wp_0 = soil_df.iloc[0]['WP'] * soil_df.iloc[0]['Depth_mm']
            available_0 = SW_layers[0] - wp_0
            actual_rm = min(ev_rem, available_0)
            SW_layers[0] -= actual_rm
            ev_rem -= actual_rm
        
        if infiltration >= 4.0:
            E_count = 0.0
        else:
            E_count += Etc_evap
        E_count = max(0, min(E_count, TEW))
        
        new_SWroot = 0
        tot_depth = 0
        sum_FC2 = 0
        for j in range(len(SW_layers)):
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
        
        yield_val = None
        if enable_yield:
            if Ym > 0 and Ky > 0 and ETc_ > 0:
                y_val = Ym * (1 - Ky * (1 - (ETc_ / (Kcb_i * ET0_i + 1e-9))))
                yield_val = max(0, y_val)
            if use_transp and WP_yield > 0:
                yield_val = WP_yield * Etc_trans
        
        leaching_val = 0
        if enable_leaching:
            leaching_val = drainage * 10 * (nitrate_conc * 1e-6) * 1000
        
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
        if yield_val is not None:
            day_out["Yield (ton/ha)"] = yield_val
        if enable_leaching:
            day_out["Leaching (kg/ha)"] = leaching_val
        for j in range(len(SW_layers)):
            day_out[f"Layer{j}_SW (mm)"] = SW_layers[j]
        
        results.append(day_out)
    
    outdf = pd.DataFrame(results)
    return outdf

def create_auto_stages_for_crop(crop_name):
    total_d = CROP_DATABASE[crop_name]["total_days_default"]
    init_d = int(total_d * 0.2)
    dev_d = int(total_d * 0.3)
    mid_d = int(total_d * 0.3)
    late_d = total_d - (init_d + dev_d + mid_d)
    kcb_mid = CROP_DATABASE[crop_name]["Kcb_mid"]
    kcb_end = CROP_DATABASE[crop_name]["Kcb_end"]
    stages = []
    stages.append({"Start_Day": 1, "End_Day": init_d, "Kcb": 0.15, "Root_Depth_mm": 100, "p": 0.5, "Ke": 1.0})
    stages.append({"Start_Day": init_d + 1, "End_Day": init_d + dev_d, "Kcb": kcb_mid, "Root_Depth_mm": 400, "p": 0.5, "Ke": 0.5})
    stages.append({"Start_Day": init_d + dev_d + 1, "End_Day": init_d + dev_d + mid_d, "Kcb": kcb_mid, "Root_Depth_mm": 600, "p": 0.5, "Ke": 0.2})
    stages.append({"Start_Day": init_d + dev_d + mid_d + 1, "End_Day": total_d, "Kcb": kcb_end, "Root_Depth_mm": 600, "p": 0.5, "Ke": 0.1})
    return pd.DataFrame(stages)

def produce_irrigation_calendar(results_df):
    """
    Produces a wall-calendar–style display for the current month (next 30 days).
    The calendar grid shows the day number, ET0, ETa, and recommended irrigation.
    Recommended irrigation is based on a simple threshold: if Dr_end > RAW then recommend (Dr_end - RAW) mm.
    """
    import calendar
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    cal = calendar.monthcalendar(current_year, current_month)
    
    # Create a dictionary of results for the current month from results_df
    # We assume Date column is in datetime format
    results_df["Date"] = pd.to_datetime(results_df["Date"])
    month_data = results_df[results_df["Date"].dt.month == current_month].copy()
    # We'll create a dict keyed by day
    data_dict = {}
    for _, row in month_data.iterrows():
        day = row["Date"].day
        rec = 0
        if "Dr_end (mm)" in row and "RAW (mm)" in row and not pd.isnull(row["Dr_end (mm)"]) and not pd.isnull(row["RAW (mm)"]):
            if row["Dr_end (mm)"] > row["RAW (mm)"]:
                rec = round(row["Dr_end (mm)"] - row["RAW (mm)"], 1)
        data_dict[day] = {
            "ET0": round(row["ET0 (mm)"], 1),
            "ETa": round(row["ETa (mm)"], 1),
            "Irrigation": rec
        }
    
    # Build HTML calendar table
    html = f"<h3 style='text-align: center;'>{now.strftime('%B %Y')}</h3>"
    html += "<table style='width:100%; border-collapse: collapse; text-align: center;'>"
    # Table header with weekday names
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    html += "<tr style='background-color:#1E3A8A; color:white;'>"
    for wd in weekdays:
        html += f"<th style='padding: 5px; border: 1px solid #ddd; font-size: 12px;'>{wd}</th>"
    html += "</tr>"
    
    for week in cal:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += "<td style='padding: 5px; border: 1px solid #ddd; font-size: 12px;'>&nbsp;</td>"
            else:
                cell_data = data_dict.get(day, None)
                if cell_data:
                    cell_text = f"<strong>{day}</strong><br/>ET0: {cell_data['ET0']} mm<br/>ETa: {cell_data['ETa']} mm<br/>Irrig: {cell_data['Irrigation']} mm"
                else:
                    cell_text = f"<strong>{day}</strong><br/>&nbsp;"
                html += f"<td style='padding: 5px; border: 1px solid #ddd; font-size: 12px; vertical-align: top;'>{cell_text}</td>"
        html += "</tr>"
    html += "</table>"
    return html

def download_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
    buf.seek(0)
    return buf

# --------------------------------------------------------------------------------
# 6. SETUP TAB
# Use small headers via CSS class "small-header"
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown("<span class='small-header'>1) Select Crop</span>", unsafe_allow_html=True)
    crop_list = list(CROP_DATABASE.keys())
    selected_crop = st.selectbox("Choose your crop", crop_list)
    st.write(f"**Selected Crop**: {selected_crop}")
    st.write(f"- Kc_mid = {CROP_DATABASE[selected_crop]['Kc_mid']},  Kc_end = {CROP_DATABASE[selected_crop]['Kc_end']}")
    st.write(f"- Kcb_mid = {CROP_DATABASE[selected_crop]['Kcb_mid']}, Kcb_end = {CROP_DATABASE[selected_crop]['Kcb_end']}")
    
    st.markdown("<span class='small-header'>2) Weather Data</span>", unsafe_allow_html=True)
    weather_file = st.file_uploader("Upload CSV with Date,ET0,Precipitation,Irrigation, or rely on forecast", type=["csv","txt"])
    
    st.markdown("<span class='small-header'>3) Crop Stage Data</span>", unsafe_allow_html=True)
    use_custom_stage = st.checkbox("Upload custom Crop Stage Data?", value=False)
    st.write("*Otherwise, we'll compute automatically from known stage durations.*")
    custom_crop_file = None
    if use_custom_stage:
        custom_crop_file = st.file_uploader("Upload Crop Stage CSV (columns: Start_Day, End_Day, Kcb, Root_Depth_mm, p, Ke)",
                                             type=["csv","txt"])
    
    st.markdown("<span class='small-header'>4) Soil Layers Data</span>", unsafe_allow_html=True)
    soil_file = st.file_uploader("Upload soil data (Depth_mm, FC, WP, TEW, REW) or use default", type=["csv","txt"])
    
    st.markdown("<span class='small-header'>5) Additional Options</span>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        track_drainage = st.checkbox("Track Drainage", value=True)
        enable_yield = st.checkbox("Enable Yield Estimation?", value=False)
        if enable_yield:
            st.write("**Yield Options**")
            ym_ = st.number_input("Max Yield (ton/ha)?", min_value=0.0, value=10.0)
            ky_ = st.number_input("Ky (yield response factor)", min_value=0.0, value=1.0)
            use_transp = st.checkbox("Use Transp-based approach (WP_yield)?", value=False)
            if use_transp:
                wp_yield = st.number_input("WP_yield (ton/ha per mm)?", min_value=0.0, value=0.012, step=0.001)
            else:
                wp_yield = 0
        else:
            ym_ = ky_ = 0
            use_transp = False
            wp_yield = 0
    with colB:
        enable_leaching = st.checkbox("Enable Leaching?", value=False)
        nitrate_conc = 10.0
        totalN = 100.0
        lf = 0.1
        if enable_leaching:
            nitrate_conc = st.number_input("Nitrate mg/L", min_value=0.0, value=10.0)
            totalN = st.number_input("Total N input (kg/ha)?", min_value=0.0, value=100.0)
            lf = st.number_input("Leaching Fraction (0-1)?", min_value=0.0, max_value=1.0, value=0.1)
    
    st.markdown("<span class='small-header'>6) ETA Forecast (5-day) Options</span>", unsafe_allow_html=True)
    enable_forecast = st.checkbox("Enable 5-Day Forecast?", value=True)
    # Removed manual forecast option—always use API forecast.
    lat_ = st.number_input("Latitude?", value=35.0)
    lon_ = st.number_input("Longitude?", value=-80.0)
    manual_forecast_data = None
    
    st.markdown("<span class='small-header'>7) Dynamic Root Growth?</span>", unsafe_allow_html=True)
    dynamic_root = st.checkbox("Enable dynamic root growth?", value=False)
    init_rd = 300
    max_rd = 800
    days_mx = 60
    if dynamic_root:
        init_rd = st.number_input("Initial Root Depth (mm)", min_value=50, value=300)
        max_rd = st.number_input("Max Root Depth (mm)", min_value=50, value=800)
        days_mx = st.number_input("Days to reach max root depth?", min_value=1, value=60)
    
    st.markdown("<span class='small-header'>8) Run Simulation</span>", unsafe_allow_html=True)
    run_button = st.button("Run Simulation")
    
    if run_button:
        st.success("Simulation is now complete! Please open the 'Results' tab to see them.")
        st.session_state["simulation_done"] = True
        
        if weather_file is not None:
            try:
                wdf = pd.read_csv(weather_file)
                if "Date" not in wdf.columns:
                    st.error("Weather file missing 'Date' column.")
                    st.stop()
                if pd.api.types.is_string_dtype(wdf["Date"]):
                    wdf["Date"] = pd.to_datetime(wdf["Date"])
                wdf = wdf.sort_values("Date").reset_index(drop=True)
            except:
                st.warning("Could not parse the weather file. Using forecast.")
                startdt = datetime.now().date()
                enddt = startdt + timedelta(days=4)
                wdf = fetch_weather_data(lat_, lon_, startdt, enddt, forecast=True)
        else:
            if enable_forecast:
                startdt = datetime.now().date()
                enddt = startdt + timedelta(days=4)
                wdf = fetch_weather_data(lat_, lon_, startdt, enddt, forecast=True)
            else:
                st.warning("No weather file and forecast disabled => no data. Stopping.")
                st.stop()
        
        if wdf is None or wdf.empty:
            st.error("No valid weather data. Stopping.")
            st.stop()
        
        if use_custom_stage and custom_crop_file is not None:
            try:
                stage_df = pd.read_csv(custom_crop_file)
            except:
                st.error("Could not parse custom stage file. Using auto-generated stages.")
                stage_df = create_auto_stages_for_crop(selected_crop)
        elif use_custom_stage and custom_crop_file is None:
            st.error("Custom stage selected but no file uploaded. Using auto-generated stages.")
            stage_df = create_auto_stages_for_crop(selected_crop)
        else:
            stage_df = create_auto_stages_for_crop(selected_crop)
        
        if soil_file is not None:
            try:
                soil_df = pd.read_csv(soil_file)
            except:
                st.error("Could not read soil file. Using default 2-layer soil.")
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

# --------------------------------------------------------------------------------
# 7. RESULTS TAB
# --------------------------------------------------------------------------------
with results_tab:
    if not st.session_state.get("simulation_done", False):
        st.info("Please run the simulation in the 'Setup Simulation' tab.")
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
            
            # Dropdown for selecting chart type
            chart_choice = st.selectbox("Select Chart", 
                                        ["Daily ET Components", "Root Zone Depletion", "Daily Drainage", "Soil Water in Root Zone", "Yield", "Leaching"])
            # Create and display chart based on selection with clean style (no grid lines) and 600 dpi download option
            if chart_choice == "Daily ET Components":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["ETa (mm)"], label="ETa total")
                ax.plot(results_df["Date"], results_df["ETa_transp (mm)"], label="ETa transp")
                ax.plot(results_df["Date"], results_df["ETa_evap (mm)"], label="ETa evap")
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf = download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="daily_et_components.png", mime="image/png")
            elif chart_choice == "Root Zone Depletion":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Dr_start (mm)"], label="Dr start")
                ax.plot(results_df["Date"], results_df["Dr_end (mm)"], label="Dr end")
                ax.set_xlabel("Date"); ax.set_ylabel("Depletion (mm)")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf = download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="root_zone_depletion.png", mime="image/png")
            elif chart_choice == "Daily Drainage":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Drainage (mm)"], label="Drainage")
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf = download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="daily_drainage.png", mime="image/png")
            elif chart_choice == "Soil Water in Root Zone":
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["SW_root_start (mm)"], label="RootZ start")
                ax.plot(results_df["Date"], results_df["SW_root_end (mm)"], label="RootZ end")
                ax.set_xlabel("Date"); ax.set_ylabel("mm water")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf = download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="soil_water_root_zone.png", mime="image/png")
            elif chart_choice == "Yield" and "Yield (ton/ha)" in results_df.columns:
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Yield (ton/ha)"], label="Yield")
                ax.set_xlabel("Date"); ax.set_ylabel("ton/ha")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf = download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="yield.png", mime="image/png")
            elif chart_choice == "Leaching" and "Leaching (kg/ha)" in results_df.columns:
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(results_df["Date"], results_df["Leaching (kg/ha)"], label="Leaching")
                ax.set_xlabel("Date"); ax.set_ylabel("kg/ha")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf = download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="leaching.png", mime="image/png")
                
# --------------------------------------------------------------------------------
# 8. IRRIGATION CALENDAR TAB
# --------------------------------------------------------------------------------
with irrig_calendar_tab:
    if not st.session_state.get("simulation_done", False):
        st.info("Please run the simulation first.")
    else:
        if st.session_state.results_df is None or st.session_state.results_df.empty:
            st.warning("No results available.")
        else:
            st.markdown("## Irrigation Calendar for the Current Month")
            # Generate a calendar for the current month (next 30 days forecast)
            cal_html = produce_irrigation_calendar(st.session_state.results_df)
            st.markdown(cal_html, unsafe_allow_html=True)

st.markdown('<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>', unsafe_allow_html=True)
