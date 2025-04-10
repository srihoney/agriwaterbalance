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

# --------------------------------------------------------------------------------
# 1. Built-in Kc Database (from Pereira et al. 2021 updates for vegetables & field crops)
# --------------------------------------------------------------------------------
KC_DATABASE = {
    # Vegetables
    "Carrot":        {"Kc_mid": 1.05, "Kc_end": 0.95, "Kcb_mid": 1.00, "Kcb_end": 0.90},
    "Beet":          {"Kc_mid": 1.10, "Kc_end": 0.95, "Kcb_mid": 1.05, "Kcb_end": 0.85},
    "Garlic":        {"Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65},
    "Onion (fresh)": {"Kc_mid": 1.10, "Kc_end": 0.80, "Kcb_mid": 1.05, "Kcb_end": 0.75},
    "Onion (dry)":   {"Kc_mid": 1.10, "Kc_end": 0.65, "Kcb_mid": 1.05, "Kcb_end": 0.60},
    "Cabbage":       {"Kc_mid": 1.00, "Kc_end": 0.90, "Kcb_mid": 0.95, "Kcb_end": 0.85},
    "Lettuce":       {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90},
    "Spinach":       {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90},
    "Broccoli":      {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90},
    "Cauliflower":   {"Kc_mid": 1.00, "Kc_end": 0.95, "Kcb_mid": 0.95, "Kcb_end": 0.90},
    "Green bean":    {"Kc_mid": 1.05, "Kc_end": 0.90, "Kcb_mid": 1.00, "Kcb_end": 0.85},
    "Tomato (fresh)":{"Kc_mid": 1.15, "Kc_end": 0.80, "Kcb_mid": 1.10, "Kcb_end": 0.75},
    "Tomato (proc)": {"Kc_mid": 1.15, "Kc_end": 0.70, "Kcb_mid": 1.10, "Kcb_end": 0.65},
    "Pepper":        {"Kc_mid": 1.15, "Kc_end": 0.90, "Kcb_mid": 1.10, "Kcb_end": 0.85},
    "Eggplant":      {"Kc_mid": 1.10, "Kc_end": 0.90, "Kcb_mid": 1.05, "Kcb_end": 0.85},
    "Zucchini":      {"Kc_mid": 1.05, "Kc_end": 0.80, "Kcb_mid": 1.00, "Kcb_end": 0.75},
    "Cucumber":      {"Kc_mid": 1.00, "Kc_end": 0.75, "Kcb_mid": 0.95, "Kcb_end": 0.70},
    "Melon":         {"Kc_mid": 1.05, "Kc_end": 0.65, "Kcb_mid": 1.00, "Kcb_end": 0.60},
    "Watermelon":    {"Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65},
    "Pumpkin":       {"Kc_mid": 1.05, "Kc_end": 0.70, "Kcb_mid": 1.00, "Kcb_end": 0.65},
    "Okra":          {"Kc_mid": 1.15, "Kc_end": 0.75, "Kcb_mid": 1.10, "Kcb_end": 0.70},
    "Basil":         {"Kc_mid": 1.00, "Kc_end": 0.80, "Kcb_mid": 0.95, "Kcb_end": 0.75},
    "Parsley":       {"Kc_mid": 1.00, "Kc_end": 0.85, "Kcb_mid": 0.95, "Kcb_end": 0.80},
    "Coriander":     {"Kc_mid": 1.00, "Kc_end": 0.85, "Kcb_mid": 0.95, "Kcb_end": 0.80},

    # Field Crops
    "Wheat":         {"Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30},
    "Barley":        {"Kc_mid": 1.15, "Kc_end": 0.25, "Kcb_mid": 1.10, "Kcb_end": 0.20},
    "Maize":         {"Kc_mid": 1.20, "Kc_end": 0.60, "Kcb_mid": 1.15, "Kcb_end": 0.55},
    "Rice":          {"Kc_mid": 1.20, "Kc_end": 0.90, "Kcb_mid": 1.15, "Kcb_end": 0.85},
    "Sorghum":       {"Kc_mid": 1.05, "Kc_end": 0.40, "Kcb_mid": 1.00, "Kcb_end": 0.35},
    "Soybean":       {"Kc_mid": 1.15, "Kc_end": 0.50, "Kcb_mid": 1.10, "Kcb_end": 0.45},
    "Bean":          {"Kc_mid": 1.15, "Kc_end": 0.90, "Kcb_mid": 1.10, "Kcb_end": 0.85},
    "Peanut":        {"Kc_mid": 1.10, "Kc_end": 0.60, "Kcb_mid": 1.05, "Kcb_end": 0.55},
    "Cotton":        {"Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60},
    "Sugarbeet":     {"Kc_mid": 1.20, "Kc_end": 0.60, "Kcb_mid": 1.15, "Kcb_end": 0.55},
    "Sugarcane":     {"Kc_mid": 1.25, "Kc_end": 1.10, "Kcb_mid": 1.20, "Kcb_end": 1.05},
    "Sunflower":     {"Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30},
    "Rapeseed":      {"Kc_mid": 1.15, "Kc_end": 0.40, "Kcb_mid": 1.10, "Kcb_end": 0.35},
    "Mustard":       {"Kc_mid": 1.15, "Kc_end": 0.35, "Kcb_mid": 1.10, "Kcb_end": 0.30},
    "Faba bean":     {"Kc_mid": 1.15, "Kc_end": 0.65, "Kcb_mid": 1.10, "Kcb_end": 0.60},
    "Chickpea":      {"Kc_mid": 1.15, "Kc_end": 0.25, "Kcb_mid": 1.10, "Kcb_end": 0.20},
}

# --------------------------------------------------------------------------------
# 2. Configure Requests Session with Retries
# --------------------------------------------------------------------------------
session = requests.Session()
retries = Retry(total=5, backoff_factor=1.0, 
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"])
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
        margin-top: 0;
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
st.markdown("**A More Advanced Tool for Crop Water Management**", unsafe_allow_html=True)

# Create 2 main tabs
setup_tab, results_tab = st.tabs(["Setup Simulation", "View Results"])

# --------------------------------------------------------------------------------
# 4. Session State
# --------------------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'soil_profile' not in st.session_state:
    st.session_state.soil_profile = None
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
# 5. Core Water Balance & ET Functions
# --------------------------------------------------------------------------------
def compute_Ks(SW, WP, FC, p):
    """
    Computes the water stress coefficient Ks based on:
      Ks = 1 if depletion <= RAW
      else linearly down to 0 at TAW
    SW, WP, FC as fraction volumes, or
    SW in mm if converted, then TAW in mm, etc.
    """
    TAW = (FC - WP) * 1000.0  # total available water (mm/m)
    RAW = p * TAW
    Dr = (FC - SW) * 1000.0
    if Dr <= RAW:
        Ks = 1.0
    else:
        Ks = (TAW - Dr) / ((1 - p) * TAW)
        Ks = max(0.0, min(1.0, Ks))
    return Ks

def compute_Kr(TEW, REW, E):
    """
    Evaporation reduction coefficient for topsoil layer.
    TEW: total evaporable water, REW: readily evaporable water
    E: cumulative evaporation since last major wetting
    """
    if E <= REW:
        Kr = 1.0
    else:
        Kr = (TEW - E) / (TEW - REW)
        Kr = max(0.0, min(1.0, Kr))
    return Kr

def compute_ETc(Kcb, Ks, Ke, ET0):
    """
    Dual crop coefficient approach:
      ETc = (Kcb*Ks + Ke) * ET0
    """
    return (Kcb * Ks + Ke) * ET0

def interpolate_crop_stages(crop_df, total_days):
    """
    Builds daily arrays of (Kcb, root_depth, p, Ke)
    by linear interpolation over user-defined stage intervals.
    """
    Kcb = np.zeros(total_days)
    root_depth = np.zeros(total_days)
    p = np.zeros(total_days)
    Ke = np.zeros(total_days)
    
    for i in range(len(crop_df) - 1):
        start_day = int(crop_df.iloc[i]['Start_Day'])
        end_day   = min(int(crop_df.iloc[i]['End_Day']), total_days)
        
        Kcb_start  = crop_df.iloc[i]['Kcb']
        Kcb_end    = crop_df.iloc[i+1]['Kcb']
        rd_start   = crop_df.iloc[i]['Root_Depth_mm']
        rd_end     = crop_df.iloc[i+1]['Root_Depth_mm']
        p_start    = crop_df.iloc[i]['p']
        p_end      = crop_df.iloc[i+1]['p']
        Ke_start   = crop_df.iloc[i]['Ke']
        Ke_end     = crop_df.iloc[i+1]['Ke']
        
        if i == 0 and start_day > 1:
            Kcb[0:start_day-1]        = 0
            root_depth[0:start_day-1] = rd_start
            p[0:start_day-1]          = p_start
            Ke[0:start_day-1]         = Ke_start
        
        idx = np.arange(start_day - 1, end_day)
        if len(idx) > 0:
            Kcb[idx]        = np.linspace(Kcb_start, Kcb_end, len(idx))
            root_depth[idx] = np.linspace(rd_start, rd_end, len(idx))
            p[idx]          = np.linspace(p_start, p_end, len(idx))
            Ke[idx]         = np.linspace(Ke_start, Ke_end, len(idx))
    
    # fill last stage
    last_i = len(crop_df) - 1
    last_start = int(crop_df.iloc[last_i]['Start_Day'])
    last_end   = min(int(crop_df.iloc[last_i]['End_Day']), total_days)
    lastKcb    = crop_df.iloc[last_i]['Kcb']
    lastRd     = crop_df.iloc[last_i]['Root_Depth_mm']
    lastP      = crop_df.iloc[last_i]['p']
    lastKe     = crop_df.iloc[last_i]['Ke']
    
    if last_start <= total_days:
        idx_last = np.arange(last_start - 1, last_end)
        if len(idx_last) > 0:
            Kcb[idx_last]        = lastKcb
            root_depth[idx_last] = lastRd
            p[idx_last]          = lastP
            Ke[idx_last]         = lastKe
    
    if last_end < total_days:
        Kcb[last_end:]        = lastKcb
        root_depth[last_end:] = lastRd
        p[last_end:]          = lastP
        Ke[last_end:]         = lastKe
    
    return Kcb, root_depth, p, Ke

def SIMdualKc(weather_df, crop_df, soil_df, track_drainage=True, enable_yield=False,
              use_fao33=False, Ym=0, Ky=0, use_transp=False, WP_yield=0,
              enable_leaching=False, leaching_method="", nitrate_conc=0,
              total_N_input=0, leaching_fraction=0,
              enable_dynamic_root=False, initial_root_depth=None, max_root_depth=None, days_to_max=None,
              return_soil_profile=False, initial_SW_layers=None):
    """
    This function runs a dual Kc daily water balance over the entire
    weather_df range. It produces a results table with daily:
     - ET0, ETa_total, ETa_transp, ETa_evap
     - Soil Water in root zone (SW_root)
     - Per-layer SW as well
     - Drainage, etc.
    """
    if weather_df.empty:
        st.error("Weather data is empty.")
        return None
    
    total_days = len(weather_df)
    results = []
    
    # Initialize layer water content
    if initial_SW_layers is None:
        # start at Field Capacity
        SW_layers = [soil['FC'] * soil['Depth_mm'] for _, soil in soil_df.iterrows()]
    else:
        SW_layers = initial_SW_layers.copy()
    
    # Cumulative surface evap since last major wetting
    E = soil_df['REW'].sum()
    cum_irrig = 0.0
    cum_precip = 0.0
    
    # Get daily arrays from crop_df
    Kcb_daily, root_depth_daily, p_daily, Ke_daily = interpolate_crop_stages(crop_df, total_days)
    
    # If dynamic root growth, override root_depth
    if enable_dynamic_root and initial_root_depth and max_root_depth and days_to_max:
        dyn = np.linspace(initial_root_depth, max_root_depth, min(days_to_max, total_days)).tolist()
        if total_days > days_to_max:
            dyn += [max_root_depth]*(total_days - days_to_max)
        root_depth_daily = np.array(dyn[:total_days])
    
    for day_i in range(total_days):
        date_i = weather_df.iloc[day_i]['Date']
        ET0_i = max(0, weather_df.iloc[day_i]['ET0'])
        prcp_i = max(0, weather_df.iloc[day_i]['Precipitation'])
        irr_i  = max(0, weather_df.iloc[day_i]['Irrigation'])
        
        cum_irrig  += irr_i
        cum_precip += prcp_i
        
        # daily crop props
        Kcb_i  = max(0, Kcb_daily[day_i])
        p_i    = max(0, min(1, p_daily[day_i]))
        ke0_i  = max(0, Ke_daily[day_i])
        rd_i   = max(1, root_depth_daily[day_i])  # ensure >0
        
        # Determine average WP, FC in root zone
        total_depth_covered = 0
        SW_root = 0
        for j, soil in soil_df.iterrows():
            layer_d = soil['Depth_mm']
            new_d = total_depth_covered + layer_d
            fraction_in_root = 0
            if new_d <= rd_i:
                fraction_in_root = 1.0
            elif (total_depth_covered < rd_i < new_d):
                fraction_in_root = (rd_i - total_depth_covered)/layer_d
            # sum water in that portion
            SW_root += SW_layers[j]* fraction_in_root
            total_depth_covered = new_d
        
        # approximate average FC, WP in root zone:
        # sum( FC_j * d_j ) over root zone / root_depth ...
        # We'll do a simpler approach:
        FC_avg = soil_df['FC'].mean()
        WP_avg = soil_df['WP'].mean()
        
        # compute Ks
        avg_SW_frac = SW_root/float(rd_i)
        Ks_i = compute_Ks(avg_SW_frac, WP_avg, FC_avg, p_i)
        
        # Kr for topsoil
        TEW = soil_df['TEW'].sum()
        REW = soil_df['REW'].sum()
        Kr_i = compute_Kr(TEW, REW, E)
        Ke_i = Kr_i * ke0_i
        
        # daily ETc
        ETc_i = compute_ETc(Kcb_i, Ks_i, Ke_i, ET0_i)
        ETa_transp = max(0, Kcb_i*Ks_i*ET0_i)
        ETa_evap   = max(0, Ke_i*ET0_i)
        ETa_total  = ETa_transp + ETa_evap
        
        water_in   = prcp_i + irr_i
        excess     = water_in - ETa_total
        
        drainage = 0
        if track_drainage:
            # fill layers to FC if we have excess
            for j in range(len(SW_layers)):
                capacity_j = (soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']) - SW_layers[j]
                if capacity_j>0 and excess>0:
                    add_j = min(excess, capacity_j)
                    SW_layers[j] += add_j
                    excess -= add_j
            # leftover -> drainage
            drainage = max(0, excess)
            # ensure SW doesn't exceed FC or drop below WP
            for j in range(len(SW_layers)):
                max_sw = soil_df.iloc[j]['FC']* soil_df.iloc[j]['Depth_mm']
                min_sw = soil_df.iloc[j]['WP']* soil_df.iloc[j]['Depth_mm']
                SW_layers[j] = min(max_sw, SW_layers[j])
                SW_layers[j] = max(min_sw, SW_layers[j])
        else:
            drainage=0
        
        # remove ET from layers in proportion to fraction in root zone
        if ETa_total>0 and SW_root>0:
            to_remove = ETa_total
            total_depth_covered=0
            for j,soil in soil_df.iterrows():
                layer_d = soil['Depth_mm']
                old_d = total_depth_covered
                new_d = old_d + layer_d
                total_depth_covered = new_d
                fraction_in_root=0
                if new_d <= rd_i:
                    fraction_in_root=1.0
                elif old_d < rd_i < new_d:
                    fraction_in_root = (rd_i - old_d)/layer_d
                if fraction_in_root>0:
                    # remove from SW_layers[j] up to fraction_in_root
                    maxrem_j = SW_layers[j] - (soil['WP']*layer_d)
                    rem_j = min(to_remove*fraction_in_root, maxrem_j)
                    SW_layers[j] -= rem_j
                    to_remove   -= rem_j
                    if to_remove<=0:
                        break
        
        # update E for topsoil evaporation
        if water_in >= 4.0:
            E=0.0
        else:
            E+= ETa_evap
        E = max(0, min(E, TEW))
        
        daily_result = {
            "Date": date_i,
            "ET0 (mm)": ET0_i,
            "Precip (mm)": prcp_i,
            "Irrig (mm)": irr_i,
            "ETa_total (mm)": ETa_total,
            "ETa_transp (mm)": ETa_transp,
            "ETa_evap (mm)": ETa_evap,
            "Ks": Ks_i,
            "Ke": Ke_i,
            "SW_root (mm)": SW_root,  # at start-of-day
            "Root_Depth (mm)": rd_i,
            "Daily_Drainage (mm)": drainage,
            "Cumulative_Irrigation (mm)": cum_irrig,
            "Cumulative_Precip (mm)": cum_precip
        }
        
        # optional yield
        if enable_yield:
            if use_fao33 and Ym>0 and Ky>0 and ETc_i>0:
                Ya = Ym*(1 - Ky*(1 - (ETa_total/ETc_i)))
                daily_result["Yield (ton/ha)"] = max(0, Ya)
            if use_transp and WP_yield>0:
                Ya_transp = WP_yield*ETa_transp
                daily_result["Yield (ton/ha)"] = Ya_transp
        
        # optional leaching
        if enable_leaching:
            leach_val=0
            if leaching_method=="Method 1: Drainage × nitrate concentration" and drainage>0:
                # drainage mm => 1 mm/ha = 10 m3/ha => mg/L => kg/ha
                leach_val = drainage*10.0*(nitrate_conc*1e-6)*1000.0
            elif leaching_method=="Method 2: Leaching Fraction × total N input":
                # daily fraction
                leach_val = leaching_fraction*(total_N_input/float(total_days))
            daily_result["Leaching (kg/ha)"] = leach_val
        
        # store each layer's SW
        for j in range(len(SW_layers)):
            daily_result[f"Layer{j}_SW(mm)"] = SW_layers[j]
        
        results.append(daily_result)
    
    results_df = pd.DataFrame(results)
    
    if return_soil_profile:
        final_soil_profile = []
        for j, soil in soil_df.iterrows():
            final_soil_profile.append({
                "Layer": j,
                "Depth_mm": soil['Depth_mm'],
                "SW (mm)": SW_layers[j]
            })
        return results_df, final_soil_profile
    else:
        return results_df

def fetch_weather_data(lat, lon, start_date, end_date, forecast=True, manual_data=None):
    """
    Grabs either the next 5-day forecast from OpenWeather
    or manual forecast data if provided.
    """
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}_{forecast}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    # Use manual forecast data if provided
    if manual_data is not None:
        dates = pd.date_range(start_date, end_date)
        weather_df = pd.DataFrame({
            "Date": dates,
            "ET0": manual_data['eto'],
            "Precipitation": manual_data['precip'],
            "Irrigation": [0]*len(dates)
        })
        st.session_state.weather_cache[cache_key] = weather_df
        return weather_df
    
    if forecast:
        if st.session_state.api_calls >= 1000:
            st.warning("Daily API call limit reached.")
            return None
        
        if lat == 0.0 and lon == 0.0:
            st.warning("Invalid coordinates. Provide valid lat/lon.")
            return None
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=5)
        if start_date < today or end_date> max_forecast_date:
            st.warning("Forecast date range is outside valid period. Adjusting.")
            start_date = today
            end_date   = today + timedelta(days=4)
        
        api_key = "fe2d869569674a4afbfca57707bdf691"
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            st.session_state.api_calls += 1
            data = response.json()
            
            daily_data = {}
            for entry in data['list']:
                dt_date = datetime.fromtimestamp(entry['dt']).date()
                if start_date<= dt_date <= end_date:
                    dstr = dt_date.strftime("%Y-%m-%d")
                    if dstr not in daily_data:
                        daily_data[dstr] = {
                            'tmax': entry['main']['temp_max'],
                            'tmin': entry['main']['temp_min'],
                            'precip': entry.get('rain',{}).get('3h',0)
                        }
                    else:
                        daily_data[dstr]['tmax'] = max(daily_data[dstr]['tmax'], entry['main']['temp_max'])
                        daily_data[dstr]['tmin'] = min(daily_data[dstr]['tmin'], entry['main']['temp_min'])
                        daily_data[dstr]['precip'] += entry.get('rain',{}).get('3h',0)
            
            dates, ETo_list, precip_list = [], [], []
            for dstr, val in daily_data.items():
                d_ = pd.to_datetime(dstr)
                dates.append(d_)
                tx, tn = val['tmax'], val['tmin']
                if tx<tn: tx, tn= tn, tx
                Ra = 10.0
                Tmean = (tx+tn)/2
                # simplified Hargreaves
                ETo_ = 0.0023*Ra*(Tmean+17.8)*((tx-tn)**0.5)
                ETo_ = max(0,ETo_)
                ETo_list.append(ETo_)
                precip_list.append(val['precip'])
            
            weather_df = pd.DataFrame({
                "Date": dates,
                "ET0": ETo_list,
                "Precipitation": precip_list,
                "Irrigation": [0]*len(dates)
            }).sort_values("Date").reset_index(drop=True)
            
            st.session_state.weather_cache[cache_key] = weather_df
            return weather_df
        except:
            st.error("Unable to fetch forecast data now.")
            return None
    else:
        # historical data from NASA or others
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str   = end_date.strftime("%Y%m%d")
            url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            dates = pd.date_range(start_date, end_date)
            ET0_arr = []
            precip_arr = []
            for d_ in dates:
                ds = d_.strftime("%Y%m%d")
                rad_ = data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(ds,0)
                eto_ = rad_*0.2
                p_   = data['properties']['parameter']['PRECTOTCORR'].get(ds,0)
                ET0_arr.append(eto_)
                precip_arr.append(p_)
            weather_df = pd.DataFrame({
                "Date": dates,
                "ET0": ET0_arr,
                "Precipitation": precip_arr,
                "Irrigation": [0]*len(dates)
            })
            st.session_state.weather_cache[cache_key] = weather_df
            return weather_df
        except:
            st.warning("Unable to fetch historical data now.")
            return None

def irrigation_schedule(results_df, soil_df):
    """
    Simple post-processing approach to produce an irrigation scheduling calendar.
    For demonstration, we define a threshold = 50% TAW:
      If SW_root < 0.5 * TAW => recommend irrigation = TAW - SW_root
    That is purely illustrative. You can refine as needed.
    """
    schedule = []
    for i, row in results_df.iterrows():
        date_i = row["Date"]
        sw_root_i = row["SW_root (mm)"]  # at start-of-day
        root_d_i  = row["Root_Depth (mm)"]
        # approximate average FC, WP
        FC_avg = soil_df['FC'].mean()
        WP_avg = soil_df['WP'].mean()
        TAW_i  = (FC_avg - WP_avg)*root_d_i
        threshold = 0.5*TAW_i
        if sw_root_i < threshold:
            recommend = (TAW_i - sw_root_i)
            if recommend>0:
                schedule.append({
                    "Date": date_i.strftime("%Y-%m-%d"),
                    "Recommended_Irrigation_mm": round(recommend,1)
                })
    schedule_df = pd.DataFrame(schedule)
    return schedule_df

# --------------------------------------------------------------------------------
# 6. Streamlit UI - Setup Tab
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown('<div class="sub-header">Input Data Configuration</div>', unsafe_allow_html=True)
    
    st.write("#### Available Crops (built-in) for Forecasting Actual ET:")
    st.write(list(KC_DATABASE.keys()))
    
    # Let user choose crop
    st.write("Please select your crop type (for 5-day ET forecast):")
    selected_crop = st.selectbox("Pick a crop", list(KC_DATABASE.keys()))
    st.write(f"**Selected Crop** = {selected_crop}")
    st.write(f" - Kc_mid={KC_DATABASE[selected_crop]['Kc_mid']}, Kc_end={KC_DATABASE[selected_crop]['Kc_end']}")
    st.write(f" - Kcb_mid={KC_DATABASE[selected_crop]['Kcb_mid']}, Kcb_end={KC_DATABASE[selected_crop]['Kcb_end']}")
    
    with st.container():
        st.markdown("### Upload Input Files")
        
        with st.expander("Weather Data"):
            st.write("Upload a text file with columns: Date,ET0,Precipitation,Irrigation (or rely on forecast).")
            weather_file = st.file_uploader("Weather Data File (.txt)", type="txt", key="weather")
            sample_weather = pd.DataFrame({
                "Date": ["2025-01-01","2025-01-02","2025-01-03"],
                "ET0": [5.0, 5.2, 4.8],
                "Precipitation": [0.0, 2.0, 0.0],
                "Irrigation": [0.0, 0.0, 10.0]
            })
            st.download_button("Download Sample Weather Data",
                               sample_weather.to_csv(index=False),
                               file_name="weather_sample.txt",mime="text/plain")
        
        with st.expander("Crop Stage Data"):
            st.write("""If you prefer a multi-stage approach, upload a custom CSV with 
            columns: Start_Day,End_Day,Kcb,Root_Depth_mm,p,Ke. 
            Or do a minimal approach using mid/end from built-in DB.""")
            use_custom_crop_file = st.checkbox("Use a custom crop file instead of a minimal approach?", value=False)
            if use_custom_crop_file:
                crop_file = st.file_uploader("Crop Stage Data File (.txt)", type="txt", key="crop")
                sample_crop = pd.DataFrame({
                    "Start_Day":[1, 31],
                    "End_Day":[30, 60],
                    "Kcb":[0.3, 1.20],
                    "Root_Depth_mm":[300, 1000],
                    "p":[0.5, 0.55],
                    "Ke":[1.0, 0.2]
                })
                st.download_button("Download Sample Crop Data",
                                   sample_crop.to_csv(index=False),
                                   file_name="crop_sample.txt", mime="text/plain")
            else:
                st.write("We'll generate a minimal 2-stage approach using mid/end from the built-in DB.")
                approximate_duration = st.number_input("Approx Crop Cycle (days)?", min_value=30, value=90, step=1)
                st.session_state["approximate_duration"] = approximate_duration
        
        with st.expander("Soil Layers Data"):
            st.write("Upload text file w/ columns: Depth_mm, FC, WP, TEW, REW.")
            soil_file = st.file_uploader("Soil Layers File (.txt)", type="txt", key="soil")
            sample_soil = pd.DataFrame({
                "Depth_mm":[200, 100],
                "FC":[0.30, 0.30],
                "WP":[0.15, 0.15],
                "TEW":[20, 0],
                "REW":[5, 0]
            })
            st.download_button("Download Sample Soil Data",
                               sample_soil.to_csv(index=False),
                               file_name="soil_sample.txt", mime="text/plain")
    
    with st.container():
        st.markdown("### Additional Features")
        
        with st.expander("Simulation Options"):
            track_drainage = st.checkbox("Track Drainage?", value=True)
            enable_yield   = st.checkbox("Enable Yield Estimation?", value=False)
            if enable_yield:
                st.markdown("**Yield Options**")
                use_fao33 = st.checkbox("FAO-33 Ky-based method?", value=True)
                if use_fao33:
                    Ym = st.number_input("Max Yield (ton/ha)", min_value=0.0, value=10.0, step=0.1)
                    Ky = st.number_input("Yield Response Factor Ky", min_value=0.0, value=1.0, step=0.1)
                else:
                    Ym = Ky = 0
                use_transp = st.checkbox("Transpiration-based (WP_yield)?", value=False)
                if use_transp:
                    WP_yield = st.number_input("WP_yield (ton/ha per mm)", min_value=0.0, value=0.01, step=0.001)
                else:
                    WP_yield=0
            else:
                use_fao33=use_transp=False
                Ym=Ky=WP_yield=0
            
            enable_leaching = st.checkbox("Enable Leaching?", value=False)
            if enable_leaching:
                leaching_method = st.radio("Leaching Method", ["Method 1: Drainage × nitrate concentration",
                                                               "Method 2: Leaching Fraction × total N input"])
                if leaching_method=="Method 1: Drainage × nitrate concentration":
                    nitrate_conc = st.number_input("Nitrate mg/L", value=10.0, step=0.1)
                    total_N_input=0
                    leaching_fraction=0
                else:
                    total_N_input     = st.number_input("Total N input (kg/ha)?", value=100.0, step=1.0)
                    leaching_fraction = st.number_input("Leaching Fraction (0-1)?", value=0.1, step=0.01)
                    nitrate_conc=0
            else:
                leaching_method=""
                nitrate_conc=total_N_input=leaching_fraction=0
            
            enable_etaforecast = st.checkbox("Enable 5-Day ET Forecast?", value=True)
            manual_forecast_data = None
            if enable_etaforecast:
                st.write("Enter your field coordinates for the forecast.")
                forecast_lat = st.number_input("Lat", value=35.0)
                forecast_lon = st.number_input("Lon", value=-80.0)
                use_manual_input = st.checkbox("Manual 5-day forecast input?", value=False)
                if use_manual_input:
                    tmax_vals, tmin_vals, prcp_vals=[],[],[]
                    for i in range(5):
                        st.write(f"Day {i+1}")
                        tx_ = st.number_input(f"Max Temp (°C)? day {i+1}", value=25.0, key=f"tx_{i}")
                        tn_ = st.number_input(f"Min Temp (°C)? day {i+1}", value=15.0, key=f"tn_{i}")
                        pp_ = st.number_input(f"Precip (mm)? day {i+1}", value=0.0, key=f"pp_{i}")
                        tmax_vals.append(tx_)
                        tmin_vals.append(tn_)
                        prcp_vals.append(pp_)
                    eto_vals=[]
                    for tx_, tn_ in zip(tmax_vals, tmin_vals):
                        if tx_<tn_: tx_, tn_= tn_, tx_
                        Ra=10
                        Tmean=(tx_+tn_)/2
                        ETo_ = 0.0023*Ra*(Tmean+17.8)*((tx_-tn_)**0.5)
                        ETo_ = max(0,ETo_)
                        eto_vals.append(ETo_)
                    manual_forecast_data = {
                        "tmax": tmax_vals,
                        "tmin": tmin_vals,
                        "precip": prcp_vals,
                        "eto": eto_vals
                    }
            else:
                forecast_lat=forecast_lon=0.0
            
            enable_dynamic_root= st.checkbox("Dynamic Root Growth?", value=False)
            if enable_dynamic_root:
                initial_root_depth= st.number_input("Initial Root Depth (mm)", min_value=50, value=300, step=10)
                max_root_depth    = st.number_input("Max Root Depth (mm)", min_value=50, value=1000, step=10)
                days_to_max       = st.number_input("Days to Max Root Depth", min_value=1, value=60, step=1)
            else:
                initial_root_depth=max_root_depth=days_to_max=None
            
            show_soil_profile = st.checkbox("Show final soil profile water storage?", value=False)
    
    st.button("Run Simulation", key="run_simulation")

# --------------------------------------------------------------------------------
# 7. Streamlit UI - Results Tab
# --------------------------------------------------------------------------------
with results_tab:
    if st.session_state.get('run_simulation', False):
        # Attempt reading weather
        # If user uploaded weather_file => read it, else we do a demonstration
        if 'approximate_duration' not in st.session_state:
            st.session_state['approximate_duration'] = 90
        
        if weather_file is not None:
            try:
                weather_df = pd.read_csv(weather_file)
                if pd.api.types.is_string_dtype(weather_df['Date']):
                    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
                weather_df = weather_df.sort_values("Date").reset_index(drop=True)
            except:
                st.error("Failed reading weather file.")
                st.stop()
        else:
            # do a minimal 5-day fetch
            st.warning("No weather file => We'll fetch 5-day forecast from your lat/lon as demonstration.")
            start_dt = datetime.now().date()
            end_dt   = start_dt + timedelta(days=4)
            weather_df = fetch_weather_data(forecast_lat, forecast_lon, start_dt, end_dt, forecast=True, 
                                            manual_data=None)
            if weather_df is None or weather_df.empty:
                st.error("No valid weather data. Please re-check.")
                st.stop()
        
        # Crop data
        use_custom_crop_file = st.session_state.get('use_custom_crop_file', False)
        if 'approximate_duration' in st.session_state:
            approximate_duration= st.session_state['approximate_duration']
        else:
            approximate_duration=90
        
        # Because user might not have used a toggle, let's see:
        if use_custom_crop_file and 'crop_file' in st.session_state:
            pass
        # We'll just define a local approach:
        if use_custom_crop_file:
            # we try to read from the UI
            if 'crop' in st.session_state and st.session_state['crop'] is not None:
                try:
                    crop_file = st.session_state['crop']
                except:
                    crop_file=None
            else:
                crop_file=None
            if crop_file is not None:
                try:
                    crop_df = pd.read_csv(crop_file)
                    crop_df = crop_df.sort_values("Start_Day").reset_index(drop=True)
                except:
                    st.error("Could not parse uploaded crop file. Stopping.")
                    st.stop()
            else:
                st.error("You selected 'use custom crop file' but no file uploaded.")
                st.stop()
        else:
            # build minimal 2-stage from Kcb_mid, Kcb_end from KC_DATABASE
            kcb_mid = KC_DATABASE[selected_crop]['Kcb_mid']
            kcb_end = KC_DATABASE[selected_crop]['Kcb_end']
            stage1_end = int(round(0.7*approximate_duration))
            auto_crop_df = pd.DataFrame({
                "Start_Day":[1, stage1_end+1],
                "End_Day":[stage1_end, approximate_duration],
                "Kcb":[kcb_mid, kcb_end],
                "Root_Depth_mm":[300, 600],
                "p":[0.5,0.5],
                "Ke":[0.15,0.1]
            })
            crop_df = auto_crop_df
        
        # Soil data
        if soil_file is not None:
            try:
                soil_df = pd.read_csv(soil_file)
            except:
                st.error("Failed to read soil file => using a 2-layer default.")
                soil_df = pd.DataFrame({
                    "Depth_mm":[200, 100],
                    "FC":[0.30, 0.30],
                    "WP":[0.15, 0.15],
                    "TEW":[20, 0],
                    "REW":[5, 0]
                })
        else:
            soil_df = pd.DataFrame({
                "Depth_mm":[200, 100],
                "FC":[0.30, 0.30],
                "WP":[0.15, 0.15],
                "TEW":[20, 0],
                "REW":[5, 0]
            })
        
        # run main simulation
        total_days = len(weather_df)
        # force last row in crop_df
        crop_df.loc[crop_df.index[-1],"End_Day"] = total_days
        
        results_df, final_soil_profile = SIMdualKc(
            weather_df=weather_df,
            crop_df=crop_df,
            soil_df=soil_df,
            track_drainage=track_drainage,
            enable_yield=enable_yield,
            use_fao33=use_fao33,
            Ym=Ym,
            Ky=Ky,
            use_transp=use_transp,
            WP_yield=WP_yield,
            enable_leaching=enable_leaching,
            leaching_method=leaching_method,
            nitrate_conc=nitrate_conc,
            total_N_input=total_N_input,
            leaching_fraction=leaching_fraction,
            enable_dynamic_root=enable_dynamic_root,
            initial_root_depth=initial_root_depth,
            max_root_depth=max_root_depth,
            days_to_max=days_to_max,
            return_soil_profile=True
        )
        st.session_state.results_df = results_df
        st.session_state.soil_profile = final_soil_profile
        
        # do the 5-day forecast if requested
        if enable_etaforecast:
            last_layer_sw = [ly['SW (mm)'] for ly in final_soil_profile]
            # minimal 1-stage approach for next 5 days
            forecast_crop_df = pd.DataFrame({
                "Start_Day":[1],
                "End_Day":[5],
                "Kcb":[KC_DATABASE[selected_crop]["Kcb_end"]], 
                "Root_Depth_mm":[600],
                "p":[0.5],
                "Ke":[0.1]
            })
            forecast_start = datetime.now().date()
            forecast_end   = forecast_start + timedelta(days=4)
            forecast_wdf   = fetch_weather_data(forecast_lat, forecast_lon,
                                                forecast_start, forecast_end,
                                                forecast=True,
                                                manual_data=manual_forecast_data)
            if forecast_wdf is not None and not forecast_wdf.empty:
                forecast_res = SIMdualKc(
                    weather_df=forecast_wdf,
                    crop_df=forecast_crop_df,
                    soil_df=soil_df,
                    track_drainage=track_drainage,
                    enable_yield=False,
                    return_soil_profile=False,
                    initial_SW_layers=last_layer_sw
                )
                st.session_state.forecast_results = forecast_res
            else:
                st.session_state.forecast_results = None
        
        st.success("Simulation completed successfully!")
        
        # display results with irrigation scheduling side by side
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown('<div class="sub-header">Daily Simulation Results</div>', unsafe_allow_html=True)
            st.dataframe(results_df)
            st.download_button("Download Results (.txt)",
                               results_df.to_csv(index=False),
                               file_name="results.txt", mime="text/plain")
            
            if enable_etaforecast and st.session_state.forecast_results is not None and not st.session_state.forecast_results.empty:
                st.markdown('<div class="sub-header">5-Day ETa Forecast</div>', unsafe_allow_html=True)
                frc = st.session_state.forecast_results
                st.dataframe(frc[["Date","ET0 (mm)","ETa_total (mm)","ETa_transp (mm)","ETa_evap (mm)"]])
            
            # Graphing
            st.markdown('<div class="sub-header">Graphs</div>', unsafe_allow_html=True)
            plot_options = ["ETa Components", "Soil Water in Root Zone", "Drainage", "Cumulative Inputs"]
            if "Yield (ton/ha)" in results_df.columns:
                plot_options.append("Yield")
            if "Leaching (kg/ha)" in results_df.columns:
                plot_options.append("Leaching")
            
            choice = st.selectbox("Select a plot", plot_options)
            if choice=="ETa Components":
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(results_df["Date"], results_df["ETa_transp (mm)"], label="Transp")
                ax.plot(results_df["Date"], results_df["ETa_evap (mm)"], label="Evap")
                ax.plot(results_df["Date"], results_df["ETa_total (mm)"], label="ETa total")
                ax.legend(); ax.grid(True)
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                st.pyplot(fig)
            elif choice=="Soil Water in Root Zone":
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(results_df["Date"], results_df["SW_root (mm)"], label="SW_root")
                ax.legend(); ax.grid(True)
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                st.pyplot(fig)
            elif choice=="Drainage":
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(results_df["Date"], results_df["Daily_Drainage (mm)"], label="Drainage")
                ax.legend(); ax.grid(True)
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                st.pyplot(fig)
            elif choice=="Cumulative Inputs":
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(results_df["Date"], results_df["Cumulative_Irrigation (mm)"], label="Irrigation")
                ax.plot(results_df["Date"], results_df["Cumulative_Precip (mm)"], label="Precip")
                ax.legend(); ax.grid(True)
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                st.pyplot(fig)
            elif choice=="Yield":
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(results_df["Date"], results_df["Yield (ton/ha)"], label="Yield")
                ax.legend(); ax.grid(True)
                ax.set_xlabel("Date"); ax.set_ylabel("ton/ha")
                st.pyplot(fig)
            elif choice=="Leaching":
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(results_df["Date"], results_df["Leaching (kg/ha)"], label="Leaching")
                ax.legend(); ax.grid(True)
                ax.set_xlabel("Date"); ax.set_ylabel("kg/ha")
                st.pyplot(fig)
            
            # show final soil profile
            if show_soil_profile and final_soil_profile:
                st.markdown('<div class="sub-header">Final Soil Profile</div>', unsafe_allow_html=True)
                pf = pd.DataFrame(final_soil_profile)
                st.dataframe(pf)
        
        with col2:
            st.markdown("## Irrigation Scheduling Calendar")
            sched_df = irrigation_schedule(results_df, soil_df)
            if sched_df.empty:
                st.write("No irrigation is recommended based on the threshold logic.")
            else:
                st.dataframe(sched_df)
            st.write("""
            *Note: This is a simple threshold approach (SW_root < 50% TAW) 
            -> recommended = TAW - SW_root. Adjust logic as needed.*
            """)

    else:
        st.info("Please click 'Run Simulation' on the Setup tab.")

st.markdown('<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>', unsafe_allow_html=True)
