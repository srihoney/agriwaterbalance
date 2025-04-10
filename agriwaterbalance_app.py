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
# 1.  Built-in Kc Database
# --------------------------------------------------------------------------------
# The dictionaries below store mid-season (Kc_mid, Kcb_mid) and end-season (Kc_end, Kcb_end)
# values for a variety of crops, following Pereira et al. (2021a) [vegetables]
# and Pereira et al. (2021b) [field crops].
# The user will select a crop name, and these data will be used in the simulation.

KC_DATABASE = {
    # --- Vegetables ---
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

    # --- Field Crops ---
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
    # ... add more as needed ...
}


# --------------------------------------------------------------------------------
# 2.  Configure Requests Session with Retries
# --------------------------------------------------------------------------------
session = requests.Session()
retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
session.mount('https://', HTTPAdapter(max_retries=retries))

# --------------------------------------------------------------------------------
# 3.  Streamlit Page Configuration & Logo
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
    .footer {{ background-color: #1E3A8A; color: white; padding: 10px; text-align: center; position: fixed; bottom: 0; width: 100%; border-radius: 5px 5px 0 0; }}
    .stButton>button {{ background-color: #2563EB; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; }}
    .stButton>button:hover {{ background-color: #1E40AF; }}
    .stFileUploader {{ border: 2px dashed #1E3A8A; border-radius: 5px; padding: 10px; }}
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

setup_tab, results_tab = st.tabs(["Setup Simulation", "View Results"])

# --------------------------------------------------------------------------------
# 4.  Session State
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
# 5.  Core Water Balance & ET Functions
# --------------------------------------------------------------------------------

def compute_Ks(SW, WP, FC, p):
    """
    Computes the water stress coefficient Ks based on:
    Ks=1 when root zone depletion <= RAW
    Otherwise Ks linearly decreases to 0 as depletion approaches TAW
    SW in mm of water per m of soil depth (or per root depth).
    WP, FC in volumetric fraction [m3/m3].
    """
    TAW = (FC - WP) * 1000.0  # total available water (mm/m)
    RAW = p * TAW            # readily available water (mm/m)
    Dr = (FC - SW) * 1000.0  # Depletion from field capacity
    if Dr <= RAW:
        Ks = 1.0
    else:
        Ks = (TAW - Dr) / ((1 - p) * TAW)
        Ks = max(0.0, min(1.0, Ks))
    return Ks

def compute_Kr(TEW, REW, E):
    """
    Kr is the evaporation reduction coefficient, depends on how much
    of the TEW has been used for evaporation.
    TEW: total evaporable water in the topsoil
    REW: readily evaporable water
    E: cumulative evaporation since the last major wetting
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
    Interpolates the daily crop parameters (Kcb, root_depth, p, Ke)
    from user-provided stage data (start_day -> end_day).
    Typically we store the final arrays to be used day-by-day.
    """
    Kcb = np.zeros(total_days)
    root_depth = np.zeros(total_days)
    p = np.zeros(total_days)
    Ke = np.zeros(total_days)
    
    for i in range(len(crop_df) - 1):
        start_day = int(crop_df.iloc[i]['Start_Day'])
        end_day   = min(int(crop_df.iloc[i]['End_Day']), total_days)
        
        Kcb_start       = crop_df.iloc[i]['Kcb']
        Kcb_end         = crop_df.iloc[i+1]['Kcb']
        root_start      = crop_df.iloc[i]['Root_Depth_mm']
        root_end        = crop_df.iloc[i+1]['Root_Depth_mm']
        p_start         = crop_df.iloc[i]['p']
        p_end           = crop_df.iloc[i+1]['p']
        Ke_start        = crop_df.iloc[i]['Ke']
        Ke_end          = crop_df.iloc[i+1]['Ke']
        
        if i == 0 and start_day > 1:
            Kcb[0:start_day-1]        = 0
            root_depth[0:start_day-1] = root_start
            p[0:start_day-1]          = p_start
            Ke[0:start_day-1]         = Ke_start
        
        idx = np.arange(start_day - 1, end_day)
        if len(idx) > 0:
            Kcb[idx]        = np.linspace(Kcb_start, Kcb_end, len(idx))
            root_depth[idx] = np.linspace(root_start, root_end, len(idx))
            p[idx]          = np.linspace(p_start, p_end, len(idx))
            Ke[idx]         = np.linspace(Ke_start, Ke_end, len(idx))
    
    last_i = len(crop_df) - 1
    last_start_day = int(crop_df.iloc[last_i]['Start_Day'])
    last_end_day   = min(int(crop_df.iloc[last_i]['End_Day']), total_days)
    last_Kcb       = crop_df.iloc[last_i]['Kcb']
    last_root      = crop_df.iloc[last_i]['Root_Depth_mm']
    last_p         = crop_df.iloc[last_i]['p']
    last_Ke        = crop_df.iloc[last_i]['Ke']
    
    if last_start_day <= total_days:
        idx_last = np.arange(last_start_day - 1, last_end_day)
        if len(idx_last) > 0:
            Kcb[idx_last]        = last_Kcb
            root_depth[idx_last] = last_root
            p[idx_last]          = last_p
            Ke[idx_last]         = last_Ke
    
    if last_end_day < total_days:
        Kcb[last_end_day:]        = last_Kcb
        root_depth[last_end_day:] = last_root
        p[last_end_day:]          = last_p
        Ke[last_end_day:]         = last_Ke
    
    return Kcb, root_depth, p, Ke

def SIMdualKc(weather_df, crop_df, soil_df, track_drainage=True, enable_yield=False,
              use_fao33=False, Ym=0, Ky=0, use_transp=False, WP_yield=0,
              enable_leaching=False, leaching_method="", nitrate_conc=0,
              total_N_input=0, leaching_fraction=0,
              enable_dynamic_root=False, initial_root_depth=None, max_root_depth=None, days_to_max=None,
              return_soil_profile=False, initial_SW_layers=None):
    """
    This function implements a dual Kc water balance, computing daily ETa,
    transpiration, evaporation, soil-water content in the root zone,
    drainage, etc.
    """
    if weather_df.empty:
        st.error("Weather data is empty. Please check your input.")
        return None
    
    total_days = len(weather_df)
    results = []
    
    if initial_SW_layers is None:
        # Start soil water content at field capacity:
        SW_layers = [soil['FC'] * soil['Depth_mm'] for _, soil in soil_df.iterrows()]
    else:
        SW_layers = initial_SW_layers.copy()
    
    # Cumulative soil evaporation since last wetting:
    E = soil_df['REW'].sum()  # start "dry"
    
    cumulative_irrigation = 0.0
    cumulative_precip = 0.0
    
    Kcb_daily, root_depth_daily, p_daily, Ke_daily = interpolate_crop_stages(crop_df, total_days)
    
    # If dynamic root growth is enabled, override root_depth_daily with a linear function:
    if enable_dynamic_root and initial_root_depth is not None and max_root_depth is not None and days_to_max is not None:
        # linearly from initial_root_depth to max_root_depth over days_to_max:
        # If total_days > days_to_max, remain constant after that
        dynamic_depth = np.linspace(initial_root_depth, max_root_depth, min(days_to_max, total_days)).tolist()
        if total_days > days_to_max:
            dynamic_depth += [max_root_depth] * (total_days - days_to_max)
        root_depth_daily = np.array(dynamic_depth[:total_days])
    
    for day in range(total_days):
        date = weather_df.iloc[day]['Date']
        ET0 = max(0, weather_df.iloc[day]['ET0'])
        precip = max(0, weather_df.iloc[day]['Precipitation'])
        irrig = max(0, weather_df.iloc[day]['Irrigation'])
        
        cumulative_irrigation += irrig
        cumulative_precip += precip
        
        # Retrieve daily Kcb, p, root depth, Ke
        Kcb = max(0, Kcb_daily[day])
        p   = max(0, min(1, p_daily[day]))
        ke0 = max(0, Ke_daily[day])  # 'potential' Ke if no shortage in surface layer
        root_depth = max(1, root_depth_daily[day])  # mm
        
        # Compute average WP, FC in the root zone:
        total_depth = 0
        SW_root = 0
        TAW_root = 0
        RAW_root = 0
        
        for j, soil in soil_df.iterrows():
            layer_depth = soil['Depth_mm']
            # track cumulative depth
            new_total = total_depth + layer_depth
            if new_total <= root_depth:
                # entire layer is in root zone
                SW_root += SW_layers[j]
                TAW_root += (soil['FC'] - soil['WP']) * layer_depth
                RAW_root += p * (soil['FC'] - soil['WP']) * layer_depth
            elif total_depth < root_depth < new_total:
                # partial layer
                fraction = (root_depth - total_depth) / layer_depth
                SW_root += SW_layers[j] * fraction
                TAW_root += (soil['FC'] - soil['WP']) * layer_depth * fraction
                RAW_root += p * (soil['FC'] - soil['WP']) * layer_depth * fraction
            total_depth = new_total
        
        # Convert SW_root from mm of water to fraction? We'll do it as volumetric approach:
        # Ks = f(SW fraction) -> we need SW fraction in root zone
        # But below we do a simplified approach with average WP, FC in the root zone:
        # We'll define average SW frac = SW_root / root_depth (since each mm of depth has 1 mm of water capacity)
        
        avg_SW_frac = SW_root / float(root_depth)
        # average WP, FC for the root zone
        if TAW_root <= 1e-6:
            # fallback
            zone_FC = soil_df['FC'].mean()
            zone_WP = soil_df['WP'].mean()
        else:
            # approximate average
            zone_FC = zone_WP = 0
            # We'll guess an average from TAW_root => zone_FC - zone_WP
            # This is a simpler approach for demonstration
            zone_FC = zone_WP + (TAW_root / float(root_depth))  # WP + fraction that yields TAW in mm
            # we guess WP from soil_df average or from fraction of TAW
            zone_WP = soil_df['WP'].mean()
        
        Ks = compute_Ks(avg_SW_frac, zone_WP, zone_FC, p)
        # Soil evaporation coefficient:
        # We adopt Kr * ke0 approach:
        TEW = soil_df['TEW'].sum()
        REW = soil_df['REW'].sum()
        Kr = compute_Kr(TEW, REW, E)
        Ke = Kr * ke0
        
        # Compute ETc
        ETc = compute_ETc(Kcb, Ks, Ke, ET0)
        
        # Partition into transpiration and evaporation:
        ETa_transp = max(0, Kcb * Ks * ET0)
        ETa_evap   = max(0, Ke * ET0)
        ETa_total  = ETa_transp + ETa_evap
        
        # Water input
        water_input = precip + irrig
        
        # Soil water update
        drainage = 0
        # If track drainage is True, allow infiltration up to layer capacity,
        # any excess is drainage
        excess = water_input - ETa_total
        if track_drainage:
            for j in range(len(SW_layers)):
                # Fill up layer j to FC if there's water
                capacity = (soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']) - SW_layers[j]
                if capacity > 0 and excess > 0:
                    added = min(excess, capacity)
                    SW_layers[j] += added
                    excess -= added
            # leftover is deep drainage
            drainage = max(0, excess)
            # ensure we do not exceed FC or drop below WP:
            for j in range(len(SW_layers)):
                max_sw = soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm']
                min_sw = soil_df.iloc[j]['WP'] * soil_df.iloc[j]['Depth_mm']
                SW_layers[j] = min(max_sw, SW_layers[j])
                SW_layers[j] = max(min_sw, SW_layers[j])
        else:
            # simpler approach: remove ETa_total from SW_layers
            drainage = 0
        
        # remove transpiration portion from the root zone in proportion to that layer's fraction:
        # remove evaporation portion from top layer(s) or from a topsoil reservoir E:
        # For simplicity, we do it proportionally from each layer in the root zone
        # Weighted by the fraction of SW in each layer
        if ETa_total > 0 and SW_root > 0:
            # fraction of total transpiration from each root layer
            # total transp = ETa_transp, total SW in root zone is SW_root
            # remove daily_result from layers in root zone
            # Evap presumably from top layer, we can unify it:
            transp_fraction = ETa_transp / ETa_total  if ETa_total>1e-9 else 0
            evap_fraction   = ETa_evap / ETa_total    if ETa_total>1e-9 else 0
            
            # transp depletion from root zone only
            # evap depletion from top "surface" (we skip the full detail).
            # Here we do a simple approach: remove ETa_total proportionally from all layers in the root zone
            to_remove = ETa_total
            # gather total SW in root zone again
            total_depth = 0
            for j, soil in soil_df.iterrows():
                layer_depth = soil['Depth_mm']
                layer_capacity = SW_layers[j]
                # check how much of this layer is in root zone:
                # do the same partial approach
                old_total = total_depth
                new_total = old_total + layer_depth
                fraction_in_root = 0
                if new_total <= root_depth:
                    fraction_in_root = 1.0
                elif old_total < root_depth < new_total:
                    fraction_in_root = (root_depth - old_total)/layer_depth
                total_depth = new_total
                
                # remove a fraction_in_root of the ET from this layer
                # Weighted by how much water is there
                if fraction_in_root>0:
                    # proportion of the total root zone water that is in this layer
                    # but we can do a simpler approach to avoid overcomplication
                    # remove in proportion to fraction_in_root:
                    remove_here = to_remove * fraction_in_root
                    # can't remove more than what's in the layer
                    maxrem = layer_capacity - (soil['WP']*layer_depth)
                    actual_rem = min(remove_here, maxrem)
                    SW_layers[j] = layer_capacity - actual_rem
                    to_remove -= actual_rem
                    if to_remove <= 0:
                        break
        
        # Next, track E (cumulative evaporation from topsoil):
        # E is used to compute Kr in subsequent days
        # Increase E by ETa_evap if no major wetting (or if precip+irrig is small)
        # If large wetting occurs (> ~4 mm), reset E=0
        # We'll define a small threshold
        if water_input >= 4.0:
            E = 0.0
        else:
            E += ETa_evap
        E = max(0, min(E, TEW))  # clamp
        
        # Collect daily results
        daily_result = {
            "Date": date,
            "ET0 (mm)": ET0,
            "ETa_total (mm)": ETa_total,
            "ETa_transp (mm)": ETa_transp,
            "ETa_evap (mm)": ETa_evap,
            "Ks": Ks,
            "Ke": Ke,
            "SW_root (mm)": SW_root,  # water in root zone at start-of-day
            "Cumulative_Irrigation (mm)": cumulative_irrigation,
            "Cumulative_Precip (mm)": cumulative_precip,
            "Root_Depth (mm)": root_depth,
            "Daily_Drainage (mm)": drainage,
        }
        
        if enable_yield:
            if use_fao33 and Ym > 0 and Ky > 0 and ETc>0:
                # approximate daily yield (just a day-slice; typically you do end-of-season)
                Ya = Ym * (1 - Ky * (1 - (ETa_total / ETc)))
                daily_result["Yield (ton/ha)"] = max(0, Ya)
            if use_transp and WP_yield > 0:
                # daily transp-based yield estimate
                Ya_transp = WP_yield * ETa_transp
                daily_result["Yield (ton/ha)"] = Ya_transp
        
        if enable_leaching:
            leaching = 0
            if leaching_method == "Method 1: Drainage × nitrate concentration" and drainage > 0:
                # drainage mm -> m  => * concentration mg/L => kg/ha factor
                # This is a simplistic approach
                # 1 mm/ha = 10 m3/ha => convert mg->kg
                # so (drainage mm) * 10 => m3/ha
                # mg/L -> kg/m3 => factor 1e-6
                leaching = drainage*10.0*(nitrate_conc*1e-6)*1000.0  # rough
            elif leaching_method == "Method 2: Leaching Fraction × total N input":
                # assume a daily fraction from total N input
                leaching = leaching_fraction*(total_N_input/ total_days)
            
            daily_result["Leaching (kg/ha)"] = leaching
        
        if enable_yield and total_N_input>0 and "Yield (ton/ha)" in daily_result:
            # approximate Nitrogen use efficiency
            daily_result["NUE (kg/ha)"] = daily_result["Yield (ton/ha)"]/ total_N_input
        
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
            st.warning("Daily API call limit reached. Please try again later.")
            return None
        
        if lat == 0.0 and lon == 0.0:
            st.warning("Invalid coordinates. Please enter valid latitude and longitude.")
            return None
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=5)
        if start_date < today or end_date > max_forecast_date:
            st.warning("Forecast date range is outside the valid period. Adjusting dates.")
            start_date = today
            end_date = today + timedelta(days=4)
        
        api_key = "fe2d869569674a4afbfca57707bdf691"  # please use your own or environment variable
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            st.session_state.api_calls += 1
            data = response.json()
            
            daily_data = {}
            for entry in data['list']:
                dt_date = datetime.fromtimestamp(entry['dt']).date()
                if start_date <= dt_date <= end_date:
                    date_str = dt_date.strftime("%Y-%m-%d")
                    if date_str not in daily_data:
                        daily_data[date_str] = {
                            'tmax': entry['main']['temp_max'],
                            'tmin': entry['main']['temp_min'],
                            'precip': entry.get('rain', {}).get('3h', 0)
                        }
                    else:
                        daily_data[date_str]['tmax'] = max(daily_data[date_str]['tmax'], entry['main']['temp_max'])
                        daily_data[date_str]['tmin'] = min(daily_data[date_str]['tmin'], entry['main']['temp_min'])
                        daily_data[date_str]['precip'] += entry.get('rain', {}).get('3h', 0)
            
            dates = []
            ETo_list = []
            precip_list = []
            for date_str, values in daily_data.items():
                dates.append(pd.to_datetime(date_str))
                tmax = values['tmax']
                tmin = values['tmin']
                if tmax < tmin:
                    tmax, tmin = tmin, tmax
                # Simple Hargreaves-ish approach
                Ra = 10  # fixed estimate of daily solar radiation [MJ/m2/d] for demonstration
                Tmean = (tmax + tmin)/2
                # ETo = 0.0023 * Ra * (Tmean+17.8)*((tmax - tmin)**0.5) # or
                # We'll keep it consistent with your prior code
                ETo = 0.0023 * Ra*(Tmean+17.8)*((tmax - tmin)**0.5)
                ETo = max(0, ETo)
                ETo_list.append(ETo)
                precip_list.append(values['precip'])
            
            weather_df = pd.DataFrame({
                "Date": dates,
                "ET0": ETo_list,
                "Precipitation": precip_list,
                "Irrigation": [0]*len(dates)
            }).sort_values("Date").reset_index(drop=True)
            
            st.session_state.weather_cache[cache_key] = weather_df
            return weather_df
        except:
            st.error("Unable to fetch forecast data at this time. Please try again later.")
            return None
    else:
        # Historical data from NASA or any other source could be fetched
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str   = end_date.strftime("%Y%m%d")
            # NASA POWER example (not guaranteed)
            url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            dates = pd.date_range(start_date, end_date)
            ET0_arr = [data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(d.strftime("%Y%m%d"),0)*0.2 for d in dates]
            precip_arr = [data['properties']['parameter']['PRECTOTCORR'].get(d.strftime("%Y%m%d"),0) for d in dates]
            weather_df = pd.DataFrame({
                "Date": dates,
                "ET0": ET0_arr,
                "Precipitation": precip_arr,
                "Irrigation": [0]*len(dates)
            })
            st.session_state.weather_cache[cache_key] = weather_df
            return weather_df
        except:
            st.warning("Unable to fetch historical weather data. Please try again later.")
            return None


# --------------------------------------------------------------------------------
# 6.  Streamlit UI
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown('<div class="sub-header">Input Data Configuration</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### Upload Input Files")
        
        with st.expander("Weather Data"):
            st.write("Upload a text file with columns: Date, ET0, Precipitation, Irrigation")
            weather_file = st.file_uploader("Weather Data File (.txt)", type="txt", key="weather")
            sample_weather = pd.DataFrame({
                "Date": ["2025-01-01","2025-01-02","2025-01-03"],
                "ET0": [5.0, 5.2, 4.8],
                "Precipitation": [0.0, 2.0, 0.0],
                "Irrigation": [0.0, 0.0, 10.0]
            })
            st.download_button("Download Sample Weather Data", sample_weather.to_csv(index=False), file_name="weather_sample.txt", mime="text/plain")
        
        with st.expander("Select Crop Type (From Built-In Database)"):
            st.write("Below is a curated list of crops with mid/end-season Kc and Kcb from Pereira et al. (2021).")
            selected_crop = st.selectbox("Pick Your Crop", list(KC_DATABASE.keys()), index=0)
            st.write(f"**Selected Crop**: {selected_crop}")
            st.write("Standard Kc_mid = ", KC_DATABASE[selected_crop]["Kc_mid"],
                     ", Kc_end = ", KC_DATABASE[selected_crop]["Kc_end"])
            st.write("Basal Kcb_mid = ", KC_DATABASE[selected_crop]["Kcb_mid"],
                     ", Kcb_end = ", KC_DATABASE[selected_crop]["Kcb_end"])
            st.markdown("""
                *Note:* These values assume sub-humid climate (RHmin≈45%, u2≈2 m/s). 
                For significantly different climates, you may adjust them.
            """)
        
        with st.expander("Crop Stage Data"):
            st.write("""
                If you prefer a 2-stage or 4-stage approach with daily Kcb, root depth, p, and Ke, 
                upload your custom data. Otherwise, we’ll just create a minimal 2-stage table 
                that uses the mid-/end-season Kc or Kcb from the built-in database.
            """)
            use_custom_crop_file = st.checkbox("Use a custom crop file instead of the built-in database?", value=False)
            
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
                st.download_button("Download Sample Crop Data", sample_crop.to_csv(index=False),
                                  file_name="crop_sample.txt", mime="text/plain")
            else:
                # We'll auto-build a 2-stage table: from day=1 to day= (some user-chosen?), 
                # with mid-season and end-season
                st.write("Enter an approximate total crop duration (days). We’ll build a minimal 2-stage Kcb table.")
                approximate_duration = st.number_input("Crop Cycle (days)", min_value=30, value=90, step=1)
                # We can store a simple 2-row DF:
                # row 1 -> start=1, end=round(approx_duration*0.7), Kcb= Kcb_mid*0.5 as an example
                # row 2 -> start= round(approx_duration*0.7+1), end=approx_duration, Kcb=Kcb_end
                # For demonstration, let's do a 2-stage approach: 
                stage1_end = int(round(0.7*approx_duration))
                # We'll guess the Kcb at mid for stage 1, and Kcb_end for stage 2, but 
                # typically you'd want a 4-stage approach. 
                
                # We'll define p and Ke somewhat arbitrarily or let the user define them:
                default_p = 0.5
                default_Ke = 0.1
                
                auto_crop_df = pd.DataFrame({
                    "Start_Day":[1, stage1_end+1],
                    "End_Day":[stage1_end, approximate_duration],
                    "Kcb":[KC_DATABASE[selected_crop]["Kcb_mid"], KC_DATABASE[selected_crop]["Kcb_end"]],
                    "Root_Depth_mm":[300, 600],
                    "p":[default_p, default_p],
                    "Ke":[default_Ke, default_Ke]
                })
                st.dataframe(auto_crop_df)
        
        with st.expander("Soil Layers Data"):
            st.write("Upload a text file with columns: Depth_mm, FC, WP, TEW, REW")
            soil_file = st.file_uploader("Soil Layers File (.txt)", type="txt", key="soil")
            sample_soil = pd.DataFrame({
                "Depth_mm":[200, 100],
                "FC":[0.30, 0.30],
                "WP":[0.15, 0.15],
                "TEW":[20, 0],
                "REW":[5, 0]
            })
            st.download_button("Download Sample Soil Data", sample_soil.to_csv(index=False), file_name="soil_sample.txt", mime="text/plain")
    
    with st.container():
        st.markdown("#### Additional Features")
        
        with st.expander("Simulation Options"):
            track_drainage = st.checkbox("Track Drainage", value=True)
            enable_yield = st.checkbox("Enable Yield Estimation", value=False)
            if enable_yield:
                st.markdown("**Yield Estimation Options**")
                use_fao33 = st.checkbox("Use FAO-33 Ky-based method", value=True)
                if use_fao33:
                    Ym = st.number_input("Maximum Yield (Ym, ton/ha)", min_value=0.0, value=10.0, step=0.1)
                    Ky = st.number_input("Yield Response Factor (Ky)", min_value=0.0, value=1.0, step=0.1)
                else:
                    Ym = 0
                    Ky = 0
                use_transp = st.checkbox("Use Transpiration-based method (WP_yield)", value=False)
                if use_transp:
                    WP_yield = st.number_input("Yield Water Productivity (WP_yield, ton/ha per mm)", 
                                               min_value=0.0, value=0.01, step=0.001)
                else:
                    WP_yield = 0
            else:
                use_fao33 = use_transp = False
                Ym = Ky = WP_yield = 0
            
            enable_leaching = st.checkbox("Enable Leaching Estimation", value=False)
            if enable_leaching:
                leaching_method = st.radio("Select Leaching Method", 
                                           ["Method 1: Drainage × nitrate concentration", 
                                            "Method 2: Leaching Fraction × total N input"])
                if leaching_method == "Method 1: Drainage × nitrate concentration":
                    nitrate_conc = st.number_input("Nitrate Concentration (mg/L)", min_value=0.0, value=10.0, step=0.1)
                    total_N_input = 0
                    leaching_fraction = 0
                else:
                    total_N_input = st.number_input("Total N Input (kg/ha)", min_value=0.0, value=100.0, step=1.0)
                    leaching_fraction = st.number_input("Leaching Fraction (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                    nitrate_conc = 0
            else:
                leaching_method = ""
                nitrate_conc = total_N_input = leaching_fraction = 0
            
            enable_etaforecast = st.checkbox("Enable 5-Day ETa Forecast", value=True)
            manual_forecast_data = None
            if enable_etaforecast:
                st.write("Enter your field's coordinates for the next 5-day weather forecast.")
                forecast_lat = st.number_input("Field Latitude", value=35.0)
                forecast_lon = st.number_input("Field Longitude", value=-80.0)
                
                use_manual_input = st.checkbox("Use Manual Weather Forecast Input", value=False)
                if use_manual_input:
                    st.write("Provide daily weather data for the next 5 days.")
                    tmax_values, tmin_values, precip_values = [], [], []
                    for i in range(5):
                        st.write(f"**Day {i+1}**")
                        tmax = st.number_input(f"Max Temp (°C) Day {i+1}", value=25.0, key=f"tmax_{i}")
                        tmin = st.number_input(f"Min Temp (°C) Day {i+1}", value=15.0, key=f"tmin_{i}")
                        prcp = st.number_input(f"Precip (mm) Day {i+1}", value=0.0, key=f"precip_{i}")
                        tmax_values.append(tmax)
                        tmin_values.append(tmin)
                        precip_values.append(prcp)
                    eto_values = []
                    for (tx, tn) in zip(tmax_values, tmin_values):
                        if tx<tn: tx, tn = tn, tx
                        Ra = 10
                        Tmean = (tx+tn)/2
                        ETo_temp = 0.0023*Ra*(Tmean+17.8)*((tx-tn)**0.5)
                        ETo_temp = max(0,ETo_temp)
                        eto_values.append(ETo_temp)
                    manual_forecast_data = {
                        'tmax': tmax_values,
                        'tmin': tmin_values,
                        'precip': precip_values,
                        'eto': eto_values
                    }
            else:
                forecast_lat = forecast_lon = 0.0
            
            enable_nue = st.checkbox("Enable NUE Estimation (only if yield is computed)?", value=False)
            enable_dynamic_root = st.checkbox("Enable Dynamic Root Growth", value=False)
            if enable_dynamic_root:
                initial_root_depth = st.number_input("Initial Root Depth (mm)", min_value=50, value=300, step=10)
                max_root_depth = st.number_input("Maximum Root Depth (mm)", min_value=50, value=1000, step=10)
                days_to_max = st.number_input("Days to Max Root Depth", min_value=1, value=60, step=1)
            else:
                initial_root_depth = max_root_depth = days_to_max = None
            
            show_soil_profile = st.checkbox("Show Soil Profile Water Storage after simulation?", value=False)
    
    st.button("Run Simulation", key="run_simulation")

    # If user clicked "Run Simulation":
    if st.session_state.get('run_simulation', False):
        # 1) Read weather data (uploaded or not)
        if weather_file is not None:
            try:
                weather_df = pd.read_csv(weather_file)
                # parse Date
                if pd.api.types.is_string_dtype(weather_df['Date']):
                    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
                weather_df = weather_df.sort_values('Date').reset_index(drop=True)
            except:
                st.error("Failed to read uploaded weather file.")
                st.stop()
        else:
            # if no file, user must have a daily weather for sim
            # minimal approach: we do an example
            st.warning("No weather data file. Will attempt a demonstration 5-day forecast for the chosen lat/lon.")
            start_date = datetime.now().date()
            end_date   = start_date + timedelta(days=4)
            weather_df = fetch_weather_data(forecast_lat, forecast_lon, start_date, end_date, forecast=True,
                                            manual_data=None)
            if weather_df is None or weather_df.empty:
                st.error("No weather data available. Please upload or provide data.")
                st.stop()
        
        # 2) Crop data
        if use_custom_crop_file:
            if crop_file is not None:
                try:
                    crop_df = pd.read_csv(crop_file)
                    crop_df = crop_df.sort_values("Start_Day").reset_index(drop=True)
                except:
                    st.error("Failed to read uploaded crop stage file.")
                    st.stop()
            else:
                st.error("Please upload a crop stage file or uncheck 'Use custom crop file'.")
                st.stop()
        else:
            # create a minimal 2-row DataFrame from user selection
            approximate_duration = st.session_state.get('approximate_duration', 90)
            # we re-construct the same logic as above:
            # we have "auto_crop_df" in the UI but we need to store it or reconstruct
            # for simplicity, let's re-do it quickly:
            if 'approximate_duration' in st.session_state:
                approximate_duration = st.session_state['approximate_duration']
            else:
                approximate_duration = 90
            
            stage1_end = int(round(0.7*approximate_duration))
            default_p = 0.5
            default_Ke = 0.1
            kcb_mid = KC_DATABASE[selected_crop]["Kcb_mid"]
            kcb_end = KC_DATABASE[selected_crop]["Kcb_end"]
            
            auto_crop_df = pd.DataFrame({
                "Start_Day":[1, stage1_end+1],
                "End_Day":[stage1_end, approximate_duration],
                "Kcb":[kcb_mid, kcb_end],
                "Root_Depth_mm":[300, 600],
                "p":[default_p, default_p],
                "Ke":[default_Ke, default_Ke]
            })
            crop_df = auto_crop_df
        
        # 3) Soil data
        if soil_file is not None:
            try:
                soil_df = pd.read_csv(soil_file)
            except:
                st.error("Failed to read soil file.")
                st.stop()
        else:
            st.warning("No soil file. Will use a default 2-layer soil.")
            soil_df = pd.DataFrame({
                "Depth_mm":[200, 100],
                "FC":[0.30, 0.30],
                "WP":[0.15, 0.15],
                "TEW":[20, 0],
                "REW":[5, 0]
            })
        
        # 4) Run water balance on uploaded weather data
        if weather_df is not None and not weather_df.empty:
            total_days = len(weather_df)
            # Force the last day in crop_df to match total_days
            crop_df.loc[crop_df.index[-1],"End_Day"] = total_days
            
            results_df, soil_profile = SIMdualKc(
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
            st.session_state.soil_profile = soil_profile
            
            # 5) If 5-day forecast is enabled, do next 5 days from today with final soil moisture
            if enable_etaforecast:
                final_SW_layers = [layer['SW (mm)'] for layer in soil_profile]
                # Build a minimal 1-row crop df for next 5 days with the same Kcb as end-season
                # or we can keep Kcb_end:
                forecast_crop_df = pd.DataFrame({
                    "Start_Day":[1],
                    "End_Day":[5],
                    "Kcb":[KC_DATABASE[selected_crop]["Kcb_end"]],  # or keep it simpler
                    "Root_Depth_mm":[600],
                    "p":[0.5],
                    "Ke":[0.1]
                })
                
                forecast_start = datetime.now().date()
                forecast_end   = forecast_start + timedelta(days=4)
                forecast_weather = fetch_weather_data(forecast_lat, forecast_lon, forecast_start, forecast_end,
                                                      forecast=True, manual_data=manual_forecast_data)
                if forecast_weather is not None and not forecast_weather.empty:
                    forecast_results = SIMdualKc(
                        forecast_weather, 
                        forecast_crop_df, 
                        soil_df, 
                        track_drainage=track_drainage,
                        enable_yield=False,  # usually yield not relevant to just 5 day
                        return_soil_profile=False,
                        initial_SW_layers=final_SW_layers
                    )
                    st.session_state.forecast_results = forecast_results
                else:
                    st.session_state.forecast_results = None
                    st.warning("Forecast data is not available.")
            
            st.success("Simulation completed successfully!")
        else:
            st.error("Weather data is empty or invalid. Please check your inputs.")


with results_tab:
    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        forecast_results = st.session_state.forecast_results
        soil_profile = st.session_state.soil_profile
        
        st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)
        st.dataframe(results_df)
        st.download_button("Download Results (.txt)", results_df.to_csv(index=False), 
                           file_name="results.txt", mime="text/plain")
        
        if forecast_results is not None and not forecast_results.empty:
            st.markdown('<div class="sub-header">5-Day ETa Forecast Results</div>', unsafe_allow_html=True)
            st.dataframe(forecast_results[["Date","ET0 (mm)","ETa_total (mm)","ETa_transp (mm)","ETa_evap (mm)"]])
        
        if soil_profile and len(soil_profile)>0:
            st.markdown('<div class="sub-header">Final Soil Profile</div>', unsafe_allow_html=True)
            if "SW (mm)" in soil_profile[0]:
                profile_df = pd.DataFrame(soil_profile)
                st.dataframe(profile_df)
        
        # Graphing
        st.markdown('<div class="sub-header">Graphs</div>', unsafe_allow_html=True)
        plot_options = ["ETa Components", "Soil Water in Root Zone", "Drainage", "Cumulative Metrics"]
        if "Yield (ton/ha)" in results_df.columns:
            plot_options.append("Yield")
        if "Leaching (kg/ha)" in results_df.columns:
            plot_options.append("Leaching")
        if "NUE (kg/ha)" in results_df.columns:
            plot_options.append("NUE")
        
        plot_choice = st.selectbox("Choose a plot", plot_options)
        
        if plot_choice == "ETa Components":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["ETa_transp (mm)"], label="Transpiration")
            ax.plot(results_df["Date"], results_df["ETa_evap (mm)"], label="Evaporation")
            ax.plot(results_df["Date"], results_df["ETa_total (mm)"], label="ETa total")
            ax.set_xlabel("Date")
            ax.set_ylabel("ET (mm)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        elif plot_choice == "Soil Water in Root Zone":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["SW_root (mm)"], label="SW Root Zone")
            ax.set_xlabel("Date")
            ax.set_ylabel("Soil Water (mm)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        elif plot_choice == "Drainage":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["Daily_Drainage (mm)"], label="Drainage")
            ax.set_xlabel("Date")
            ax.set_ylabel("Drainage (mm)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        elif plot_choice == "Cumulative Metrics":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["Cumulative_Irrigation (mm)"], label="Cumulative Irrigation")
            ax.plot(results_df["Date"], results_df["Cumulative_Precip (mm)"], label="Cumulative Precip")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative (mm)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        elif plot_choice == "Yield":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["Yield (ton/ha)"], label="Yield (ton/ha)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Yield (ton/ha)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        elif plot_choice == "Leaching":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["Leaching (kg/ha)"], label="Leaching")
            ax.set_xlabel("Date")
            ax.set_ylabel("Leaching (kg/ha)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        elif plot_choice == "NUE":
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(results_df["Date"], results_df["NUE (kg/ha)"], label="NUE")
            ax.set_xlabel("Date")
            ax.set_ylabel("NUE (kg/ha)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
    else:
        st.info("Please complete the setup and run the simulation to view results.")

st.markdown('<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>', unsafe_allow_html=True)
