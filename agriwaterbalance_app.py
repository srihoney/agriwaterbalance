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
# 1. Large Kc Database: merging Pereira et al. (2021a,b) plus additional references
# --------------------------------------------------------------------------------
# NOTE: The references are:
#    (a) Standard single and basal crop coefficients for vegetable crops, an update of FAO56 (Pereira et al., 2021)
#    (b) Standard single and basal crop coefficients for field crops. Updates to FAO56 (Pereira et al., 2021)
# Below includes additional typical Kc from broader FAO-56 or other sources.
# For each crop, we store Kc_mid, Kc_end, Kcb_mid, Kcb_end in sub-humid reference climate.
# Also, default "total crop length" or stage lengths (for auto stage generation).

CROP_DATABASE = {
    # ---------- Vegetables ----------
    "Carrot": {
        "Kc_mid":1.05, "Kc_end":0.95, "Kcb_mid":1.00, "Kcb_end":0.90,
        "total_days_default":90  # approximate
    },
    "Beet": {
        "Kc_mid":1.10, "Kc_end":0.95, "Kcb_mid":1.05, "Kcb_end":0.85,
        "total_days_default":100
    },
    "Garlic": {
        "Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65,
        "total_days_default":120
    },
    "Onion (fresh)": {
        "Kc_mid":1.10, "Kc_end":0.80, "Kcb_mid":1.05, "Kcb_end":0.75,
        "total_days_default":110
    },
    "Onion (dry)": {
        "Kc_mid":1.10, "Kc_end":0.65, "Kcb_mid":1.05, "Kcb_end":0.60,
        "total_days_default":120
    },
    "Cabbage": {
        "Kc_mid":1.00, "Kc_end":0.90, "Kcb_mid":0.95, "Kcb_end":0.85,
        "total_days_default":90
    },
    "Lettuce": {
        "Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90,
        "total_days_default":65
    },
    "Spinach": {
        "Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90,
        "total_days_default":55
    },
    "Broccoli": {
        "Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90,
        "total_days_default":80
    },
    "Cauliflower": {
        "Kc_mid":1.00, "Kc_end":0.95, "Kcb_mid":0.95, "Kcb_end":0.90,
        "total_days_default":95
    },
    "Green bean": {
        "Kc_mid":1.05, "Kc_end":0.90, "Kcb_mid":1.00, "Kcb_end":0.85,
        "total_days_default":75
    },
    "Tomato (fresh)": {
        "Kc_mid":1.15, "Kc_end":0.80, "Kcb_mid":1.10, "Kcb_end":0.75,
        "total_days_default":120
    },
    "Tomato (proc)": {
        "Kc_mid":1.15, "Kc_end":0.70, "Kcb_mid":1.10, "Kcb_end":0.65,
        "total_days_default":110
    },
    "Pepper": {
        "Kc_mid":1.15, "Kc_end":0.90, "Kcb_mid":1.10, "Kcb_end":0.85,
        "total_days_default":130
    },
    "Eggplant": {
        "Kc_mid":1.10, "Kc_end":0.90, "Kcb_mid":1.05, "Kcb_end":0.85,
        "total_days_default":130
    },
    "Zucchini": {
        "Kc_mid":1.05, "Kc_end":0.80, "Kcb_mid":1.00, "Kcb_end":0.75,
        "total_days_default":60
    },
    "Cucumber": {
        "Kc_mid":1.00, "Kc_end":0.75, "Kcb_mid":0.95, "Kcb_end":0.70,
        "total_days_default":70
    },
    "Melon": {
        "Kc_mid":1.05, "Kc_end":0.65, "Kcb_mid":1.00, "Kcb_end":0.60,
        "total_days_default":85
    },
    "Watermelon": {
        "Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65,
        "total_days_default":90
    },
    "Pumpkin": {
        "Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65,
        "total_days_default":100
    },
    "Okra": {
        "Kc_mid":1.15, "Kc_end":0.75, "Kcb_mid":1.10, "Kcb_end":0.70,
        "total_days_default":100
    },
    "Basil": {
        "Kc_mid":1.00, "Kc_end":0.80, "Kcb_mid":0.95, "Kcb_end":0.75,
        "total_days_default":60
    },
    "Parsley": {
        "Kc_mid":1.00, "Kc_end":0.85, "Kcb_mid":0.95, "Kcb_end":0.80,
        "total_days_default":70
    },
    "Coriander": {
        "Kc_mid":1.00, "Kc_end":0.85, "Kcb_mid":0.95, "Kcb_end":0.80,
        "total_days_default":65
    },
    # Additional leafy greens or veggies not in original shorter list:
    "Celery": {
        "Kc_mid":1.05, "Kc_end":0.90, "Kcb_mid":1.00, "Kcb_end":0.85,
        "total_days_default":120
    },
    "Turnip": {
        "Kc_mid":1.05, "Kc_end":0.80, "Kcb_mid":1.00, "Kcb_end":0.75,
        "total_days_default":85
    },
    "Radish": {
        "Kc_mid":1.00, "Kc_end":0.80, "Kcb_mid":0.95, "Kcb_end":0.75,
        "total_days_default":45
    },

    # ---------- Field Crops ----------
    "Wheat": {
        "Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30,
        "total_days_default":150
    },
    "Barley": {
        "Kc_mid":1.15, "Kc_end":0.25, "Kcb_mid":1.10, "Kcb_end":0.20,
        "total_days_default":130
    },
    "Maize": {
        "Kc_mid":1.20, "Kc_end":0.60, "Kcb_mid":1.15, "Kcb_end":0.55,
        "total_days_default":140
    },
    "Rice": {
        "Kc_mid":1.20, "Kc_end":0.90, "Kcb_mid":1.15, "Kcb_end":0.85,
        "total_days_default":160
    },
    "Sorghum": {
        "Kc_mid":1.05, "Kc_end":0.40, "Kcb_mid":1.00, "Kcb_end":0.35,
        "total_days_default":120
    },
    "Soybean": {
        "Kc_mid":1.15, "Kc_end":0.50, "Kcb_mid":1.10, "Kcb_end":0.45,
        "total_days_default":130
    },
    "Bean": {
        "Kc_mid":1.15, "Kc_end":0.90, "Kcb_mid":1.10, "Kcb_end":0.85,
        "total_days_default":95
    },
    "Peanut": {
        "Kc_mid":1.10, "Kc_end":0.60, "Kcb_mid":1.05, "Kcb_end":0.55,
        "total_days_default":135
    },
    "Cotton": {
        "Kc_mid":1.15, "Kc_end":0.65, "Kcb_mid":1.10, "Kcb_end":0.60,
        "total_days_default":160
    },
    "Sugarbeet": {
        "Kc_mid":1.20, "Kc_end":0.60, "Kcb_mid":1.15, "Kcb_end":0.55,
        "total_days_default":180
    },
    "Sugarcane": {
        "Kc_mid":1.25, "Kc_end":1.10, "Kcb_mid":1.20, "Kcb_end":1.05,
        "total_days_default":300
    },
    "Sunflower": {
        "Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30,
        "total_days_default":120
    },
    "Rapeseed": {
        "Kc_mid":1.15, "Kc_end":0.40, "Kcb_mid":1.10, "Kcb_end":0.35,
        "total_days_default":150
    },
    "Mustard": {
        "Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30,
        "total_days_default":120
    },
    "Faba bean": {
        "Kc_mid":1.15, "Kc_end":0.65, "Kcb_mid":1.10, "Kcb_end":0.60,
        "total_days_default":130
    },
    "Chickpea": {
        "Kc_mid":1.15, "Kc_end":0.25, "Kcb_mid":1.10, "Kcb_end":0.20,
        "total_days_default":120
    },
    # Additional field crops from older references or not in the short list:
    "Millet": {
        "Kc_mid":1.10, "Kc_end":0.40, "Kcb_mid":1.05, "Kcb_end":0.35,
        "total_days_default":100
    },
    "Quinoa": {
        "Kc_mid":1.05, "Kc_end":0.45, "Kcb_mid":1.00, "Kcb_end":0.40,
        "total_days_default":120
    },
    "Lentil": {
        "Kc_mid":1.10, "Kc_end":0.25, "Kcb_mid":1.05, "Kcb_end":0.20,
        "total_days_default":110
    },
    "Potato": {
        "Kc_mid":1.15, "Kc_end":0.75, "Kcb_mid":1.10, "Kcb_end":0.70,
        "total_days_default":110
    }
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

# Header
st.markdown(f"""
    <div class="header-container">
        <div class="header-logo">
            <img src="{logo_url}" alt="Logo">
        </div>
        <div class="header-title">Advanced AgriWaterBalance</div>
    </div>
""", unsafe_allow_html=True)
st.markdown("**A Professional, All-in-One Tool for Crop Water Management**", unsafe_allow_html=True)

# We create 3 main tabs: Setup, Results, Irrigation Calendar
setup_tab, results_tab, irrig_calendar_tab = st.tabs(["Setup Simulation", "Results", "Irrigation Calendar"])

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
# 5. Water Balance Functions (SIMDualKc-like approach)
#    We add more daily columns: infiltration, depletion Dr, TAW, RAW, etc.
# --------------------------------------------------------------------------------

def compute_Ks(Dr, RAW, TAW):
    """
    SIMDualKc style:
    Dr = depletion at the end of previous day
    RAW = readily available water
    TAW = total available water
    if Dr <= RAW => Ks=1
    elif Dr >= TAW => Ks=0
    else => linear
    """
    if Dr <= RAW:
        return 1.0
    elif Dr >= TAW:
        return 0.0
    else:
        return max(0.0, (TAW - Dr)/ (TAW - RAW))

def compute_Kr(TEW, REW, Ew):
    """
    Evaporation reduction factor for topsoil
    TEW: total evaporable water in topsoil
    REW: readily evaporable water
    Ew: cumulative evaporation since last big wetting
    """
    if Ew <= REW:
        return 1.0
    elif Ew >= TEW:
        return 0.0
    else:
        return (TEW - Ew)/(TEW - REW)

def fetch_weather_data(lat, lon, start_date, end_date, forecast=True, manual_data=None):
    """
    Grabs either next 5-day forecast or manual data
    """
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}_{forecast}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    # manual
    if manual_data is not None:
        dates = pd.date_range(start_date, end_date)
        wdf = pd.DataFrame({
            "Date": dates,
            "ET0": manual_data["eto"],
            "Precipitation": manual_data["precip"],
            "Irrigation": [0]*len(dates)
        })
        st.session_state.weather_cache[cache_key] = wdf
        return wdf
    
    if forecast:
        if st.session_state.api_calls>=1000:
            st.warning("Daily API call limit reached.")
            return None
        if lat==0 and lon==0:
            st.warning("Invalid lat/lon.")
            return None
        today = datetime.now().date()
        maxf = today+timedelta(days=5)
        if start_date<today or end_date>maxf:
            st.warning("Adjusting forecast to next 5 days.")
            start_date=today
            end_date= today+timedelta(days=4)
        api_key="fe2d869569674a4afbfca57707bdf691"
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            st.session_state.api_calls +=1
            data = resp.json()
            dd={}
            for entry in data['list']:
                dt_ = datetime.fromtimestamp(entry['dt']).date()
                if start_date<= dt_ <= end_date:
                    ds = dt_.strftime("%Y-%m-%d")
                    if ds not in dd:
                        dd[ds]={
                            'tmax':entry['main']['temp_max'],
                            'tmin':entry['main']['temp_min'],
                            'precip': entry.get('rain',{}).get('3h',0)
                        }
                    else:
                        dd[ds]['tmax'] = max(dd[ds]['tmax'], entry['main']['temp_max'])
                        dd[ds]['tmin'] = min(dd[ds]['tmin'], entry['main']['temp_min'])
                        dd[ds]['precip'] += entry.get('rain',{}).get('3h',0)
            ds_=sorted(dd.keys())
            dates,ET0_,PP_=[],[],[]
            for dsi in ds_:
                dat_ = pd.to_datetime(dsi)
                dates.append(dat_)
                tmax_ = dd[dsi]['tmax']
                tmin_ = dd[dsi]['tmin']
                if tmax_<tmin_:
                    tmax_,tmin_= tmin_,tmax_
                Ra=10
                Tmean=(tmax_+tmin_)/2
                eto_ = 0.0023*Ra*(Tmean+17.8)*((tmax_-tmin_)**0.5)
                ET0_.append(max(0,eto_))
                PP_.append(dd[dsi]['precip'])
            wdf= pd.DataFrame({
                "Date":dates,
                "ET0":ET0_,
                "Precipitation":PP_,
                "Irrigation":[0]*len(dates)
            }).sort_values("Date").reset_index(drop=True)
            st.session_state.weather_cache[cache_key]= wdf
            return wdf
        except:
            st.error("Unable to fetch forecast data.")
            return None
    else:
        # historical
        try:
            start_str= start_date.strftime("%Y%m%d")
            end_str= end_date.strftime("%Y%m%d")
            url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
            r = session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            dts = pd.date_range(start_date, end_date)
            ET0_,PP_=[],[]
            for dt_ in dts:
                ds = dt_.strftime("%Y%m%d")
                rad_ = data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(ds,0)
                eto_v = rad_*0.2
                ET0_.append(eto_v)
                prec= data['properties']['parameter']['PRECTOTCORR'].get(ds,0)
                PP_.append(prec)
            wdf= pd.DataFrame({
                "Date":dts,
                "ET0":ET0_,
                "Precipitation":PP_,
                "Irrigation":[0]*len(dts)
            })
            st.session_state.weather_cache[cache_key]= wdf
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
    """
    This function is an expanded daily water balance akin to SIMDualKc:
     - Tracks Dr: depletion in the root zone
     - TAW, RAW
     - Ks
     - Evap in topsoil (Kr, etc.)
     - Outputs full daily data
    """
    if weather_df.empty:
        st.error("Weather data is empty.")
        return None
    
    NDAYS = len(weather_df)
    
    # Prepare arrays for each stage param:
    # We'll define daily (Kcb, p, root_depth, Ke) by linear interpolation from crop_stages_df
    # but we'll track Dr daily in a single-bucket style for the root zone.
    
    # Sort by Start_Day
    crop_stages_df = crop_stages_df.sort_values("Start_Day").reset_index(drop=True)
    
    # Build daily arrays:
    day_kcb = np.zeros(NDAYS)
    day_p   = np.zeros(NDAYS)
    day_ke  = np.zeros(NDAYS)
    day_root = np.zeros(NDAYS)
    
    # Step 1: Interpolate
    for i in range(len(crop_stages_df)-1):
        st_i = int(crop_stages_df.iloc[i]['Start_Day'])-1
        en_i = int(crop_stages_df.iloc[i]['End_Day'])-1
        if en_i<0: continue
        en_i= min(en_i, NDAYS-1)
        st_i= max(0, st_i)
        if st_i>en_i: continue
        
        kcb_s = crop_stages_df.iloc[i]['Kcb']
        kcb_e = crop_stages_df.iloc[i+1]['Kcb']
        p_s   = crop_stages_df.iloc[i]['p']
        p_e   = crop_stages_df.iloc[i+1]['p']
        ke_s  = crop_stages_df.iloc[i]['Ke']
        ke_e  = crop_stages_df.iloc[i+1]['Ke']
        rd_s  = crop_stages_df.iloc[i]['Root_Depth_mm']
        rd_e  = crop_stages_df.iloc[i+1]['Root_Depth_mm']
        L = en_i-st_i+1
        day_kcb[st_i:en_i+1]  = np.linspace(kcb_s, kcb_e, L)
        day_p[st_i:en_i+1]    = np.linspace(p_s, p_e, L)
        day_ke[st_i:en_i+1]   = np.linspace(ke_s, ke_e, L)
        day_root[st_i:en_i+1] = np.linspace(rd_s, rd_e, L)
    
    # Fill last stage
    i_last= len(crop_stages_df)-1
    st_l= int(crop_stages_df.iloc[i_last]['Start_Day'])-1
    en_l= int(crop_stages_df.iloc[i_last]['End_Day'])-1
    if st_l<0: st_l=0
    if en_l<0: en_l=0
    if en_l>NDAYS-1: en_l= NDAYS-1
    if st_l<=en_l:
        day_kcb[st_l:en_l+1]  = crop_stages_df.iloc[i_last]['Kcb']
        day_p[st_l:en_l+1]    = crop_stages_df.iloc[i_last]['p']
        day_ke[st_l:en_l+1]   = crop_stages_df.iloc[i_last]['Ke']
        day_root[st_l:en_l+1] = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    if en_l<NDAYS-1:
        day_kcb[en_l+1:]      = crop_stages_df.iloc[i_last]['Kcb']
        day_p[en_l+1:]        = crop_stages_df.iloc[i_last]['p']
        day_ke[en_l+1:]       = crop_stages_df.iloc[i_last]['Ke']
        day_root[en_l+1:]     = crop_stages_df.iloc[i_last]['Root_Depth_mm']
    
    # If dynamic root is on, override the daily root with a linear from init->max-> ...
    if dynamic_root:
        root_lin = np.linspace(init_root, max_root, min(days_to_max, NDAYS)).tolist()
        if NDAYS>days_to_max:
            root_lin+= [max_root]*(NDAYS-days_to_max)
        day_root= np.array(root_lin[:NDAYS])
    
    # We read soil info
    # We'll assume a multi-layer approach, track each layer's water.
    # Then Dr is the difference from FC?
    # We'll do topsoil evaporation with TEW, REW, etc.
    TEW = soil_df['TEW'].sum()
    REW = soil_df['REW'].sum()
    # Start evaporation count
    E_count= REW  # start "dry"
    
    # Initialize SW in each layer at FC
    SW_layers = [ (soil_df.iloc[j]['FC']*soil_df.iloc[j]['Depth_mm']) for j in range(len(soil_df)) ]
    
    results=[]
    cumIrr=0
    cumPrec=0
    
    for d_i in range(NDAYS):
        date_i = weather_df.iloc[d_i]['Date']
        ET0_i  = max(0, weather_df.iloc[d_i]['ET0'])
        PR_i   = max(0, weather_df.iloc[d_i]['Precipitation'])
        IR_i   = max(0, weather_df.iloc[d_i]['Irrigation'])
        
        cumIrr+= IR_i
        cumPrec+=PR_i
        
        # daily crop param
        Kcb_i= day_kcb[d_i]
        p_i  = day_p[d_i]
        ke0_i= day_ke[d_i]
        rd_i= max(1, day_root[d_i])
        
        # Weighted average soil WP,FC in root zone:
        # We'll do the TAW, RAW approach
        # sum( layer TAW if within root zone )
        tot_depth=0
        TAW=0
        RAW=0
        Dr_prev=0
        SW_root=0
        for j in range(len(soil_df)):
            layer_d= soil_df.iloc[j]['Depth_mm']
            WP_j=   soil_df.iloc[j]['WP']
            FC_j=   soil_df.iloc[j]['FC']
            new_d= tot_depth+ layer_d
            fraction=0
            if new_d<=rd_i:
                fraction=1.0
            elif tot_depth<rd_i<new_d:
                fraction= (rd_i- tot_depth)/layer_d
            # TAW in that fraction
            if fraction>0:
                TAW += (FC_j- WP_j)* layer_d* fraction*1000.0/ layer_d # Actually simpler approach
                # we'll sum the actual mm as: (FC_j-WP_j)*layer_d
                # but let's do the final in mm not per meter.
                TAW += 0  # actually let's do a direct approach below
            tot_depth= new_d
        
        # Actually let's do a direct approach: we must do partial sums:
        # We'll track the water in each layer inside the root zone => SW_root
        # Then Dr= (FC - SWroot)
        # But we need the sum(FC_j * fraction) too
        # We'll do it layer by layer while also updating Dr
        tot_depth=0
        sum_FC=0
        sum_WP=0
        for j in range(len(SW_layers)):
            layer_d= soil_df.iloc[j]['Depth_mm']
            WP_j= soil_df.iloc[j]['WP']
            FC_j= soil_df.iloc[j]['FC']
            new_d= tot_depth+ layer_d
            fraction=0
            if new_d<= rd_i:
                fraction=1.0
            elif tot_depth<rd_i<new_d:
                fraction= (rd_i- tot_depth)/ layer_d
            # sum partial FC, WP, SW
            sw_j = SW_layers[j]
            if fraction>0:
                sum_FC+= FC_j* layer_d*fraction
                sum_WP+= WP_j* layer_d*fraction
                SW_root+= sw_j * fraction
            tot_depth=new_d
        # TAW in mm:
        TAW_ = (sum_FC- sum_WP)
        RAW_ = p_i*TAW_
        # Dr is how much less water than FC
        Dr_ = (sum_FC- SW_root)
        
        # compute Ks
        Ks_ = compute_Ks(Dr_, RAW_, TAW_)
        
        # topsoil evaporation factor
        Kr_ = compute_Kr(TEW, REW, E_count)
        Ke_ = Kr_* ke0_i
        
        # daily ETc
        ETc_ = (Kcb_i*Ks_ + Ke_)* ET0_i
        Etc_trans= Kcb_i*Ks_* ET0_i
        Etc_evap = Ke_* ET0_i
        # infiltration
        infiltration= PR_i+ IR_i
        # We'll keep run-off=0 for simplicity
        runoff=0
        
        # water that can fill the layers up to FC
        excess= infiltration - ETc_
        drainage=0
        if track_drainage:
            # fill each layer up to FC
            for j in range(len(SW_layers)):
                layer_fc= soil_df.iloc[j]['FC']* soil_df.iloc[j]['Depth_mm']
                gap_j= layer_fc- SW_layers[j]
                if gap_j>0 and excess>0:
                    added= min(excess, gap_j)
                    SW_layers[j]+= added
                    excess-= added
            drainage= max(0, excess)
            # check min WP
            for j in range(len(SW_layers)):
                layer_wp= soil_df.iloc[j]['WP']*soil_df.iloc[j]['Depth_mm']
                layer_fc= soil_df.iloc[j]['FC']*soil_df.iloc[j]['Depth_mm']
                SW_layers[j]= max(layer_wp, min(layer_fc, SW_layers[j]))
        else:
            # no drainage tracking
            drainage=0
        
        # remove ETc from the root zone & topsoil
        # We remove Etc_trans from the root zone proportionally
        tr_rem= Etc_trans
        if tr_rem>0 and SW_root>0:
            tot_depth=0
            for j in range(len(SW_layers)):
                layer_d= soil_df.iloc[j]['Depth_mm']
                new_d= tot_depth+ layer_d
                fraction=0
                if new_d<= rd_i:
                    fraction=1.0
                elif tot_depth< rd_i< new_d:
                    fraction= (rd_i- tot_depth)/ layer_d
                tot_depth=new_d
                if fraction>0:
                    # max that can be removed from that layer up to WP
                    fc_j= soil_df.iloc[j]['FC']* layer_d
                    wp_j= soil_df.iloc[j]['WP']* layer_d
                    available_j= SW_layers[j] - wp_j
                    # portion to remove
                    portion= tr_rem * fraction
                    actual_remove= min(portion, available_j)
                    SW_layers[j]-= actual_remove
                    tr_rem-= actual_remove
                    if tr_rem<=0: break
        
        # For evaporation portion, remove from top soil layer(s). We'll assume top layer is j=0:
        ev_rem= Etc_evap
        if ev_rem>0:
            # remove from first layer only for simplicity:
            fc_0= soil_df.iloc[0]['FC']* soil_df.iloc[0]['Depth_mm']
            wp_0= soil_df.iloc[0]['WP']* soil_df.iloc[0]['Depth_mm']
            available_0= SW_layers[0] - wp_0
            actual_rm= min(ev_rem, available_0)
            SW_layers[0]-= actual_rm
            ev_rem-= actual_rm
        
        # update E_count
        if infiltration>=4.0:
            E_count=0
        else:
            E_count+= Etc_evap
        E_count= max(0, min(E_count, TEW))
        
        # recompute final SW_root:
        new_SWroot=0
        tot_depth=0
        sum_FC2=0
        for j in range(len(SW_layers)):
            layer_d= soil_df.iloc[j]['Depth_mm']
            new_d= tot_depth+ layer_d
            fraction=0
            if new_d<= rd_i:
                fraction=1.0
            elif tot_depth< rd_i< new_d:
                fraction= (rd_i- tot_depth)/ layer_d
            new_SWroot+= SW_layers[j]* fraction
            sum_FC2+= soil_df.iloc[j]['FC']* layer_d* fraction
            tot_depth=new_d
        Dr_end= sum_FC2- new_SWroot
        
        # optional yield, leaching
        yield_= None
        if enable_yield:
            if (Ym>0 and Ky>0 and ETc_>0):
                # daily approach is approximate
                y_ = Ym*(1- Ky*(1- (ETc_/ (Kcb_i*ET0_i+1e-9))))  # approximate
                yield_= max(0,y_)
            if use_transp and WP_yield>0:
                yield_= (WP_yield* Etc_trans)
        
        leaching_=0
        if enable_leaching:
            # method 1 => drainage * nitrate
            # method 2 => fraction * totalN
            # We'll do a simple approach: drainage mm => 10m3/ha per mm => mg->kg
            leaching_= drainage*10*(nitrate_conc*1e-6)*1000
        
        # gather daily output
        day_out={
            "Date":date_i,
            "ET0 (mm)":ET0_i,
            "Precip (mm)":PR_i,
            "Irrigation (mm)":IR_i,
            "Runoff (mm)":runoff,
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
            "Cumulative_Irrig (mm)":cumIrr,
            "Cumulative_Precip (mm)":cumPrec
        }
        if yield_ is not None:
            day_out["Yield (ton/ha)"]= yield_
        if enable_leaching:
            day_out["Leaching (kg/ha)"]= leaching_
        
        # store each layer SW
        for j in range(len(SW_layers)):
            day_out[f"Layer{j}_SW (mm)"]= SW_layers[j]
        
        results.append(day_out)
    
    outdf= pd.DataFrame(results)
    return outdf

def create_auto_stages_for_crop(crop_name):
    """
    Creates a simple 2-stage or 4-stage DataFrame for the chosen crop
    using the default total_days_default in CROP_DATABASE.
    We do a 4-stage approach: initial, dev, mid, late
    """
    total_d = CROP_DATABASE[crop_name]["total_days_default"]
    init_d  = int(total_d*0.2)   # first 20% 
    dev_d   = int(total_d*0.3)   # next 30%
    mid_d   = int(total_d*0.3)   # next 30%
    late_d  = total_d- (init_d+ dev_d+ mid_d)
    
    # Kcb references
    kcb_mid= CROP_DATABASE[crop_name]["Kcb_mid"]
    kcb_end= CROP_DATABASE[crop_name]["Kcb_end"]
    
    # Let Kcb initial=0.15, dev goes from 0.15-> kcb_mid
    # mid= kcb_mid, late goes from kcb_mid->kcb_end
    # We'll guess p=0.5 for all, Ke= 1.0 initial -> 0.1 final
    # We'll guess root depth from 100mm to 600mm
    # This is purely an example. You can refine.
    
    stg = []
    # Stage1: initial
    stg.append({
        "Start_Day":1,
        "End_Day":init_d,
        "Kcb":0.15,
        "Root_Depth_mm":100,
        "p":0.5,
        "Ke":1.0
    })
    # Stage2: dev
    stg.append({
        "Start_Day":init_d+1,
        "End_Day":init_d+dev_d,
        "Kcb":kcb_mid,
        "Root_Depth_mm":400,
        "p":0.5,
        "Ke":0.5
    })
    # Stage3: mid
    stg.append({
        "Start_Day":init_d+dev_d+1,
        "End_Day":init_d+dev_d+mid_d,
        "Kcb":kcb_mid,
        "Root_Depth_mm":600,
        "p":0.5,
        "Ke":0.2
    })
    # Stage4: late
    stg.append({
        "Start_Day":init_d+dev_d+mid_d+1,
        "End_Day":total_d,
        "Kcb":kcb_end,
        "Root_Depth_mm":600,
        "p":0.5,
        "Ke":0.1
    })
    df = pd.DataFrame(stg)
    return df

def produce_irrigation_calendar(results_df, forecast_df=None):
    """
    Creates a monthly style table for the next forecast days.
    We'll guess a simple threshold approach:
       if Dr_end > RAW => propose irrigation
    We'll combine it into a 'calendar' style for the 5-day forecast or the entire results.
    """
    # We'll just do a minimal approach for demonstration:
    # gather the final 5 days or next 5 days from forecast, if available
    if forecast_df is not None and not forecast_df.empty:
        # let's join if needed
        # but we don't have the daily Dr from forecast unless we also run the model on forecast data
        # We'll assume the user used "results_df" for the forecast as well, or we skip
        pass
    
    # We'll produce a "calendar" of date, ETa, recommended irrigation
    # We'll do monthly grouping if there's more than 5 days
    # But typically we only have 5 days. Let's do day by day with a "Month - Day" approach
    cal_data = []
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    for i, row in results_df.iterrows():
        dt_ = row["Date"]
        Dr_e= row["Dr_end (mm)"] if "Dr_end (mm)" in row else None
        RAW_ = row["RAW (mm)"] if "RAW (mm)" in row else None
        ETa_ = row["ETa (mm)"] if "ETa (mm)" in row else None
        rec_irr=0
        if Dr_e is not None and RAW_ is not None:
            if Dr_e> RAW_:
                rec_irr= round(Dr_e- RAW_,1)
        cal_data.append({
            "Month": dt_.strftime("%b"),
            "Day":   dt_.day,
            "ETa (mm)": round(ETa_,2) if ETa_ else 0,
            "Dr_end (mm)": round(Dr_e,1) if Dr_e else None,
            "RAW (mm)": round(RAW_,1) if RAW_ else None,
            "Irrigation_Recommendation (mm)": rec_irr
        })
    cal_df= pd.DataFrame(cal_data)
    return cal_df

# --------------------------------------------------------------------------------
# 6. SETUP TAB
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown("### 1) Select Crop")
    # Single dropdown at the top
    crop_list = list(CROP_DATABASE.keys())
    selected_crop = st.selectbox("Choose your crop", crop_list)
    
    st.write(f"**Selected Crop**: {selected_crop}")
    st.write(f" - Kc_mid={CROP_DATABASE[selected_crop]['Kc_mid']},  Kc_end={CROP_DATABASE[selected_crop]['Kc_end']}")
    st.write(f" - Kcb_mid={CROP_DATABASE[selected_crop]['Kcb_mid']}, Kcb_end={CROP_DATABASE[selected_crop]['Kcb_end']}")
    
    st.markdown("### 2) Weather Data")
    weather_file = st.file_uploader("Upload CSV with Date,ET0,Precipitation,Irrigation, or rely on forecast", type=["csv","txt"])
    
    st.markdown("### 3) Crop Stage Data")
    use_custom_stage = st.checkbox("Upload custom Crop Stage Data?", value=False)
    st.write("*Otherwise, we'll compute automatically from known stage durations.*")
    custom_crop_file = None
    if use_custom_stage:
        custom_crop_file= st.file_uploader("Upload Crop Stage CSV (columns: Start_Day, End_Day, Kcb, Root_Depth_mm, p, Ke)",
                                           type=["csv","txt"])
    
    st.markdown("### 4) Soil Layers Data")
    soil_file= st.file_uploader("Upload soil data (Depth_mm, FC, WP, TEW, REW) or we use default", type=["csv","txt"])
    
    st.markdown("### 5) Additional Options")
    colA, colB= st.columns(2)
    with colA:
        track_drainage= st.checkbox("Track Drainage", value=True)
        enable_yield= st.checkbox("Enable Yield Estimation?", value=False)
        if enable_yield:
            st.write("**Yield Options**")
            ym_ = st.number_input("Max Yield (ton/ha)?", min_value=0.0, value=10.0)
            ky_ = st.number_input("Ky (yield response factor)", min_value=0.0, value=1.0)
            use_transp= st.checkbox("Use Transp-based approach (WP_yield)?", value=False)
            if use_transp:
                wp_yield= st.number_input("WP_yield (ton/ha per mm)?", min_value=0.0, value=0.012, step=0.001)
            else:
                wp_yield=0
        else:
            ym_= ky_=0
            use_transp=False
            wp_yield=0
    with colB:
        enable_leaching= st.checkbox("Enable Leaching?", value=False)
        nitrate_conc=10.0
        totalN=100.0
        lf=0.1
        if enable_leaching:
            nitrate_conc= st.number_input("Nitrate mg/L", min_value=0.0, value=10.0)
            totalN= st.number_input("Total N input (kg/ha)?", min_value=0.0, value=100.0)
            lf= st.number_input("Leaching Fraction (0-1)?", min_value=0.0, max_value=1.0, value=0.1)
    
    st.markdown("### 6) ETA Forecast (5-day) Options")
    enable_forecast= st.checkbox("Enable 5-Day Forecast?", value=True)
    manual_forecast_data=None
    lat_=0.0
    lon_=0.0
    if enable_forecast:
        lat_= st.number_input("Latitude?", value=35.0)
        lon_= st.number_input("Longitude?", value=-80.0)
        use_manual_fore= st.checkbox("Manual input for 5 days?", value=False)
        if use_manual_fore:
            tmax_, tmin_, prec_=[],[],[]
            for i in range(5):
                st.write(f"**Day {i+1}**")
                tx_= st.number_input(f"Max Temp(°C) day {i+1}", value=25.0, key=f"tx{i}")
                tn_= st.number_input(f"Min Temp(°C) day {i+1}", value=15.0, key=f"tn{i}")
                pr_= st.number_input(f"Precip(mm) day {i+1}", value=0.0, key=f"pr{i}")
                tmax_.append(tx_)
                tmin_.append(tn_)
                prec_.append(pr_)
            et0_=[]
            for tx_, tn_ in zip(tmax_, tmin_):
                if tx_<tn_: tx_, tn_= tn_, tx_
                Ra=10
                Tm=(tx_+tn_)/2
                eto_val= 0.0023*Ra*(Tm+17.8)*((tx_-tn_)**0.5)
                et0_.append(max(0,eto_val))
            manual_forecast_data={
                "tmax": tmax_,
                "tmin": tmin_,
                "precip": prec_,
                "eto": et0_
            }

    st.markdown("### 7) Dynamic Root Growth?")
    dynamic_root= st.checkbox("Enable dynamic root growth?", value=False)
    init_rd=300
    max_rd=800
    days_mx=60
        # Remainder of the code (continuation from your snippet):

    # user input for days to reach max root depth
    if dynamic_root:
        init_rd= st.number_input("Initial Root Depth (mm)", min_value=50, value=300)
        max_rd= st.number_input("Max Root Depth (mm)", min_value=50, value=800)
        days_mx= st.number_input("Days to reach max root depth?", min_value=1, value=60)

    st.markdown("### 8) Run Simulation")
    run_button= st.button("Run Simulation")

    if run_button:
        st.success("Simulation is now complete! Please open the 'Results' tab to see them.")

        # We'll store a flag in session to indicate sim run
        st.session_state["simulation_done"]= True
        
        # Prepare weather df
        if weather_file is not None:
            try:
                wdf= pd.read_csv(weather_file)
                if "Date" not in wdf.columns:
                    st.error("Weather file missing 'Date' column.")
                    st.stop()
                if pd.api.types.is_string_dtype(wdf["Date"]):
                    wdf["Date"]= pd.to_datetime(wdf["Date"])
                wdf= wdf.sort_values("Date").reset_index(drop=True)
            except:
                st.warning("Could not parse the weather file. Using forecast if enabled.")
                if enable_forecast:
                    startdt= datetime.now().date()
                    enddt= startdt+ timedelta(days=4)
                    wdf= fetch_weather_data(lat_, lon_, startdt, enddt, forecast=True, manual_data=None)
        else:
            if enable_forecast:
                startdt= datetime.now().date()
                enddt= startdt+ timedelta(days=4)
                wdf= fetch_weather_data(lat_, lon_, startdt, enddt, forecast=True, manual_data=manual_forecast_data)
            else:
                st.warning("No weather file & forecast disabled => no data. Stopping.")
                st.stop()
        
        if wdf is None or wdf.empty:
            st.error("No valid weather data. Stopping.")
            st.stop()

        # Crop stages
        if use_custom_stage and custom_crop_file is not None:
            try:
                stage_df= pd.read_csv(custom_crop_file)
            except:
                st.error("Could not parse uploaded stage file => using auto.")
                stage_df= create_auto_stages_for_crop(selected_crop)
        elif use_custom_stage and custom_crop_file is None:
            st.error("You checked custom stage but didn't upload a file. Using auto.")
            stage_df= create_auto_stages_for_crop(selected_crop)
        else:
            # auto
            stage_df= create_auto_stages_for_crop(selected_crop)
        
        # Soil
        if soil_file is not None:
            try:
                soil_df= pd.read_csv(soil_file)
            except:
                st.error("Could not read soil file => using default 2-layer.")
                soil_df= pd.DataFrame({
                    "Depth_mm":[200, 100],
                    "FC":[0.30, 0.30],
                    "WP":[0.15, 0.15],
                    "TEW":[20, 0],
                    "REW":[5, 0]
                })
        else:
            soil_df= pd.DataFrame({
                "Depth_mm":[200, 100],
                "FC":[0.30, 0.30],
                "WP":[0.15, 0.15],
                "TEW":[20, 0],
                "REW":[5, 0]
            })
        
        # Now run simulation
        res_df= simulate_SIMdualKc(
            weather_df= wdf,
            crop_stages_df= stage_df,
            soil_df= soil_df,
            track_drainage= track_drainage,
            enable_yield= enable_yield,
            Ym= ym_,
            Ky= ky_,
            use_transp= use_transp,
            WP_yield= wp_yield,
            enable_leaching= enable_leaching,
            nitrate_conc= nitrate_conc,
            total_N_input= totalN,
            leaching_fraction= lf,
            dynamic_root= dynamic_root,
            init_root= init_rd,
            max_root= max_rd,
            days_to_max= days_mx
        )
        st.session_state.results_df= res_df
        
        # If forecast, do next 5-day from final day
        if enable_forecast:
            # We only have 5 days, but if user wants a separate forecast, they'd need more data
            # We'll store it in st.session_state, or do a second run if they'd like.
            # For demonstration we won't do a second run. We'll just keep res_df
            pass
        
        # We also store the soil final state if needed
        st.session_state.soil_profile= soil_df

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
            results_df= st.session_state.results_df
            st.markdown("## Simulation Results")
            st.dataframe(results_df)
            st.download_button("Download Results (.csv)", 
                               results_df.to_csv(index=False),
                               file_name="results.csv",
                               mime="text/csv")
            
            st.markdown("## Charts")
            # We'll show multiple charts, each in a subheader
            # 1) ETa, 2) Dr, 3) Drainage, 4) Soil Water, etc.
            # user can scroll them without re-run
            # a) ETa
            st.markdown("### Daily ET Components")
            fig1, ax1= plt.subplots(figsize=(10,4))
            ax1.plot(results_df["Date"], results_df["ETa (mm)"], label="ETa total")
            ax1.plot(results_df["Date"], results_df["ETa_transp (mm)"], label="ETa transp")
            ax1.plot(results_df["Date"], results_df["ETa_evap (mm)"], label="ETa evap")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("mm")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
            
            # b) Dr
            st.markdown("### Root Zone Depletion (start & end)")
            fig2, ax2= plt.subplots(figsize=(10,4))
            ax2.plot(results_df["Date"], results_df["Dr_start (mm)"], label="Dr start of day")
            ax2.plot(results_df["Date"], results_df["Dr_end (mm)"], label="Dr end of day")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Depletion (mm)")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
            
            # c) Drainage
            st.markdown("### Daily Drainage")
            fig3, ax3= plt.subplots(figsize=(10,4))
            ax3.plot(results_df["Date"], results_df["Drainage (mm)"], label="Drainage")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("mm")
            ax3.legend()
            ax3.grid(True)
            st.pyplot(fig3)
            
            # d) Soil Water in Root zone
            st.markdown("### Soil Water in Root Zone (start, end)")
            fig4, ax4= plt.subplots(figsize=(10,4))
            ax4.plot(results_df["Date"], results_df["SW_root_start (mm)"], label="RootZ Start")
            ax4.plot(results_df["Date"], results_df["SW_root_end (mm)"], label="RootZ End")
            ax4.set_xlabel("Date")
            ax4.set_ylabel("mm water")
            ax4.legend()
            ax4.grid(True)
            st.pyplot(fig4)
            
            # e) If yield
            if "Yield (ton/ha)" in results_df.columns:
                st.markdown("### Daily Estimated Yield")
                figy, axy= plt.subplots(figsize=(10,4))
                axy.plot(results_df["Date"], results_df["Yield (ton/ha)"], label="Yield (ton/ha)")
                axy.set_xlabel("Date")
                axy.set_ylabel("ton/ha")
                axy.grid(True)
                axy.legend()
                st.pyplot(figy)
            
            # f) If leaching
            if "Leaching (kg/ha)" in results_df.columns:
                st.markdown("### Leaching (kg/ha)")
                figl, axl= plt.subplots(figsize=(10,4))
                axl.plot(results_df["Date"], results_df["Leaching (kg/ha)"], label="Leaching")
                axl.set_xlabel("Date")
                axl.set_ylabel("kg/ha")
                axl.grid(True)
                axl.legend()
                st.pyplot(figl)

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
            st.markdown("## Irrigation Calendar (Monthly Style)")
            # produce a next-likely irrigation schedule
            cal_df= produce_irrigation_calendar(st.session_state.results_df)
            if cal_df.empty:
                st.write("No calendar data found. Possibly no extended forecast or no depletion above RAW.")
            else:
                st.write("**Recommended Irrigation** occurs whenever Dr_end>RAW.")
                st.dataframe(cal_df)
                # We can also pivot it by month if we want a bigger style, but let's keep it simple.

st.markdown('<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>', unsafe_allow_html=True)

