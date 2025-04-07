import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta  # Updated import
import requests
import math
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure Requests Session with Retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
session.mount('https://', HTTPAdapter(max_retries=retries))

# App Configuration
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main-header {
        background-color: #1E3A8A;
        color: white;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
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
""", unsafe_allow_html=True)

# Cache for weather data
if 'weather_cache' not in st.session_state:
    st.session_state.weather_cache = {}

# Core Simulation Functions
def compute_Ks(SW, WP, FC, p):
    TAW = (FC - WP) * 1000  # mm/m
    RAW = p * TAW
    Dr = (FC - SW) * 1000
    if Dr <= RAW:
        Ks = 1.0
    else:
        Ks = (TAW - Dr) / ((1 - p) * TAW)
        Ks = max(0.0, min(1.0, Ks))
    return Ks

def compute_Kr(TEW, REW, E):
    if E <= REW:
        Kr = 1.0
    else:
        Kr = (TEW - E) / (TEW - REW)
        Kr = max(0.0, min(1.0, Kr))
    return Kr

def compute_ETc(Kcb, Ks, Ke, ET0):
    ETc = (Kcb * Ks + Ke) * ET0
    return ETc

def interpolate_crop_stages(crop_df, total_days):
    Kcb = np.zeros(total_days)
    root_depth = np.zeros(total_days)
    p = np.zeros(total_days)
    Ke = np.zeros(total_days)
    
    for i in range(len(crop_df) - 1):
        start_day = int(crop_df.iloc[i]['Start_Day'])
        end_day = min(int(crop_df.iloc[i]['End_Day']), total_days)
        
        Kcb_start = crop_df.iloc[i]['Kcb']
        Kcb_end = crop_df.iloc[i + 1]['Kcb']
        root_start = crop_df.iloc[i]['Root_Depth_mm']
        root_end = crop_df.iloc[i + 1]['Root_Depth_mm']
        p_start = crop_df.iloc[i]['p']
        p_end = crop_df.iloc[i + 1]['p']
        Ke_start = crop_df.iloc[i]['Ke']
        Ke_end = crop_df.iloc[i + 1]['Ke']
        
        if i == 0 and start_day > 1:
            Kcb[0:start_day-1] = 0
            root_depth[0:start_day-1] = root_start
            p[0:start_day-1] = p_start
            Ke[0:start_day-1] = Ke_start
        
        idx = np.arange(start_day - 1, end_day)
        if len(idx) > 0:
            Kcb[idx] = np.linspace(Kcb_start, Kcb_end, len(idx))
            root_depth[idx] = np.linspace(root_start, root_end, len(idx))
            p[idx] = np.linspace(p_start, p_end, len(idx))
            Ke[idx] = np.linspace(Ke_start, Ke_end, len(idx))
    
    last_start_day = int(crop_df.iloc[-1]['Start_Day'])
    last_end_day = min(int(crop_df.iloc[-1]['End_Day']), total_days)
    last_Kcb = crop_df.iloc[-1]['Kcb']
    last_root = crop_df.iloc[-1]['Root_Depth_mm']
    last_p = crop_df.iloc[-1]['p']
    last_Ke = crop_df.iloc[-1]['Ke']
    
    if last_start_day <= total_days:
        idx_last = np.arange(last_start_day - 1, last_end_day)
        if len(idx_last) > 0:
            Kcb[idx_last] = last_Kcb
            root_depth[idx_last] = last_root
            p[idx_last] = last_p
            Ke[idx_last] = last_Ke
    
    if last_end_day < total_days:
        Kcb[last_end_day:] = last_Kcb
        root_depth[last_end_day:] = last_root
        p[last_end_day:] = last_p
        Ke[last_end_day:] = last_Ke
    
    return Kcb, root_depth, p, Ke

def SIMdualKc(weather_df, crop_df, soil_df, track_drainage=True, enable_yield=False,
              use_fao33=False, Ym=0, Ky=0, use_transp=False, WP_yield=0,
              enable_leaching=False, leaching_method="", nitrate_conc=0,
              total_N_input=0, leaching_fraction=0,
              enable_dynamic_root=False, initial_root_depth=None, max_root_depth=None, days_to_max=None,
              return_soil_profile=False):
    total_days = len(weather_df)
    results = []
    SW_layers = [soil['FC'] * soil['Depth_mm'] for _, soil in soil_df.iterrows()]
    E = soil_df['REW'].sum()
    cumulative_irrigation = 0
    cumulative_precip = 0
    
    Kcb_daily, root_depth_daily, p_daily, Ke_daily = interpolate_crop_stages(crop_df, total_days)
    
    if enable_dynamic_root and initial_root_depth is not None and max_root_depth is not None and days_to_max is not None:
        root_depth_daily = np.linspace(initial_root_depth, max_root_depth, min(days_to_max, total_days))
        if total_days > days_to_max:
            root_depth_daily = np.concatenate([root_depth_daily, [max_root_depth] * (total_days - days_to_max)])
    
    for day in range(total_days):
        date = weather_df.iloc[day]['Date']
        ET0 = max(0, weather_df.iloc[day]['ET0'])  # Ensure ETo is non-negative
        precip = max(0, weather_df.iloc[day]['Precipitation'])  # Ensure precipitation is non-negative
        irrig = max(0, weather_df.iloc[day]['Irrigation'])  # Ensure irrigation is non-negative
        
        cumulative_irrigation += irrig
        cumulative_precip += precip
        
        Kcb = max(0, Kcb_daily[day])
        p = max(0, min(1, p_daily[day]))
        Ke = max(0, Ke_daily[day])
        root_depth = max(1, root_depth_daily[day])
        
        total_depth = 0
        SW_root = 0
        TAW_root = 0
        RAW_root = 0
        for j, soil in soil_df.iterrows():
            layer_depth = soil['Depth_mm']
            total_depth += layer_depth
            if total_depth <= root_depth:
                SW_root += SW_layers[j]
                TAW_root += (soil['FC'] - soil['WP']) * soil['Depth_mm']
                RAW_root += p * (soil['FC'] - soil['WP']) * soil['Depth_mm']
            elif total_depth > root_depth and (total_depth - layer_depth) < root_depth:
                fraction = (root_depth - (total_depth - layer_depth)) / layer_depth
                SW_root += SW_layers[j] * fraction
                TAW_root += (soil['FC'] - soil['WP']) * soil['Depth_mm'] * fraction
                RAW_root += p * (soil['FC'] - soil['WP']) * soil['Depth_mm'] * fraction
        
        Ks = compute_Ks(SW_root / root_depth if root_depth > 0 else 0, soil_df['WP'].mean(), soil_df['FC'].mean(), p)
        Kr = compute_Kr(soil_df['TEW'].sum(), soil_df['REW'].sum(), E)
        ETc = compute_ETc(Kcb, Ks, Ke, ET0)
        ETa_transp = max(0, Kcb * Ks * ET0)
        ETa_evap = max(0, Ke * Kr * ET0)
        ETa_total = ETa_transp + ETa_evap
        
        water_input = precip + irrig
        drainage = 0
        if track_drainage:
            excess = water_input - ETa_total
            for j, soil in soil_df.iterrows():
                capacity = (soil['FC'] * soil['Depth_mm']) - SW_layers[j]
                if excess > 0:
                    added = min(excess, capacity)
                    SW_layers[j] += added
                    excess -= added
            drainage = max(0, excess)
            for j in range(len(SW_layers)):
                SW_layers[j] = min(SW_layers[j], soil_df.iloc[j]['FC'] * soil_df.iloc[j]['Depth_mm'])
                SW_layers[j] = max(SW_layers[j], soil_df.iloc[j]['WP'] * soil_df.iloc[j]['Depth_mm'])
        
        for j in range(len(SW_layers)):
            depletion = ETa_total * (SW_layers[j] / SW_root) if SW_root > 0 else 0
            SW_layers[j] = max(SW_layers[j] - depletion, soil_df.iloc[j]['WP'] * soil_df.iloc[j]['Depth_mm'])
        
        E += ETa_evap - (precip + irrig)
        E = max(0, min(E, soil_df['TEW'].sum()))
        
        daily_result = {
            "Date": date,
            "ETc (mm)": ETc,
            "ETa_transp (mm)": ETa_transp,
            "ETa_evap (mm)": ETa_evap,
            "ETa_total (mm)": ETa_total,
            "SW_root (mm)": SW_root,
            "Cumulative_Irrigation (mm)": cumulative_irrigation,
            "Cumulative_Precip (mm)": cumulative_precip,
            "Root_Depth (mm)": root_depth,
            "Daily_Drainage": drainage if track_drainage else 0
        }
        
        if enable_yield:
            if use_fao33 and Ym > 0 and Ky > 0:
                Ya = Ym * (1 - Ky * (1 - (ETa_total / ETc))) if ETc > 0 else 0
                daily_result["Yield (ton/ha)"] = Ya
            if use_transp and WP_yield > 0:
                Ya_transp = WP_yield * ETa_transp
                daily_result["Yield (ton/ha)"] = Ya_transp
        
        if enable_leaching:
            if leaching_method == "Method 1: Drainage × nitrate concentration" and drainage > 0:
                leaching = drainage * nitrate_conc * 0.001
                daily_result["Leaching (kg/ha)"] = leaching
            elif leaching_method == "Method 2: Leaching Fraction × total N input":
                leaching = leaching_fraction * total_N_input / total_days
                daily_result["Leaching (kg/ha)"] = leaching
        
        if enable_yield and total_N_input > 0 and "Yield (ton/ha)" in daily_result:
            daily_result["NUE (kg/ha)"] = daily_result["Yield (ton/ha)"] / total_N_input
        
        results.append(daily_result)
    
    results_df = pd.DataFrame(results)
    
    if return_soil_profile:
        final_soil_profile = [{"Layer": j, "Depth_mm": soil['Depth_mm'], "SW (mm)": SW_layers[j]} for j, soil in soil_df.iterrows()]
        return results_df, final_soil_profile
    return results_df

# Data Fetching Function with OpenWeatherMap
def fetch_weather_data(lat, lon, start_date, end_date, forecast=False):
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}_{forecast}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]

    if forecast:
        # OpenWeatherMap API for forecast
        api_key = "fe2d869569674a4afbfca57707bdf691"  # Your API key
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"  # Use api_key variable
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Aggregate 3-hourly data into daily data
            daily_data = {}
            for entry in data['list']:
                dt = datetime.fromtimestamp(entry['dt'])
                if start_date <= dt <= end_date:
                    date_str = dt.strftime("%Y-%m-%d")
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
                # Ensure tmax >= tmin to avoid negative square root
                if tmax < tmin:
                    tmax, tmin = tmin, tmax
                # Hargreaves method for ETo
                Ra = 10  # Simplified; ideally calculate based on latitude and day of year
                Tmean = (tmax + tmin) / 2
                ETo = 0.0023 * Ra * (Tmean + 17.8) * (tmax - tmin) ** 0.5
                ETo = max(0, ETo)  # Ensure ETo is non-negative
                ETo_list.append(ETo)
                precip_list.append(values['precip'])
            
            weather_df = pd.DataFrame({
                "Date": dates,
                "ET0": ETo_list,
                "Precipitation": precip_list,
                "Irrigation": [0] * len(dates)
            })
            
            # Sort by date
            weather_df = weather_df.sort_values("Date").reset_index(drop=True)
            
            # Extrapolate to 7 days by averaging the last 3 days
            if len(weather_df) < 7:
                last_days = weather_df.tail(3)
                avg_data = last_days.mean(numeric_only=True)
                extra_days = 7 - len(weather_df)
                extra_dates = pd.date_range(start=weather_df['Date'].max() + timedelta(days=1), periods=extra_days)
                extra_data = pd.DataFrame([avg_data] * extra_days)
                extra_data['Date'] = extra_dates
                weather_df = pd.concat([weather_df, extra_data], ignore_index=True)
            
            st.session_state.weather_cache[cache_key] = weather_df
            return weather_df
        except Exception as e:
            st.error(f"Failed to fetch forecast data: {e}")
            return None
    else:
        # Historical data via NASA POWER API
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_str}&end={end_str}&format=JSON"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            dates = pd.date_range(start_date, end_date)
            ET0 = [data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(d.strftime("%Y%m%d"), 0) * 0.2 for d in dates]
            precip = [data['properties']['parameter']['PRECTOTCORR'].get(d.strftime("%Y%m%d"), 0) for d in dates]
            
            weather_df = pd.DataFrame({
                "Date": dates,
                "ET0": ET0,
                "Precipitation": precip,
                "Irrigation": [0] * len(dates)
            })
            st.session_state.weather_cache[cache_key] = weather_df
            return weather_df
        except Exception as e:
            st.warning(f"Failed to fetch historical weather data: {e}")
            return None

# User Interface
st.markdown('<div class="main-header">AgriWaterBalance</div>', unsafe_allow_html=True)
st.markdown("**A Professional Tool for Soil Water Management**", unsafe_allow_html=True)

# Navigation Tabs
setup_tab, results_tab = st.tabs(["Setup Simulation", "View Results"])

if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'soil_profile' not in st.session_state:
    st.session_state.soil_profile = None

with setup_tab:
    st.markdown('<div class="sub-header">Input Data Configuration</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### Upload Input Files")
        
        with st.expander("Weather Data"):
            st.write("Upload a text file with columns: Date, ET0, Precipitation, Irrigation")
            weather_file = st.file_uploader("Weather Data File (.txt)", type="txt", key="weather")
            sample_weather = pd.DataFrame({
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "ET0": [5.0, 5.2, 4.8],
                "Precipitation": [0.0, 2.0, 0.0],
                "Irrigation": [0.0, 0.0, 10.0]
            })
            st.download_button("Download Sample Weather Data", sample_weather.to_csv(index=False), file_name="weather_sample.txt", mime="text/plain")
        
        with st.expander("Crop Stage Data"):
            crop_input_method = st.selectbox("Select Crop Input Method", ["Upload My Own", "Maize", "Wheat", "Soybean", "Rice", "Almond", "Tomato", "Custom"])
            if crop_input_method == "Upload My Own":
                crop_file = st.file_uploader("Crop Stage Data File (.txt)", type="txt", key="crop")
                sample_crop = pd.DataFrame({
                    "Start_Day": [1, 31],
                    "End_Day": [30, 60],
                    "Kcb": [0.3, 1.2],
                    "Root_Depth_mm": [300, 1000],
                    "p": [0.5, 0.55],
                    "Ke": [0.7, 0.8]
                })
                st.download_button("Download Sample Crop Data", sample_crop.to_csv(index=False), file_name="crop_sample.txt", mime="text/plain")
            else:
                templates = {
                    "Maize": {"Kcb": 1.2, "Root_Depth_mm": 1500, "p": 0.55, "Ke": 0.7},
                    "Wheat": {"Kcb": 1.1, "Root_Depth_mm": 1200, "p": 0.5, "Ke": 0.6},
                    "Soybean": {"Kcb": 1.0, "Root_Depth_mm": 1000, "p": 0.5, "Ke": 0.7},
                    "Rice": {"Kcb": 1.3, "Root_Depth_mm": 800, "p": 0.2, "Ke": 1.0},
                    "Almond": {"Kcb": 0.9, "Root_Depth_mm": 1000, "p": 0.45, "Ke": 0.8},
                    "Tomato": {"Kcb": 1.0, "Root_Depth_mm": 900, "p": 0.5, "Ke": 0.75},
                    "Custom": {"Kcb": 1.0, "Root_Depth_mm": 1000, "p": 0.5, "Ke": 1.0}
                }
                template = templates[crop_input_method]
                crop_df = pd.DataFrame({
                    "Start_Day": [1],
                    "End_Day": [100],
                    "Kcb": [template["Kcb"]],
                    "Root_Depth_mm": [template["Root_Depth_mm"]],
                    "p": [template["p"]],
                    "Ke": [template["Ke"]]
                })
                st.write("Edit Crop Stage Data (Optional)")
                crop_df = st.data_editor(crop_df, num_rows="dynamic")
        
        with st.expander("Soil Layers Data"):
            st.write("Upload a text file with columns: Depth_mm, FC, WP, TEW, REW")
            soil_file = st.file_uploader("Soil Layers File (.txt)", type="txt", key="soil")
            sample_soil = pd.DataFrame({
                "Depth_mm": [200, 100],
                "FC": [0.30, 0.30],
                "WP": [0.15, 0.15],
                "TEW": [20, 0],
                "REW": [5, 0]
            })
            st.download_button("Download Sample Soil Data", sample_soil.to_csv(index=False), file_name="soil_sample.txt", mime="text/plain")
    
    with st.container():
        st.markdown("#### Additional Features")
        
        with st.expander("Simulation Options"):
            track_drainage = st.checkbox("Track Drainage", value=True, help="Monitor water drainage from the soil.")
            enable_yield = st.checkbox("Enable Yield Estimation", value=False, help="Estimate crop yield based on water use.")
            if enable_yield:
                st.markdown("**Yield Estimation Options**")
                use_fao33 = st.checkbox("Use FAO-33 Ky-based method", value=True, help="Standard method for yield estimation.")
                if use_fao33:
                    Ym = st.number_input("Maximum Yield (Ym, ton/ha)", min_value=0.0, value=10.0, step=0.1)
                    Ky = st.number_input("Yield Response Factor (Ky)", min_value=0.0, value=1.0, step=0.1)
                use_transp = st.checkbox("Use Transpiration-based method", value=False, help="Estimate yield based on transpiration.")
                if use_transp:
                    WP_yield = st.number_input("Yield Water Productivity (WP_yield, ton/ha per mm)", min_value=0.0, value=0.01, step=0.001)
            else:
                use_fao33 = use_transp = False
                Ym = Ky = WP_yield = 0
            
            enable_leaching = st.checkbox("Enable Leaching Estimation", value=False, help="Estimate nitrate leaching from the soil.")
            if enable_leaching:
                leaching_method = st.radio("Select Leaching Method", ["Method 1: Drainage × nitrate concentration", "Method 2: Leaching Fraction × total N input"])
                if leaching_method == "Method 1: Drainage × nitrate concentration":
                    nitrate_conc = st.number_input("Nitrate Concentration (mg/L)", min_value=0.0, value=10.0, step=0.1)
                    total_N_input = leaching_fraction = 0
                else:
                    total_N_input = st.number_input("Total N Input (kg/ha)", min_value=0.0, value=100.0, step=1.0)
                    leaching_fraction = st.number_input("Leaching Fraction (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                    nitrate_conc = 0
            else:
                leaching_method = ""
                nitrate_conc = total_N_input = leaching_fraction = 0
            
            enable_etaforecast = st.checkbox("Enable 7-Day ETa Forecast", value=False, help="Generate a 7-day forecast of actual evapotranspiration.")
            if enable_etaforecast:
                st.write("Enter your field's coordinates for weather forecasting. Find these using Google Maps by right-clicking on your field's location.")
                forecast_lat = st.number_input("Field Latitude", value=0.0)
                forecast_lon = st.number_input("Field Longitude", value=0.0)
            else:
                forecast_lat = forecast_lon = 0.0
            
            enable_nue = st.checkbox("Enable NUE Estimation", value=False, help="Estimate Nitrogen Use Efficiency (requires yield and nitrogen input).")
            enable_dynamic_root = st.checkbox("Enable Dynamic Root Growth", value=False, help="Simulate root growth over time.")
            if enable_dynamic_root:
                initial_root_depth = st.number_input("Initial Root Depth (mm)", min_value=50, value=300, step=10)
                max_root_depth = st.number_input("Maximum Root Depth (mm)", min_value=50, value=1000, step=10)
                days_to_max = st.number_input("Days to Reach Maximum Root Depth", min_value=1, value=60, step=1)
            else:
                initial_root_depth = max_root_depth = days_to_max = None
            
            show_soil_profile = st.checkbox("Show Soil Profile Water Storage", value=False, help="Visualize water storage in soil layers.")
    
    st.button("Run Simulation", key="run_simulation")

    if st.session_state.get('run_simulation', False):
        if weather_file and (crop_file or crop_input_method != "Upload My Own") and soil_file:
            try:
                weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
                if crop_input_method == "Upload My Own":
                    crop_df = pd.read_csv(crop_file)
                soil_df = pd.read_csv(soil_file)
                total_days = len(weather_df)
                crop_df.loc[crop_df.index[-1], "End_Day"] = total_days
                
                if show_soil_profile:
                    results_df, soil_profile = SIMdualKc(weather_df, crop_df, soil_df, track_drainage, enable_yield,
                                                        use_fao33, Ym, Ky, use_transp, WP_yield,
                                                        enable_leaching, leaching_method, nitrate_conc,
                                                        total_N_input, leaching_fraction,
                                                        enable_dynamic_root, initial_root_depth, max_root_depth, days_to_max,
                                                        return_soil_profile=True)
                    st.session_state.soil_profile = soil_profile
                else:
                    results_df = SIMdualKc(weather_df, crop_df, soil_df, track_drainage, enable_yield,
                                           use_fao33, Ym, Ky, use_transp, WP_yield,
                                           enable_leaching, leaching_method, nitrate_conc,
                                           total_N_input, leaching_fraction,
                                           enable_dynamic_root, initial_root_depth, max_root_depth, days_to_max,
                                           return_soil_profile=False)
                    st.session_state.soil_profile = None
                
                st.session_state.results_df = results_df
                
                if enable_etaforecast:
                    last_date = weather_df['Date'].max()
                    forecast_start = last_date + timedelta(days=1)  # Use timedelta directly
                    forecast_end = forecast_start + timedelta(days=6)
                    forecast_weather = fetch_weather_data(forecast_lat, forecast_lon, forecast_start, forecast_end, forecast=True)
                    if forecast_weather is not None:
                        forecast_results = SIMdualKc(forecast_weather, crop_df, soil_df, track_drainage, enable_yield,
                                                     use_fao33, Ym, Ky, use_transp, WP_yield,
                                                     enable_leaching, leaching_method, nitrate_conc,
                                                     total_N_input, leaching_fraction,
                                                     enable_dynamic_root, initial_root_depth, max_root_depth, days_to_max)
                        st.session_state.forecast_results = forecast_results
                    else:
                        st.session_state.forecast_results = None
                        st.warning("Unable to fetch forecast data. Please try again later.")
                else:
                    st.session_state.forecast_results = None
                st.success("Simulation completed successfully!")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
        else:
            st.error("Please upload all required files.")

with results_tab:
    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        forecast_results = st.session_state.forecast_results
        soil_profile = st.session_state.soil_profile
        
        st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)
        with st.container():
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False)
            st.download_button("Download Results (.txt)", csv, file_name="results.txt", mime="text/plain")
        
        if enable_etaforecast and forecast_results is not None:
            st.markdown('<div class="sub-header">7-Day ETa Forecast</div>', unsafe_allow_html=True)
            with st.container():
                st.dataframe(forecast_results[["Date", "ETa_total (mm)"]])
                st.write("Note: Forecast is based on OpenWeatherMap data. Values are ensured to be non-negative.")
        
        st.markdown('<div class="sub-header">Graphs</div>', unsafe_allow_html=True)
        with st.container():
            plot_options = ["ETa Components", "Cumulative Metrics", "Soil Water Storage", "Drainage", "Root Depth"]
            if 'Yield (ton/ha)' in results_df.columns:
                plot_options.append("Yield")
            if 'Leaching (kg/ha)' in results_df.columns:
                plot_options.append("Leaching")
            if 'NUE (kg/ha)' in results_df.columns:
                plot_options.append("NUE")
            if show_soil_profile and soil_profile is not None:
                plot_options.append("Soil Profile Water")
            
            plot_option = st.selectbox("Select Graph to Display", plot_options)
            
            if plot_option == "ETa Components":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['ETa_transp (mm)'], label="Transpiration")
                ax.plot(results_df['Date'], results_df['ETa_evap (mm)'], label="Evaporation")
                ax.plot(results_df['Date'], results_df['ETc (mm)'], label="ETc")
                ax.set_xlabel("Date")
                ax.set_ylabel("ET (mm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Cumulative Metrics":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['Cumulative_Irrigation (mm)'], label="Cumulative Irrigation")
                ax.plot(results_df['Date'], results_df['Cumulative_Precip (mm)'], label="Cumulative Precipitation")
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative (mm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Soil Water Storage":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['SW_root (mm)'], label="Soil Water in Root Zone")
                ax.set_xlabel("Date")
                ax.set_ylabel("Soil Water (mm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Drainage":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['Daily_Drainage'], label="Daily Drainage")
                ax.set_xlabel("Date")
                ax.set_ylabel("Drainage (mm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Root Depth":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['Root_Depth (mm)'], label="Root Depth")
                ax.set_xlabel("Date")
                ax.set_ylabel("Depth (mm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Yield" and 'Yield (ton/ha)' in results_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['Yield (ton/ha)'], label="Yield")
                ax.set_xlabel("Date")
                ax.set_ylabel("Yield (ton/ha)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Leaching" and 'Leaching (kg/ha)' in results_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['Leaching (kg/ha)'], label="Leaching")
                ax.set_xlabel("Date")
                ax.set_ylabel("Leaching (kg/ha)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "NUE" and 'NUE (kg/ha)' in results_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Date'], results_df['NUE (kg/ha)'], label="NUE")
                ax.set_xlabel("Date")
                ax.set_ylabel("NUE (kg/ha)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            elif plot_option == "Soil Profile Water" and show_soil_profile and soil_profile is not None:
                st.markdown('<div class="sub-header">Soil Profile Water Storage</div>', unsafe_allow_html=True)
                profile_df = pd.DataFrame(soil_profile)
                st.dataframe(profile_df)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(profile_df['Layer'], profile_df['SW (mm)'], color='skyblue')
                ax.set_xlabel("Soil Layer")
                ax.set_ylabel("Soil Water (mm)")
                ax.set_title("Water Storage per Soil Layer")
                st.pyplot(fig)
    else:
        st.info("Please complete the setup and click 'Run Simulation' to view results.")

# Footer
st.markdown('<div class="footer">© 2025 AgriWaterBalance | Contact: support@agriwaterbalance.com</div>', unsafe_allow_html=True)
