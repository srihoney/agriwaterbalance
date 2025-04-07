import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, Point, Polygon
import datetime
import requests
import math
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.interpolate import griddata
import branca

# -------------------
# Configure Requests Session with Retries
# -------------------
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1.0,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

# -------------------
# App Configuration
# -------------------
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# -------------------
# Core Simulation Functions
# -------------------
def compute_Ks(depletion, TAW, p):
    RAW = p * TAW
    if depletion <= RAW:
        return 1.0
    elif depletion >= TAW:
        return 0.0
    else:
        return (TAW - depletion) / (TAW - RAW)

def compute_Kr(depletion, TEW, REW):
    if depletion <= REW:
        return 1.0
    elif depletion >= TEW:
        return 0.0
    else:
        return (TEW - depletion) / (TEW - REW)

def compute_ETc(Kcb, Ks, Kr, Ke, ET0):
    return (Kcb * Ks + Kr * Ke) * ET0

def interpolate_crop_stages(crop_df, total_days):
    kcb_list, RD_list, p_list, ke_list = [], [], [], []
    for i in range(len(crop_df)):
        row = crop_df.iloc[i]
        start, end = row['Start_Day'], row['End_Day']
        days = int(end - start + 1)
        for d in range(days):
            frac = d/(days-1) if days > 1 else 0
            if i > 0:
                prev = crop_df.iloc[i-1]
                kcb = prev['Kcb'] + frac*(row['Kcb']-prev['Kcb'])
                rd  = prev['Root_Depth_mm'] + frac*(row['Root_Depth_mm']-prev['Root_Depth_mm'])
                p   = prev['p'] + frac*(row['p']-prev['p'])
                ke  = prev['Ke'] + frac*(row['Ke']-prev['Ke'])
            else:
                kcb, rd, p, ke = row['Kcb'], row['Root_Depth_mm'], row['p'], row['Ke']
            kcb_list.append(kcb)
            RD_list.append(rd)
            p_list.append(p)
            ke_list.append(ke)
    return kcb_list[:total_days], RD_list[:total_days], p_list[:total_days], ke_list[:total_days]

def SIMdualKc(weather_df, crop_df, soil_df, track_drainage=True, enable_yield=False,
              use_fao33=False, Ym=0, Ky=0, use_transp=False, WP_yield=0,
              enable_leaching=False, leaching_method="", nitrate_conc=0,
              total_N_input=0, leaching_fraction=0,
              dynamic_root_growth=False, initial_root_depth=None, max_root_depth=None, days_to_max=None,
              return_soil_profile=False):
    days = len(weather_df)
    profile_depth = soil_df['Depth_mm'].sum()
    Kcb_list, RD_list, p_list, ke_list = interpolate_crop_stages(crop_df, days)
    if dynamic_root_growth and initial_root_depth is not None and max_root_depth is not None and days_to_max is not None:
        RD_list = [initial_root_depth + (max_root_depth - initial_root_depth)*min(1, (i+1)/days_to_max) for i in range(days)]
    TEW, REW = soil_df.iloc[0]['TEW'], soil_df.iloc[0]['REW']
    SW_surface = TEW
    SW_layers = [row['FC']*row['Depth_mm'] for _, row in soil_df.iterrows()]
    cum_ETc = cum_ETa = cum_Irr = cum_P = cum_drain = 0
    stress_days = 0
    cum_transp = 0
    cum_evap = 0
    results = []
    for i in range(days):
        row = weather_df.iloc[i]
        ET0, P, I = row['ET0'], row['Precipitation'], row['Irrigation']
        Kcb, RD, p, Ke = Kcb_list[i], min(RD_list[i], profile_depth), p_list[i], ke_list[i]
        cum_P += P
        cum_Irr += I
        FC_total = WP_total = SW_root = 0.0
        cum_depth = 0.0
        for j, soil in soil_df.iterrows():
            if cum_depth >= RD:
                break
            d = min(soil['Depth_mm'], RD-cum_depth)
            FC_total += soil['FC']*d
            WP_total += soil['WP']*d
            SW_root += (SW_layers[j]/soil['Depth_mm'])*d
            cum_depth += d
        TAW = FC_total - WP_total
        depletion = FC_total - SW_root
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW-SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks*Kcb*ET0
        ETa_evap = Kr*Ke*ET0
        cum_ETc += ETc
        cum_ETa += (ETa_transp+ETa_evap)
        cum_transp += ETa_transp
        cum_evap += ETa_evap
        if Ks < 1.0:
            stress_days += 1

        # Irrigation Scheduling Recommendation
        RAW = p*TAW
        recommended_irrigation = (depletion-RAW) if depletion>RAW else 0

        SW_surface += P+I
        excess_surface = max(0, SW_surface-TEW)
        SW_surface = max(0, SW_surface-ETa_evap)
        water = excess_surface
        for j, soil in soil_df.iterrows():
            max_SW = soil['FC']*soil['Depth_mm']
            SW_layers[j] += water
            drain = max(0, SW_layers[j]-max_SW)
            cum_drain += drain
            SW_layers[j] = min(SW_layers[j], max_SW)
            water = drain
            if cum_depth < RD:
                transp = ETa_transp*(soil['Depth_mm']/RD)
                SW_layers[j] -= transp
                SW_layers[j] = max(soil['WP']*soil['Depth_mm'], SW_layers[j])
        results.append({
            "Date": row['Date'],
            "ET0 (mm)": ET0,
            "Kcb": Kcb,
            "Ks": Ks,
            "Kr": Kr,
            "ETc (mm)": ETc,
            "ETa_transp (mm)": ETa_transp,
            "ETa_evap (mm)": ETa_evap,
            "ETa_total (mm)": ETa_transp+ETa_evap,
            "SW_surface (mm)": SW_surface,
            "SW_root (mm)": SW_root,
            "Root_Depth (mm)": RD,
            "Depletion (mm)": depletion,
            "TAW (mm)": TAW,
            "Recommended_Irrigation (mm)": recommended_irrigation,
            "Cumulative_ETc (mm)": cum_ETc,
            "Cumulative_ETa (mm)": cum_ETa,
            "Cumulative_Transp (mm)": cum_transp,
            "Cumulative_Evap (mm)": cum_evap,
            "Cumulative_Irrigation (mm)": cum_Irr,
            "Cumulative_Precip (mm)": cum_P,
            "Cumulative_Drainage (mm)": cum_drain,
            "Stress_Days": stress_days
        })
    results_df = pd.DataFrame(results)
    # Soil water stress and drainage alerts
    results_df['SWC (%)'] = (results_df['SW_root (mm)']/results_df['Root_Depth (mm)'])*100
    results_df['Stress_Risk'] = np.where((results_df['Ks']<0.5)|(results_df['SWC (%)']<50), "Alert", "Normal")
    results_df['Daily_Drainage'] = results_df['Cumulative_Drainage (mm)'].diff().fillna(results_df['Cumulative_Drainage (mm)'])
    # Yield estimation
    if enable_yield:
        if use_fao33:
            ETc_total = results_df['ETc (mm)'].sum()
            ETa_total = results_df['ETa_transp (mm)'].sum() + results_df['ETa_evap (mm)'].sum()
            Ya = Ym*(1-Ky*(1-ETa_total/ETc_total))
            results_df['Yield (ton/ha)'] = Ya
        if use_transp:
            Ya_transp = WP_yield * cum_transp
            results_df['Yield (ton/ha)'] = Ya_transp
    # Leaching estimation
    if enable_leaching:
        if leaching_method=="Method 1: Drainage × nitrate concentration":
            leaching = cum_drain * nitrate_conc / 1000
            results_df['Leaching (kg/ha)'] = leaching
        elif leaching_method=="Method 2: Leaching Fraction × total N input":
            leaching = leaching_fraction * total_N_input
            results_df['Leaching (kg/ha)'] = leaching
    # NUE estimation
    if enable_yield and total_N_input>0:
        results_df['NUE (kg/ha)'] = results_df['Yield (ton/ha)'] / total_N_input
    # Water productivity metrics
    if enable_yield:
        results_df['WP_ET'] = results_df['Yield (ton/ha)'] / results_df['ETa_total (mm)'].replace(0, np.nan)
        results_df['WP_Irrigation'] = results_df['Yield (ton/ha)'] / results_df['Cumulative_Irrigation (mm)'].replace(0, np.nan)
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

# -------------------
# Data Fetching Functions
# -------------------
def fetch_weather_data(lat, lon, start_date, end_date):
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOT,WS2M,RH2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        response = session.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params)
        response.raise_for_status()
        data = response.json()['properties']['parameter']
        precip_key = "PRECTOT" if "PRECTOT" in data else ("PRECTOTCORG" if "PRECTOTCORG" in data else None)
        dates, et0_list, precip_list = [], [], []
        for date_str in data['T2M_MAX']:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            dates.append(dt)
            Tmax = data['T2M_MAX'][date_str]
            Tmin = data['T2M_MIN'][date_str]
            Tmean = (Tmax+Tmin)/2
            Rs = data['ALLSKY_SFC_SW_DWN'][date_str]
            u2 = data['WS2M'][date_str]
            RH = data['RH2M'][date_str]
            delta = 4098*(0.6108*math.exp((17.27*Tmean)/(Tmean+237.3)))/(Tmean+237.3)**2
            P = 101.3
            gamma = 0.000665 * P
            es = (0.6108*math.exp(17.27*Tmax/(Tmax+237.3)) + 0.6108*math.exp(17.27*Tmin/(Tmin+237.3)))/2
            ea = es * RH/100
            ET0 = (0.408*delta*Rs+gamma*(900/(Tmean+273))*u2*(es-ea))/(delta+gamma*(1+0.34*u2))
            et0_list.append(ET0)
            precip_list.append(data[precip_key][date_str] if precip_key else 0)
        weather_df = pd.DataFrame({
            "Date": dates,
            "ET0": et0_list,
            "Precipitation": precip_list,
            "Irrigation": [0]*len(dates)
        })
        return weather_df
    except Exception as e:
        st.error(f"Weather data fetch failed: {e}")
        return None

def fetch_soil_data(lat, lon):
    max_retries = 5
    retry_delay = 2
    timeout = 10
    default_soil_df = pd.DataFrame({
        "Depth_mm": [200, 100],
        "FC": [0.30, 0.30],
        "WP": [0.15, 0.15],
        "TEW": [200, 0],
        "REW": [50, 0]
    })
    for attempt in range(max_retries):
        try:
            url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
            params = {
                'lon': lon,
                'lat': lat,
                'property': 'bdod,sand,clay,ocd',
                'depth': '0-5cm,5-15cm',
                'value': 'mean'
            }
            response = session.get(url, params=params, timeout=timeout,
                                   headers={'User-Agent': 'AgriWaterBalance/1.0'})
            response.raise_for_status()
            data = response.json()
            properties = data['properties']
            layers = []
            for depth in ['0-5cm', '5-15cm']:
                bdod = properties.get('bdod', {}).get(depth, {}).get('mean', 140)/100
                sand = properties.get('sand', {}).get(depth, {}).get('mean', 40)
                clay = properties.get('clay', {}).get(depth, {}).get('mean', 20)
                ocd = properties.get('ocd', {}).get(depth, {}).get('mean', 1.0)/100
                FC = max(0.1, min(0.4, (-0.251*sand/100+0.195*clay/100+0.011*ocd+0.006*(sand/100)*ocd-0.027*(clay/100)*ocd+0.452*(sand/100)*(clay/100)+0.299)*bdod))
                WP = max(0.01, min(0.2, (-0.024*sand/100+0.487*clay/100+0.006*ocd+0.005*(sand/100)*ocd-0.013*(clay/100)*ocd+0.068*(sand/100)*(clay/100)+0.031)*bdod))
                layers.append({
                    "Depth_mm": 50 if depth=='0-5cm' else 100,
                    "FC": FC,
                    "WP": WP,
                    "TEW": 200 if depth=='0-5cm' else 0,
                    "REW": 50 if depth=='0-5cm' else 0
                })
            return pd.DataFrame(layers)
        except Exception as e:
            if attempt < max_retries-1:
                time.sleep(retry_delay*(attempt+1))
                continue
            st.warning(f"Soil data fetch failed: {str(e)[:100]}... Using default values.")
            return default_soil_df

# -------------------
# User Interface – NORMAL MODE (Spatial mode removed)
# -------------------
st.title("AgriWaterBalance")
st.markdown("**Soil Water Balance Modeling with Enhanced Features**")

with st.sidebar:
    st.header("Upload Input Files (.txt)")
    weather_file = st.file_uploader("Weather Data (.txt)", type="txt")
    crop_input_method = st.selectbox("Crop Stage Input Method", ["Use Uploaded File", "Maize", "Wheat", "Soybean", "Rice", "Almond", "Tomato", "Custom"])
    if crop_input_method=="Use Uploaded File":
        crop_file = st.file_uploader("Crop Stage Data (.txt)", type="txt")
    else:
        crop_file = None
    soil_file = st.file_uploader("Soil Layers (.txt)", type="txt")
    
    st.header("Options")
    show_monthly_summary = st.checkbox("Show Monthly Summary", value=True)
    track_drainage = st.checkbox("Track Drainage", value=True)
    
    st.header("Yield Estimation")
    enable_yield = st.checkbox("Enable Yield Estimation", value=False)
    if enable_yield:
        st.subheader("Select Methods")
        use_fao33 = st.checkbox("Use FAO-33 Ky-based method", value=True)
        use_transp = st.checkbox("Use Transpiration-based method", value=False)
        if use_fao33:
            Ym = st.number_input("Maximum Yield (Ym, ton/ha)", min_value=0.0, value=10.0, step=0.1)
            Ky = st.number_input("Yield Response Factor (Ky)", min_value=0.0, value=1.0, step=0.1)
        if use_transp:
            WP_yield = st.number_input("Yield Water Productivity (WP_yield, ton/ha per mm)", min_value=0.0, value=0.01, step=0.001)
    else:
        use_fao33 = use_transp = Ym = Ky = WP_yield = 0

    st.header("Leaching Estimation")
    enable_leaching = st.checkbox("Enable Leaching Estimation", value=False)
    if enable_leaching:
        leaching_method = st.radio("Select Leaching Method", ["Method 1: Drainage × nitrate concentration", "Method 2: Leaching Fraction × total N input"])
        if leaching_method=="Method 1: Drainage × nitrate concentration":
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

    st.header("Additional Features")
    # List of additional features (line by line)
    enable_etaforecast = st.checkbox("Enable ETa Forecast (7-day forecast of actual ETa)", value=False)
    if enable_etaforecast:
        forecast_lat = st.number_input("Forecast Latitude", value=0.0)
        forecast_lon = st.number_input("Forecast Longitude", value=0.0)
    enable_nue = st.checkbox("Enable NUE Estimation", value=False)
    enable_dynamic_root = st.checkbox("Enable Dynamic Root Growth", value=False)
    if enable_dynamic_root:
        initial_root_depth = st.number_input("Initial Root Depth (mm)", min_value=50, value=300, step=10)
        max_root_depth = st.number_input("Maximum Root Depth (mm)", min_value=50, value=1000, step=10)
        days_to_max = st.number_input("Days to Reach Maximum Root Depth", min_value=1, value=60, step=1)
    else:
        initial_root_depth = max_root_depth = days_to_max = None
    show_soil_profile = st.checkbox("Show Soil Profile Water Storage Visualization", value=False)
    
    st.header("Graph Enhancements")
    # Graph tab will use a dropdown menu for plot selection.
    st.header("Comparison Run (Optional)")
    comp_file = st.file_uploader("Upload Comparison Run Results (.txt)", type="txt")
    
    run_button = st.button("🚀 Run Simulation")

# ----------- Generate Crop Data -----------
if crop_input_method=="Use Uploaded File":
    if crop_file:
        crop_df = pd.read_csv(crop_file)
    else:
        crop_df = None
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
    template = templates.get(crop_input_method, templates["Custom"])
    crop_df = pd.DataFrame({
        "Start_Day": [1],
        "End_Day": [100],
        "Kcb": [template["Kcb"]],
        "Root_Depth_mm": [template["Root_Depth_mm"]],
        "p": [template["p"]],
        "Ke": [template["Ke"]]
    })
    st.subheader("Edit Crop Stage Template")
    try:
        crop_df = st.experimental_data_editor(crop_df, num_rows="dynamic")
    except Exception as e:
        st.warning("Your Streamlit version does not support the experimental data editor. Using non-editable table.")
        st.write(crop_df)

# ----------- Run Simulation -----------
if run_button and weather_file and (crop_df is not None) and soil_file:
    with st.spinner("Running simulation..."):
        try:
            weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
            soil_df = pd.read_csv(soil_file)
            total_days = len(weather_df)
            crop_df.loc[crop_df.index[0], "End_Day"] = total_days

            # Run historical simulation
            results_df = SIMdualKc(weather_df, crop_df, soil_df, track_drainage, enable_yield,
                                   use_fao33, Ym, Ky, use_transp, WP_yield,
                                   enable_leaching, leaching_method, nitrate_conc,
                                   total_N_input, leaching_fraction,
                                   dynamic_root_growth=enable_dynamic_root,
                                   initial_root_depth=initial_root_depth,
                                   max_root_depth=max_root_depth,
                                   days_to_max=days_to_max,
                                   return_soil_profile=show_soil_profile)

            # If ETa Forecast is enabled, fetch forecast weather for next 7 days and simulate
            forecast_results = None
            if enable_etaforecast:
                last_date = weather_df['Date'].max()
                forecast_start = last_date + datetime.timedelta(days=1)
                forecast_end = forecast_start + datetime.timedelta(days=6)  # 7-day forecast
                forecast_weather = fetch_weather_data(forecast_lat, forecast_lon, forecast_start, forecast_end)
                if forecast_weather is not None:
                    forecast_results = SIMdualKc(forecast_weather, crop_df, soil_df, track_drainage, enable_yield,
                                                 use_fao33, Ym, Ky, use_transp, WP_yield,
                                                 enable_leaching, leaching_method, nitrate_conc,
                                                 total_N_input, leaching_fraction,
                                                 dynamic_root_growth=enable_dynamic_root,
                                                 initial_root_depth=initial_root_depth,
                                                 max_root_depth=max_root_depth,
                                                 days_to_max=days_to_max)
            # If a comparison run file is provided, read it.
            if comp_file:
                comp_df = pd.read_csv(comp_file, parse_dates=['Date'])
            
            st.success("Simulation completed successfully!")
            
            # Create three tabs: Daily Results, Graphs, Additional Features
            tab_daily, tab_graphs, tab_features = st.tabs(["📄 Daily Results", "📊 Graphs", "⚙️ Additional Features"])
            
            with tab_daily:
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Results (.txt)", csv, file_name="results.txt")
                # If irrigation recommendations exist
                irr_sched = results_df[results_df["Recommended_Irrigation (mm)"] > 0]
                if not irr_sched.empty:
                    st.subheader("Irrigation Scheduling Recommendations")
                    st.dataframe(irr_sched[["Date", "Recommended_Irrigation (mm)"]])
                    csv2 = irr_sched.to_csv(index=False)
                    st.download_button("📥 Download Irrigation Schedule", csv2, file_name="irrigation_schedule.txt")
                    
            with tab_graphs:
                plot_option = st.selectbox("Select Plot", 
                                             options=["ETa Components", "Cumulative Metrics", "Soil Water Storage", "Yield & Leaching"] + (["Comparison Graph"] if comp_file else []))
                if plot_option=="ETa Components":
                    fig, ax = plt.subplots()
                    ax.plot(results_df['Date'], results_df['ETa_transp (mm)'], label="Transpiration")
                    ax.plot(results_df['Date'], results_df['ETa_evap (mm)'], label="Evaporation")
                    ax.plot(results_df['Date'], results_df['ETc (mm)'], label="ETc")
                    ax.set_ylabel("ET (mm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                elif plot_option=="Cumulative Metrics":
                    fig, ax = plt.subplots()
                    ax.plot(results_df['Date'], results_df['Cumulative_Irrigation (mm)'], label="Cumulative Irrigation")
                    ax.plot(results_df['Date'], results_df['Cumulative_Precip (mm)'], label="Cumulative Precipitation")
                    ax.set_ylabel("Cumulative (mm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                elif plot_option=="Soil Water Storage":
                    fig, ax = plt.subplots()
                    ax.plot(results_df['Date'], results_df['SW_root (mm)'], label="Soil Water in Root Zone")
                    ax.set_ylabel("Soil Water (mm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                elif plot_option=="Yield & Leaching":
                    if enable_yield and ('Yield (ton/ha)' in results_df.columns):
                        fig, ax = plt.subplots()
                        ax.plot(results_df['Date'], results_df['Yield (ton/ha)'], label="Yield (ton/ha)")
                        ax.set_ylabel("Yield")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                    if enable_leaching and ('Leaching (kg/ha)' in results_df.columns):
                        fig2, ax2 = plt.subplots()
                        ax2.plot(results_df['Date'], results_df['Leaching (kg/ha)'], label="Leaching (kg/ha)", color="red")
                        ax2.set_ylabel("Leaching")
                        ax2.legend()
                        ax2.grid(True)
                        st.pyplot(fig2)
                elif plot_option=="Comparison Graph" and comp_file:
                    fig, ax = plt.subplots()
                    ax.plot(results_df['Date'], results_df['ETa_total (mm)'], label="Run1 ETa")
                    ax.plot(comp_df['Date'], comp_df['ETa_total (mm)'], label="Run2 ETa", linestyle="--")
                    ax.set_ylabel("ETa (mm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
            with tab_features:
                st.subheader("Additional Features Status")
                st.markdown("**1. Yield Estimation:** " + ("Enabled" if enable_yield else "Disabled"))
                st.markdown("**2. Track Drainage:** " + ("Enabled" if track_drainage else "Disabled"))
                st.markdown("**3. Leaching Estimation:** " + ("Enabled" if enable_leaching else "Disabled"))
                st.markdown("**4. ETa Forecast (7-day forecast of actual ETa):** " + ("Enabled" if enable_etaforecast else "Disabled"))
                st.markdown("**5. NUE Estimation:** " + ("Enabled" if enable_nue else "Disabled"))
                st.markdown("**6. Dynamic Root Growth:** " + ("Enabled" if enable_dynamic_root else "Disabled"))
                st.markdown("**7. Soil Profile Water Storage Visualization:** " + ("Enabled" if show_soil_profile else "Disabled"))
                # If ETa Forecast was enabled, show the forecast table
                if enable_etaforecast and forecast_results is not None:
                    st.subheader("7-Day ETa Forecast")
                    st.dataframe(forecast_results[["Date", "ETa_transp (mm)", "ETa_evap (mm)", "ETa_total (mm)"]])
        except Exception as e:
            st.error(f"⚠️ Simulation failed: {e}")
else:
    st.info("📂 Please upload all required files and click 'Run Simulation'.")
