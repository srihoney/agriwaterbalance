import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
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

# Core Simulation Functions
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
            frac = d / (days - 1) if days > 1 else 0
            if i > 0:
                prev = crop_df.iloc[i - 1]
                kcb = prev['Kcb'] + frac * (row['Kcb'] - prev['Kcb'])
                rd = prev['Root_Depth_mm'] + frac * (row['Root_Depth_mm'] - prev['Root_Depth_mm'])
                p = prev['p'] + frac * (row['p'] - prev['p'])
                ke = prev['Ke'] + frac * (row['Ke'] - prev['Ke'])
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
        RD_list = [initial_root_depth + (max_root_depth - initial_root_depth) * min(1, (i + 1) / days_to_max) for i in range(days)]
    TEW, REW = soil_df.iloc[0]['TEW'], soil_df.iloc[0]['REW']
    SW_surface = TEW
    SW_layers = [row['FC'] * row['Depth_mm'] for _, row in soil_df.iterrows()]
    cum_ETc = cum_ETa = cum_Irr = cum_P = cum_drain = 0
    stress_days = 0
    cum_transp = cum_evap = 0
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
            d = min(soil['Depth_mm'], RD - cum_depth)
            FC_total += soil['FC'] * d
            WP_total += soil['WP'] * d
            SW_root += (SW_layers[j] / soil['Depth_mm']) * d
            cum_depth += d
        TAW = FC_total - WP_total
        depletion = FC_total - SW_root
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0
        cum_ETc += ETc
        cum_ETa += (ETa_transp + ETa_evap)
        cum_transp += ETa_transp
        cum_evap += ETa_evap
        if Ks < 1.0:
            stress_days += 1
        SW_surface += P + I
        excess_surface = max(0, SW_surface - TEW)
        SW_surface = max(0, SW_surface - ETa_evap)
        water = excess_surface
        for j, soil in soil_df.iterrows():
            max_SW = soil['FC'] * soil['Depth_mm']
            SW_layers[j] += water
            drain = max(0, SW_layers[j] - max_SW)
            cum_drain += drain
            SW_layers[j] = min(SW_layers[j], max_SW)
            water = drain
            if cum_depth < RD:
                transp = ETa_transp * (soil['Depth_mm'] / RD)
                SW_layers[j] -= transp
                SW_layers[j] = max(soil['WP'] * soil['Depth_mm'], SW_layers[j])
        results.append({
            "Date": row['Date'],
            "ET0 (mm)": ET0,
            "Kcb": Kcb,
            "Ks": Ks,
            "Kr": Kr,
            "ETc (mm)": ETc,
            "ETa_transp (mm)": ETa_transp,
            "ETa_evap (mm)": ETa_evap,
            "ETa_total (mm)": ETa_transp + ETa_evap,
            "SW_surface (mm)": SW_surface,
            "SW_root (mm)": SW_root,
            "Root_Depth (mm)": RD,
            "Depletion (mm)": depletion,
            "TAW (mm)": TAW,
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
    results_df['Daily_Drainage'] = results_df['Cumulative_Drainage (mm)'].diff().fillna(results_df['Cumulative_Drainage (mm)'])
    if enable_yield:
        if use_fao33:
            ETc_total = results_df['ETc (mm)'].sum()
            ETa_total = results_df['ETa_transp (mm)'].sum() + results_df['ETa_evap (mm)'].sum()
            Ya = Ym * (1 - Ky * (1 - ETa_total / ETc_total))
            results_df['Yield (ton/ha)'] = Ya
        if use_transp:
            Ya_transp = WP_yield * cum_transp
            results_df['Yield (ton/ha)'] = Ya_transp
    if enable_leaching:
        if leaching_method == "Method 1: Drainage × nitrate concentration":
            leaching = cum_drain * nitrate_conc / 1000
            results_df['Leaching (kg/ha)'] = leaching
        elif leaching_method == "Method 2: Leaching Fraction × total N input":
            leaching = leaching_fraction * total_N_input
            results_df['Leaching (kg/ha)'] = leaching
    if enable_yield and total_N_input > 0:
        results_df['NUE (kg/ha)'] = results_df['Yield (ton/ha)'] / total_N_input
    if return_soil_profile:
        final_soil_profile = [{"Layer": j, "Depth_mm": soil['Depth_mm'], "SW (mm)": SW_layers[j]} for j, soil in soil_df.iterrows()]
        return results_df, final_soil_profile
    return results_df

# Data Fetching Functions
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
            Tmean = (Tmax + Tmin) / 2
            Rs = data['ALLSKY_SFC_SW_DWN'][date_str]
            u2 = data['WS2M'][date_str]
            RH = data['RH2M'][date_str]
            delta = 4098 * (0.6108 * math.exp((17.27 * Tmean) / (Tmean + 237.3))) / (Tmean + 237.3) ** 2
            P = 101.3
            gamma = 0.000665 * P
            es = (0.6108 * math.exp(17.27 * Tmax / (Tmax + 237.3)) + 0.6108 * math.exp(17.27 * Tmin / (Tmin + 237.3))) / 2
            ea = es * RH / 100
            ET0 = (0.408 * delta * Rs + gamma * (900 / (Tmean + 273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
            et0_list.append(ET0)
            precip_list.append(data[precip_key][date_str] if precip_key else 0)
        return pd.DataFrame({"Date": dates, "ET0": et0_list, "Precipitation": precip_list, "Irrigation": [0] * len(dates)})
    except Exception as e:
        st.error(f"Weather data fetch failed: {e}")
        return None

# User Interface
st.title("AgriWaterBalance")
st.markdown("**A Simple Tool for Soil Water Management**")

tab1, tab2, tab3 = st.tabs(["📂 Upload Files", "⚙️ Additional Features", "📊 Graphs"])

with tab1:
    st.header("Upload Your Files")
    st.write("Upload the files below to start the simulation.")
    weather_file = st.file_uploader("Weather Data (.txt)", type="txt", help="Upload a text file with Date, ET0, Precipitation, and Irrigation columns.")
    crop_input_method = st.selectbox("Crop Type", ["Upload My Own", "Maize", "Wheat", "Soybean", "Rice", "Almond", "Tomato", "Custom"], help="Choose a crop or upload your own data.")
    if crop_input_method == "Upload My Own":
        crop_file = st.file_uploader("Crop Stage Data (.txt)", type="txt", help="Upload a text file with Start_Day, End_Day, Kcb, Root_Depth_mm, p, and Ke columns.")
    else:
        crop_file = None
    soil_file = st.file_uploader("Soil Layers (.txt)", type="txt", help="Upload a text file with Depth_mm, FC, WP, TEW, and REW columns.")
    run_button = st.button("🚀 Run Simulation", help="Click to start the simulation after uploading files.")

with tab2:
    st.header("Additional Features")
    st.write("Check the boxes to turn on extra tools.")
    track_drainage = st.checkbox("Track Drainage", value=True, help="See how much water drains from the soil.")
    enable_yield = st.checkbox("Estimate Yield", value=False, help="Predict crop yield.")
    if enable_yield:
        st.subheader("Yield Options")
        use_fao33 = st.checkbox("Use FAO-33 Method", value=True, help="A standard method to estimate yield.")
        if use_fao33:
            Ym = st.number_input("Maximum Yield (ton/ha)", min_value=0.0, value=10.0, step=0.1, help="The highest possible yield.")
            Ky = st.number_input("Yield Factor (Ky)", min_value=0.0, value=1.0, step=0.1, help="How water affects yield.")
        use_transp = st.checkbox("Use Transpiration Method", value=False, help="Estimate yield based on water use.")
        if use_transp:
            WP_yield = st.number_input("Yield per Water (ton/ha per mm)", min_value=0.0, value=0.01, step=0.001, help="Yield per unit of water used.")
    else:
        use_fao33 = use_transp = Ym = Ky = WP_yield = 0

    enable_leaching = st.checkbox("Estimate Leaching", value=False, help="Check how much nitrate leaves the soil.")
    if enable_leaching:
        leaching_method = st.radio("Leaching Method", ["Drainage × Nitrate", "Fraction × Nitrogen Input"], help="Choose how to calculate leaching.")
        if leaching_method == "Drainage × Nitrate":
            nitrate_conc = st.number_input("Nitrate Level (mg/L)", min_value=0.0, value=10.0, step=0.1, help="Nitrate in the drainage water.")
            total_N_input = leaching_fraction = 0
        else:
            total_N_input = st.number_input("Total Nitrogen (kg/ha)", min_value=0.0, value=100.0, step=1.0, help="Nitrogen added to the field.")
            leaching_fraction = st.number_input("Leaching Fraction (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="Fraction of nitrogen lost.")
            nitrate_conc = 0
    else:
        leaching_method = ""
        nitrate_conc = total_N_input = leaching_fraction = 0

    enable_etaforecast = st.checkbox("Forecast ETa (7 Days)", value=False, help="Predict water use for the next week.")
    if enable_etaforecast:
        forecast_lat = st.number_input("Latitude", value=0.0, help="Your field's latitude.")
        forecast_lon = st.number_input("Longitude", value=0.0, help="Your field's longitude.")
    enable_nue = st.checkbox("Estimate NUE", value=False, help="Check nitrogen use efficiency (needs yield and nitrogen input).")
    enable_dynamic_root = st.checkbox("Dynamic Root Growth", value=False, help="Simulate how roots grow over time.")
    if enable_dynamic_root:
        initial_root_depth = st.number_input("Starting Root Depth (mm)", min_value=50, value=300, step=10, help="Root depth at the start.")
        max_root_depth = st.number_input("Maximum Root Depth (mm)", min_value=50, value=1000, step=10, help="Deepest roots can go.")
        days_to_max = st.number_input("Days to Max Depth", min_value=1, value=60, step=1, help="Days to reach deepest roots.")
    else:
        initial_root_depth = max_root_depth = days_to_max = None
    show_soil_profile = st.checkbox("Show Soil Water Profile", value=False, help="View water storage in soil layers.")

with tab3:
    st.header("Graphs")
    st.write("View results after running the simulation.")

# Generate Crop Data
if crop_input_method == "Upload My Own":
    crop_df = pd.read_csv(crop_file) if crop_file else None
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
    with tab1:
        st.subheader("Edit Crop Details (Optional)")
        try:
            crop_df = st.experimental_data_editor(crop_df, num_rows="dynamic", help="Adjust crop settings if needed.")
        except:
            st.write(crop_df)

# Run Simulation
if run_button and weather_file and (crop_df is not None) and soil_file:
    with st.spinner("Running simulation..."):
        try:
            weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
            soil_df = pd.read_csv(soil_file)
            total_days = len(weather_df)
            crop_df.loc[crop_df.index[0], "End_Day"] = total_days

            results_df, soil_profile = SIMdualKc(weather_df, crop_df, soil_df, track_drainage, enable_yield,
                                                use_fao33, Ym, Ky, use_transp, WP_yield,
                                                enable_leaching, leaching_method, nitrate_conc,
                                                total_N_input, leaching_fraction,
                                                enable_dynamic_root, initial_root_depth, max_root_depth, days_to_max,
                                                return_soil_profile=True) if show_soil_profile else (SIMdualKc(weather_df, crop_df, soil_df, track_drainage, enable_yield,
                                                                                                                use_fao33, Ym, Ky, use_transp, WP_yield,
                                                                                                                enable_leaching, leaching_method, nitrate_conc,
                                                                                                                total_N_input, leaching_fraction,
                                                                                                                enable_dynamic_root, initial_root_depth, max_root_depth, days_to_max), None)

            forecast_results = None
            if enable_etaforecast:
                last_date = weather_df['Date'].max()
                forecast_start = last_date + datetime.timedelta(days=1)
                forecast_end = forecast_start + datetime.timedelta(days=6)
                forecast_weather = fetch_weather_data(forecast_lat, forecast_lon, forecast_start, forecast_end)
                if forecast_weather is not None:
                    forecast_results = SIMdualKc(forecast_weather, crop_df, soil_df, track_drainage, enable_yield,
                                                use_fao33, Ym, Ky, use_transp, WP_yield,
                                                enable_leaching, leaching_method, nitrate_conc,
                                                total_N_input, leaching_fraction,
                                                enable_dynamic_root, initial_root_depth, max_root_depth, days_to_max)

            st.success("Simulation completed!")
            with tab3:
                plot_options = ["ETa Components", "Cumulative Metrics", "Soil Water Storage", "Drainage", "Root Depth"]
                if enable_yield:
                    plot_options.append("Yield")
                if enable_leaching:
                    plot_options.append("Leaching")
                if enable_nue and total_N_input > 0:
                    plot_options.append("NUE")
                if show_soil_profile and soil_profile:
                    plot_options.append("Soil Profile Water")
                plot_option = st.selectbox("Choose a Graph", plot_options, help="Pick a graph to view.")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                if plot_option == "ETa Components":
                    ax.plot(results_df['Date'], results_df['ETa_transp (mm)'], label="Transpiration")
                    ax.plot(results_df['Date'], results_df['ETa_evap (mm)'], label="Evaporation")
                    ax.plot(results_df['Date'], results_df['ETc (mm)'], label="Potential ET")
                    ax.set_ylabel("Water Use (mm)")
                elif plot_option == "Cumulative Metrics":
                    ax.plot(results_df['Date'], results_df['Cumulative_Irrigation (mm)'], label="Irrigation")
                    ax.plot(results_df['Date'], results_df['Cumulative_Precip (mm)'], label="Precipitation")
                    ax.plot(results_df['Date'], results_df['Cumulative_ETa (mm)'], label="Total ETa")
                    ax.set_ylabel("Cumulative (mm)")
                elif plot_option == "Soil Water Storage":
                    ax.plot(results_df['Date'], results_df['SW_root (mm)'], label="Soil Water (Root Zone)")
                    ax.set_ylabel("Soil Water (mm)")
                elif plot_option == "Drainage":
                    ax.plot(results_df['Date'], results_df['Daily_Drainage'], label="Daily Drainage")
                    ax.set_ylabel("Drainage (mm)")
                elif plot_option == "Root Depth":
                    ax.plot(results_df['Date'], results_df['Root_Depth (mm)'], label="Root Depth")
                    ax.set_ylabel("Depth (mm)")
                elif plot_option == "Yield" and enable_yield:
                    ax.plot(results_df['Date'], results_df['Yield (ton/ha)'], label="Yield")
                    ax.set_ylabel("Yield (ton/ha)")
                elif plot_option == "Leaching" and enable_leaching:
                    ax.plot(results_df['Date'], results_df['Leaching (kg/ha)'], label="Leaching")
                    ax.set_ylabel("Leaching (kg/ha)")
                elif plot_option == "NUE" and enable_nue and total_N_input > 0:
                    ax.plot(results_df['Date'], results_df['NUE (kg/ha)'], label="NUE")
                    ax.set_ylabel("NUE (kg/ha)")
                elif plot_option == "Soil Profile Water" and show_soil_profile:
                    depths = [layer["Depth_mm"] for layer in soil_profile]
                    sw = [layer["SW (mm)"] for layer in soil_profile]
                    ax.bar(range(len(depths)), sw, width=0.8, align='center')
                    ax.set_xticks(range(len(depths)))
                    ax.set_xticklabels([f"Layer {i+1}" for i in range(len(depths))])
                    ax.set_ylabel("Water Stored (mm)")
                
                ax.set_xlabel("Date")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            if enable_etaforecast and forecast_results is not None:
                with tab2:
                    st.subheader("7-Day ETa Forecast")
                    st.write("Predicted water use for the next week:")
                    st.dataframe(forecast_results[["Date", "ETa_total (mm)"]].rename(columns={"ETa_total (mm)": "Predicted ETa (mm)"}))
        except Exception as e:
            st.error(f"Simulation failed: {e}")
else:
    with tab1:
        st.info("Please upload all files and click 'Run Simulation' to see results.")
