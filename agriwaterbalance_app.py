import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import Draw, HeatMap
from streamlit_folium import st_folium
from shapely.geometry import shape, Point, Polygon
import datetime
import requests
import math
import time
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------
# Configure Requests Session with Retries
# -------------------
session = requests.Session()
retries = Retry(
    total=5,  # Increased from 3 to 5
    backoff_factor=1.0,  # Increased from 0.5 to 1.0
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
    kcb_list, root_list, p_list, ke_list = [], [], [], []
    for i in range(len(crop_df)):
        row = crop_df.iloc[i]
        start, end = row['Start_Day'], row['End_Day']
        days = int(end - start + 1)
        for d in range(days):
            frac = d / (days - 1) if days > 1 else 0
            if i > 0:
                prev = crop_df.iloc[i - 1]
                kcb = prev['Kcb'] + frac * (row['Kcb'] - prev['Kcb'])
                root = prev['Root_Depth_mm'] + frac * (row['Root_Depth_mm'] - prev['Root_Depth_mm'])
                p = prev['p'] + frac * (row['p'] - prev['p'])
                ke = prev['Ke'] + frac * (row['Ke'] - prev['Ke'])
            else:
                kcb, root, p, ke = row['Kcb'], row['Root_Depth_mm'], row['p'], row['Ke']
            kcb_list.append(kcb)
            root_list.append(root)
            p_list.append(p)
            ke_list.append(ke)
    return kcb_list[:total_days], root_list[:total_days], p_list[:total_days], ke_list[:total_days]

def SIMdualKc(weather_df, crop_df, soil_df, track_drain=True, enable_yield=False, 
              use_fao33=False, Ym=0, Ky=0, use_transp=False, WP_yield=0, 
              enable_leaching=False, leaching_method="", nitrate_conc=0, 
              total_N_input=0, leaching_fraction=0):
    days = len(weather_df)
    profile_depth = soil_df['Depth_mm'].sum()
    Kcb_list, RD_list, p_list, Ke_list = interpolate_crop_stages(crop_df, days)
    TEW, REW = soil_df.iloc[0]['TEW'], soil_df.iloc[0]['REW']
    SW_surface = TEW
    SW_layers = [row['FC'] * row['Depth_mm'] for _, row in soil_df.iterrows()]
    cum_ETc = cum_ETa = cum_Irr = cum_P = cum_drain = 0
    stress_days = 0
    cum_transp = 0
    cum_evap = 0  # Added to track cumulative evaporation
    results = []
    for i in range(days):
        row = weather_df.iloc[i]
        ET0, P, I = row['ET0'], row['Precipitation'], row['Irrigation']
        Kcb, RD, p, Ke = Kcb_list[i], min(RD_list[i], profile_depth), p_list[i], Ke_list[i]
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
        cum_evap += ETa_evap  # Track cumulative evaporation
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
            "ETa_total (mm)": ETa_transp + ETa_evap,  # Added total ETa
            "SW_surface (mm)": SW_surface,
            "SW_root (mm)": SW_root,
            "Root_Depth (mm)": RD,
            "Depletion (mm)": depletion,
            "TAW (mm)": TAW,
            "Cumulative_ETc (mm)": cum_ETc,
            "Cumulative_ETa (mm)": cum_ETa,
            "Cumulative_Transp (mm)": cum_transp,
            "Cumulative_Evap (mm)": cum_evap,  # Added cumulative evaporation
            "Cumulative_Irrigation (mm)": cum_Irr,
            "Cumulative_Precip (mm)": cum_P,
            "Cumulative_Drainage (mm)": cum_drain,
            "Stress_Days": stress_days
        })
    results_df = pd.DataFrame(results)
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
        if "PRECTOT" in data:
            precip_key = "PRECTOT"
        elif "PRECTOTCORG" in data:
            precip_key = "PRECTOTCORG"
        else:
            precip_key = None
        dates = []
        et0_list = []
        precip_list = []
        for date_str in data['T2M_MAX']:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            dates.append(dt)
            Tmax = data['T2M_MAX'][date_str]
            Tmin = data['T2M_MIN'][date_str]
            Tmean = (Tmax + Tmin) / 2
            Rs = data['ALLSKY_SFC_SW_DWN'][date_str]
            u2 = data['WS2M'][date_str]
            RH = data['RH2M'][date_str]
            delta = 4098 * (0.6108 * math.exp((17.27 * Tmean)/(Tmean + 237.3))) / (Tmean + 237.3)**2
            P = 101.3
            gamma = 0.000665 * P
            es = (0.6108 * math.exp(17.27 * Tmax/(Tmax + 237.3)) + 0.6108 * math.exp(17.27 * Tmin/(Tmin + 237.3))) / 2
            ea = es * RH / 100
            ET0 = (0.408 * delta * Rs + gamma * (900/(Tmean+273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34*u2))
            et0_list.append(ET0)
            if precip_key:
                precip_list.append(data[precip_key][date_str])
            else:
                precip_list.append(0)
        weather_df = pd.DataFrame({
            "Date": dates,
            "ET0": et0_list,
            "Precipitation": precip_list,
            "Irrigation": [0] * len(dates)
        })
        return weather_df
    except Exception as e:
        st.error(f"Weather data fetch failed: {str(e)}")
        return None

def fetch_soil_data(lat, lon):
    max_retries = 5  # Increased from 3 to 5
    retry_delay = 2  # Increased from 1 to 2 seconds
    timeout = 10  # Increased from 5 to 10 seconds
    
    # Use default values as fallback
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
                bdod = properties.get('bdod', {}).get(depth, {}).get('mean', 140) / 100
                sand = properties.get('sand', {}).get(depth, {}).get('mean', 40)
                clay = properties.get('clay', {}).get(depth, {}).get('mean', 20)
                ocd = properties.get('ocd', {}).get(depth, {}).get('mean', 1.0) / 100
                FC = max(0.1, min(0.4, (-0.251 * sand/100 + 0.195 * clay/100 + 0.011 * ocd +
                          0.006 * (sand/100) * ocd - 0.027 * (clay/100) * ocd +
                          0.452 * (sand/100) * (clay/100) + 0.299) * bdod))
                WP = max(0.01, min(0.2, (-0.024 * sand/100 + 0.487 * clay/100 + 0.006 * ocd +
                          0.005 * (sand/100) * ocd - 0.013 * (clay/100) * ocd +
                          0.068 * (sand/100) * (clay/100) + 0.031) * bdod))
                layers.append({
                    "Depth_mm": 50 if depth == '0-5cm' else 100,
                    "FC": FC,
                    "WP": WP,
                    "TEW": 200 if depth == '0-5cm' else 0,
                    "REW": 50 if depth == '0-5cm' else 0
                })
            return pd.DataFrame(layers)
        except (requests.exceptions.RequestException, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            
            st.warning(f"Soil data fetch failed: {str(e)[:100]}... Using default values.")
            return default_soil_df

def fetch_ndvi_data(study_area, start_date, end_date):
    return 0.6

def get_crop_data(ndvi, num_days):
    kcb = 0.1 + 1.1 * ndvi
    crop_df = pd.DataFrame({
        "Start_Day": [1],
        "End_Day": [num_days],
        "Kcb": [kcb],
        "Root_Depth_mm": [300],
        "p": [0.5],
        "Ke": [1.0]
    })
    return crop_df

def create_grid_in_polygon(polygon, spacing=0.01):
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    polygon_utm = gdf_utm.iloc[0].geometry
    bounds = polygon_utm.bounds
    spacing_m = spacing * 1000
    x_coords = np.arange(bounds[0], bounds[2], spacing_m)
    y_coords = np.arange(bounds[1], bounds[3], spacing_m)
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            pt = Point(x, y)
            if polygon_utm.contains(pt):
                pt_lonlat = gpd.GeoSeries([pt], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
                grid_points.append(pt_lonlat)
    return grid_points

# Function to calculate polygon area in hectares
def calculate_polygon_area_ha(polygon):
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    # Area in square meters
    area_m2 = gdf_utm.area[0]
    # Convert to hectares
    area_ha = area_m2 / 10000
    return area_ha

# Function to calculate polygon-level water balance
def calculate_polygon_water_balance(point_results_df, polygon_area_ha):
    """
    Calculate water balance for the entire polygon based on point results
    
    Parameters:
    - point_results_df: DataFrame with point-level results
    - polygon_area_ha: Area of the polygon in hectares
    
    Returns:
    - DataFrame with polygon-level water balance
    """
    # Get the last day results for each point
    last_day_results = []
    
    # Group by lat/lon and get the last day for each point
    for (lat, lon), group in point_results_df.groupby(['lat', 'lon']):
        last_day = group.iloc[-1].to_dict()
        last_day['lat'] = lat
        last_day['lon'] = lon
        last_day_results.append(last_day)
    
    last_day_df = pd.DataFrame(last_day_results)
    
    # Calculate average values across all points
    avg_values = {
        'Mean_Transp (mm/day)': last_day_df['ETa_transp (mm)'].mean(),
        'Mean_Evap (mm/day)': last_day_df['ETa_evap (mm)'].mean(),
        'Mean_ET (mm/day)': last_day_df['ETa_total (mm)'].mean(),
        'Final_SW_root (mm)': last_day_df['SW_root (mm)'].mean(),
        'Cum_Transp (mm)': last_day_df['Cumulative_Transp (mm)'].mean(),
        'Cum_Evap (mm)': last_day_df['Cumulative_Evap (mm)'].mean(),
        'Cum_ET (mm)': last_day_df['Cumulative_ETa (mm)'].mean(),
        'Cum_Drainage (mm)': last_day_df['Cumulative_Drainage (mm)'].mean(),
        'Cum_Precip (mm)': last_day_df['Cumulative_Precip (mm)'].mean(),
        'Cum_Irrigation (mm)': last_day_df['Cumulative_Irrigation (mm)'].mean(),
    }
    
    # Calculate total volumes (m³) based on area
    total_values = {}
    for key, value in avg_values.items():
        if key.startswith('Cum_'):
            # Convert mm to m³ for the entire polygon
            # 1 mm over 1 ha = 10 m³
            volume_m3 = value * polygon_area_ha * 10
            total_values[f'Total_{key[4:].replace(" (mm)", " (m³)")}'] = volume_m3
    
    # Combine average and total values
    polygon_results = {**avg_values, **total_values}
    
    return polygon_results

# -------------------
# User Interface
# -------------------
st.title("AgriWaterBalance")
st.markdown("**Spatiotemporal Soil Water Balance Modeling**")
mode = st.radio("Operation Mode:", ["Normal Mode", "Spatial Mode"], horizontal=True)

# ========= NORMAL MODE =========
if mode == "Normal Mode":
    with st.sidebar:
        st.header("Upload Input Files (.txt)")
        weather_file = st.file_uploader("Weather Data (.txt)", type="txt")
        crop_file = st.file_uploader("Crop Stage Data (.txt)", type="txt")
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
        st.header("Leaching Estimation")
        enable_leaching = st.checkbox("Enable Leaching Estimation", value=False)
        if enable_leaching:
            leaching_method = st.radio("Select Leaching Method", [
                "Method 1: Drainage × nitrate concentration",
                "Method 2: Leaching Fraction × total N input"
            ])
            if leaching_method == "Method 1: Drainage × nitrate concentration":
                nitrate_conc = st.number_input("Nitrate Concentration (mg/L)", min_value=0.0, value=10.0, step=0.1)
            elif leaching_method == "Method 2: Leaching Fraction × total N input":
                total_N_input = st.number_input("Total N Input (kg/ha)", min_value=0.0, value=100.0, step=1.0)
                leaching_fraction = st.number_input("Leaching Fraction (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        run_button = st.button("🚀 Run Simulation")
    if run_button and weather_file and crop_file and soil_file:
        with st.spinner("Running simulation..."):
            try:
                weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
                crop_df = pd.read_csv(crop_file)
                soil_df = pd.read_csv(soil_file)
                results_df = SIMdualKc(
                    weather_df, crop_df, soil_df, track_drainage,
                    enable_yield=enable_yield,
                    use_fao33=use_fao33 if enable_yield else False,
                    Ym=Ym if enable_yield else 0,
                    Ky=Ky if enable_yield else 0,
                    use_transp=use_transp if enable_yield else False,
                    WP_yield=WP_yield if enable_yield else 0,
                    enable_leaching=enable_leaching,
                    leaching_method=leaching_method if enable_leaching else "",
                    nitrate_conc=nitrate_conc if enable_leaching and leaching_method=="Method 1: Drainage × nitrate concentration" else 0,
                    total_N_input=total_N_input if enable_leaching and leaching_method=="Method 2: Leaching Fraction × total N input" else 0,
                    leaching_fraction=leaching_fraction if enable_leaching and leaching_method=="Method 2: Leaching Fraction × total N input" else 0
                )
                results_df['SWC (%)'] = (results_df['SW_root (mm)'] / results_df['Root_Depth (mm)']) * 100
                st.success("Simulation completed successfully!")
                tab1, tab2, tab3, tab4 = st.tabs(["📄 Daily Results", "📈 ET Graphs", "💧 Storage", "🌾 Yield and Leaching"])
                with tab1:
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results (.txt)", csv, file_name="results.txt")
                with tab2:
                    fig, ax = plt.subplots()
                    ax.plot(results_df['Date'], results_df['ETa_transp (mm)'], label='Transpiration')
                    ax.plot(results_df['Date'], results_df['ETa_evap (mm)'], label='Evaporation')
                    ax.plot(results_df['Date'], results_df['ETc (mm)'], label='ETc')
                    ax.set_ylabel("ET (mm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                with tab3:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(results_df['Date'], results_df['SW_root (mm)'], label='Root Zone SW')
                    ax2.set_ylabel("Soil Water (mm)")
                    ax2.legend()
                    ax2.grid(True)
                    st.pyplot(fig2)
                with tab4:
                    if enable_yield and 'Yield (ton/ha)' in results_df.columns:
                        st.write("### Yield Estimation")
                        st.write(results_df[['Date', 'Yield (ton/ha)']])
                    if enable_leaching and 'Leaching (kg/ha)' in results_df.columns:
                        st.write("### Leaching Estimation")
                        st.write(results_df[['Date', 'Leaching (kg/ha)']])
                if show_monthly_summary:
                    st.subheader("📆 Monthly Summary")
                    monthly = results_df.copy()
                    monthly['Month'] = monthly['Date'].dt.to_period('M')
                    summary = monthly.groupby('Month').agg({
                        'ET0 (mm)': 'mean',
                        'ETc (mm)': 'mean',
                        'ETa_transp (mm)': 'mean',
                        'ETa_evap (mm)': 'mean',
                        'Cumulative_Irrigation (mm)': 'max',
                        'Cumulative_Precip (mm)': 'max',
                        'Stress_Days': 'max'
                    }).reset_index()
                    st.dataframe(summary)
            except Exception as e:
                st.error(f"⚠️ Simulation failed: {e}")
    else:
        st.info("📂 Please upload all required files and click 'Run Simulation'.")
        
# ========= SPATIAL MODE =========
elif mode == "Spatial Mode":
    st.markdown("### 🌍 Spatial Analysis Mode")
    col1, col2 = st.columns([3, 1])
    with col1:
        m = folium.Map(location=[40, -100], zoom_start=4, tiles="Esri.WorldImagery")
        Draw(export=True, draw_options={"polygon": True, "marker": False, "circlemarker": False}).add_to(m)
        map_data = st_folium(m, key="spatial_map", height=600)
    with col2:
        st.markdown("### Spatial Parameters")
        spacing = st.slider("Grid spacing (km)", 0.1, 5.0, 1.0)
        start_date = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=30))
        end_date = st.date_input("End date", datetime.date.today())
        
        st.markdown("### Crop Parameters")
        crop_type = st.selectbox("Crop Type", ["Custom", "Maize", "Wheat", "Soybean", "Rice"])
        
        if crop_type == "Custom":
            kcb = st.slider("Basal Crop Coefficient (Kcb)", 0.1, 1.5, 1.0, 0.05)
            root_depth = st.slider("Root Depth (mm)", 100, 2000, 1000, 100)
            p_factor = st.slider("Depletion Factor (p)", 0.1, 0.8, 0.5, 0.05)
            ke_factor = st.slider("Evaporation Coefficient (Ke)", 0.1, 1.2, 0.8, 0.05)
        else:
            # Predefined crop parameters
            if crop_type == "Maize":
                kcb, root_depth, p_factor, ke_factor = 1.2, 1500, 0.55, 0.7
            elif crop_type == "Wheat":
                kcb, root_depth, p_factor, ke_factor = 1.1, 1200, 0.5, 0.6
            elif crop_type == "Soybean":
                kcb, root_depth, p_factor, ke_factor = 1.0, 1000, 0.5, 0.7
            elif crop_type == "Rice":
                kcb, root_depth, p_factor, ke_factor = 1.3, 800, 0.2, 1.0
        
        st.markdown("### Output Options")
        output_variable = st.selectbox(
            "Select Primary Output Variable", 
            ["Evaporation", "Transpiration", "Evapotranspiration", "Soil Water Content"]
        )
        
    if st.button("🚀 Run Spatial Analysis"):
        if map_data and map_data.get("all_drawings"):
            try:
                shapes = [shape(d["geometry"]) for d in map_data["all_drawings"] if d["geometry"]["type"] == "Polygon"]
                if not shapes:
                    st.error("No polygon drawn.")
                    st.stop()
                
                # Use union_all() instead of deprecated unary_union
                study_area = gpd.GeoSeries(shapes).union_all()
                
                # Calculate polygon area in hectares
                polygon_area_ha = calculate_polygon_area_ha(study_area)
                
                st.markdown("### Your Field Boundary")
                st.write(f"Field Area: {polygon_area_ha:.2f} hectares ({polygon_area_ha*2.47:.2f} acres)")
                
                grid_points = create_grid_in_polygon(study_area, spacing=spacing)
                if not grid_points:
                    st.warning("No grid points generated! Check your polygon and spacing.")
                    st.stop()
                
                st.write(f"Generated {len(grid_points)} sample points within the field.")
                
                grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
                grid_gdf["lat"] = grid_gdf.geometry.y
                grid_gdf["lon"] = grid_gdf.geometry.x
                
                # Display grid points on map
                st.map(grid_gdf[["lat", "lon"]])
                
                st.subheader("Running Spatial Analysis")
                point_results_all = []  # Store all daily results for all points
                results = []  # Store final results for each point
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get centroid for representative weather data
                centroid = study_area.centroid[0]
                centroid_lat, centroid_lon = centroid.y, centroid.x
                
                # Fetch weather data for the centroid
                centroid_weather = fetch_weather_data(centroid_lat, centroid_lon, start_date, end_date)
                if centroid_weather is None:
                    st.error("Failed to fetch weather data for the field centroid. Please try again.")
                    st.stop()
                
                for i, point in enumerate(grid_points):
                    lat_pt, lon_pt = point.y, point.x
                    status_text.text(f"Processing point {i+1} of {len(grid_points)} ({lat_pt:.4f}, {lon_pt:.4f})")
                    
                    # Use the same weather data for all points (from centroid)
                    weather = centroid_weather.copy()
                    
                    # Fetch soil data for each point
                    soil = fetch_soil_data(lat_pt, lon_pt)
                    
                    # Create crop data
                    crop = pd.DataFrame({
                        "Start_Day": [1],
                        "End_Day": [len(weather)],
                        "Kcb": [kcb],
                        "Root_Depth_mm": [root_depth],
                        "p": [p_factor],
                        "Ke": [ke_factor]
                    })
                    
                    if soil is not None:
                        # Run water balance simulation
                        result = SIMdualKc(weather, crop, soil)
                        
                        # Add location information to each daily result
                        result['lat'] = lat_pt
                        result['lon'] = lon_pt
                        
                        # Store all daily results for this point
                        point_results_all.append(result)
                        
                        # Extract results for the last day
                        final_row = result.iloc[-1]
                        
                        # Calculate average values
                        mean_transp = result["ETa_transp (mm)"].mean()
                        mean_evap = result["ETa_evap (mm)"].mean()
                        mean_et = result["ETa_total (mm)"].mean()
                        
                        # Store results
                        results.append({
                            "lat": lat_pt,
                            "lon": lon_pt,
                            "SW_root (mm)": final_row["SW_root (mm)"],
                            "Mean_Transp (mm/day)": mean_transp,
                            "Mean_Evap (mm/day)": mean_evap,
                            "Mean_ET (mm/day)": mean_et,
                            "Cum_Transp (mm)": final_row["Cumulative_Transp (mm)"],
                            "Cum_Evap (mm)": final_row["Cumulative_Evap (mm)"],
                            "Cum_ET (mm)": final_row["Cumulative_ETa (mm)"],
                            "Cum_Drainage (mm)": final_row["Cumulative_Drainage (mm)"]
                        })
                    else:
                        st.warning(f"Skipping point ({lat_pt:.4f}, {lon_pt:.4f}) due to missing soil data.")
                    
                    progress_bar.progress((i+1)/len(grid_points))
                
                if results:
                    # Create results DataFrame for points
                    results_df = pd.DataFrame(results)
                    
                    # Combine all daily results into one DataFrame
                    all_results_df = pd.concat(point_results_all)
                    
                    # Calculate polygon-level water balance
                    polygon_results = calculate_polygon_water_balance(all_results_df, polygon_area_ha)
                    
                    # Display results
                    st.markdown("## 📊 Spatial Results")
                    tab1, tab2, tab3, tab4 = st.tabs(["Field Summary", "Map Visualization", "Data Table", "Export Data"])
                    
                    with tab1:
                        st.markdown("### Field-Level Water Balance")
                        
                        # Create two columns for average and total values
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Average Values (per unit area)")
                            avg_df = pd.DataFrame({
                                "Component": [
                                    "Transpiration (daily avg)",
                                    "Evaporation (daily avg)",
                                    "Evapotranspiration (daily avg)",
                                    "Final Soil Water Content",
                                    "Cumulative Transpiration",
                                    "Cumulative Evaporation",
                                    "Cumulative Evapotranspiration",
                                    "Cumulative Drainage",
                                    "Cumulative Precipitation",
                                    "Cumulative Irrigation"
                                ],
                                "Value": [
                                    f"{polygon_results['Mean_Transp (mm/day)']:.2f} mm/day",
                                    f"{polygon_results['Mean_Evap (mm/day)']:.2f} mm/day",
                                    f"{polygon_results['Mean_ET (mm/day)']:.2f} mm/day",
                                    f"{polygon_results['Final_SW_root (mm)']:.2f} mm",
                                    f"{polygon_results['Cum_Transp (mm)']:.2f} mm",
                                    f"{polygon_results['Cum_Evap (mm)']:.2f} mm",
                                    f"{polygon_results['Cum_ET (mm)']:.2f} mm",
                                    f"{polygon_results['Cum_Drainage (mm)']:.2f} mm",
                                    f"{polygon_results['Cum_Precip (mm)']:.2f} mm",
                                    f"{polygon_results['Cum_Irrigation (mm)']:.2f} mm"
                                ]
                            })
                            st.table(avg_df)
                        
                        with col2:
                            st.markdown("#### Total Values (entire field)")
                            total_df = pd.DataFrame({
                                "Component": [
                                    "Total Transpiration",
                                    "Total Evaporation",
                                    "Total Evapotranspiration",
                                    "Total Drainage",
                                    "Total Precipitation",
                                    "Total Irrigation"
                                ],
                                "Value": [
                                    f"{polygon_results['Total_Transp (m³)']:.2f} m³",
                                    f"{polygon_results['Total_Evap (m³)']:.2f} m³",
                                    f"{polygon_results['Total_ET (m³)']:.2f} m³",
                                    f"{polygon_results['Total_Drainage (m³)']:.2f} m³",
                                    f"{polygon_results['Total_Precip (m³)']:.2f} m³",
                                    f"{polygon_results['Total_Irrigation (m³)']:.2f} m³"
                                ]
                            })
                            st.table(total_df)
                        
                        # Create bar chart for water balance components
                        st.markdown("#### Water Balance Components")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        components = ["Transpiration", "Evaporation", "Drainage", "Soil Storage"]
                        values = [
                            polygon_results['Cum_Transp (mm)'],
                            polygon_results['Cum_Evap (mm)'],
                            polygon_results['Cum_Drainage (mm)'],
                            polygon_results['Final_SW_root (mm)']
                        ]
                        colors = ['green', 'blue', 'orange', 'brown']
                        ax.bar(components, values, color=colors)
                        ax.set_ylabel("Water (mm)")
                        ax.set_title("Water Balance Components")
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        for i, v in enumerate(values):
                            ax.text(i, v + 1, f"{v:.1f}", ha='center')
                        
                        st.pyplot(fig)
                        
                        # Create pie chart for ET components
                        fig2, ax2 = plt.subplots(figsize=(8, 8))
                        et_components = ["Transpiration", "Evaporation"]
                        et_values = [
                            polygon_results['Cum_Transp (mm)'],
                            polygon_results['Cum_Evap (mm)']
                        ]
                        ax2.pie(et_values, labels=et_components, autopct='%1.1f%%', colors=['green', 'blue'])
                        ax2.set_title("Evapotranspiration Components")
                        st.pyplot(fig2)
                    
                    with tab2:
                        st.markdown("### Spatial Distribution Maps")
                        
                        # Select output variable for visualization
                        if output_variable == "Evaporation":
                            value_col = "Mean_Evap (mm/day)"
                            cum_col = "Cum_Evap (mm)"
                            title = "Evaporation"
                            color = "Blues"
                        elif output_variable == "Transpiration":
                            value_col = "Mean_Transp (mm/day)"
                            cum_col = "Cum_Transp (mm)"
                            title = "Transpiration"
                            color = "Greens"
                        elif output_variable == "Evapotranspiration":
                            value_col = "Mean_ET (mm/day)"
                            cum_col = "Cum_ET (mm)"
                            title = "Evapotranspiration"
                            color = "YlGnBu"
                        else:
                            value_col = "SW_root (mm)"
                            cum_col = None
                            title = "Soil Water Content"
                            color = "YlOrBr"
                        
                        # Create map for daily values
                        st.subheader(f"Daily {title} (mm/day)")
                        
                        # Convert study area to GeoJSON for map display
                        field_gdf = gpd.GeoDataFrame(geometry=[study_area], crs="EPSG:4326")
                        field_geojson = json.loads(field_gdf.to_json())
                        
                        # Create base map
                        m_daily = folium.Map(
                            location=[results_df.lat.mean(), results_df.lon.mean()], 
                            zoom_start=10,
                            tiles="CartoDB positron"
                        )
                        
                        # Add field boundary
                        folium.GeoJson(
                            data=field_geojson,
                            name="Field Boundary",
                            style_function=lambda x: {'fillColor': 'none', 'color': '#000000', 'weight': 2}
                        ).add_to(m_daily)
                        
                        # Add heatmap
                        heat_data = [[float(r[0]), float(r[1]), float(r[2])] 
                                     for r in results_df[['lat', 'lon', value_col]].values.tolist()]
                        HeatMap(
                            data=heat_data, 
                            radius=15, 
                            blur=10, 
                            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
                            name=f"{title} Heatmap"
                        ).add_to(m_daily)
                        
                        # Add markers
                        for idx, row in results_df.iterrows():
                            folium.CircleMarker(
                                location=[row['lat'], row['lon']],
                                radius=3,
                                popup=f"{title}: {row[value_col]:.2f} mm/day<br>SW: {row['SW_root (mm)']:.1f} mm",
                                color='#3186cc',
                                fill=True,
                                fill_opacity=0.7
                            ).add_to(m_daily)
                        
                        # Add layer control
                        folium.LayerControl().add_to(m_daily)
                        
                        # Display the map
                        daily_map = st_folium(m_daily, width=800, height=500, key="daily_map")
                        
                        # Create map for cumulative values if applicable
                        if cum_col:
                            st.subheader(f"Cumulative {title} (mm)")
                            m_cum = folium.Map(
                                location=[results_df.lat.mean(), results_df.lon.mean()], 
                                zoom_start=10,
                                tiles="CartoDB positron"
                            )
                            
                            # Add field boundary
                            folium.GeoJson(
                                data=field_geojson,
                                name="Field Boundary",
                                style_function=lambda x: {'fillColor': 'none', 'color': '#000000', 'weight': 2}
                            ).add_to(m_cum)
                            
                            # Add heatmap
                            heat_data = [[float(r[0]), float(r[1]), float(r[2])] 
                                         for r in results_df[['lat', 'lon', cum_col]].values.tolist()]
                            HeatMap(
                                data=heat_data, 
                                radius=15, 
                                blur=10, 
                                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
                                name=f"Cumulative {title} Heatmap"
                            ).add_to(m_cum)
                            
                            # Add markers
                            for idx, row in results_df.iterrows():
                                folium.CircleMarker(
                                    location=[row['lat'], row['lon']],
                                    radius=3,
                                    popup=f"{title}: {row[cum_col]:.2f} mm<br>SW: {row['SW_root (mm)']:.1f} mm",
                                    color='#3186cc',
                                    fill=True,
                                    fill_opacity=0.7
                                ).add_to(m_cum)
                            
                            # Add layer control
                            folium.LayerControl().add_to(m_cum)
                            
                            # Display the map
                            cum_map = st_folium(m_cum, width=800, height=500, key="cum_map")
                    
                    with tab3:
                        st.markdown("### Raw Results Data")
                        
                        # Point-level results
                        st.subheader("Point-Level Results")
                        st.dataframe(results_df, height=300)
                        
                        # Field-level results
                        st.subheader("Field-Level Results")
                        field_df = pd.DataFrame({
                            "Component": list(polygon_results.keys()),
                            "Value": list(polygon_results.values())
                        })
                        st.dataframe(field_df, height=300)
                    
                    with tab4:
                        st.markdown("### Export Options")
                        
                        # Create tabs for different export options
                        export_tab1, export_tab2, export_tab3 = st.tabs(["Point Data", "Field Summary", "GIS Data"])
                        
                        with export_tab1:
                            # Point-level CSV export
                            csv_points = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Point Results (CSV)", 
                                csv_points, 
                                file_name="spatial_water_balance_points.csv", 
                                mime="text/csv",
                                key="download_points_csv"
                            )
                        
                        with export_tab2:
                            # Field-level CSV export
                            field_df = pd.DataFrame({
                                "Component": list(polygon_results.keys()),
                                "Value": list(polygon_results.values())
                            })
                            csv_field = field_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Field Summary (CSV)", 
                                csv_field, 
                                file_name="field_water_balance_summary.csv", 
                                mime="text/csv",
                                key="download_field_csv"
                            )
                            
                            # Create a summary report with more details
                            summary_report = f"""# Field Water Balance Summary Report

## Field Information
- **Field Area:** {polygon_area_ha:.2f} hectares ({polygon_area_ha*2.47:.2f} acres)
- **Analysis Period:** {start_date} to {end_date}
- **Crop Type:** {crop_type}
- **Number of Sample Points:** {len(grid_points)}

## Water Balance Components (Average per unit area)
- **Transpiration (daily average):** {polygon_results['Mean_Transp (mm/day)']:.2f} mm/day
- **Evaporation (daily average):** {polygon_results['Mean_Evap (mm/day)']:.2f} mm/day
- **Evapotranspiration (daily average):** {polygon_results['Mean_ET (mm/day)']:.2f} mm/day
- **Final Soil Water Content:** {polygon_results['Final_SW_root (mm)']:.2f} mm
- **Cumulative Transpiration:** {polygon_results['Cum_Transp (mm)']:.2f} mm
- **Cumulative Evaporation:** {polygon_results['Cum_Evap (mm)']:.2f} mm
- **Cumulative Evapotranspiration:** {polygon_results['Cum_ET (mm)']:.2f} mm
- **Cumulative Drainage:** {polygon_results['Cum_Drainage (mm)']:.2f} mm
- **Cumulative Precipitation:** {polygon_results['Cum_Precip (mm)']:.2f} mm
- **Cumulative Irrigation:** {polygon_results['Cum_Irrigation (mm)']:.2f} mm

## Total Water Volumes (entire field)
- **Total Transpiration:** {polygon_results['Total_Transp (m³)']:.2f} m³
- **Total Evaporation:** {polygon_results['Total_Evap (m³)']:.2f} m³
- **Total Evapotranspiration:** {polygon_results['Total_ET (m³)']:.2f} m³
- **Total Drainage:** {polygon_results['Total_Drainage (m³)']:.2f} m³
- **Total Precipitation:** {polygon_results['Total_Precip (m³)']:.2f} m³
- **Total Irrigation:** {polygon_results['Total_Irrigation (m³)']:.2f} m³

## Evapotranspiration Components
- **Transpiration:** {polygon_results['Cum_Transp (mm)']:.2f} mm ({polygon_results['Cum_Transp (mm)']/polygon_results['Cum_ET (mm)']*100:.1f}%)
- **Evaporation:** {polygon_results['Cum_Evap (mm)']:.2f} mm ({polygon_results['Cum_Evap (mm)']/polygon_results['Cum_ET (mm)']*100:.1f}%)

## Crop Parameters
- **Basal Crop Coefficient (Kcb):** {kcb}
- **Root Depth:** {root_depth} mm
- **Depletion Factor (p):** {p_factor}
- **Evaporation Coefficient (Ke):** {ke_factor}

Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
                            
                            st.download_button(
                                "📥 Download Detailed Report (MD)", 
                                summary_report, 
                                file_name="water_balance_report.md", 
                                mime="text/markdown",
                                key="download_report"
                            )
                        
                        with export_tab3:
                            # GeoJSON export for points
                            gdf = gpd.GeoDataFrame(
                                results_df,
                                geometry=gpd.points_from_xy(results_df.lon, results_df.lat),
                                crs="EPSG:4326"
                            )
                            geojson = gdf.to_json()
                            st.download_button(
                                "🌐 Download Points as GeoJSON", 
                                geojson, 
                                file_name="spatial_results.geojson", 
                                mime="application/json",
                                key="download_points_geojson"
                            )
                            
                            # Field boundary GeoJSON export
                            field_gdf = gpd.GeoDataFrame(
                                {"area_ha": [polygon_area_ha]},
                                geometry=[study_area],
                                crs="EPSG:4326"
                            )
                            field_geojson_str = field_gdf.to_json()
                            st.download_button(
                                "🌐 Download Field Boundary as GeoJSON", 
                                field_geojson_str, 
                                file_name="field_boundary.geojson", 
                                mime="application/json",
                                key="download_field_geojson"
                            )
                            
                            # Create a combined GeoJSON with both points and field boundary
                            # First create a feature collection for the field
                            field_feature = {
                                "type": "Feature",
                                "properties": {"type": "field", "area_ha": polygon_area_ha},
                                "geometry": json.loads(field_gdf.geometry.to_json())["features"][0]["geometry"]
                            }
                            
                            # Then create features for each point
                            point_features = []
                            for idx, row in gdf.iterrows():
                                properties = {k: v for k, v in row.items() if k != 'geometry'}
                                properties["type"] = "point"
                                point_feature = {
                                    "type": "Feature",
                                    "properties": properties,
                                    "geometry": json.loads(gdf.iloc[[idx]].geometry.to_json())["features"][0]["geometry"]
                                }
                                point_features.append(point_feature)
                            
                            # Combine into a single feature collection
                            combined_geojson = {
                                "type": "FeatureCollection",
                                "features": [field_feature] + point_features
                            }
                            
                            st.download_button(
                                "🌐 Download Combined GeoJSON (Field + Points)", 
                                json.dumps(combined_geojson), 
                                file_name="combined_results.geojson", 
                                mime="application/json",
                                key="download_combined_geojson"
                            )
                else:
                    st.warning("No results generated! Check input data sources.")
            except Exception as e:
                st.error(f"Spatial analysis failed: {str(e)}")
                st.exception(e)  # This will show the full traceback
        else:
            st.warning("Please draw a polygon on the map first!")
