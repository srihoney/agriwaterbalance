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
import json
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
    cum_evap = 0
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

def calculate_polygon_area_ha(polygon):
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    area_m2 = gdf_utm.area[0]
    area_ha = area_m2 / 10000
    return area_ha

# -------------------
# New Functions for Daily Spatial Output
# -------------------
def create_fishnet(polygon, cell_size_m=100):
    """
    Create a fishnet (grid of small polygons) covering the given polygon.
    The polygon is assumed to be in EPSG:4326; it is first projected to an estimated UTM.
    """
    gdf = gpd.GeoDataFrame({'geometry':[polygon]}, crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    poly_utm = gdf_utm.geometry.iloc[0]
    minx, miny, maxx, maxy = poly_utm.bounds
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = Polygon([(x, y), (x+cell_size_m, y), (x+cell_size_m, y+cell_size_m), (x, y+cell_size_m)])
            if poly_utm.intersects(cell):
                cells.append(cell.intersection(poly_utm))
            y += cell_size_m
        x += cell_size_m
    fishnet = gpd.GeoDataFrame({'geometry': cells}, crs=utm_crs)
    fishnet = fishnet.to_crs("EPSG:4326")
    return fishnet

def interpolate_to_fishnet(fishnet, points_df, value_col):
    """
    Given a fishnet GeoDataFrame and a DataFrame of points with columns 'lon', 'lat' and value_col,
    interpolate the values (using linear griddata interpolation) to the centroids of the fishnet cells.
    """
    # Compute centroids
    fishnet = fishnet.copy()
    fishnet["centroid"] = fishnet.geometry.centroid
    fishnet["lon"] = fishnet.centroid.x
    fishnet["lat"] = fishnet.centroid.y
    coords = points_df[['lon', 'lat']].values
    values = points_df[value_col].values
    fishnet_coords = fishnet[['lon', 'lat']].values
    interpolated_values = griddata(coords, values, fishnet_coords, method='linear')
    fishnet[value_col] = interpolated_values
    return fishnet.drop(columns=["centroid", "lon", "lat"])

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
    with st.sidebar:
        st.header("Spatial and Crop Parameters")
        spacing = st.slider("Grid spacing (km)", 0.1, 5.0, 1.0)
        cell_size_m = st.number_input("Fishnet Cell Size (m)", min_value=10, max_value=500, value=100, step=10)
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
    # Draw polygon using folium
    col1, col2 = st.columns([3, 1])
    with col1:
        m = folium.Map(location=[40, -100], zoom_start=4, tiles="Esri.WorldImagery")
        Draw(export=True, draw_options={"polygon": True, "marker": False, "circlemarker": False}).add_to(m)
        map_data = st_folium(m, key="spatial_map", height=600)
    with col2:
        st.info("Draw your field polygon on the map and then click 'Run Spatial Analysis'.")
    if st.button("🚀 Run Spatial Analysis"):
        if map_data and map_data.get("all_drawings"):
            try:
                shapes = [shape(d["geometry"]) for d in map_data["all_drawings"] if d["geometry"]["type"] == "Polygon"]
                if not shapes:
                    st.error("No polygon drawn.")
                    st.stop()
                # Combine drawn polygons into one study area
                study_area = gpd.GeoSeries(shapes).unary_union
                polygon_area_ha = calculate_polygon_area_ha(study_area)
                st.markdown("### Your Field Boundary")
                st.write(f"Field Area: {polygon_area_ha:.2f} hectares")
                # Create sample grid points within the polygon
                grid_points = create_grid_in_polygon(study_area, spacing=spacing)
                if not grid_points:
                    st.warning("No grid points generated! Check your polygon and spacing.")
                    st.stop()
                st.write(f"Generated {len(grid_points)} sample points within the field.")
                grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
                grid_gdf["lat"] = grid_gdf.geometry.y
                grid_gdf["lon"] = grid_gdf.geometry.x
                st.map(grid_gdf[["lat", "lon"]])
                
                st.subheader("Running Daily Simulations at Grid Points")
                point_results_all = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                # Get centroid for weather data
                centroid = study_area.centroid
                centroid_lat, centroid_lon = centroid.y, centroid.x
                centroid_weather = fetch_weather_data(centroid_lat, centroid_lon, start_date, end_date)
                if centroid_weather is None:
                    st.error("Failed to fetch weather data for the field centroid.")
                    st.stop()
                # For each grid point, run the water balance simulation (daily outputs)
                for i, point in enumerate(grid_points):
                    lat_pt, lon_pt = point.y, point.x
                    status_text.text(f"Processing point {i+1} of {len(grid_points)} ({lat_pt:.4f}, {lon_pt:.4f})")
                    weather = centroid_weather.copy()
                    soil = fetch_soil_data(lat_pt, lon_pt)
                    crop = pd.DataFrame({
                        "Start_Day": [1],
                        "End_Day": [len(weather)],
                        "Kcb": [kcb],
                        "Root_Depth_mm": [root_depth],
                        "p": [p_factor],
                        "Ke": [ke_factor]
                    })
                    if soil is not None:
                        result = SIMdualKc(weather, crop, soil)
                        result['lat'] = lat_pt
                        result['lon'] = lon_pt
                        point_results_all.append(result)
                    else:
                        st.warning(f"Skipping point ({lat_pt:.4f}, {lon_pt:.4f}) due to missing soil data.")
                    progress_bar.progress((i+1)/len(grid_points))
                if not point_results_all:
                    st.error("No simulation results generated!")
                    st.stop()
                # Combine daily results for all points into one DataFrame
                all_results_df = pd.concat(point_results_all)
                # Determine the output variable column based on user selection
                if output_variable == "Evaporation":
                    value_col = "ETa_evap (mm)"
                elif output_variable == "Transpiration":
                    value_col = "ETa_transp (mm)"
                elif output_variable == "Evapotranspiration":
                    value_col = "ETa_total (mm)"
                else:
                    value_col = "SW_root (mm)"
                
                st.subheader("Interpolating to Create Daily Spatial Outputs")
                # Create a fishnet (grid) covering the study area using the specified cell size
                fishnet = create_fishnet(study_area, cell_size_m=cell_size_m)
                daily_gdfs = []
                unique_dates = sorted(all_results_df['Date'].unique())
                for d in unique_dates:
                    points_day = all_results_df[all_results_df['Date'] == d][['lon', 'lat', value_col]]
                    # Interpolate simulation values from grid points to fishnet cells
                    fishnet_day = interpolate_to_fishnet(fishnet.copy(), points_day, value_col)
                    fishnet_day['Date'] = d
                    daily_gdfs.append(fishnet_day)
                daily_spatial_gdf = gpd.GeoDataFrame(pd.concat(daily_gdfs), crs="EPSG:4326")
                
                # Allow the user to select a day to visualize
                st.subheader("Visualize Daily Spatial Output")
                selected_date = st.selectbox("Select Date", unique_dates)
                selected_gdf = daily_spatial_gdf[daily_spatial_gdf['Date'] == selected_date]
                
                # Create a folium map with choropleth display using a color scale
                m_daily = folium.Map(location=[study_area.centroid.y, study_area.centroid.x], zoom_start=12, tiles="CartoDB positron")
                # Create a colormap
                min_val = float(selected_gdf[value_col].min())
                max_val = float(selected_gdf[value_col].max())
                colormap = branca.colormap.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=min_val, vmax=max_val)
                def style_function(feature):
                    val = feature['properties'][value_col]
                    return {
                        'fillColor': colormap(val) if val is not None else '#gray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                folium.GeoJson(
                    selected_gdf.__geo_interface__,
                    style_function=style_function,
                    tooltip=folium.GeoJsonTooltip(fields=[value_col])
                ).add_to(m_daily)
                colormap.caption = output_variable
                colormap.add_to(m_daily)
                st_folium(m_daily, width=800, height=500)
                
                # Provide a download button for the full daily spatial GeoJSON
                geojson_str = daily_spatial_gdf.to_json()
                st.download_button("📥 Download Daily Spatial Results (GeoJSON)", geojson_str, file_name="daily_spatial_results.geojson", mime="application/json")
            except Exception as e:
                st.error(f"Spatial analysis failed: {str(e)}")
                st.exception(e)
        else:
            st.warning("Please draw a polygon on the map first!")
