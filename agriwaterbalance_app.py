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
from io import BytesIO
import math

# -------------------
# App Configuration
# -------------------
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# -------------------
# Core Simulation Functions
# -------------------
def compute_Ks(depletion, TAW, p):
    """Compute water stress coefficient (Ks)."""
    RAW = p * TAW
    if depletion <= RAW:
        return 1.0
    elif depletion >= TAW:
        return 0.0
    else:
        return (TAW - depletion) / (TAW - RAW)

def compute_Kr(depletion, TEW, REW):
    """Compute evaporation reduction coefficient (Kr)."""
    if depletion <= REW:
        return 1.0
    elif depletion >= TEW:
        return 0.0
    else:
        return (TEW - depletion) / (TEW - REW)

def compute_ETc(Kcb, Ks, Kr, Ke, ET0):
    """Compute crop evapotranspiration (ETc)."""
    return (Kcb * Ks + Kr * Ke) * ET0

def interpolate_crop_stages(crop_df, total_days):
    """Interpolate crop parameters over simulation days."""
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
    """Run SIMDualKc water balance simulation with yield and leaching options."""
    days = len(weather_df)
    profile_depth = soil_df['Depth_mm'].sum()
    Kcb_list, RD_list, p_list, Ke_list = interpolate_crop_stages(crop_df, days)

    TEW, REW = soil_df.iloc[0]['TEW'], soil_df.iloc[0]['REW']
    SW_surface = TEW
    SW_layers = [row['FC'] * row['Depth_mm'] for _, row in soil_df.iterrows()]

    cum_ETc = cum_ETa = cum_Irr = cum_P = cum_drain = 0
    stress_days = 0
    cum_transp = 0
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
            "SW_surface (mm)": SW_surface,
            "SW_root (mm)": SW_root,
            "Root_Depth (mm)": RD,
            "Depletion (mm)": depletion,
            "TAW (mm)": TAW,
            "Cumulative_ETc (mm)": cum_ETc,
            "Cumulative_ETa (mm)": cum_ETa,
            "Cumulative_Irrigation (mm)": cum_Irr,
            "Cumulative_Precip (mm)": cum_P,
            "Cumulative_Drainage (mm)": cum_drain,
            "Stress_Days": stress_days
        })

    results_df = pd.DataFrame(results)

    # Yield calculations
    if enable_yield:
        if use_fao33:
            ETc_total = results_df['ETc (mm)'].sum()
            ETa_total = results_df['ETa_transp (mm)'].sum() + results_df['ETa_evap (mm)'].sum()
            Ya = Ym * (1 - Ky * (1 - ETa_total / ETc_total))
            results_df['Yield (ton/ha)'] = Ya
        if use_transp:
            Ya_transp = WP_yield * cum_transp
            results_df['Yield (ton/ha)'] = Ya_transp

    # Leaching calculations
    if enable_leaching:
        if leaching_method == "Method 1: Drainage × nitrate concentration":
            leaching = cum_drain * nitrate_conc / 1000  # Convert mg/L to kg/ha
            results_df['Leaching (kg/ha)'] = leaching
        elif leaching_method == "Method 2: Leaching Fraction × total N input":
            leaching = leaching_fraction * total_N_input
            results_df['Leaching (kg/ha)'] = leaching

    return results_df

# -------------------
# Data Fetching Functions
# -------------------
def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetch weather data from NASA POWER API with ET0 calculation."""
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORG,WS2M,RH2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    
    try:
        response = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params)
        response.raise_for_status()
        data = response.json()['properties']['parameter']
        
        # Calculate reference evapotranspiration (FAO-56 Penman-Monteith)
        dates = []
        et0_list = []
        for date_str in data['T2M_MAX']:
            dates.append(datetime.datetime.strptime(date_str, "%Y%m%d"))
            Tmax = data['T2M_MAX'][date_str]
            Tmin = data['T2M_MIN'][date_str]
            Tmean = (Tmax + Tmin) / 2
            Rs = data['ALLSKY_SFC_SW_DWN'][date_str]  # Solar radiation (MJ/m²/day)
            u2 = data['WS2M'][date_str]               # Wind speed at 2m (m/s)
            RH = data['RH2M'][date_str]               # Relative humidity (%)
            
            # ET0 calculation steps
            delta = 4098 * (0.6108 * math.exp((17.27 * Tmean)/(Tmean + 237.3))) / (Tmean + 237.3)**2
            P = 101.3 * ((293 - 0.0065 * 0) / 293)**5.26  # Atmospheric pressure
            gamma = 0.000665 * P
            es = (0.6108 * math.exp(17.27 * Tmax/(Tmax + 237.3)) + 0.6108 * math.exp(17.27 * Tmin/(Tmin + 237.3))) / 2
            ea = es * RH / 100
            
            ET0 = (0.408 * delta * (Rs - 0) + gamma * (900/(Tmean + 273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
            et0_list.append(ET0)
        
        weather_df = pd.DataFrame({
            "Date": dates,
            "ET0": et0_list,
            "Precipitation": [data['PRECTOTCORG'][d] for d in data['PRECTOTCORG']],
            "Irrigation": [0] * len(dates)
        })
        return weather_df
    except Exception as e:
        st.error(f"Weather data fetch failed: {str(e)}")
        return None

def fetch_soil_data(lat, lon):
    """Fetch soil data from SoilGrids API."""
    try:
        url = f"https://rest.soilgrids.org/soilgrids/v2.0/properties/query?"
        params = {
            'lon': lon,
            'lat': lat,
            'property': 'bdod,sand,silt,clay,ocd',
            'depth': '0-5cm,5-15cm',
            'value': 'mean'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process soil properties
        properties = data['properties']
        layers = []
        for depth in ['0-5cm', '5-15cm']:
            bdod = properties['bdod'][depth]['mean'] / 100  # kg/dm³
            sand = properties['sand'][depth]['mean']
            clay = properties['clay'][depth]['mean']
            ocd = properties['ocd'][depth]['mean'] / 100  # %
            
            # Calculate field capacity and wilting point using Saxton equations
            FC = (-0.251 * sand/100 + 0.195 * clay/100 + 0.011 * ocd +
                0.006 * (sand/100) * ocd - 0.027 * (clay/100) * ocd +
                0.452 * (sand/100) * (clay/100) + 0.299) * bdod
            WP = (-0.024 * sand/100 + 0.487 * clay/100 + 0.006 * ocd +
                0.005 * (sand/100) * ocd - 0.013 * (clay/100) * ocd +
                0.068 * (sand/100) * (clay/100) + 0.031) * bdod
            
            layers.append({
                "Depth_mm": 50 if depth == '0-5cm' else 100,
                "FC": FC,
                "WP": WP,
                "TEW": 200 if depth == '0-5cm' else 0,
                "REW": 50 if depth == '0-5cm' else 0
            })
        
        return pd.DataFrame(layers)
    except Exception as e:
        st.warning(f"Soil data fetch failed: {str(e)}. Using default values.")
        return pd.DataFrame({
            "Depth_mm": [200, 100],
            "FC": [0.30, 0.30],
            "WP": [0.15, 0.15],
            "TEW": [200, 0],
            "REW": [50, 0]
        })

def create_grid_in_polygon(polygon, spacing=0.01):
    """Create grid points within a polygon with UTM projection."""
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
    bounds = gdf_utm.total_bounds
    x_coords = np.arange(bounds[0], bounds[2], spacing*1000)  # Convert degrees to meters
    y_coords = np.arange(bounds[1], bounds[3], spacing*1000)
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            point = gpd.GeoSeries([Point(x, y)], crs=gdf_utm.crs).to_crs(4326)[0]
            if polygon.contains(point):
                grid_points.append(point)
    return grid_points

# -------------------
# User Interface
# -------------------
st.title("AgriWaterBalance")
st.markdown("**Spatiotemporal Soil Water Balance Modeling**")

# Mode selection
mode = st.radio("Operation Mode:", ["Normal Mode", "Spatial Mode"], horizontal=True)

if mode == "Normal Mode":
    # [Keep original Normal Mode implementation here]
    # (Same as in user's original code)

elif mode == "Spatial Mode":
    st.markdown("### 🌍 Spatial Analysis Mode")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        m = folium.Map(location=[40, -100], zoom_start=4)
        Draw(export=True, draw_options={"polygon": True, "marker": False, "circlemarker": False}).add_to(m)
        map_data = st_folium(m, key="spatial_map", height=600)
    
    with col2:
        st.markdown("### Spatial Parameters")
        spacing = st.slider("Grid spacing (km)", 0.1, 5.0, 1.0)
        start_date = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=30))
        end_date = st.date_input("End date", datetime.date.today())
        
        if st.button("🚀 Run Spatial Analysis"):
            if map_data and map_data.get("all_drawings"):
                try:
                    # Process drawn polygon
                    shapes = [shape(d["geometry"]) for d in map_data["all_drawings"] if d["geometry"]["type"] == "Polygon"]
                    study_area = gpd.GeoSeries(shapes).unary_union
                    
                    # Create grid points
                    grid_points = create_grid_in_polygon(study_area, spacing/111)
                    
                    # Run simulation for each point
                    results = []
                    progress_bar = st.progress(0)
                    for i, point in enumerate(grid_points):
                        lat, lon = point.y, point.x
                        
                        # Fetch spatial inputs
                        weather = fetch_weather_data(lat, lon, start_date, end_date)
                        soil = fetch_soil_data(lat, lon)
                        crop = pd.DataFrame({  # Simplified crop parameterization
                            "Start_Day": [1],
                            "End_Day": [len(weather)],
                            "Kcb": [1.0],
                            "Root_Depth_mm": [1000],
                            "p": [0.5],
                            "Ke": [0.8]
                        })
                        
                        # Run water balance model
                        if weather is not None and soil is not None:
                            result = SIMdualKc(weather, crop, soil)
                            final_sw = result.iloc[-1]["SW_root (mm)"]
                            results.append({"lat": lat, "lon": lon, "SW_root": final_sw})
                        
                        progress_bar.progress((i+1)/len(grid_points))
                    
                    # Visualize results
                    if results:
                        results_df = pd.DataFrame(results)
                        st.markdown("### Spatial Results")
                        
                        # Heatmap visualization
                        m_results = folium.Map(location=[results_df.lat.mean(), results_df.lon.mean()], zoom_start=10)
                        HeatMap(results_df[['lat', 'lon', 'SW_root']].values.tolist(), radius=10).add_to(m_results)
                        st_folium(m_results, width=800, height=500)
                        
                        # Data export
                        st.download_button(
                            "💾 Download Spatial Results",
                            results_df.to_csv(index=False),
                            file_name="spatial_water_balance.csv"
                        )
                except Exception as e:
                    st.error(f"Spatial analysis failed: {str(e)}")
            else:
                st.warning("Please draw a polygon on the map first!")

# -------------------
# Footer
# -------------------
st.markdown("---")
st.markdown("*Developed by Srinivasa Rao Peddinti* | *Model: Soil water balance web tool*")
