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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------
# Configure Requests Session with Retries
# -------------------
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

# -------------------
# App Configuration
# -------------------
st.set_page_config(page_title="Spatial Soil Water Balance", layout="wide")

# -------------------
# Core Simulation Functions
# -------------------
def compute_Ks(depletion, TAW, p):
    """
    Compute water stress coefficient (Ks)
    
    Parameters:
    - depletion: Current depletion in root zone (mm)
    - TAW: Total available water in root zone (mm)
    - p: Depletion fraction for no stress
    
    Returns:
    - Ks: Water stress coefficient (0-1)
    """
    RAW = p * TAW
    if depletion <= RAW:
        return 1.0
    elif depletion >= TAW:
        return 0.0
    else:
        return (TAW - depletion) / (TAW - RAW)

def compute_Kr(depletion, TEW, REW):
    """
    Compute evaporation reduction coefficient (Kr)
    
    Parameters:
    - depletion: Current depletion in surface layer (mm)
    - TEW: Total evaporable water (mm)
    - REW: Readily evaporable water (mm)
    
    Returns:
    - Kr: Evaporation reduction coefficient (0-1)
    """
    if depletion <= REW:
        return 1.0
    elif depletion >= TEW:
        return 0.0
    else:
        return (TEW - depletion) / (TEW - REW)

def compute_ETc(Kcb, Ks, Kr, Ke, ET0):
    """
    Compute crop evapotranspiration (ETc)
    
    Parameters:
    - Kcb: Basal crop coefficient
    - Ks: Water stress coefficient
    - Kr: Evaporation reduction coefficient
    - Ke: Evaporation coefficient
    - ET0: Reference evapotranspiration (mm)
    
    Returns:
    - ETc: Crop evapotranspiration (mm)
    """
    return (Kcb * Ks + Kr * Ke) * ET0

def interpolate_crop_stages(crop_df, total_days):
    """
    Interpolate crop parameters for each day of the simulation
    
    Parameters:
    - crop_df: DataFrame with crop stage data
    - total_days: Total number of simulation days
    
    Returns:
    - Interpolated lists of Kcb, root depth, p, and Ke values
    """
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
    """
    Dual Kc soil water balance simulation
    
    Parameters:
    - weather_df: DataFrame with weather data
    - crop_df: DataFrame with crop stage data
    - soil_df: DataFrame with soil layer data
    - track_drain: Whether to track drainage
    - enable_yield: Whether to enable yield estimation
    - use_fao33: Whether to use FAO-33 method for yield estimation
    - Ym: Maximum yield (ton/ha)
    - Ky: Yield response factor
    - use_transp: Whether to use transpiration-based method for yield estimation
    - WP_yield: Yield water productivity (ton/ha per mm)
    - enable_leaching: Whether to enable leaching estimation
    - leaching_method: Method for leaching estimation
    - nitrate_conc: Nitrate concentration (mg/L)
    - total_N_input: Total N input (kg/ha)
    - leaching_fraction: Leaching fraction (0-1)
    
    Returns:
    - DataFrame with daily water balance results
    """
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
        
        # Calculate water content in root zone
        for j, soil in soil_df.iterrows():
            if cum_depth >= RD:
                break
            d = min(soil['Depth_mm'], RD - cum_depth)
            FC_total += soil['FC'] * d
            WP_total += soil['WP'] * d
            SW_root += (SW_layers[j] / soil['Depth_mm']) * d
            cum_depth += d
        
        # Calculate stress coefficients
        TAW = FC_total - WP_total
        depletion = FC_total - SW_root
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        
        # Calculate evapotranspiration components
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0
        
        # Update cumulative values
        cum_ETc += ETc
        cum_ETa += (ETa_transp + ETa_evap)
        cum_transp += ETa_transp
        cum_evap += ETa_evap
        
        if Ks < 1.0:
            stress_days += 1
        
        # Update surface water content
        SW_surface += P + I
        excess_surface = max(0, SW_surface - TEW)
        SW_surface = max(0, SW_surface - ETa_evap)
        
        # Update soil layer water contents
        water = excess_surface
        for j, soil in soil_df.iterrows():
            max_SW = soil['FC'] * soil['Depth_mm']
            SW_layers[j] += water
            drain = max(0, SW_layers[j] - max_SW)
            cum_drain += drain
            SW_layers[j] = min(SW_layers[j], max_SW)
            water = drain
            
            # Extract transpiration from root zone
            if cum_depth < RD:
                transp = ETa_transp * (soil['Depth_mm'] / RD)
                SW_layers[j] -= transp
                SW_layers[j] = max(soil['WP'] * soil['Depth_mm'], SW_layers[j])
        
        # Store daily results
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
    
    # Add yield estimation if enabled
    if enable_yield:
        if use_fao33:
            ETc_total = results_df['ETc (mm)'].sum()
            ETa_total = results_df['ETa_total (mm)'].sum()
            Ya = Ym * (1 - Ky * (1 - ETa_total / ETc_total))
            results_df['Yield (ton/ha)'] = Ya
        if use_transp:
            Ya_transp = WP_yield * cum_transp
            results_df['Yield (ton/ha)'] = Ya_transp
    
    # Add leaching estimation if enabled
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
    """
    Fetch weather data from NASA POWER API
    
    Parameters:
    - lat: Latitude
    - lon: Longitude
    - start_date: Start date
    - end_date: End date
    
    Returns:
    - DataFrame with daily weather data
    """
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
            
            # Extract weather parameters
            Tmax = data['T2M_MAX'][date_str]
            Tmin = data['T2M_MIN'][date_str]
            Tmean = (Tmax + Tmin) / 2
            Rs = data['ALLSKY_SFC_SW_DWN'][date_str]
            u2 = data['WS2M'][date_str]
            RH = data['RH2M'][date_str]
            
            # Calculate ET0 using FAO Penman-Monteith equation
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
    """
    Fetch soil data from ISRIC SoilGrids API
    
    Parameters:
    - lat: Latitude
    - lon: Longitude
    
    Returns:
    - DataFrame with soil layer data
    """
    max_retries = 3
    retry_delay = 1  # seconds
    timeout = 5
    
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
                                   headers={'User-Agent': 'SpatialSoilWaterBalance/1.0'})
            response.raise_for_status()
            data = response.json()
            properties = data['properties']
            
            layers = []
            for depth in ['0-5cm', '5-15cm']:
                bdod = properties.get('bdod', {}).get(depth, {}).get('mean', 140) / 100
                sand = properties.get('sand', {}).get(depth, {}).get('mean', 40)
                clay = properties.get('clay', {}).get(depth, {}).get('mean', 20)
                ocd = properties.get('ocd', {}).get(depth, {}).get('mean', 1.0) / 100
                
                # Calculate field capacity and wilting point using pedotransfer functions
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
            
            st.warning(f"Soil data fetch failed: {str(e)[:200]}... Using default values.")
            return pd.DataFrame({
                "Depth_mm": [200, 100],
                "FC": [0.30, 0.30],
                "WP": [0.15, 0.15],
                "TEW": [200, 0],
                "REW": [50, 0]
            })

def create_grid_in_polygon(polygon, spacing=0.01):
    """
    Create a grid of points within a polygon
    
    Parameters:
    - polygon: Shapely polygon
    - spacing: Grid spacing in degrees (approximately km)
    
    Returns:
    - List of points within the polygon
    """
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    polygon_utm = gdf_utm.iloc[0].geometry
    
    bounds = polygon_utm.bounds
    spacing_m = spacing * 1000  # Convert to meters
    
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

# -------------------
# Main Application
# -------------------
def main():
    st.title("Spatial Soil Water Balance")
    st.markdown("**Spatiotemporal Soil Water Balance Modeling with Evaporation, Transpiration, and Evapotranspiration Outputs**")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Simulation Parameters")
        spacing = st.slider("Grid spacing (km)", 0.1, 5.0, 1.0)
        start_date = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=30))
        end_date = st.date_input("End date", datetime.date.today())
        
        st.header("Crop Parameters")
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
        
        st.header("Output Options")
        output_variable = st.selectbox(
            "Select Primary Output Variable", 
            ["Evaporation", "Transpiration", "Evapotranspiration", "Soil Water Content"]
        )
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🌍 Draw Field Boundary")
        m = folium.Map(location=[40, -100], zoom_start=4, tiles="Esri.WorldImagery")
        Draw(export=True, draw_options={"polygon": True, "marker": False, "circlemarker": False}).add_to(m)
        map_data = st_folium(m, key="spatial_map", height=600)
    
    with col2:
        st.markdown("### Instructions")
        st.info("""
        1. Draw a polygon on the map to define your field boundary
        2. Adjust parameters in the sidebar
        3. Click 'Run Spatial Analysis' to calculate water balance
        4. View results in the tabs below
        """)
    
    if st.button("🚀 Run Spatial Analysis"):
        if map_data and map_data.get("all_drawings"):
            try:
                shapes = [shape(d["geometry"]) for d in map_data["all_drawings"] if d["geometry"]["type"] == "Polygon"]
                if not shapes:
                    st.error("No polygon drawn.")
                    st.stop()
                
                # Use union_all() instead of deprecated unary_union
                study_area = gpd.GeoSeries(shapes).union_all()
                
                st.markdown("### Your Field Boundary")
                st.write(f"Area: {study_area.area * 111**2:.2f} km²")
                
                # Create grid points within the field boundary
                grid_points = create_grid_in_polygon(study_area, spacing=spacing)
                
                if not grid_points:
                    st.warning("No grid points generated! Check your polygon and spacing.")
                    st.stop()
                
                st.write(f"Generated {len(grid_points)} sample points within the field.")
                
                # Create GeoDataFrame from grid points
                grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
                grid_gdf["lat"] = grid_gdf.geometry.y
                grid_gdf["lon"] = grid_gdf.geometry.x
                
                # Display grid points on map
                st.map(grid_gdf[["lat", "lon"]])
                
                # Run spatial analysis
                st.subheader("Running Spatial Analysis")
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, point in enumerate(grid_points):
                    lat_pt, lon_pt = point.y, point.x
                    status_text.text(f"Processing point {i+1} of {len(grid_points)} ({lat_pt:.4f}, {lon_pt:.4f})")
                    
                    # Fetch weather and soil data
                    weather = fetch_weather_data(lat_pt, lon_pt, start_date, end_date)
                    soil = fetch_soil_data(lat_pt, lon_pt)
                    
                    # Create crop data
                    crop = pd.DataFrame({
                        "Start_Day": [1],
                        "End_Day": [len(weather)] if weather is not None else [30],
                        "Kcb": [kcb],
                        "Root_Depth_mm": [root_depth],
                        "p": [p_factor],
                        "Ke": [ke_factor]
                    })
                    
                    if weather is not None and soil is not None:
                        # Run water balance simulation
                        result = SIMdualKc(weather, crop, soil)
                        
                        # Extract results for the last day
                        final_row = result.iloc[-1]
                        
                        # Calculate average values
                        mean_transp = result["ETa_transp (mm)"].mean()
                        mean_evap = result["ETa_evap (mm)"].mean()
                        mean_et = result["ETa_total (mm)"].mean()
                        
                        # Calculate cumulative values
                        cum_transp = final_row["Cumulative_Transp (mm)"]
                        cum_evap = final_row["Cumulative_Evap (mm)"]
                        cum_et = final_row["Cumulative_ETa (mm)"]
                        
                        # Store results
                        results.append({
                            "lat": lat_pt,
                            "lon": lon_pt,
                            "SW_root (mm)": final_row["SW_root (mm)"],
                            "Mean_Transp (mm/day)": mean_transp,
                            "Mean_Evap (mm/day)": mean_evap,
                            "Mean_ET (mm/day)": mean_et,
                            "Cum_Transp (mm)": cum_transp,
                            "Cum_Evap (mm)": cum_evap,
                            "Cum_ET (mm)": cum_et
                        })
                    else:
                        st.warning(f"Skipping point ({lat_pt:.4f}, {lon_pt:.4f}) due to missing data.")
                    
                    progress_bar.progress((i+1)/len(grid_points))
                
                if results:
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.markdown("## 📊 Spatial Results")
                    tab1, tab2, tab3, tab4 = st.tabs(["Map Visualization", "Data Table", "Statistics", "Export Data"])
                    
                    with tab1:
                        st.markdown("### Spatial Distribution Maps")
                        
                        # Select output variable for visualization
                        if output_variable == "Evaporation":
                            value_col = "Mean_Evap (mm/day)"
                            cum_col = "Cum_Evap (mm)"
                            title = "Evaporation"
                        elif output_variable == "Transpiration":
                            value_col = "Mean_Transp (mm/day)"
                            cum_col = "Cum_Transp (mm)"
                            title = "Transpiration"
                        elif output_variable == "Evapotranspiration":
                            value_col = "Mean_ET (mm/day)"
                            cum_col = "Cum_ET (mm)"
                            title = "Evapotranspiration"
                        else:
                            value_col = "SW_root (mm)"
                            cum_col = None
                            title = "Soil Water Content"
                        
                        # Create map for daily values
                        st.subheader(f"Daily {title} (mm/day)")
                        m_daily = folium.Map(
                            location=[results_df.lat.mean(), results_df.lon.mean()], 
                            zoom_start=10,
                            tiles="CartoDB positron"
                        )
                        
                        # Add heatmap
                        heat_data = [[float(r[0]), float(r[1]), float(r[2])] 
                                     for r in results_df[['lat', 'lon', value_col]].values.tolist()]
                        HeatMap(data=heat_data, radius=12, blur=20).add_to(m_daily)
                        
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
                        
                        st_folium(m_daily, width=800, height=400)
                        
                        # Create map for cumulative values if applicable
                        if cum_col:
                            st.subheader(f"Cumulative {title} (mm)")
                            m_cum = folium.Map(
                                location=[results_df.lat.mean(), results_df.lon.mean()], 
                                zoom_start=10,
                                tiles="CartoDB positron"
                            )
                            
                            # Add heatmap
                            heat_data = [[float(r[0]), float(r[1]), float(r[2])] 
                                         for r in results_df[['lat', 'lon', cum_col]].values.tolist()]
                            HeatMap(data=heat_data, radius=12, blur=20).add_to(m_cum)
                            
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
                            
                            st_folium(m_cum, width=800, height=400)
                    
                    with tab2:
                        st.markdown("### Raw Results Data")
                        st.dataframe(results_df, height=400)
                    
                    with tab3:
                        st.markdown("### Statistical Summary")
                        
                        # Calculate statistics
                        stats_df = pd.DataFrame({
                            "Statistic": ["Mean", "Median", "Min", "Max", "Std Dev"],
                            "Evaporation (mm/day)": [
                                results_df["Mean_Evap (mm/day)"].mean(),
                                results_df["Mean_Evap (mm/day)"].median(),
                                results_df["Mean_Evap (mm/day)"].min(),
                                results_df["Mean_Evap (mm/day)"].max(),
                                results_df["Mean_Evap (mm/day)"].std()
                            ],
                            "Transpiration (mm/day)": [
                                results_df["Mean_Transp (mm/day)"].mean(),
                                results_df["Mean_Transp (mm/day)"].median(),
                                results_df["Mean_Transp (mm/day)"].min(),
                                results_df["Mean_Transp (mm/day)"].max(),
                                results_df["Mean_Transp (mm/day)"].std()
                            ],
                            "Evapotranspiration (mm/day)": [
                                results_df["Mean_ET (mm/day)"].mean(),
                                results_df["Mean_ET (mm/day)"].median(),
                                results_df["Mean_ET (mm/day)"].min(),
                                results_df["Mean_ET (mm/day)"].max(),
                                results_df["Mean_ET (mm/day)"].std()
                            ],
                            "Soil Water Content (mm)": [
                                results_df["SW_root (mm)"].mean(),
                                results_df["SW_root (mm)"].median(),
                                results_df["SW_root (mm)"].min(),
                                results_df["SW_root (mm)"].max(),
                                results_df["SW_root (mm)"].std()
                            ]
                        })
                        
                        st.dataframe(stats_df)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        x = ["Evaporation", "Transpiration", "Evapotranspiration"]
                        y = [
                            results_df["Mean_Evap (mm/day)"].mean(),
                            results_df["Mean_Transp (mm/day)"].mean(),
                            results_df["Mean_ET (mm/day)"].mean()
                        ]
                        ax.bar(x, y, color=['blue', 'green', 'red'])
                        ax.set_ylabel("Mean Rate (mm/day)")
                        ax.set_title("Average Water Balance Components")
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        for i, v in enumerate(y):
                            ax.text(i, v + 0.05, f"{v:.2f}", ha='center')
                        
                        st.pyplot(fig)
                    
                    with tab4:
                        st.markdown("### Export Options")
                        
                        # CSV export
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV", 
                            csv, 
                            file_name="spatial_water_balance.csv", 
                            mime="text/csv"
                        )
                        
                        # GeoJSON export
                        gdf = gpd.GeoDataFrame(
                            results_df,
                            geometry=gpd.points_from_xy(results_df.lon, results_df.lat),
                            crs="EPSG:4326"
                        )
                        geojson = gdf.to_json()
                        st.download_button(
                            "🌐 Download GeoJSON", 
                            geojson, 
                            file_name="spatial_results.geojson", 
                            mime="application/json"
                        )
                else:
                    st.warning("No results generated! Check input data sources.")
            except Exception as e:
                st.error(f"Spatial analysis failed: {str(e)}")
        else:
            st.warning("Please draw a polygon on the map first!")

if __name__ == "__main__":
    main()
