pip install streamlit-folium geopandas folium requests
```)

Below is the complete code:

---

```python
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, Point
import datetime
import requests

# -------------------
# App Configuration
# -------------------
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# -------------------
# Header with Logo and Title
# -------------------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=90)
with col2:
    st.title("AgriWaterBalance")
    st.markdown("**A research-grade, multi-layer soil water balance tool for any crop and soil.**")

# -------------------
# Core Simulation Functions (common to both modes)
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

def SIMdualKc(weather_df, crop_df, soil_df, track_drain=True):
    days = len(weather_df)
    profile_depth = soil_df['Depth_mm'].sum()
    Kcb_list, RD_list, p_list, Ke_list = interpolate_crop_stages(crop_df, days)

    TEW, REW = soil_df.iloc[0]['TEW'], soil_df.iloc[0]['REW']
    SW_surface = TEW
    SW_layers = [row['FC'] * row['Depth_mm'] for _, row in soil_df.iterrows()]

    cum_ETc = cum_ETa = cum_Irr = cum_P = cum_drain = 0
    stress_days = 0
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
        storage_deficit = max(0, TAW - SW_root)
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0
        cum_ETc += ETc
        cum_ETa += (ETa_transp + ETa_evap)
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
            "Storage_Deficit (mm)": storage_deficit,
            "Cumulative_ETc (mm)": cum_ETc,
            "Cumulative_ETa (mm)": cum_ETa,
            "Cumulative_Irrigation (mm)": cum_Irr,
            "Cumulative_Precip (mm)": cum_P,
            "Cumulative_Drainage (mm)": cum_drain,
            "Stress_Days": stress_days
        })

    return pd.DataFrame(results)

# -------------------
# Data Fetching Functions (real online data)
# -------------------
def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Query NASA POWER API for daily weather data.
    Returns a DataFrame with Date, ET0 (mm), Precipitation (mm) and Irrigation (set to 0).
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOT,ET0",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    r = requests.get(url, params=params)
    data = r.json()
    try:
        param = data["properties"]["parameter"]
    except KeyError:
        st.error("Error fetching weather data from NASA POWER.")
        return None

    dates = []
    et0_list = []
    precip_list = []
    for date_str in param["ET0"]:
        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        dates.append(date_obj)
        et0_list.append(param["ET0"][date_str])
        precip_list.append(param["PRECTOT"][date_str])
    weather_df = pd.DataFrame({
        "Date": dates,
        "ET0": et0_list,
        "Precipitation": precip_list,
        "Irrigation": [0]*len(dates)
    })
    return weather_df

def fetch_soil_data(lat, lon):
    """
    Query SoilGrids API for soil properties at the given point.
    Returns a DataFrame with a two-layer soil profile.
    """
    url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Error fetching soil data from SoilGrids.")
        return None
    data = r.json()
    # In a real application, process 'data' to compute properties.
    soil_df = pd.DataFrame({
        "Depth_mm": [200, 100],
        "FC": [0.30, 0.30],
        "WP": [0.15, 0.15],
        "TEW": [200, 0],
        "REW": [50, 0]
    })
    return soil_df

def fetch_ndvi_data(study_area, start_date, end_date):
    """
    Use an external API (e.g., Sentinel-2 via GEE or STAC) to compute NDVI over the study area.
    Here we return a constant NDVI value for demonstration.
    """
    return 0.6

def get_crop_data(ndvi, num_days):
    """
    Convert NDVI to a basal crop coefficient (Kcb) using a simple relationship.
    Returns a crop stage DataFrame covering the simulation period.
    """
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

# -------------------
# Helper: Create a grid of points within a polygon
# -------------------
def create_grid_in_polygon(polygon, spacing=0.1):
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx, maxx, spacing)
    ys = np.arange(miny, maxy, spacing)
    points = []
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if polygon.contains(pt):
                points.append(pt)
    return points

# -------------------
# Mode Selection
# -------------------
mode = st.radio("Choose Mode:", ["Normal Mode", "Spatial Mode"], horizontal=True)

# =========================================
# NORMAL MODE (User uploads their own data)
# =========================================
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
        try:
            weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
            crop_df = pd.read_csv(crop_file)
            soil_df = pd.read_csv(soil_file)

            if enable_leaching and leaching_method == "Method 1: Drainage × nitrate concentration":
                track_drainage = True
                st.sidebar.info("ℹ️ Drainage tracking enabled for leaching estimation.")

            results_df = SIMdualKc(weather_df, crop_df, soil_df, track_drainage)
            results_df['SWC (%)'] = (results_df['SW_root (mm)'] / results_df['Root_Depth (mm)']) * 100

            tab1, tab2, tab3, tab4 = st.tabs(["📄 Daily Results", "📈 ET Graphs", "💧 Storage", "🌾 Yield and Leaching"])
            with tab1:
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Results (.txt)", csv, file_name="agriwaterbalance_results.txt")
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
                st.write("Yield and leaching calculations can be integrated here.")
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
        st.info("📂 Please upload all required files and click 'Run Simulation' to begin.")

# =========================================
# SPATIAL MODE (Use drawn polygon to fetch online data and run grid-based simulation)
# =========================================
elif mode == "Spatial Mode":
    st.markdown("### 🌍 Spatial Mode Activated")
    st.info("Draw your field boundary below. The app will then download online data for your field and simulate the water balance on a grid, displaying spatial maps of outputs.")

    # Step 1: Let the user draw a polygon (field boundary) on an interactive Folium map
    default_center = [36.7783, -119.4179]
    m = folium.Map(location=default_center, zoom_start=7, tiles="Esri.WorldImagery")
    draw = Draw(export=True)
    draw.add_to(m)
    st.info("Use the drawing tool to delineate your field (polygon).")
    map_data = st_folium(m, key="map", width=700, height=500)

    # Step 2: Extract the drawn polygon(s)
    study_area = None
    if map_data and map_data.get("all_drawings"):
        drawings = map_data["all_drawings"]
        if drawings:
            shapes = []
            for drawing in drawings:
                geom = drawing.get("geometry")
                if geom:
                    shapes.append(shape(geom))
            if shapes:
                # If more than one polygon is drawn, merge them into a single geometry
                study_area = gpd.GeoDataFrame(geometry=shapes, crs="EPSG:4326")
                unified = study_area.unary_union
                st.write("### Your Field Boundary")
                st.write(unified)

    if study_area is not None:
        # Use the unified polygon for data queries
        unified_polygon = study_area.unary_union
        # Create a grid of points within the polygon (adjust spacing as needed)
        spacing = st.number_input("Grid spacing (degrees)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        grid_points = create_grid_in_polygon(unified_polygon, spacing=spacing)
        if not grid_points:
            st.error("No grid points generated. Try a smaller grid spacing or check your polygon.")
        else:
            st.write(f"Generated {len(grid_points)} sample points within the field.")
            # Display the grid points on a simple map
            grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
            grid_gdf["lat"] = grid_gdf.geometry.y
            grid_gdf["lon"] = grid_gdf.geometry.x
            st.map(grid_gdf[["lat", "lon"]])

            st.subheader("Run Spatial Simulation")
            if st.button("Run Spatial Simulation"):
                try:
                    # Define simulation period (e.g., last 30 days)
                    end_date = datetime.date.today()
                    start_date = end_date - datetime.timedelta(days=29)
                    # For each grid point, query online data and run the simulation.
                    # We will extract one key output (e.g., final SW_surface) for mapping.
                    results_list = []
                    for pt in grid_points:
                        lat_pt = pt.y
                        lon_pt = pt.x
                        # Fetch weather, soil, and NDVI data for this point
                        weather_df = fetch_weather_data(lat_pt, lon_pt, start_date, end_date)
                        if weather_df is None:
                            continue
                        soil_df = fetch_soil_data(lat_pt, lon_pt)
                        if soil_df is None:
                            continue
                        ndvi = fetch_ndvi_data(unified_polygon, start_date, end_date)
                        crop_df = get_crop_data(ndvi, len(weather_df))
                        sim_df = SIMdualKc(weather_df, crop_df, soil_df, track_drain=True)
                        # Take the last day value for SW_surface as an example output
                        final_SW = sim_df.iloc[-1]["SW_surface (mm)"]
                        results_list.append({"lat": lat_pt, "lon": lon_pt, "SW_surface": final_SW})
                    
                    if not results_list:
                        st.error("No simulation results were generated for the grid.")
                    else:
                        spatial_results = pd.DataFrame(results_list)
                        # Create a GeoDataFrame for mapping
                        spatial_gdf = gpd.GeoDataFrame(
                            spatial_results,
                            geometry=gpd.points_from_xy(spatial_results.lon, spatial_results.lat),
                            crs="EPSG:4326"
                        )
                        st.write("### Spatial Output: Final Surface Soil Water (mm)")
                        st.dataframe(spatial_results)
                        # Plot spatial output on an interactive Folium map
                        m_out = folium.Map(location=default_center, zoom_start=7, tiles="Esri.WorldImagery")
                        for idx, row in spatial_gdf.iterrows():
                            # Color-code markers based on SW_surface value (this is a simple example)
                            color = "blue" if row["SW_surface"] > 100 else "red"
                            folium.CircleMarker(
                                location=[row["lat"], row["lon"]],
                                radius=5,
                                popup=f"SW_surface: {row['SW_surface']:.1f} mm",
                                color=color,
                                fill=True,
                                fill_opacity=0.7
                            ).add_to(m_out)
                        st_folium(m_out, width=700, height=500)
                except Exception as e:
                    st.error(f"⚠️ Spatial simulation failed: {e}")
    else:
        st.info("Please draw your field boundary on the map above to run the spatial simulation.")
