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
# Header
# -------------------
st.title("AgriWaterBalance")
st.markdown("**A research-grade, multi-layer soil water balance tool for any crop and soil.**")

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
    """Fetch weather data from NASA POWER API."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        param = data["properties"]["parameter"]
        dates, precip_list, et0_list = [], [], []
        for date_str in param["PRECTOTCORR"]:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            dates.append(date_obj)
            precip_list.append(param["PRECTOTCORR"][date_str])
            et0_list.append(5.0)  # Placeholder for ET0
        weather_df = pd.DataFrame({
            "Date": dates,
            "ET0": et0_list,
            "Precipitation": precip_list,
            "Irrigation": [0] * len(dates)
        })
        return weather_df
    except Exception as e:
        st.warning(f"Failed to fetch weather data for lat={lat}, lon={lon}: {e}")
        return None

def fetch_soil_data(lat, lon):
    """Fetch hardcoded soil data (placeholder)."""
    soil_df = pd.DataFrame({
        "Depth_mm": [200, 100],
        "FC": [0.30, 0.30],
        "WP": [0.15, 0.15],
        "TEW": [200, 0],
        "REW": [50, 0]
    })
    return soil_df

def fetch_ndvi_data(study_area, start_date, end_date):
    """Fetch placeholder NDVI value."""
    return 0.6

def get_crop_data(ndvi, num_days):
    """Generate crop data based on NDVI."""
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
# Helper: Create Grid Points
# -------------------
def create_grid_in_polygon(polygon, spacing=0.001):
    """Create grid points within a polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx, maxx, spacing)
    ys = np.arange(miny, maxy, spacing)
    points = [Point(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))]
    return points

# -------------------
# Mode Selection
# -------------------
mode = st.radio("Choose Mode:", ["Normal Mode", "Spatial Mode"], horizontal=True)

# =========================================
# NORMAL MODE
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
        with st.spinner("Running simulation..."):
            try:
                weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
                crop_df = pd.read_csv(crop_file)
                soil_df = pd.read_csv(soil_file)

                results_df = SIMdualKc(
                    weather_df, crop_df, soil_df, track_drainage,
                    enable_yield=enable_yield,
                    use_fao33=use_fao33 if 'use_fao33' in locals() else False,
                    Ym=Ym if 'Ym' in locals() else 0,
                    Ky=Ky if 'Ky' in locals() else 0,
                    use_transp=use_transp if 'use_transp' in locals() else False,
                    WP_yield=WP_yield if 'WP_yield' in locals() else 0,
                    enable_leaching=enable_leaching,
                    leaching_method=leaching_method if 'leaching_method' in locals() else "",
                    nitrate_conc=nitrate_conc if 'nitrate_conc' in locals() else 0,
                    total_N_input=total_N_input if 'total_N_input' in locals() else 0,
                    leaching_fraction=leaching_fraction if 'leaching_fraction' in locals() else 0
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

# =========================================
# SPATIAL MODE
# =========================================
elif mode == "Spatial Mode":
    st.markdown("### 🌍 Spatial Mode Activated")
    st.info("Draw your field boundary below. The app will fetch online data and simulate the water balance on a grid.")

    # Explanation of data sources
    st.markdown("### About Spatial Mode")
    st.write("In Spatial Mode, the app fetches weather data from NASA POWER API, uses hardcoded soil data, and a placeholder NDVI value to simulate soil water balance for each grid point within your drawn polygon.")

    default_center = [36.7783, -119.4179]  # Central California as default
    m = folium.Map(location=default_center, zoom_start=7, tiles="Esri.WorldImagery")
    Draw(export=True).add_to(m)
    st.info("Use the drawing tool to delineate your field (polygon).")
    map_data = st_folium(m, key="map", width=700, height=500)

    study_area = None
    if map_data and map_data.get("all_drawings"):
        drawings = map_data["all_drawings"]
        if drawings:
            shapes = [shape(drawing.get("geometry")) for drawing in drawings if drawing.get("geometry")]
            if shapes:
                study_area = gpd.GeoDataFrame(geometry=shapes, crs="EPSG:4326")
                unified = study_area.union_all()
                st.write("### Your Field Boundary")
                st.write(unified)

    if study_area is not None:
        unified_polygon = study_area.union_all()
        spacing = st.number_input("Grid spacing (degrees)", min_value=0.0001, max_value=0.5, value=0.001, step=0.0001)
        grid_points = create_grid_in_polygon(unified_polygon, spacing=spacing)
        if not grid_points:
            st.error("No grid points generated. Try a smaller grid spacing or check your polygon.")
        else:
            st.write(f"Generated {len(grid_points)} sample points within the field.")
            grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
            grid_gdf["lat"] = grid_gdf.geometry.y
            grid_gdf["lon"] = grid_gdf.geometry.x
            st.map(grid_gdf[["lat", "lon"]])

            st.subheader("Run Spatial Simulation")
            if st.button("Run Spatial Simulation"):
                with st.spinner("Running simulation..."):
                    try:
                        end_date = datetime.date.today()
                        start_date = end_date - datetime.timedelta(days=29)
                        results_list = []
                        progress_bar = st.progress(0)
                        total_points = len(grid_points)
                        for i, pt in enumerate(grid_points):
                            lat_pt = pt.y
                            lon_pt = pt.x
                            st.write(f"Processing point {i+1}/{total_points}: Lat {lat_pt:.4f}, Lon {lon_pt:.4f}")
                            weather_df = fetch_weather_data(lat_pt, lon_pt, start_date, end_date)
                            if weather_df is None:
                                st.warning(f"Skipping point {i+1} due to weather data fetch failure.")
                                continue
                            soil_df = fetch_soil_data(lat_pt, lon_pt)
                            ndvi = fetch_ndvi_data(unified_polygon, start_date, end_date)
                            crop_df = get_crop_data(ndvi, len(weather_df))
                            sim_df = SIMdualKc(weather_df, crop_df, soil_df, track_drain=True)
                            final_SW = sim_df.iloc[-1]["SW_surface (mm)"]
                            results_list.append({"lat": lat_pt, "lon": lon_pt, "SW_surface": final_SW})
                            progress_bar.progress((i + 1) / total_points)
                            st.write(f"Point {i+1} processed successfully. SW_surface: {final_SW:.1f} mm")

                        if not results_list:
                            st.error("No simulation results were generated. Please check if data is available for the selected area.")
                        else:
                            spatial_results = pd.DataFrame(results_list)
                            spatial_gdf = gpd.GeoDataFrame(
                                spatial_results,
                                geometry=gpd.points_from_xy(spatial_results.lon, spatial_results.lat),
                                crs="EPSG:4326"
                            )
                            st.write("### Spatial Output: Final Surface Soil Water (mm)")
                            st.dataframe(spatial_results)

                            m_out = folium.Map(location=default_center, zoom_start=7, tiles="Esri.WorldImagery")
                            for idx, row in spatial_gdf.iterrows():
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
                            st.success("Spatial simulation completed successfully!")
                    except Exception as e:
                        st.error(f"⚠️ Spatial simulation failed: {e}")
    else:
        st.info("Please draw your field boundary on the map to proceed.")
