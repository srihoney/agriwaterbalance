import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# App Configuration & Branding
# -----------------------------
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# Round logo styling
with st.container():
    col1, col2 = st.columns([1, 8])
    with col1:
        st.markdown("""
            <style>
                .round-logo {
                    border-radius: 50%;
                    width: 100px;
                    height: 100px;
                    object-fit: cover;
                    border: 2px solid #ccc;
                }
            </style>
        """, unsafe_allow_html=True)
        st.image("logo.png", use_column_width=False, output_format="auto", caption="", width=100)
    with col2:
        st.title("AgriWaterBalance")
        st.markdown("""
        **A professional-grade, multi-layer soil water balance tool for any crop and soil type.**
        Upload your soil, weather, and crop stage data to simulate evapotranspiration, stress, and water storage.
        """)

# ------------------
# File Upload Inputs
# ------------------
with st.sidebar:
    st.header("Upload Inputs")
    weather_file = st.file_uploader("Weather Data (ET0, Precip, Irrigation)", type="csv")
    stage_file = st.file_uploader("Crop Stage Data (Kcb, Days, Root Depth)", type="csv")
    soil_file = st.file_uploader("Soil Layers Data (FC, WP, Depth, TEW, REW)", type="csv")

# --------------
# Run Simulation
# --------------
simulate = False
if weather_file and stage_file and soil_file:
    simulate = st.button("🚀 Run Simulation")
else:
    st.warning("Please upload all required input files to enable simulation.")

# ------------------
# Utility Functions
# ------------------
def compute_Ks(depletion, TAW, p):
    RAW = p * TAW
    if depletion <= RAW:
        return 1.0
    elif depletion >= TAW:
        return 0.0
    else:
        return (TAW - depletion) / (TAW - RAW)

def compute_Kr(surface_depletion, TEW, REW):
    if surface_depletion <= REW:
        return 1.0
    elif surface_depletion >= TEW:
        return 0.0
    else:
        return (TEW - surface_depletion) / (TEW - REW)

def compute_ETc(Kcb, Ks, Kr, Ke, ET0):
    return (Kcb * Ks + Kr * Ke) * ET0

def interpolate_crop_stages(stage_df, total_days):
    kcb_values = []
    root_depths = []
    for i in range(len(stage_df)):
        row = stage_df.iloc[i]
        start = int(row['Start_Day'])
        end = int(row['End_Day'])
        kcb_start = stage_df.iloc[i - 1]['Kcb'] if i > 0 else row['Kcb']
        kcb_end = row['Kcb']
        rd_start = stage_df.iloc[i - 1]['Root_Depth_mm'] if i > 0 else row['Root_Depth_mm']
        rd_end = row['Root_Depth_mm']
        days = end - start + 1
        for d in range(days):
            frac = d / (days - 1) if days > 1 else 0
            kcb = kcb_start + frac * (kcb_end - kcb_start)
            rd = rd_start + frac * (rd_end - rd_start)
            kcb_values.append(kcb)
            root_depths.append(rd)
    kcb_values += [kcb_values[-1]] * (total_days - len(kcb_values))
    root_depths += [root_depths[-1]] * (total_days - len(root_depths))
    return kcb_values[:total_days], root_depths[:total_days]

# ------------------
# Simulation Engine
# ------------------
def simulate_water_balance(weather_df, stage_df, soil_df):
    days = len(weather_df)
    Kcb_list, root_depth_list = interpolate_crop_stages(stage_df, days)

    # Initialize values
    surface_layer = soil_df.iloc[0]  # Assume top layer used for evaporation
    TEW = surface_layer['TEW']
    REW = surface_layer['REW']
    Ke = 1.1
    p = 0.5

    SW_surface = TEW
    SW_layers = [row['FC'] * row['Depth_mm'] for _, row in soil_df.iterrows()]

    results = []
    for i, row in weather_df.iterrows():
        ET0 = row['ET0']
        P = row['Precipitation']
        I = row['Irrigation']
        input_water = P + I
        Kcb = Kcb_list[i]
        RD = root_depth_list[i]

        # Root zone TAW & depletion
        FC_total = 0
        WP_total = 0
        SW_root = 0
        for j, soil in soil_df.iterrows():
            if RD > sum(soil_df.iloc[:j+1]['Depth_mm']):
                depth = soil['Depth_mm']
            else:
                depth = max(0, RD - sum(soil_df.iloc[:j]['Depth_mm']))
            FC_total += soil['FC'] * depth
            WP_total += soil['WP'] * depth
            SW_root += SW_layers[j] if j < len(SW_layers) else 0

        TAW = FC_total - WP_total
        depletion = FC_total - SW_root
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0

        # Water movement
        SW_surface += input_water
        excess_surface = max(0, SW_surface - TEW)
        SW_surface = min(SW_surface, TEW) - ETa_evap
        SW_surface = max(0, SW_surface)

        # Distribute to layers
        water_to_layers = excess_surface
        for j, soil in soil_df.iterrows():
            max_sw = soil['FC'] * soil['Depth_mm']
            SW_layers[j] += water_to_layers
            drain = max(0, SW_layers[j] - max_sw)
            SW_layers[j] = min(SW_layers[j], max_sw)
            water_to_layers = drain

            transp = ETa_transp * (soil['Depth_mm'] / RD)
            SW_layers[j] -= transp
            SW_layers[j] = max(soil['WP'] * soil['Depth_mm'], SW_layers[j])

        results.append({
            "Date": row['Date'],
            "ET0": ET0,
            "Kcb": Kcb,
            "Ks": Ks,
            "Kr": Kr,
            "ETc": ETc,
            "ETa_transp": ETa_transp,
            "ETa_evap": ETa_evap,
            "SW_surface": SW_surface,
            "SW_root": sum(SW_layers),
            "Depletion": depletion,
            "TAW": TAW
        })

    return pd.DataFrame(results)

# -----------------
# Run If Triggered
# -----------------
if simulate:
    weather_df = pd.read_csv(weather_file, parse_dates=["Date"])
    stage_df = pd.read_csv(stage_file)
    soil_df = pd.read_csv(soil_file)
    sim_df = simulate_water_balance(weather_df, stage_df, soil_df)

    with st.tabs(["📊 Results", "🌿 ET Breakdown", "💧 Storage", "⬇ Download"]):
        with st.container():
            st.subheader("Daily Water Balance Output")
            st.dataframe(sim_df)

        with st.container():
            st.subheader("ET Components")
            fig, ax = plt.subplots()
            ax.plot(sim_df['Date'], sim_df['ETc'], label='ETc')
            ax.plot(sim_df['Date'], sim_df['ETa_transp'], label='Transpiration')
            ax.plot(sim_df['Date'], sim_df['ETa_evap'], label='Evaporation')
            ax.legend(); ax.grid(); st.pyplot(fig)

        with st.container():
            st.subheader("Root Zone Soil Water")
            fig2, ax2 = plt.subplots()
            ax2.plot(sim_df['Date'], sim_df['SW_root'], label='SW Root Zone', color='green')
            ax2.grid(); ax2.legend(); st.pyplot(fig2)

        with st.container():
            st.subheader("Download Results")
            st.download_button("Download CSV", data=sim_df.to_csv(index=False).encode("utf-8"), file_name="agriwaterbalance_results.csv")
