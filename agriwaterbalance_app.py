# AgriWaterBalance: Research-Grade Soil Water Balance App
# ----------------------------------------------------------------------------------
# Features:
# - Dual crop coefficient (Kcb + Ke)
# - Multi-layer soil water balance
# - Dynamic root depth
# - Ks and Kr computation
# - Cumulative ETc, ETa, P, I, stress tracking
# - Input/output as .txt
# - Fancy UI with logo, tabs, and charts

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Set page configuration
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# Display logo and title
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("""
    <style>
    .circle-img {
        border-radius: 50%;
        width: 90px;
        height: 90px;
        object-fit: cover;
        border: 2px solid #ccc;
    }
    </style>
    <img src='data:image/png;base64,{}' class='circle-img'>
    """.format(base64.b64encode(open("logo.png", "rb").read()).decode()), unsafe_allow_html=True)

with col_title:
    st.title("AgriWaterBalance")
    st.markdown("**A research-grade, multi-layer soil water balance tool for any crop and soil.**")

# Upload section
st.sidebar.header("Upload Input Files (.txt)")
weather_file = st.sidebar.file_uploader("Weather Data", type="txt")
crop_file = st.sidebar.file_uploader("Crop Stages", type="txt")
soil_file = st.sidebar.file_uploader("Soil Layers", type="txt")

simulate = st.sidebar.button("Run Simulation")

# Helper functions
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

def interpolate_stages(stage_df, total_days):
    kcb_list, rd_list = [], []
    for i in range(len(stage_df)):
        row = stage_df.iloc[i]
        days = row['End_Day'] - row['Start_Day'] + 1
        for d in range(days):
            f = d / (days - 1) if days > 1 else 0
            Kcb = stage_df.iloc[i - 1]['Kcb'] + f * (row['Kcb'] - stage_df.iloc[i - 1]['Kcb']) if i > 0 else row['Kcb']
            RD = stage_df.iloc[i - 1]['Root_Depth_mm'] + f * (row['Root_Depth_mm'] - stage_df.iloc[i - 1]['Root_Depth_mm']) if i > 0 else row['Root_Depth_mm']
            kcb_list.append(Kcb)
            rd_list.append(RD)
    return kcb_list[:total_days], rd_list[:total_days]

def simulate_balance(weather_df, crop_df, soil_df):
    days = len(weather_df)
    Kcb_list, RD_list = interpolate_stages(crop_df, days)

    soil_layers = soil_df.to_dict("records")
    surface = soil_layers[0]
    TEW, REW = surface['TEW'], surface['REW']
    Ke, p = 1.1, 0.5

    SW_surface = TEW
    SW_layers = [lyr['FC'] * lyr['Depth_mm'] for lyr in soil_layers]

    cum_ETc = cum_ETa = cum_Irr = cum_P = 0
    stress_days = 0
    results = []

    for i, row in weather_df.iterrows():
        ET0, P, I = row['ET0'], row['Precipitation'], row['Irrigation']
        Kcb = Kcb_list[i]; RD = RD_list[i]
        cum_P += P; cum_Irr += I

        FC_total, WP_total, SW_root = 0, 0, 0
        depth_cum = 0
        for j, lyr in enumerate(soil_layers):
            if depth_cum >= RD:
                break
            d = min(lyr['Depth_mm'], RD - depth_cum)
            FC_total += lyr['FC'] * d
            WP_total += lyr['WP'] * d
            SW_root += SW_layers[j]
            depth_cum += d

        TAW = FC_total - WP_total
        depletion = FC_total - SW_root
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0
        cum_ETc += ETc; cum_ETa += (ETa_transp + ETa_evap)
        if Ks < 1: stress_days += 1

        SW_surface += P + I
        excess = max(0, SW_surface - TEW)
        SW_surface = max(0, SW_surface - ETa_evap)

        # Distribute to layers
        water = excess
        for j, lyr in enumerate(soil_layers):
            max_SW = lyr['FC'] * lyr['Depth_mm']
            SW_layers[j] += water
            drain = max(0, SW_layers[j] - max_SW)
            SW_layers[j] = min(SW_layers[j], max_SW)
            water = drain

            transp = ETa_transp * (lyr['Depth_mm'] / RD)
            SW_layers[j] -= transp
            SW_layers[j] = max(lyr['WP'] * lyr['Depth_mm'], SW_layers[j])

        results.append({
            'Date': row['Date'], 'ET0': ET0, 'Kcb': Kcb, 'Ks': Ks, 'Kr': Kr,
            'ETc': ETc, 'ETa_transp': ETa_transp, 'ETa_evap': ETa_evap,
            'SW_surface': SW_surface, 'SW_root': sum(SW_layers),
            'TAW': TAW, 'Depletion': depletion,
            'Cumulative_ETc': cum_ETc, 'Cumulative_ETa': cum_ETa,
            'Cumulative_Irrigation': cum_Irr, 'Cumulative_Precip': cum_P,
            'Stress_Days': stress_days
        })

    return pd.DataFrame(results)

# Main app logic
if simulate:
    try:
        weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
        crop_df = pd.read_csv(crop_file)
        soil_df = pd.read_csv(soil_file)
        result_df = simulate_balance(weather_df, crop_df, soil_df)

        tab1, tab2, tab3 = st.tabs(["Results Table", "ET Charts", "Root Zone Storage"])

        with tab1:
            st.dataframe(result_df)
            st.download_button("📥 Download Results (.txt)", result_df.to_csv(index=False), file_name="results_agriwaterbalance.txt")

        with tab2:
            fig, ax = plt.subplots()
            ax.plot(result_df['Date'], result_df['ETa_transp'], label='Transpiration')
            ax.plot(result_df['Date'], result_df['ETa_evap'], label='Evaporation')
            ax.plot(result_df['Date'], result_df['ETc'], label='ETc')
            ax.legend(); ax.grid(True)
            st.pyplot(fig)

        with tab3:
            fig2, ax2 = plt.subplots()
            ax2.plot(result_df['Date'], result_df['SW_root'], label='Root Zone SW')
            ax2.set_ylabel("mm"); ax2.grid(True); ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Simulation failed: {e}")
