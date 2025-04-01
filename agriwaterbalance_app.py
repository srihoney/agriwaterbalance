# ---------------------------------------------
# AgriWaterBalance | Final Enhanced App (With Toggles)
# ---------------------------------------------
# Features:
# - Multi-layer soil water balance
# - Dual Kcb and Ke coefficients
# - Dynamic root depth, p, Ke by crop stage
# - Cumulative ET, Irrigation, Precipitation, Drainage
# - Optional toggles for drainage tracking and monthly summary
# - Scientifically formatted outputs and plots

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
# Sidebar: File Uploads, Toggles, and Run Button
# -------------------
st.sidebar.header("Upload Input Files (.txt)")
weather_file = st.sidebar.file_uploader("Weather Data (.txt)", type="txt")
crop_file = st.sidebar.file_uploader("Crop Stage Data (.txt)", type="txt")
soil_file = st.sidebar.file_uploader("Soil Layers (.txt)", type="txt")

st.sidebar.header("Options")
show_monthly_summary = st.sidebar.checkbox("Show Monthly Summary", value=True)
track_drainage = st.sidebar.checkbox("Track Drainage", value=True)

run_button = st.sidebar.button("🚀 Run Simulation")

# -------------------
# Helper Functions
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

# -------------------
# Core Simulation Function
# -------------------
def SIMdualKc(weather_df, crop_df, soil_df, track_drain):
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

        FC_total, WP_total, SW_root = 0.0, 0.0, 0.0
        cum_depth = 0.0
        for j, soil in soil_df.iterrows():
            if cum_depth >= RD: break
            d = min(soil['Depth_mm'], RD - cum_depth)
            FC_total += soil['FC'] * d
            WP_total += soil['WP'] * d
            SW_root += SW_layers[j]
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
        if Ks < 1.0: stress_days += 1

        SW_surface += P + I
        excess_surface = max(0, SW_surface - TEW)
        SW_surface = max(0, SW_surface - ETa_evap)

        water = excess_surface
        for j, soil in soil_df.iterrows():
            max_SW = soil['FC'] * soil['Depth_mm']
            SW_layers[j] += water
            drain = max(0, SW_layers[j] - max_SW)
            if track_drain: cum_drain += drain
            SW_layers[j] = min(SW_layers[j], max_SW)
            water = drain

            transp = ETa_transp * (soil['Depth_mm'] / RD)
            SW_layers[j] -= transp
            SW_layers[j] = max(soil['WP'] * soil['Depth_mm'], SW_layers[j])

        results.append({
            "Date": row['Date'], "ET0 (mm)": ET0, "Kcb": Kcb, "Ks": Ks, "Kr": Kr,
            "ETc (mm)": ETc, "ETa_transp (mm)": ETa_transp, "ETa_evap (mm)": ETa_evap,
            "SW_surface (mm)": SW_surface, "SW_root (mm)": sum(SW_layers),
            "Depletion (mm)": depletion, "TAW (mm)": TAW,
            "Storage_Deficit (mm)": storage_deficit,
            "Cumulative_ETc (mm)": cum_ETc, "Cumulative_ETa (mm)": cum_ETa,
            "Cumulative_Irrigation (mm)": cum_Irr, "Cumulative_Precip (mm)": cum_P,
            "Cumulative_Drainage (mm)": cum_drain if track_drain else None,
            "Stress_Days": stress_days
        })

    return pd.DataFrame(results)

# -------------------
# Run the Simulation
# -------------------
if run_button and weather_file and crop_file and soil_file:
    try:
        weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
        crop_df = pd.read_csv(crop_file)
        soil_df = pd.read_csv(soil_file)

        results_df = SIMdualKc(weather_df, crop_df, soil_df, track_drainage)

        tab1, tab2, tab3 = st.tabs(["📄 Daily Results", "📈 ET Graphs", "💧 Storage"])

        with tab1:
            st.dataframe(results_df)
            st.download_button("📥 Download Results (.txt)", results_df.to_csv(index=False), file_name="agriwaterbalance_results.txt")

        with tab2:
            fig, ax = plt.subplots()
            ax.plot(results_df['Date'], results_df['ETa_transp (mm)'], label='Transpiration')
            ax.plot(results_df['Date'], results_df['ETa_evap (mm)'], label='Evaporation')
            ax.plot(results_df['Date'], results_df['ETc (mm)'], label='ETc')
            ax.set_ylabel("ET (mm)"); ax.legend(); ax.grid(True)
            st.pyplot(fig)

        with tab3:
            fig2, ax2 = plt.subplots()
            ax2.plot(results_df['Date'], results_df['SW_root (mm)'], label='Root Zone SW', color='green')
            ax2.set_ylabel("Soil Water (mm)"); ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)

        # -------------------
        # Monthly Summary Tab (Optional)
        # -------------------
        if show_monthly_summary:
            st.subheader("📆 Monthly Summary")
            monthly = results_df.copy()
            monthly['Month'] = monthly['Date'].dt.to_period('M')
            summary = monthly.groupby('Month').agg({
                'ET0 (mm)': 'mean', 'ETc (mm)': 'mean',
                'ETa_transp (mm)': 'mean', 'ETa_evap (mm)': 'mean',
                'Cumulative_Irrigation (mm)': 'max',
                'Cumulative_Precip (mm)': 'max',
                'Stress_Days': 'max'
            }).reset_index()
            st.dataframe(summary)

    except Exception as e:
        st.error(f"⚠️ Simulation failed: {e}")
else:
    st.info("📂 Please upload all required files and click 'Run Simulation' to begin.")
