import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set app layout and title
st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# -------------------
# Header with Logo
# -------------------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=90)
with col2:
    st.title("AgriWaterBalance")
    st.markdown("**A research-grade, multi-layer soil water balance tool for any crop and soil.**")

# -------------------
# File Upload Section
# -------------------
st.sidebar.header("Upload Input Files (.txt)")
weather_file = st.sidebar.file_uploader("Weather Data (.txt)", type="txt")
crop_file = st.sidebar.file_uploader("Crop Stage Data (.txt)", type="txt")
soil_file = st.sidebar.file_uploader("Soil Layers (.txt)", type="txt")

run_button = st.sidebar.button("🚀 Run Simulation")

# -------------------
# Helper Functions
# -------------------
def compute_Ks(depletion, TAW, p=0.5):
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
    kcb_list, root_list = [], []
    for i in range(len(crop_df)):
        row = crop_df.iloc[i]
        start, end = row['Start_Day'], row['End_Day']
        days = int(end - start + 1)
        for d in range(days):
            frac = d / (days - 1) if days > 1 else 0
            if i > 0:
                kcb = crop_df.iloc[i - 1]['Kcb'] + frac * (row['Kcb'] - crop_df.iloc[i - 1]['Kcb'])
                root = crop_df.iloc[i - 1]['Root_Depth_mm'] + frac * (row['Root_Depth_mm'] - crop_df.iloc[i - 1]['Root_Depth_mm'])
            else:
                kcb = row['Kcb']
                root = row['Root_Depth_mm']
            kcb_list.append(kcb)
            root_list.append(root)
    return kcb_list[:total_days], root_list[:total_days]

# -------------------
# Simulation Function
# -------------------
def simulate_water_balance(weather_df, crop_df, soil_df):
    days = len(weather_df)
    Kcb_list, RD_list = interpolate_crop_stages(crop_df, days)
    TEW, REW = soil_df.iloc[0]['TEW'], soil_df.iloc[0]['REW']
    Ke = 1.1
    SW_surface = TEW
    SW_layers = [row['FC'] * row['Depth_mm'] for _, row in soil_df.iterrows()]

    cum_ETc = cum_ETa = cum_Irr = cum_P = stress_days = 0
    results = []

    for i in range(days):
        row = weather_df.iloc[i]
        ET0 = row['ET0']
        P = row['Precipitation']
        I = row['Irrigation']
        Kcb = Kcb_list[i]
        RD = RD_list[i]
        cum_P += P
        cum_Irr += I

        # Root zone TAW
        FC_total, WP_total, SW_root = 0, 0, 0
        cum_depth = 0
        for j, soil in soil_df.iterrows():
            if cum_depth >= RD:
                break
            d = min(soil['Depth_mm'], RD - cum_depth)
            FC_total += soil['FC'] * d
            WP_total += soil['WP'] * d
            SW_root += SW_layers[j]
            cum_depth += d

        TAW = FC_total - WP_total
        depletion = FC_total - SW_root
        Ks = compute_Ks(depletion, TAW)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0
        cum_ETc += ETc
        cum_ETa += ETa_transp + ETa_evap
        if Ks < 1.0:
            stress_days += 1

        # Surface layer balance
        SW_surface += P + I
        excess_surface = max(0, SW_surface - TEW)
        SW_surface = max(0, SW_surface - ETa_evap)

        # Root zone update
        water = excess_surface
        for j, soil in soil_df.iterrows():
            max_SW = soil['FC'] * soil['Depth_mm']
            SW_layers[j] += water
            drain = max(0, SW_layers[j] - max_SW)
            SW_layers[j] = min(SW_layers[j], max_SW)
            water = drain
            transp = ETa_transp * (soil['Depth_mm'] / RD)
            SW_layers[j] -= transp
            SW_layers[j] = max(soil['WP'] * soil['Depth_mm'], SW_layers[j])

        results.append({
            "Date": row['Date'], "ET0": ET0, "Kcb": Kcb, "Ks": Ks, "Kr": Kr,
            "ETc": ETc, "ETa_transp": ETa_transp, "ETa_evap": ETa_evap,
            "SW_surface": SW_surface, "SW_root": sum(SW_layers),
            "Depletion": depletion, "TAW": TAW,
            "Cumulative_ETc": cum_ETc, "Cumulative_ETa": cum_ETa,
            "Cumulative_Irrigation": cum_Irr, "Cumulative_Precip": cum_P,
            "Stress_Days": stress_days
        })

    return pd.DataFrame(results)

# -------------------
# Run Simulation
# -------------------
if run_button and weather_file and crop_file and soil_file:
    try:
        weather_df = pd.read_csv(weather_file, parse_dates=['Date'])
        crop_df = pd.read_csv(crop_file)
        soil_df = pd.read_csv(soil_file)

        results_df = simulate_water_balance(weather_df, crop_df, soil_df)

        tab1, tab2, tab3 = st.tabs(["📄 Results", "📈 ET Plots", "💧 Soil Water Storage"])

        with tab1:
            st.subheader("Daily Simulation Results")
            st.dataframe(results_df)
            st.download_button("📥 Download Results (.txt)", results_df.to_csv(index=False), file_name="agriwaterbalance_results.txt")

        with tab2:
            fig, ax = plt.subplots()
            ax.plot(results_df['Date'], results_df['ETa_transp'], label='Transpiration')
            ax.plot(results_df['Date'], results_df['ETa_evap'], label='Evaporation')
            ax.plot(results_df['Date'], results_df['ETc'], label='ETc')
            ax.set_ylabel("ET (mm/day)")
            ax.grid(); ax.legend()
            st.pyplot(fig)

        with tab3:
            fig2, ax2 = plt.subplots()
            ax2.plot(results_df['Date'], results_df['SW_root'], label='Root Zone SW', color='green')
            ax2.set_ylabel("Soil Water (mm)")
            ax2.grid(); ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ Simulation failed: {e}")
else:
    st.info("📂 Please upload all required files and click 'Run Simulation' to begin.")
