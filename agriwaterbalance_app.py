import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# AgriWaterBalance: Core Functions
# --------------------------------------------

def compute_root_zone_depletion(SW, FC, WP, RD_mm):
    theta_fc = FC * RD_mm / 1000
    theta_wp = WP * RD_mm / 1000
    TAW = theta_fc - theta_wp
    depletion = theta_fc - SW
    return depletion, TAW

def compute_Ks(depletion, TAW, p=0.5):
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

def generate_crop_coefficients_and_root_depth(stage_df, total_days):
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

def simulate_agriwaterbalance(weather_df, Kcb_list, root_depth_list, FC, WP, TEW=12, REW=5, Ke=1.1, p=0.5):
    results = []
    SW_root = FC * root_depth_list[0] / 1000
    SW_surface = TEW
    for i, row in weather_df.iterrows():
        date = row['Date']
        ET0 = row['ET0']
        P = row['Precipitation']
        I = row['Irrigation']
        Kcb = Kcb_list[i]
        RD_mm = root_depth_list[i]
        depletion, TAW = compute_root_zone_depletion(SW_root, FC, WP, RD_mm)
        Ks = compute_Ks(depletion, TAW, p)
        Kr = compute_Kr(TEW - SW_surface, TEW, REW)
        ETc = compute_ETc(Kcb, Ks, Kr, Ke, ET0)
        ETa_transp = Ks * Kcb * ET0
        ETa_evap = Kr * Ke * ET0
        input_water = P + I
        SW_surface += input_water
        excess_surface = max(0, SW_surface - TEW)
        SW_surface = min(SW_surface, TEW)
        SW_root += excess_surface
        SW_surface -= ETa_evap
        SW_surface = max(0, SW_surface)
        SW_root -= ETa_transp
        max_sw_root = FC * RD_mm / 1000
        min_sw_root = WP * RD_mm / 1000
        drain = max(0, SW_root - max_sw_root)
        SW_root = min(max(SW_root, min_sw_root), max_sw_root)
        results.append({
            "Date": date,
            "ET0": ET0,
            "Kcb": Kcb,
            "Ks": Ks,
            "Kr": Kr,
            "ETc": ETc,
            "ETa_transp": ETa_transp,
            "ETa_evap": ETa_evap,
            "SW_root": SW_root,
            "SW_surface": SW_surface,
            "Root_Depth_mm": RD_mm,
            "Depletion": depletion,
            "Drainage": drain
        })
    return pd.DataFrame(results)

# --------------------------------------------
# Streamlit App: AgriWaterBalance
# --------------------------------------------

st.set_page_config(page_title="AgriWaterBalance", layout="wide")

# Display logo
st.image("logo.png", width=200)

st.title("AgriWaterBalance: Crop–Soil Water Balance Tool")

st.markdown("""
AgriWaterBalance is a flexible tool to simulate daily soil water balance for **any crop and any soil**.

It supports dual crop coefficients, root growth, water stress, and soil evaporation dynamics.

**Upload your crop stages and weather files to get started.**
""")

# Upload files
weather_file = st.file_uploader("Upload Weather + Irrigation CSV", type=["csv"])
stage_file = st.file_uploader("Upload Crop Stage Definitions CSV", type=["csv"])

# Soil and model parameters
st.sidebar.header("Soil & Model Settings")
FC = st.sidebar.number_input("Field Capacity (vol/vol)", value=0.28)
WP = st.sidebar.number_input("Wilting Point (vol/vol)", value=0.14)
TEW = st.sidebar.number_input("TEW (mm)", value=12.0)
REW = st.sidebar.number_input("REW (mm)", value=5.0)
Ke = st.sidebar.number_input("Maximum Ke", value=1.1)
p = st.sidebar.slider("Depletion Fraction (p)", 0.1, 0.9, 0.5)

if weather_file and stage_file:
    weather_df = pd.read_csv(weather_file, parse_dates=["Date"])
    stage_df = pd.read_csv(stage_file)
    days = len(weather_df)
    Kcb_list, root_depth_list = generate_crop_coefficients_and_root_depth(stage_df, days)
    sim_df = simulate_agriwaterbalance(weather_df, Kcb_list, root_depth_list, FC, WP, TEW, REW, Ke, p)

    # Merge & display
    final_df = pd.concat([weather_df, sim_df.drop(columns=["Date"])], axis=1)
    st.success("Simulation completed.")
    st.subheader("Simulation Results")
    st.dataframe(final_df)

    # Plot ETc
    st.subheader("Daily ETc, Transpiration, Evaporation")
    fig1, ax1 = plt.subplots()
    ax1.plot(final_df["Date"], final_df["ETc"], label="ETc")
    ax1.plot(final_df["Date"], final_df["ETa_transp"], label="Transpiration")
    ax1.plot(final_df["Date"], final_df["ETa_evap"], label="Evaporation")
    ax1.set_ylabel("mm/day")
    ax1.legend(); ax1.grid(True); st.pyplot(fig1)

    # Plot root zone storage
    st.subheader("Root Zone Soil Water (mm)")
    fig2, ax2 = plt.subplots()
    ax2.plot(final_df["Date"], final_df["SW_root"], label="Root Zone SW", color="green")
    ax2.set_ylabel("Soil Water (mm)"); ax2.legend(); ax2.grid(True); st.pyplot(fig2)

    # Download
    st.download_button("Download Results as CSV",
                       data=final_df.to_csv(index=False).encode("utf-8"),
                       file_name="agriwaterbalance_results.csv")

else:
    st.info("Please upload both input files to begin the simulation.")
