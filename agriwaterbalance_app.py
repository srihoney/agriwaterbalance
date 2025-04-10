import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import math
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64

# --------------------------------------------------------------------------------
# 1. Built-in Kc Database and Default Growth Stage Durations 
#    (From Pereira et al. (2021) updates for vegetables/field crops)
# --------------------------------------------------------------------------------
KC_DATABASE = {
    # Example: "Carrot": {Kc_mid:..., Kc_end:..., Kcb_mid:..., Kcb_end:...}
    "Carrot":        {"Kc_mid":1.05,"Kc_end":0.95,"Kcb_mid":1.00,"Kcb_end":0.90},
    "Beet":          {"Kc_mid":1.10,"Kc_end":0.95,"Kcb_mid":1.05,"Kcb_end":0.85},
    "Garlic":        {"Kc_mid":1.05,"Kc_end":0.70,"Kcb_mid":1.00,"Kcb_end":0.65},
    "Onion (fresh)": {"Kc_mid":1.10,"Kc_end":0.80,"Kcb_mid":1.05,"Kcb_end":0.75},
    "Onion (dry)":   {"Kc_mid":1.10,"Kc_end":0.65,"Kcb_mid":1.05,"Kcb_end":0.60},
    "Cabbage":       {"Kc_mid":1.00,"Kc_end":0.90,"Kcb_mid":0.95,"Kcb_end":0.85},
    "Tomato (fresh)":{"Kc_mid":1.15,"Kc_end":0.80,"Kcb_mid":1.10,"Kcb_end":0.75},
    "Pepper":        {"Kc_mid":1.15,"Kc_end":0.90,"Kcb_mid":1.10,"Kcb_end":0.85},
    "Eggplant":      {"Kc_mid":1.10,"Kc_end":0.90,"Kcb_mid":1.05,"Kcb_end":0.85},
    "Wheat":         {"Kc_mid":1.15,"Kc_end":0.35,"Kcb_mid":1.10,"Kcb_end":0.30},
    "Maize":         {"Kc_mid":1.20,"Kc_end":0.60,"Kcb_mid":1.15,"Kcb_end":0.55},
    "Rice":          {"Kc_mid":1.20,"Kc_end":0.90,"Kcb_mid":1.15,"Kcb_end":0.85},
    "Soybean":       {"Kc_mid":1.15,"Kc_end":0.50,"Kcb_mid":1.10,"Kcb_end":0.45},
    "Bean":          {"Kc_mid":1.15,"Kc_end":0.90,"Kcb_mid":1.10,"Kcb_end":0.85},
    "Cotton":        {"Kc_mid":1.15,"Kc_end":0.65,"Kcb_mid":1.10,"Kcb_end":0.60},
    "Sugarbeet":     {"Kc_mid":1.20,"Kc_end":0.60,"Kcb_mid":1.15,"Kcb_end":0.55},
    "Sunflower":     {"Kc_mid":1.15,"Kc_end":0.35,"Kcb_mid":1.10,"Kcb_end":0.30},
    # ... add more as needed ...
}

# Default durations for a 4-stage approach: initial (ini), dev, mid, late.
# For brevity, these are approximate. Real data would come from references or user.
CROP_STAGE_LENGTHS = {
    # "Carrot":  (Ini, Dev, Mid, Late) 
    # We'll define some typical days. You can refine them further:
    "Carrot": (20, 30, 40, 20),
    "Beet":   (25, 30, 40, 25),
    "Garlic": (30, 40, 50, 20),
    "Onion (fresh)": (25, 30, 40, 25),
    "Onion (dry)":   (25, 35, 40, 20),
    "Cabbage":       (25, 30, 40, 20),
    "Tomato (fresh)":(30, 35, 40, 20),
    "Pepper":        (30, 40, 40, 20),
    "Eggplant":      (30, 40, 40, 20),
    "Wheat":         (20, 30, 40, 30),
    "Maize":         (20, 35, 40, 25),
    "Rice":          (30, 40, 50, 30),
    "Soybean":       (20, 30, 40, 25),
    "Bean":          (20, 25, 30, 20),
    "Cotton":        (30, 40, 50, 30),
    "Sugarbeet":     (25, 35, 50, 30),
    "Sunflower":     (20, 30, 40, 25),
    # add more as needed
}

# --------------------------------------------------------------------------------
# 2. Configure Requests Session with Retries
# --------------------------------------------------------------------------------
session = requests.Session()
retries = Retry(total=5, backoff_factor=1.0,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"])
session.mount('https://', HTTPAdapter(max_retries=retries))

# --------------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Logo
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Advanced AgriWaterBalance", layout="wide")

try:
    with open("logo.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    logo_url = f"data:image/png;base64,{encoded_string}"
except FileNotFoundError:
    logo_url = ""

st.markdown(f"""
    <style>
    body {{ margin: 0; padding: 0; }}
    .header-container {{
        position: relative;
        background-color: #1E3A8A;
        padding: 20px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0;
    }}
    .header-logo {{
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
    }}
    .header-logo img {{ width: 100px; height: auto; }}
    .header-title {{ color: white; font-size: 36px; font-weight: bold; text-align: center; }}
    .sub-header {{ color: #1E3A8A; font-size: 24px; font-weight: bold; margin-top: 20px; }}
    .footer {{
        background-color: #1E3A8A; color: white; padding: 10px; text-align: center;
        position: fixed; bottom: 0; width: 100%; border-radius: 5px 5px 0 0;
    }}
    .stButton>button {{
        background-color: #2563EB; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px;
    }}
    .stButton>button:hover {{ background-color: #1E40AF; }}
    .stFileUploader {{
        border: 2px dashed #1E3A8A; border-radius: 5px; padding: 10px;
    }}
    .stSelectbox {{ background-color: #F1F5F9; border-radius: 5px; }}
    .stNumberInput input {{ background-color: #F1F5F9; border-radius: 5px; }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-container">
        <div class="header-logo">
            <img src="{logo_url}" alt="Logo">
        </div>
        <div class="header-title">Advanced AgriWaterBalance</div>
    </div>
""", unsafe_allow_html=True)
st.markdown("**A More Advanced Tool for Crop Water Management**", unsafe_allow_html=True)

# We'll create 3 main tabs: Setup, Results, Irrigation Schedule
setup_tab, results_tab, schedule_tab = st.tabs(["Setup Simulation", "View Results", "Irrigation Calendar"])

# --------------------------------------------------------------------------------
# 4. Session State
# --------------------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'soil_profile' not in st.session_state:
    st.session_state.soil_profile = None
if 'weather_cache' not in st.session_state:
    st.session_state.weather_cache = {}
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.now().date()

current_date = datetime.now().date()
if st.session_state.last_reset_date != current_date:
    st.session_state.api_calls = 0
    st.session_state.last_reset_date = current_date

# --------------------------------------------------------------------------------
# 5. Water Balance & Forecast Functions
# --------------------------------------------------------------------------------

def compute_Ks(SW, WP, FC, p):
    """Compute stress coefficient Ks."""
    TAW = (FC - WP) * 1000.0
    RAW = p*TAW
    Dr  = (FC - SW)*1000.0
    if Dr <= RAW:
        Ks=1.0
    else:
        Ks= (TAW-Dr)/((1-p)*TAW)
        Ks= max(0,min(1,Ks))
    return Ks

def compute_Kr(TEW, REW, E):
    """Evaporation reduction coefficient."""
    if E<=REW:
        Kr=1.0
    else:
        Kr=(TEW - E)/(TEW-REW)
        Kr=max(0,min(1,Kr))
    return Kr

def compute_ETc(Kcb, Ks, Ke, ET0):
    """ETc from dual Kc approach."""
    return (Kcb*Ks + Ke)*ET0

def fetch_weather_data(lat, lon, start_date, end_date, forecast=True, manual_data=None):
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}_{forecast}"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    if manual_data is not None:
        dates = pd.date_range(start_date, end_date)
        wdf = pd.DataFrame({
            "Date": dates,
            "ET0": manual_data['eto'],
            "Precipitation": manual_data['precip'],
            "Irrigation": [0]*len(dates)
        })
        st.session_state.weather_cache[cache_key]= wdf
        return wdf
    
    if forecast:
        if st.session_state.api_calls>=1000:
            st.warning("Daily API call limit reached.")
            return None
        if lat==0.0 and lon==0.0:
            st.warning("Invalid lat/lon.")
            return None
        today= datetime.now().date()
        maxf= today + timedelta(days=5)
        if start_date< today or end_date>maxf:
            st.warning("Forecast date out of range => adjusting.")
            start_date=today
            end_date= today+timedelta(days=4)
        
        api_key="fe2d869569674a4afbfca57707bdf691"
        url=f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            resp= session.get(url,timeout=30)
            resp.raise_for_status()
            st.session_state.api_calls+=1
            data= resp.json()
            
            daily_data= {}
            for ent in data['list']:
                d_= datetime.fromtimestamp(ent['dt']).date()
                if start_date<= d_<= end_date:
                    dstr= d_.strftime("%Y-%m-%d")
                    if dstr not in daily_data:
                        daily_data[dstr]= {"tmax":ent['main']['temp_max'],
                                           "tmin":ent['main']['temp_min'],
                                           "precip": ent.get('rain',{}).get('3h',0)}
                    else:
                        daily_data[dstr]['tmax']= max(daily_data[dstr]['tmax'], ent['main']['temp_max'])
                        daily_data[dstr]['tmin']= min(daily_data[dstr]['tmin'], ent['main']['temp_min'])
                        daily_data[dstr]['precip'] += ent.get('rain',{}).get('3h',0)
            dd, ETo_, P_ = [], [], []
            for dstr,vv in daily_data.items():
                dd_ = pd.to_datetime(dstr)
                tx_, tn_= vv['tmax'], vv['tmin']
                if tx_<tn_: tx_, tn_= tn_, tx_
                Ra=10
                Tmean=(tx_+tn_)/2
                et0_=0.0023*Ra*(Tmean+17.8)*((tx_-tn_)**0.5)
                et0_= max(0,et0_)
                dd.append(dd_)
                ETo_.append(et0_)
                P_.append(vv['precip'])
            wdf= pd.DataFrame({
                "Date": dd,
                "ET0": ETo_,
                "Precipitation": P_,
                "Irrigation": [0]*len(dd)
            }).sort_values("Date").reset_index(drop=True)
            st.session_state.weather_cache[cache_key]= wdf
            return wdf
        except:
            st.error("Cannot fetch forecast.")
            return None
    else:
        # historical data approach
        return None  # omitted for brevity or you can implement NASA calls


def make_4stage_crop_df(selected_crop):
    """
    Build a 4-stage DataFrame (initial, dev, mid, late) from CROP_STAGE_LENGTHS 
    and the Kcb from KC_DATABASE.
    """
    if selected_crop not in KC_DATABASE or selected_crop not in CROP_STAGE_LENGTHS:
        # fallback minimal
        return pd.DataFrame({
            "Start_Day":[1, 31],
            "End_Day":[30, 60],
            "Kcb":[0.5, 1.0],
            "Root_Depth_mm":[300,600],
            "p":[0.5,0.5],
            "Ke":[0.2, 0.15]
        })
    # get durations
    ini, dev, mid, late = CROP_STAGE_LENGTHS[selected_crop]
    # total
    total_days = ini+dev+mid+late
    # Kcbini assumed 0.15 or 0.2 for initial
    kcb_ini = 0.15  
    # get mid & end from db
    kcb_mid = KC_DATABASE[selected_crop]["Kcb_mid"]
    kcb_end = KC_DATABASE[selected_crop]["Kcb_end"]
    
    # p can vary, we'll keep 0.5 or 0.55
    # root depth can vary. We'll do a guess
    # Ke can vary too
    # We'll define 4 rows:
    # row1 => initial stage 
    # row2 => dev stage
    # row3 => mid stage
    # row4 => late stage
    df = pd.DataFrame(columns=["Start_Day","End_Day","Kcb","Root_Depth_mm","p","Ke"])
    day1=1
    day2=ini
    df.loc[0]= [1, ini, kcb_ini, 200, 0.5, 1.0]  # initial
    df.loc[1]= [ini+1, ini+dev, (kcb_ini+kcb_mid)/2, 400, 0.5, 0.8]  # dev
    df.loc[2]= [ini+dev+1, ini+dev+mid, kcb_mid, 600, 0.5, 0.15]     # mid
    df.loc[3]= [ini+dev+mid+1, total_days, kcb_end, 600, 0.5, 0.10]  # late
    return df.sort_values("Start_Day").reset_index(drop=True)


# --------------------------------------------------------------------------------
# 6. The SIMDualKc-like daily water balance with more variables
# --------------------------------------------------------------------------------
def SIMdualKc(weather_df, crop_df, soil_df, 
              track_drainage=True, enable_yield=False,
              use_fao33=False, Ym=0, Ky=0, use_transp=False, WP_yield=0,
              enable_leaching=False, leaching_method="", nitrate_conc=0,
              total_N_input=0, leaching_fraction=0,
              enable_dynamic_root=False, initial_root_depth=None,
              max_root_depth=None, days_to_max=None,
              return_soil_profile=False, initial_SW_layers=None):
    
    if weather_df.empty:
        st.error("Weather data is empty.")
        return None
    
    results=[]
    total_days= len(weather_df)
    
    # Start soil water in each layer at FC if none provided
    if initial_SW_layers is None:
        SW_layers= [soil_df.iloc[i]['FC']*soil_df.iloc[i]['Depth_mm'] for i in range(len(soil_df))]
    else:
        SW_layers= initial_SW_layers.copy()
    
    # We'll track daily infiltration, Dr, TAW, RAW, infiltration, etc.
    
    # topsoil evaporation
    E = soil_df['REW'].sum()  # start "dry"
    cIrr=0
    cPrec=0
    
    # Build daily Kcb, p, rootD, Ke from crop_df
    Kcb_daily, rd_daily, p_daily, ke_daily = interpolate_crop_stages(crop_df, total_days)
    
    if enable_dynamic_root and initial_root_depth and max_root_depth and days_to_max:
        # override rd_daily with a simple linear
        dd_lin = np.linspace(initial_root_depth, max_root_depth, min(days_to_max, total_days))
        if total_days> days_to_max:
            dd_lin= np.concatenate([dd_lin, [max_root_depth]*(total_days - days_to_max)])
        rd_daily= dd_lin[:total_days]
    
    for i in range(total_days):
        date_i = weather_df.iloc[i]['Date']
        ET0_i  = weather_df.iloc[i]['ET0']
        prcp_i = weather_df.iloc[i]['Precipitation']
        irr_i  = weather_df.iloc[i]['Irrigation']
        
        cIrr+= irr_i
        cPrec+= prcp_i
        
        # stage props
        Kcb_i= Kcb_daily[i]
        p_i  = p_daily[i]
        ke0_i= ke_daily[i]
        rd_i = max(1, rd_daily[i])
        
        # average WP, FC
        FC_avg= soil_df['FC'].mean()
        WP_avg= soil_df['WP'].mean()
        
        # TAW, RAW for root zone
        TAW_i= (FC_avg - WP_avg)* rd_i
        RAW_i= p_i * TAW_i
        
        # SW_root at start of day
        # sum of water in layers that are within root zone
        SW_root=0
        depth_covered=0
        for j in range(len(soil_df)):
            layerD= soil_df.iloc[j]['Depth_mm']
            oldD= depth_covered
            newD= oldD+ layerD
            fraction_in_root=0
            if newD<= rd_i:
                fraction_in_root=1.0
            elif oldD< rd_i< newD:
                fraction_in_root= (rd_i- oldD)/layerD
            SW_root+= SW_layers[j]*fraction_in_root
            depth_covered=newD
        
        # Dr_i => depletion
        Dr_i= (FC_avg*rd_i)- SW_root  # approximate 
        if Dr_i<0: Dr_i=0
        
        # Ks
        avg_SW_frac= SW_root/ rd_i
        Ks_i= compute_Ks(avg_SW_frac, WP_avg, FC_avg, p_i)
        
        # topsoil evaporation
        TEW= soil_df['TEW'].sum()
        REW= soil_df['REW'].sum()
        Kr_i= compute_Kr(TEW, REW, E)
        Ke_i= Kr_i* ke0_i
        
        # Potential ETc
        # Kcb_noStress => Ks=1
        ETc_pot= compute_ETc(Kcb_i, 1.0, Ke_i, ET0_i)  # ignoring water stress
        # Actual ET
        ETc_act= compute_ETc(Kcb_i, Ks_i, Ke_i, ET0_i)
        # T & E
        Tact= (Kcb_i*Ks_i*ET0_i)
        Eact= (Ke_i*ET0_i)
        
        infiltration= prcp_i + irr_i
        # assume no runoff => infiltration= water_in
        # check infiltration vs capacity
        excess= infiltration - ETc_act
        drainage=0
        
        if track_drainage:
            # fill layers to FC
            for j in range(len(SW_layers)):
                cap_j= (soil_df.iloc[j]['FC']* soil_df.iloc[j]['Depth_mm'])- SW_layers[j]
                if cap_j>0 and excess>0:
                    add_ = min(excess, cap_j)
                    SW_layers[j]+= add_
                    excess-= add_
            drainage= max(0, excess)
            for j in range(len(SW_layers)):
                max_sw= soil_df.iloc[j]['FC']* soil_df.iloc[j]['Depth_mm']
                min_sw= soil_df.iloc[j]['WP']* soil_df.iloc[j]['Depth_mm']
                SW_layers[j]= min(max_sw, SW_layers[j])
                SW_layers[j]= max(min_sw, SW_layers[j])
        
        # now remove ET from root zone 
        if ETc_act>0 and SW_root>0:
            toRemove= ETc_act
            depth_covered=0
            for j in range(len(SW_layers)):
                layerD= soil_df.iloc[j]['Depth_mm']
                oldD= depth_covered
                newD= oldD+layerD
                fraction_in_root=0
                if newD<= rd_i:
                    fraction_in_root=1
                elif oldD<rd_i< newD:
                    fraction_in_root= (rd_i-oldD)/ layerD
                if fraction_in_root>0:
                    maxrem= SW_layers[j]- (soil_df.iloc[j]['WP']*layerD)
                    rem_ = min(toRemove*fraction_in_root, maxrem)
                    SW_layers[j]-= rem_
                    toRemove-= rem_
                    if toRemove<=0:
                        break
        
        # track E for topsoil dryness
        if infiltration>=4.0:
            E=0
        else:
            E+= Eact
        E= max(0, min(E, TEW))
        
        # yield
        daily_yield= None
        if enable_yield:
            if use_fao33 and Ym>0 and Ky>0 and ETc_pot>0:
                # typical approach => daily fraction 
                # We do a simplified approach
                ratio= ETc_act/ETc_pot
                daily_yield= Ym*(1 - Ky*(1-ratio))
            if use_transp and WP_yield>0:
                daily_yield= (WP_yield* Tact)
        
        # leaching
        daily_leach=0
        if enable_leaching:
            if leaching_method=="Method 1: Drainage × nitrate concentration" and drainage>0:
                # drainage mm => 10 m3/ha => mg-> kg
                daily_leach= drainage*10*(nitrate_conc*1e-6)*1000
            elif leaching_method=="Method 2: Leaching Fraction × total N input":
                daily_leach= leaching_fraction* (total_N_input/float(total_days))
        
        # gather daily results
        day_res= {
            "Date": date_i,
            "ET0 (mm)": ET0_i,
            "Precip (mm)": prcp_i,
            "Irrig (mm)": irr_i,
            "Infiltration (mm)": infiltration,
            "ETc_pot (mm)": ETc_pot,
            "ETc_act (mm)": ETc_act,
            "ETa_transp (mm)": Tact,
            "ETa_evap (mm)": Eact,
            "Ks": Ks_i,
            "Ke": Ke_i,
            "SW_root (mm)": SW_root,  # start-of-day
            "Dr (mm)": Dr_i,
            "TAW (mm)": TAW_i,
            "RAW (mm)": RAW_i,
            "Daily_Drainage (mm)": drainage,
            "Cumulative_Irrigation (mm)": cIrr,
            "Cumulative_Precip (mm)": cPrec
        }
        if daily_yield is not None:
            day_res["Yield (ton/ha)"]= daily_yield
        if enable_leaching:
            day_res["Leaching (kg/ha)"]= daily_leach
        
        # store layer SW as well
        for j in range(len(SW_layers)):
            day_res[f"Layer{j}_SW(mm)"]= SW_layers[j]
        
        results.append(day_res)
    
    resdf= pd.DataFrame(results)
    if return_soil_profile:
        final_profile=[]
        for j in range(len(soil_df)):
            final_profile.append({
                "Layer": j,
                "Depth_mm": soil_df.iloc[j]['Depth_mm'],
                "SW (mm)": SW_layers[j]
            })
        return resdf, final_profile
    else:
        return resdf

def interpolate_crop_stages(crop_df, total_days):
    """Make daily arrays for Kcb, rootD, p, Ke via linear interpolation."""
    Kcb_arr = np.zeros(total_days)
    rd_arr  = np.zeros(total_days)
    p_arr   = np.zeros(total_days)
    ke_arr  = np.zeros(total_days)
    
    for i in range(len(crop_df)-1):
        sday= int(crop_df.iloc[i]['Start_Day'])
        eday= int(crop_df.iloc[i]['End_Day'])
        if eday> total_days: eday= total_days
        Kcb_s= crop_df.iloc[i]['Kcb']
        Kcb_e= crop_df.iloc[i+1]['Kcb']
        rd_s = crop_df.iloc[i]['Root_Depth_mm']
        rd_e = crop_df.iloc[i+1]['Root_Depth_mm']
        p_s  = crop_df.iloc[i]['p']
        p_e  = crop_df.iloc[i+1]['p']
        ke_s = crop_df.iloc[i]['Ke']
        ke_e = crop_df.iloc[i+1]['Ke']
        
        idx= np.arange(sday-1, eday)
        if i==0 and (sday>1):
            Kcb_arr[0:sday-1]=0
            rd_arr[0:sday-1]= rd_s
            p_arr[0:sday-1]= p_s
            ke_arr[0:sday-1]= ke_s
        
        if len(idx)>0:
            Kcb_arr[idx]= np.linspace(Kcb_s,Kcb_e,len(idx))
            rd_arr[idx] = np.linspace(rd_s, rd_e,len(idx))
            p_arr[idx]  = np.linspace(p_s, p_e,len(idx))
            ke_arr[idx] = np.linspace(ke_s, ke_e,len(idx))
    
    # fill last stage
    last_i= len(crop_df)-1
    sday= int(crop_df.iloc[last_i]['Start_Day'])
    eday= int(crop_df.iloc[last_i]['End_Day'])
    if eday> total_days: eday= total_days
    KcbL= crop_df.iloc[last_i]['Kcb']
    rdL = crop_df.iloc[last_i]['Root_Depth_mm']
    pL  = crop_df.iloc[last_i]['p']
    keL = crop_df.iloc[last_i]['Ke']
    
    idxL= np.arange(sday-1, eday)
    if len(idxL)>0:
        Kcb_arr[idxL]= KcbL
        rd_arr[idxL]= rdL
        p_arr[idxL] = pL
        ke_arr[idxL]= keL
    
    if eday< total_days:
        Kcb_arr[eday:]= KcbL
        rd_arr[eday:]= rdL
        p_arr[eday:]= pL
        ke_arr[eday:]= keL
    
    return Kcb_arr, rd_arr, p_arr, ke_arr

# --------------------------------------------------------------------------------
# 7. The 3 tabs
# --------------------------------------------------------------------------------

with setup_tab:
    st.markdown('<div class="sub-header">1) Configure Inputs</div>', unsafe_allow_html=True)
    
    # Crop selection
    st.write("**Select your crop type for the simulation:**")
    selected_crop = st.selectbox("Pick Crop", list(KC_DATABASE.keys()))
    st.write(f"**Selected Crop**: {selected_crop}  \n"
             f"Kc_mid={KC_DATABASE[selected_crop]['Kc_mid']}, "
             f"Kc_end={KC_DATABASE[selected_crop]['Kc_end']}, "
             f"Kcb_mid={KC_DATABASE[selected_crop]['Kcb_mid']}, "
             f"Kcb_end={KC_DATABASE[selected_crop]['Kcb_end']}")
    
    # Weather input
    st.markdown("### Weather Data")
    weather_file = st.file_uploader("Upload daily weather (Date,ET0,Precipitation,Irrigation) or skip to use forecast", type="csv")
    use_manual_forecast= st.checkbox("Use manual 5-day forecast input?", value=False)
    
    if use_manual_forecast:
        tmaxvals, tminvals, pvals= [], [], []
        for i in range(5):
            st.write(f"Day {i+1}")
            tx_ = st.number_input(f"Tmax (°C) day {i+1}", value=25.0, key=f"tx_{i}")
            tn_ = st.number_input(f"Tmin (°C) day {i+1}", value=15.0, key=f"tn_{i}")
            pp_ = st.number_input(f"Precip (mm) day {i+1}", value=0.0, key=f"pp_{i}")
            tmaxvals.append(tx_)
            tminvals.append(tn_)
            pvals.append(pp_)
        # We'll compute ETo from these
        man_eto=[]
        for tx_, tn_ in zip(tmaxvals,tminvals):
            if tx_<tn_:
                tx_, tn_= tn_,tx_
            Ra=10
            Tmean= (tx_+tn_)/2
            ETo_ = 0.0023*Ra*(Tmean+17.8)*((tx_-tn_)**0.5)
            ETo_= max(0,ETo_)
            man_eto.append(ETo_)
        manual_forecast_data= {
            "tmax": tmaxvals,
            "tmin": tminvals,
            "precip": pvals,
            "eto": man_eto
        }
    else:
        manual_forecast_data=None
    
    # Crop stage data
    st.markdown("### Crop Stage Data")
    use_custom_stages = st.checkbox("Upload my own 4-stage (or multi-stage) crop data?", value=False)
    if use_custom_stages:
        stage_file= st.file_uploader("Upload crop stage CSV with columns: Start_Day,End_Day,Kcb,Root_Depth_mm,p,Ke", type="csv")
    else:
        st.write("We'll auto-generate a 4-stage table from internal defaults for this crop.")
    
    # Soil data
    st.markdown("### Soil Layers Data")
    soil_file = st.file_uploader("Upload soil layers CSV: Depth_mm,FC,WP,TEW,REW", type="csv")
    
    # Additional Options
    st.markdown("### Additional Options")
    track_drainage   = st.checkbox("Track Drainage?", value=True)
    enable_yield     = st.checkbox("Enable yield estimation?", value=False)
    if enable_yield:
        use_fao33= st.checkbox("Use FAO-33 Ky-based method?", value=True)
        if use_fao33:
            Ym= st.number_input("Max Yield (ton/ha)", min_value=0.0, value=10.0, step=0.1)
            Ky= st.number_input("Yield Response Factor Ky", min_value=0.0, value=1.0, step=0.1)
        else:
            Ym=Ky=0
        use_transp= st.checkbox("Use transpiration-based (WP_yield)?", value=False)
        if use_transp:
            WP_yield= st.number_input("WP_yield (ton/ha per mm)", min_value=0.0, value=0.01, step=0.001)
        else:
            WP_yield=0
    else:
        use_fao33
