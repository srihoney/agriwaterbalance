import os
import io
import calendar
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64

# --------------------------------------------------------------------------------
# 1. Large Kc Crop Database 
#    (merging Pereira et al. (2021a,b) plus additional references + orchard crops)
# --------------------------------------------------------------------------------
CROP_DATABASE = {
    # Vegetables
    "Carrot": {"Kc_mid":1.05, "Kc_end":0.95, "Kcb_mid":1.00, "Kcb_end":0.90, "total_days_default":90},
    "Beet":   {"Kc_mid":1.10, "Kc_end":0.95, "Kcb_mid":1.05, "Kcb_end":0.85, "total_days_default":100},
    "Garlic": {"Kc_mid":1.05, "Kc_end":0.70, "Kcb_mid":1.00, "Kcb_end":0.65, "total_days_default":120},
    # ... (You can include all from your prior code snippet) ...
    "Eggplant": {"Kc_mid":1.10, "Kc_end":0.90, "Kcb_mid":1.05, "Kcb_end":0.85, "total_days_default":130},
    # ...
    # Field Crops
    "Wheat":  {"Kc_mid":1.15, "Kc_end":0.35, "Kcb_mid":1.10, "Kcb_end":0.30, "total_days_default":150},
    "Maize":  {"Kc_mid":1.20, "Kc_end":0.60, "Kcb_mid":1.15, "Kcb_end":0.55, "total_days_default":140},
    "Rice":   {"Kc_mid":1.20, "Kc_end":0.90, "Kcb_mid":1.15, "Kcb_end":0.85, "total_days_default":160},
    # ...
    # Additional orchard/perennial
    "Almond":   {"Kc_mid":1.10, "Kc_end":0.70, "Kcb_mid":1.05, "Kcb_end":0.65, "total_days_default":300},
    "Walnuts":  {"Kc_mid":1.20, "Kc_end":0.80, "Kcb_mid":1.15, "Kcb_end":0.75, "total_days_default":300},
    "Pistachio":{"Kc_mid":1.15, "Kc_end":0.75, "Kcb_mid":1.10, "Kcb_end":0.70, "total_days_default":300},
    "Citrus":   {"Kc_mid":1.05, "Kc_end":0.65, "Kcb_mid":1.00, "Kcb_end":0.60, "total_days_default":300},
    "Olives":   {"Kc_mid":1.00, "Kc_end":0.60, "Kcb_mid":0.95, "Kcb_end":0.55, "total_days_default":300}
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
    with open("logo.png","rb") as f:
        encoded= base64.b64encode(f.read()).decode()
    logo_url = f"data:image/png;base64,{encoded}"
except FileNotFoundError:
    logo_url=""

# Custom CSS for smaller Setup headings & no grid lines in charts
st.markdown("""
    <style>
    body { margin:0; padding:0; }
    .header-container {
       position:relative; background-color:#1E3A8A; 
       padding:20px; border-radius:5px; 
       display:flex; align-items:center; justify-content:center;
    }
    .header-logo {
       position:absolute; left:20px; top:50%; transform:translateY(-50%);
    }
    .header-logo img { width:100px; height:auto; }
    .header-title { color:white; font-size:36px; font-weight:bold; text-align:center; }
    .small-header { font-size:14px; font-weight:bold; color:#1E3A8A; margin-top:10px; }
    .footer {
       background-color:#1E3A8A; color:white; padding:10px; text-align:center;
       position:fixed; bottom:0; width:100%; border-radius:5px 5px 0 0;
    }
    .stButton>button { background-color:#2563EB; color:white; 
       border-radius:5px; padding:10px 20px; font-size:16px; }
    .stButton>button:hover { background-color:#1E40AF; }
    .stFileUploader { border:2px dashed #1E3A8A; border-radius:5px; padding:10px; }
    .stSelectbox { background-color:#F1F5F9; border-radius:5px; }
    .stNumberInput input { background-color:#F1F5F9; border-radius:5px; }
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
st.markdown("**A Professional, All-in-One Tool for Crop Water Management**", unsafe_allow_html=True)

# Create 3 tabs: Setup, Results, Irrigation Calendar
setup_tab, results_tab, irrig_calendar_tab = st.tabs(["Setup Simulation", "Results", "Irrigation Calendar"])

# --------------------------------------------------------------------------------
# 4. Session State
# --------------------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'main_sim_end_date' not in st.session_state:
    st.session_state.main_sim_end_date = None
if 'forecast_5day_df' not in st.session_state:
    st.session_state.forecast_5day_df = None
if 'api_calls' not in st.session_state:
    st.session_state.api_calls= 0
if 'weather_cache' not in st.session_state:
    st.session_state.weather_cache={}
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date= datetime.now().date()

today_= datetime.now().date()
if st.session_state.last_reset_date!= today_:
    st.session_state.api_calls=0
    st.session_state.last_reset_date= today_

# --------------------------------------------------------------------------------
# 5. Water Balance & Forecast Code
# --------------------------------------------------------------------------------

def download_figure(fig):
    import io
    buf= io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
    buf.seek(0)
    return buf

def compute_Ks(Dr, RAW, TAW):
    if Dr<=RAW:
        return 1.0
    elif Dr>=TAW:
        return 0.0
    else:
        return max(0, (TAW-Dr)/(TAW-RAW))

def compute_Kr(TEW, REW, Ew):
    if Ew<=REW:
        return 1.0
    elif Ew>=TEW:
        return 0.0
    else:
        return (TEW-Ew)/(TEW-REW)

def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Fetch 3-hourly openweather forecast, aggregated daily
    """
    cache_key= f"{lat}_{lon}_{start_date}_{end_date}_forecast"
    if cache_key in st.session_state.weather_cache:
        return st.session_state.weather_cache[cache_key]
    
    if lat==0 or lon==0:
        st.warning("Invalid lat/lon for forecast.")
        return None
    
    # Force 5 day max if user tries more
    maxf= datetime.now().date()+ timedelta(days=5)
    if end_date>maxf:
        end_date= maxf
    
    api_key= "fe2d869569674a4afbfca57707bdf691"  # your openweather key
    url= f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        r= session.get(url, timeout=30)
        r.raise_for_status()
        st.session_state.api_calls+=1
        data= r.json()
        dd={}
        for entry in data['list']:
            dt_= datetime.fromtimestamp(entry['dt']).date()
            if start_date<= dt_<= end_date:
                ds= dt_.strftime("%Y-%m-%d")
                if ds not in dd:
                    dd[ds]={
                        'tmax': entry['main']['temp_max'],
                        'tmin': entry['main']['temp_min'],
                        'precip': entry.get('rain',{}).get('3h',0)
                    }
                else:
                    dd[ds]['tmax']= max(dd[ds]['tmax'], entry['main']['temp_max'])
                    dd[ds]['tmin']= min(dd[ds]['tmin'], entry['main']['temp_min'])
                    dd[ds]['precip']+= entry.get('rain',{}).get('3h',0)
        sorted_days= sorted(dd.keys())
        dates, ETo_list, prec_list=[],[],[]
        for dsi in sorted_days:
            d_= pd.to_datetime(dsi)
            tx_= dd[dsi]['tmax']
            tn_= dd[dsi]['tmin']
            pr_= dd[dsi]['precip']
            if tx_<tn_:
                tx_,tn_= tn_,tx_
            Ra=10
            Tm= (tx_+tn_)/2
            eto_= 0.0023*Ra*(Tm+17.8)*((tx_-tn_)**0.5)
            eto_= max(0,eto_)
            dates.append(d_)
            ETo_list.append(eto_)
            prec_list.append(pr_)
        df= pd.DataFrame({
            "Date": dates,
            "ET0": ETo_list,
            "Precipitation": prec_list,
            "Irrigation": [0]*len(dates)
        }).sort_values("Date").reset_index(drop=True)
        st.session_state.weather_cache[cache_key]= df
        return df
    except:
        st.error("OpenWeather forecast fetch failed.")
        return None

def create_auto_stages_for_crop(crop_name):
    total_d= CROP_DATABASE[crop_name]["total_days_default"]
    init_d= int(total_d*0.2)
    dev_d= int(total_d*0.3)
    mid_d= int(total_d*0.3)
    late_d= total_d-(init_d+dev_d+mid_d)
    
    kcb_mid= CROP_DATABASE[crop_name]["Kcb_mid"]
    kcb_end= CROP_DATABASE[crop_name]["Kcb_end"]
    
    stg=[]
    stg.append({"Start_Day":1, 
                "End_Day": init_d,
                "Kcb":0.15, "Root_Depth_mm":100, 
                "p":0.5, "Ke":1.0})
    stg.append({"Start_Day":init_d+1,
                "End_Day": init_d+dev_d,
                "Kcb":kcb_mid, "Root_Depth_mm":400,
                "p":0.5, "Ke":0.5})
    stg.append({"Start_Day":init_d+dev_d+1,
                "End_Day":init_d+dev_d+mid_d,
                "Kcb":kcb_mid, "Root_Depth_mm":600,
                "p":0.5, "Ke":0.2})
    stg.append({"Start_Day":init_d+dev_d+mid_d+1,
                "End_Day": total_d,
                "Kcb":kcb_end, "Root_Depth_mm":600,
                "p":0.5, "Ke":0.1})
    return pd.DataFrame(stg)

def simulate_SIMdualKc(weather_df, crop_stages_df, soil_df,
                       track_drainage=True, enable_yield=False, Ym=0,Ky=0,
                       use_transp=False, WP_yield=0,
                       enable_leaching=False, nitrate_conc=10.0,
                       total_N_input=100.0, leaching_fraction=0.1,
                       dynamic_root=False, init_root=300, max_root=800, days_to_max=60,
                       initial_layers_state=None):
    """
    If 'initial_layers_state' is given, we skip starting at FC and 
    start from that custom water content in each layer. 
    This is used for the 5-day forecast run.
    """
    if weather_df.empty:
        st.error("Empty weather data!")
        return None
    
    NDAYS= len(weather_df)
    crop_stages_df= crop_stages_df.sort_values("Start_Day").reset_index(drop=True)
    day_kcb= np.zeros(NDAYS)
    day_p=   np.zeros(NDAYS)
    day_ke=  np.zeros(NDAYS)
    day_root= np.zeros(NDAYS)
    # Interpolate
    for i in range(len(crop_stages_df)-1):
        st_i= int(crop_stages_df.iloc[i]['Start_Day'])-1
        en_i= int(crop_stages_df.iloc[i]['End_Day'])-1
        if en_i<0: continue
        en_i= min(en_i, NDAYS-1)
        st_i= max(0, st_i)
        if st_i>en_i: continue
        kcb_s= crop_stages_df.iloc[i]['Kcb']
        kcb_e= crop_stages_df.iloc[i+1]['Kcb']
        p_s= crop_stages_df.iloc[i]['p']
        p_e= crop_stages_df.iloc[i+1]['p']
        ke_s= crop_stages_df.iloc[i]['Ke']
        ke_e= crop_stages_df.iloc[i+1]['Ke']
        rd_s= crop_stages_df.iloc[i]['Root_Depth_mm']
        rd_e= crop_stages_df.iloc[i+1]['Root_Depth_mm']
        L= en_i-st_i+1
        day_kcb[st_i:en_i+1]= np.linspace(kcb_s,kcb_e,L)
        day_p[st_i:en_i+1]= np.linspace(p_s,p_e,L)
        day_ke[st_i:en_i+1]= np.linspace(ke_s,ke_e,L)
        day_root[st_i:en_i+1]= np.linspace(rd_s,rd_e,L)
    i_last= len(crop_stages_df)-1
    st_l= int(crop_stages_df.iloc[i_last]['Start_Day'])-1
    en_l= int(crop_stages_df.iloc[i_last]['End_Day'])-1
    if st_l<0: st_l=0
    if en_l<0: en_l=0
    if en_l>NDAYS-1: en_l=NDAYS-1
    if st_l<=en_l:
        day_kcb[st_l:en_l+1]= crop_stages_df.iloc[i_last]['Kcb']
        day_p[st_l:en_l+1]=   crop_stages_df.iloc[i_last]['p']
        day_ke[st_l:en_l+1]= crop_stages_df.iloc[i_last]['Ke']
        day_root[st_l:en_l+1]= crop_stages_df.iloc[i_last]['Root_Depth_mm']
    if en_l<NDAYS-1:
        day_kcb[en_l+1:]= crop_stages_df.iloc[i_last]['Kcb']
        day_p[en_l+1:]=   crop_stages_df.iloc[i_last]['p']
        day_ke[en_l+1:]= crop_stages_df.iloc[i_last]['Ke']
        day_root[en_l+1:]= crop_stages_df.iloc[i_last]['Root_Depth_mm']
    
    if dynamic_root:
        root_lin= np.linspace(init_root, max_root, min(days_to_max, NDAYS)).tolist()
        if NDAYS> days_to_max:
            root_lin+= [max_root]*(NDAYS-days_to_max)
        day_root= np.array(root_lin[:NDAYS])
    
    TEW= soil_df['TEW'].sum()
    REW= soil_df['REW'].sum()
    E_count= REW # start "dry" or partially
    # Initialize layer states
    if initial_layers_state is not None:
        # use the given states
        SW_layers= initial_layers_state.copy()
    else:
        # start at FC
        SW_layers= [soil_df.iloc[j]['FC']*soil_df.iloc[j]['Depth_mm'] for j in range(len(soil_df))]
    
    results=[]
    cumIrr=0
    cumPrec=0
    for di in range(NDAYS):
        date_i= weather_df.iloc[di]['Date']
        ET0_i= max(0, weather_df.iloc[di]['ET0'])
        PR_i=  max(0, weather_df.iloc[di]['Precipitation'])
        IR_i=  max(0, weather_df.iloc[di]['Irrigation'])
        cumIrr+= IR_i
        cumPrec+= PR_i
        Kcb_i= day_kcb[di]
        p_i=   day_p[di]
        ke0_i= day_ke[di]
        rd_i=  max(1, day_root[di])
        
        # compute TAW, RAW from SW in root zone
        tot_depth=0
        sum_FC=0
        sum_WP=0
        SW_root=0
        for j in range(len(SW_layers)):
            layer_d= soil_df.iloc[j]['Depth_mm']
            FC_j= soil_df.iloc[j]['FC']
            WP_j= soil_df.iloc[j]['WP']
            new_d= tot_depth+ layer_d
            fraction=0
            if new_d<=rd_i:
                fraction=1
            elif tot_depth<rd_i< new_d:
                fraction= (rd_i- tot_depth)/layer_d
            if fraction>0:
                sum_FC+= FC_j*layer_d*fraction
                sum_WP+= WP_j*layer_d*fraction
                SW_root+= SW_layers[j]* fraction
            tot_depth= new_d
        
        TAW_= (sum_FC- sum_WP)
        RAW_= p_i*TAW_
        Dr_= (sum_FC- SW_root)
        Ks_= compute_Ks(Dr_, RAW_, TAW_)
        Kr_= compute_Kr(TEW, REW, E_count)
        Ke_= Kr_* ke0_i
        
        ETc_= (Kcb_i*Ks_ + Ke_)* ET0_i
        Etc_trans= Kcb_i*Ks_*ET0_i
        Etc_evap= Ke_*ET0_i
        infiltration= PR_i + IR_i
        runoff=0
        excess= infiltration- ETc_
        drainage=0
        if track_drainage:
            for j in range(len(SW_layers)):
                fc_j= soil_df.iloc[j]['FC']* soil_df.iloc[j]['Depth_mm']
                gap_j= fc_j- SW_layers[j]
                if gap_j>0 and excess>0:
                    added= min(excess, gap_j)
                    SW_layers[j]+= added
                    excess-= added
            drainage= max(0,excess)
            # clamp to WP-FC
            for j in range(len(SW_layers)):
                fc_j= soil_df.iloc[j]['FC']*soil_df.iloc[j]['Depth_mm']
                wp_j= soil_df.iloc[j]['WP']*soil_df.iloc[j]['Depth_mm']
                SW_layers[j]= max(wp_j, min(fc_j, SW_layers[j]))
        tr_rem= Etc_trans
        if tr_rem>0 and SW_root>0:
            tot_depth=0
            for j in range(len(SW_layers)):
                layer_d= soil_df.iloc[j]['Depth_mm']
                new_d= tot_depth+ layer_d
                fraction=0
                if new_d<=rd_i:
                    fraction=1
                elif tot_depth<rd_i< new_d:
                    fraction= (rd_i- tot_depth)/layer_d
                if fraction>0:
                    fc_j= soil_df.iloc[j]['FC']* layer_d
                    wp_j= soil_df.iloc[j]['WP']* layer_d
                    available_j= SW_layers[j] - wp_j
                    portion= tr_rem*fraction
                    actual_remove= min(portion, available_j)
                    SW_layers[j]-= actual_remove
                    tr_rem-= actual_remove
                    if tr_rem<=0:
                        break
                tot_depth= new_d
        
        ev_rem= Etc_evap
        if ev_rem>0:
            # remove from top layer
            fc0= soil_df.iloc[0]['FC']* soil_df.iloc[0]['Depth_mm']
            wp0= soil_df.iloc[0]['WP']* soil_df.iloc[0]['Depth_mm']
            available_0= SW_layers[0]- wp0
            rm_= min(ev_rem, available_0)
            SW_layers[0]-= rm_
            ev_rem-= rm_
        
        if infiltration>=4.0:
            E_count=0
        else:
            E_count+= Etc_evap
        # clamp
        E_count= max(0, min(E_count, TEW))
        
        # final SW root
        new_SWroot=0
        tot_depth=0
        sum_FC2=0
        for j in range(len(SW_layers)):
            layer_d= soil_df.iloc[j]['Depth_mm']
            new_d= tot_depth+ layer_d
            fraction=0
            if new_d<=rd_i:
                fraction=1
            elif tot_depth<rd_i< new_d:
                fraction= (rd_i- tot_depth)/ layer_d
            if fraction>0:
                new_SWroot+= SW_layers[j]* fraction
                sum_FC2+= soil_df.iloc[j]['FC']* layer_d*fraction
            tot_depth= new_d
        Dr_end= sum_FC2- new_SWroot
        
        yield_val=None
        if enable_yield:
            if ETc_>0 and Ym>0 and Ky>0:
                y_ = Ym*(1- Ky*(1- (ETc_/(Kcb_i*ET0_i+1e-9))))
                yield_val= max(0,y_)
            if use_transp and WP_yield>0:
                yield_val= WP_yield* Etc_trans
        
        leach_val=0
        if enable_leaching:
            leach_val= drainage*10*(nitrate_conc*1e-6)*1000
        
        day_out= {
            "Date": date_i,
            "ET0 (mm)": ET0_i,
            "Precip (mm)": PR_i,
            "Irrigation (mm)": IR_i,
            "Runoff (mm)": runoff,
            "Infiltration (mm)": infiltration,
            "Ks": Ks_,
            "Kr": Kr_,
            "Ke": Ke_,
            "ETa (mm)": ETc_,
            "ETa_transp (mm)": Etc_trans,
            "ETa_evap (mm)": Etc_evap,
            "Drainage (mm)": drainage,
            "Dr_start (mm)": Dr_,
            "Dr_end (mm)": Dr_end,
            "TAW (mm)": TAW_,
            "RAW (mm)": RAW_,
            "SW_root_start (mm)": SW_root,
            "SW_root_end (mm)": new_SWroot,
            "Cumulative_Irrig (mm)": cumIrr,
            "Cumulative_Precip (mm)": cumPrec
        }
        if yield_val is not None:
            day_out["Yield (ton/ha)"]= yield_val
        if enable_leaching:
            day_out["Leaching (kg/ha)"]= leach_val
        for j in range(len(SW_layers)):
            day_out[f"Layer{j}_SW (mm)"]= SW_layers[j]
        results.append(day_out)
    
    return pd.DataFrame(results)

def produce_30day_calendar_with_5day_forecast(simulation_date, forecast_df):
    """
    Creates a 30-day wall calendar starting from simulation_date.
    The first 5 days (in forecast_df) show ET0, ETa, and an 'IRRIG' label if 
    ETa/ET0 < 0.8 (example threshold). The remaining days only show the date.
    """
    # We'll assume forecast_df has daily data with columns "Date","ET0 (mm)","ETa (mm)"
    # but we must run a short simulation for the forecast 5 days to get ETa. 
    # We'll store that in a dictionary keyed by date.
    forecast_dict= {}
    if forecast_df is not None and not forecast_df.empty:
        forecast_df["Date"]= pd.to_datetime(forecast_df["Date"])
        for i, row in forecast_df.iterrows():
            d_= row["Date"].date()
            et0= row["ET0 (mm)"] if "ET0 (mm)" in row else row.get("ET0",0)
            eta= row["ETa (mm)"] if "ETa (mm)" in row else 0
            ratio= 0
            if et0>0: ratio= eta/et0
            rec= ""
            if ratio<0.8:
                rec= "IRRIG"
            forecast_dict[d_]= {"ET0":round(et0,2),"ETa":round(eta,2), "Irrig":rec}
    
    # Build a 30 day calendar from simulation_date
    cal_html= f"<h3 style='text-align:center;'>{simulation_date.strftime('%B %Y')}</h3>"
    cal_html+= "<table style='width:100%; border-collapse:collapse; text-align:center;'>"
    weekdays= ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    cal_html+= "<tr style='background-color:#1E3A8A; color:white;'>"
    for wd in weekdays:
        cal_html+= f"<th style='padding:5px; border:1px solid #ddd; font-size:12px;'>{wd}</th>"
    cal_html+= "</tr>"
    
    # make a list of 30 days
    date_list= [simulation_date+ timedelta(days=x) for x in range(30)]
    # figure out the weekday of day0
    first_weekday= simulation_date.weekday() # 0=Monday
    week=[]
    # pad
    for _ in range(first_weekday):
        week.append("")
    for d_ in date_list:
        text= f"<strong>{d_.day}</strong>"
        if d_ in forecast_dict:
            et0_= forecast_dict[d_]["ET0"]
            eta_= forecast_dict[d_]["ETa"]
            irr_= forecast_dict[d_]["Irrig"]
            text+= f"<br/>ET0:{et0_}<br/>ETa:{eta_}"
            if irr_:
                text+= f"<br/><span style='color:red;font-weight:bold;'>{irr_}</span>"
        week.append(text)
        if len(week)==7:
            cal_html+= "<tr>"
            for cell in week:
                cal_html+= f"<td style='padding:5px; border:1px solid #ddd; font-size:12px; vertical-align:top;'>{cell}</td>"
            cal_html+= "</tr>"
            week=[]
    if week:
        while len(week)<7:
            week.append("")
        cal_html+="<tr>"
        for cell in week:
            cal_html+= f"<td style='padding:5px; border:1px solid #ddd; font-size:12px; vertical-align:top;'>{cell}</td>"
        cal_html+="</tr>"
    cal_html+="</table>"
    return cal_html

# --------------------------------------------------------------------------------
# 6. SETUP TAB
# --------------------------------------------------------------------------------
with setup_tab:
    st.markdown("<span class='small-header'>1) Select Crop</span>",unsafe_allow_html=True)
    all_crops= list(CROP_DATABASE.keys())
    selected_crop= st.selectbox("Pick a Crop", all_crops)
    st.write(f"**Selected Crop**: {selected_crop}")
    st.write(f"- Kc_mid={CROP_DATABASE[selected_crop]['Kc_mid']}, Kc_end={CROP_DATABASE[selected_crop]['Kc_end']}")
    st.write(f"- Kcb_mid={CROP_DATABASE[selected_crop]['Kcb_mid']}, Kcb_end={CROP_DATABASE[selected_crop]['Kcb_end']}")
    
    st.markdown("<span class='small-header'>2) Weather Data</span>", unsafe_allow_html=True)
    weather_file= st.file_uploader("Upload CSV with [Date,ET0,Precipitation,Irrigation], or use forecast", type=["csv","txt"])
    
    st.markdown("<span class='small-header'>3) Crop Stage Data</span>",unsafe_allow_html=True)
    use_custom_stages= st.checkbox("Upload custom stages for this crop?", value=False)
    custom_crop_file= None
    if use_custom_stages:
        custom_crop_file= st.file_uploader("Crop Stage CSV (Start_Day,End_Day,Kcb,Root_Depth_mm,p,Ke)", type=["csv","txt"])
    
    st.markdown("<span class='small-header'>4) Soil Layers Data</span>",unsafe_allow_html=True)
    soil_file= st.file_uploader("Soil data (Depth_mm,FC,WP,TEW,REW) or default", type=["csv","txt"])
    
    st.markdown("<span class='small-header'>5) Additional Options</span>",unsafe_allow_html=True)
    cA, cB= st.columns(2)
    with cA:
        track_drainage= st.checkbox("Track Drainage?", value=True)
        enable_yield= st.checkbox("Enable Yield Estimation?", value=False)
        if enable_yield:
            st.write("**Yield Options**")
            ym_= st.number_input("Max Yield (ton/ha)?", min_value=0.0,value=10.0)
            ky_= st.number_input("Ky factor", min_value=0.0, value=1.0)
            use_transp= st.checkbox("Use Transp-based (WP_yield)?",value=False)
            if use_transp:
                wp_yield= st.number_input("WP_yield (ton/ha per mm)?", min_value=0.0, value=0.012, step=0.001)
            else:
                wp_yield=0
        else:
            ym_=ky_=0
            use_transp=False
            wp_yield=0
    
    with cB:
        enable_leaching= st.checkbox("Enable Leaching?", value=False)
        nitrate_conc= st.number_input("Nitrate mg/L", min_value=0.0, value=10.0)
        totalN= st.number_input("Total N input (kg/ha)?", min_value=0.0, value=100.0)
        lf= st.number_input("Leaching Fraction (0-1)?", min_value=0.0, max_value=1.0, value=0.1)
    
    st.markdown("<span class='small-header'>6) 5-day ETA Forecast Option</span>",unsafe_allow_html=True)
    enable_forecast= st.checkbox("Enable 5-day forecast from OpenWeather?", value=True)
    lat_= st.number_input("Latitude", value=35.0)
    lon_= st.number_input("Longitude", value=-80.0)
    
    st.markdown("<span class='small-header'>7) Dynamic Root Growth?</span>",unsafe_allow_html=True)
    dynamic_root= st.checkbox("Dynamic Root Growth?", value=False)
    init_rd=300
    max_rd=800
    days_mx=60
    if dynamic_root:
        init_rd= st.number_input("Initial Root Depth (mm)", value=300, min_value=50)
        max_rd= st.number_input("Max Root Depth (mm)", value=800, min_value=50)
        days_mx= st.number_input("Days to Max Root Depth", min_value=1, value=60)
    
    st.markdown("<span class='small-header'>8) Run Simulation</span>",unsafe_allow_html=True)
    run_button= st.button("Run Simulation")
    if run_button:
        st.session_state["simulation_done"]= True
        st.success("Simulation done! Visit 'Results' tab for main sim and 'Irrigation Calendar' for 5-day forecast.")
        
        # 1) Load or fetch weather for main sim
        if weather_file is not None:
            try:
                main_wdf= pd.read_csv(weather_file)
                if "Date" not in main_wdf.columns:
                    st.error("No 'Date' col in weather file!")
                    st.stop()
                if pd.api.types.is_string_dtype(main_wdf["Date"]):
                    main_wdf["Date"]= pd.to_datetime(main_wdf["Date"])
                main_wdf= main_wdf.sort_values("Date").reset_index(drop=True)
            except:
                st.warning("Failed to parse weather file => fallback to forecast 5 days.")
                if enable_forecast:
                    sd_= datetime.now().date()
                    ed_= sd_+ timedelta(days=4)
                    main_wdf= fetch_weather_data(lat_, lon_, sd_, ed_)
                else:
                    st.error("No weather data => stop.")
                    st.stop()
        else:
            if enable_forecast:
                sd_= datetime.now().date()
                ed_= sd_+ timedelta(days=4)
                main_wdf= fetch_weather_data(lat_, lon_, sd_, ed_)
            else:
                st.error("No weather data => cannot run.")
                st.stop()
        
        if main_wdf is None or main_wdf.empty:
            st.error("No valid main weather data => stop.")
            st.stop()
        
        # 2) Crop stages
        if use_custom_stages and custom_crop_file is not None:
            try:
                stage_df= pd.read_csv(custom_crop_file)
            except:
                st.error("Cannot parse custom stages => using auto.")
                stage_df= create_auto_stages_for_crop(selected_crop)
        elif use_custom_stages and custom_crop_file is None:
            st.warning("No file => auto stage.")
            stage_df= create_auto_stages_for_crop(selected_crop)
        else:
            stage_df= create_auto_stages_for_crop(selected_crop)
        
        # 3) Soil
        if soil_file is not None:
            try:
                soil_df= pd.read_csv(soil_file)
            except:
                st.warning("Soil parse fail => using default 2-layer.")
                soil_df= pd.DataFrame({
                    "Depth_mm":[200, 100],
                    "FC":[0.30, 0.30],
                    "WP":[0.15, 0.15],
                    "TEW":[20,0],
                    "REW":[5,0]
                })
        else:
            soil_df= pd.DataFrame({
                "Depth_mm":[200,100],
                "FC":[0.30,0.30],
                "WP":[0.15,0.15],
                "TEW":[20,0],
                "REW":[5,0]
            })
        
        # 4) Main simulation run
        res_df= simulate_SIMdualKc(weather_df= main_wdf,
                                   crop_stages_df= stage_df,
                                   soil_df= soil_df,
                                   track_drainage= track_drainage,
                                   enable_yield= enable_yield, Ym= ym_, Ky= ky_,
                                   use_transp= use_transp, WP_yield= wp_yield,
                                   enable_leaching= enable_leaching, nitrate_conc= nitrate_conc,
                                   total_N_input= totalN, leaching_fraction= lf,
                                   dynamic_root= dynamic_root,
                                   init_root= init_rd, max_root= max_rd, days_to_max= days_mx)
        st.session_state.results_df= res_df
        # store last date
        last_date= res_df["Date"].iloc[-1]
        st.session_state.main_sim_end_date= last_date
        
        # 5) Next 5-day forecast run for ETa
        if enable_forecast:
            # fetch next 5 days from the day after last_date or from last_date?
            forecast_start= last_date.date() + timedelta(days=1)
            forecast_end= forecast_start+ timedelta(days=4)
            fore_wdf= fetch_weather_data(lat_, lon_, forecast_start, forecast_end)
            if fore_wdf is not None and not fore_wdf.empty:
                # We must do a short 5-day model run starting from the final SW profile at the end of the main sim
                # We'll get the final layer states
                final_layers=[]
                n_layers= len(soil_df)
                # look at last day row from res_df
                row_last= res_df.iloc[-1]
                for j in range(n_layers):
                    colname= f"Layer{j}_SW (mm)"
                    if colname in row_last:
                        final_layers.append(row_last[colname])
                    else:
                        # fallback
                        final_layers.append(soil_df.iloc[j]["FC"]*soil_df.iloc[j]["Depth_mm"])
                
                # We'll build a minimal 5-day stage DF: basically the final stage
                # or we can re-use the last day of the main stage
                short_stage= pd.DataFrame([{
                    "Start_Day":1,
                    "End_Day": 5,
                    "Kcb": row_last.get("Ks",0.9)*CROP_DATABASE[selected_crop]["Kcb_end"], 
                    # or just use the end Kcb?
                    "Root_Depth_mm": row_last.get("Root_Depth (mm)", 600),
                    "p":0.5, 
                    "Ke": 0.1
                }])
                
                # run forecast sim
                fore_res= simulate_SIMdualKc(weather_df= fore_wdf,
                                             crop_stages_df= short_stage,
                                             soil_df= soil_df,
                                             track_drainage= track_drainage,
                                             initial_layers_state= final_layers)
                st.session_state.forecast_5day_df= fore_res
            else:
                st.session_state.forecast_5day_df= None
        else:
            st.session_state.forecast_5day_df= None

# --------------------------------------------------------------------------------
# 7. RESULTS TAB
# --------------------------------------------------------------------------------
with results_tab:
    if not st.session_state.get("simulation_done",False):
        st.info("Please run the simulation in 'Setup Simulation' tab.")
    else:
        res_df= st.session_state.results_df
        if res_df is None or res_df.empty:
            st.warning("No results. Re-run simulation.")
        else:
            st.markdown("## Simulation Results")
            st.dataframe(res_df)
            st.download_button("Download Results (.csv)", res_df.to_csv(index=False), "results.csv", mime="text/csv")
            
            chart_choice= st.selectbox("Select Chart", 
                                       ["Daily ET Components","Root Zone Depletion","Daily Drainage","Soil Water in Root Zone","Yield","Leaching"])
            if chart_choice=="Daily ET Components":
                fig, ax= plt.subplots(figsize=(10,4))
                ax.plot(res_df["Date"], res_df["ETa (mm)"], label="ETa total")
                ax.plot(res_df["Date"], res_df["ETa_transp (mm)"], label="ETa transp")
                ax.plot(res_df["Date"], res_df["ETa_evap (mm)"], label="ETa evap")
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf= download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="et_components.png", mime="image/png")
            elif chart_choice=="Root Zone Depletion":
                fig, ax= plt.subplots(figsize=(10,4))
                ax.plot(res_df["Date"], res_df["Dr_start (mm)"], label="Dr start")
                ax.plot(res_df["Date"], res_df["Dr_end (mm)"], label="Dr end")
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf= download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="root_depletion.png", mime="image/png")
            elif chart_choice=="Daily Drainage":
                fig, ax= plt.subplots(figsize=(10,4))
                ax.plot(res_df["Date"], res_df["Drainage (mm)"], label="Drainage")
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf= download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="daily_drainage.png", mime="image/png")
            elif chart_choice=="Soil Water in Root Zone":
                fig, ax= plt.subplots(figsize=(10,4))
                ax.plot(res_df["Date"], res_df["SW_root_start (mm)"], label="RootZ start")
                ax.plot(res_df["Date"], res_df["SW_root_end (mm)"], label="RootZ end")
                ax.set_xlabel("Date"); ax.set_ylabel("mm")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf= download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="soilwater_rootzone.png", mime="image/png")
            elif chart_choice=="Yield" and "Yield (ton/ha)" in res_df.columns:
                fig, ax= plt.subplots(figsize=(10,4))
                ax.plot(res_df["Date"], res_df["Yield (ton/ha)"], label="Yield")
                ax.set_xlabel("Date"); ax.set_ylabel("ton/ha")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf= download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="yield.png", mime="image/png")
            elif chart_choice=="Leaching" and "Leaching (kg/ha)" in res_df.columns:
                fig, ax= plt.subplots(figsize=(10,4))
                ax.plot(res_df["Date"], res_df["Leaching (kg/ha)"], label="Leaching")
                ax.set_xlabel("Date"); ax.set_ylabel("kg/ha")
                ax.legend(frameon=False)
                ax.grid(False)
                st.pyplot(fig)
                buf= download_figure(fig)
                st.download_button("Download Chart", data=buf, file_name="leaching.png", mime="image/png")

# --------------------------------------------------------------------------------
# 8. IRRIGATION CALENDAR TAB
# --------------------------------------------------------------------------------
with irrig_calendar_tab:
    if not st.session_state.get("simulation_done", False):
        st.info("Run the simulation first.")
    else:
        st.markdown("## 30-Day Irrigation Calendar")
        # We'll produce a 30-day calendar starting from the main simulation date 
        # (the day the user ran the sim). Then for the next 5 days, 
        # we use st.session_state.forecast_5day_df to show ET0,ETa, IRRIG?

        # We'll define the "simulation day" as the final day from the main sim 
        # or the day we started. 
        # For clarity, let's define it as "today" 
        sim_day= datetime.now().date()
        
        # We'll run or re-check st.session_state.forecast_5day_df for ET0 and ETa
        # We expect the columns "ET0 (mm)" and "ETa (mm)" if the forecast sim was done 
        fdf= st.session_state.forecast_5day_df
        if fdf is None or fdf.empty:
            st.write("No 5-day forecast data available or forecast was disabled.")
        else:
            # rename columns to unify
            # expecting "Date","ET0 (mm)","ETa (mm)"
            # the forecast run includes columns "ETa (mm)"
            # we produce the calendar
            cal_html= produce_30day_calendar_with_5day_forecast(sim_day, fdf)
            st.markdown(cal_html, unsafe_allow_html=True)

st.markdown('<div class="footer">© 2025 Advanced AgriWaterBalance | Contact: support@agriwaterbalance.com</div>', unsafe_allow_html=True)
