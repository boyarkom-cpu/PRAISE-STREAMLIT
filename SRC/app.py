import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import html

from praise_engine import (
    load_and_validate_csv, 
    prepare_and_filter_data, 
    calculate_statistical_bounds, 
    evaluate_transaction_risk,
    BUSINESS_FLOOR_MULTIPLIER,
    EXTERNAL_BENCHMARK_MULTIPLIER
)
from data_generator import generate_data

from pathlib import Path

# --- Page Configuration ---
st.set_page_config(page_title="PRAISE Dashboard", layout="wide")

# --- 1. Data Loading & Initialization ---
# Robust Path Resolution: Find CSV in the project root regardless of working directory
project_root = Path(__file__).parent.parent
csv_path = project_root / "mockup_customs_data.csv"
current_mtime = csv_path.stat().st_mtime if csv_path.exists() else 0.0

@st.cache_data
def load_historical_data(path_str: str, mtime: float) -> tuple[pd.DataFrame, dict]:
    """
    Loads and filters historical data, caching the result to prevent UI lag.
    Cache invalidates automatically when the file's modification time (mtime) changes.
    """
    _ = mtime  # Explicitly reference mtime to ensure cache invalidation & appease static analysis
    try:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError("mockup_customs_data.csv is missing. Please generate it first.")
            
        raw_df = load_and_validate_csv(str(p))
        filtered_df, metrics = prepare_and_filter_data(raw_df)
        return filtered_df, metrics
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame(), {}

df_historical, filter_metrics = load_historical_data(str(csv_path), current_mtime)

# --- 2. State Management ---
# Anti-Pollution Rule: Store historical data as a Read-Only baseline
# Update baseline automatically if the CSV file was regenerated (mtime changed) and is valid, or if initializing
if 'historical_df' not in st.session_state or (st.session_state.csv_mtime != current_mtime and not df_historical.empty):
    st.session_state.historical_df = df_historical
    st.session_state.csv_mtime = current_mtime

if 'alert_counter' not in st.session_state:
    st.session_state.alert_counter = 1
    st.session_state.alert_date = datetime.date.today()

if 'current_alert' not in st.session_state:
    st.session_state.current_alert = None
    
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []  # Store UI inputs for plotting without polluting historical baseline

# --- Schema Migration (Clear Incompatible State on Hot-Reload) ---
SCHEMA_VERSION = 1
if st.session_state.get('schema_version', 0) < SCHEMA_VERSION:
    st.session_state.current_alert = None
    st.session_state.user_inputs = []
    if 'previous_profile_tuple' in st.session_state:
        del st.session_state['previous_profile_tuple']
    st.session_state.schema_version = SCHEMA_VERSION

def generate_alert_ref() -> str:
    """Generates a sequential alert reference number (e.g., 690101001) using a persistent file to avoid collisions across sessions."""
    today = datetime.date.today()
    date_str = today.strftime("%Y-%m-%d")
    
    # Calculate Thai Buddhist Era Year (CE + 543)
    be_year = today.year + 543
    yy = str(be_year)[-2:]
    mmdd = today.strftime("%m%d")
    
    counter_file = project_root / 'alert_counter.json'
    current_count = 1
    
    # Read existing counter
    if counter_file.exists():
        try:
            with open(counter_file, 'r') as f:
                data = json.load(f)
                if data.get('date') == date_str:
                    current_count = data.get('count', 0) + 1
        except Exception:
            pass
            
    # Write updated counter
    try:
        with open(counter_file, 'w') as f:
            json.dump({'date': date_str, 'count': current_count}, f)
    except Exception:
        pass
        
    running_no = f"{current_count:03d}"
    return f"{yy}{mmdd}{running_no}"

# --- 3. Main Dashboard UI & Interactive Input ---
st.title("ปัญญาประดิษฐ์เพื่อประเมินความเสี่ยงด้านราคาศุลกากรเชิงสถิติ (PRAISE : Pricing Risk AI & Statistical Engine)")

with st.container(border=True):
    st.subheader("📝 เริ่มประเมินความเสี่ยงด้านราคาศุลกากร")
    
    if st.session_state.historical_df.empty:
        st.markdown("<span style='color:red'>No historical data available.</span>", unsafe_allow_html=True)
        st.info("Please generate the 'mockup_customs_data.csv' dataset to continue.")
        if st.button("Generate Mockup Data"):
            with st.spinner("Generating 5,000+ rows..."):
                generate_data()
                st.cache_data.clear()
                st.rerun()
        st.stop()

    # Create Profile List for Dropdown (Use .copy() to enforce Read-Only Baseline)
    df_hist = st.session_state.historical_df.copy()
    df_hist['Profile'] = df_hist['Importer_ID'].astype(str) + " | " + df_hist['Cleaned_Description'].astype(str)

    unique_profiles = df_hist[['Importer_ID', 'Cleaned_Description']].drop_duplicates()
    profile_tuples = list(unique_profiles.itertuples(index=False, name=None))

    if not profile_tuples:
        st.warning("⚠️ ไม่พบข้อมูล Profile (Importer_ID + Description) ในฐานข้อมูลประวัติ กรุณาตรวจสอบไฟล์ข้อมูล")
        st.stop()

    selected_tuple = st.selectbox(
        "Target Profile Selection", 
        profile_tuples,
        format_func=lambda x: f"{x[0]} | {x[1]}"
    )
    importer_id, cleaned_description = selected_tuple
    selected_profile = f"{importer_id} | {cleaned_description}"

    # --- State Clear on Profile Change ---
    if 'previous_profile_tuple' not in st.session_state:
        st.session_state.previous_profile_tuple = selected_tuple

    if st.session_state.previous_profile_tuple != selected_tuple:
        # Clear alert and transient plot points when user switches target to prevent state bleeding completely
        st.session_state.current_alert = None
        st.session_state.user_inputs = []
        st.session_state.previous_profile_tuple = selected_tuple

    # Calculate bounds for the selected profile dynamically
    try:
        stats_result = calculate_statistical_bounds(df_hist, importer_id, cleaned_description)
    except Exception as e:
        stats_result = {
            'status': 'ERROR',
            'message': f"ระบบพบข้อผิดพลาดในข้อมูลประวัติ: {str(e)}",
            'cv_percent': 0.0,
            'is_volatile': False,
            'lower_bound': 0.0
        }

    # Dynamic Help Text (Hint)
    if stats_result['status'] == 'SUCCESS':
        lower_bound = stats_result['lower_bound']
        st.info(f"💡 Hint: Lower Bound is {lower_bound:,.2f} THB.")
    else:
        st.markdown(f"<span style='color:orange'>{stats_result.get('message', 'Insufficient data for this profile.')}</span>", unsafe_allow_html=True)

    user_price = st.number_input("CIF Unit Price (THB)", min_value=0.01, value=None, step=1.0)

    if st.button("💻 กดเพื่อประเมินความเสี่ยง (Run PRAISE Risk Assessment)", use_container_width=True):
        if user_price is None:
            st.error("⚠️ กรุณาระบุราคาสินค้า (CIF Unit Price) ก่อนทำการประเมิน")
        elif stats_result['status'] == 'SUCCESS':
            risk_eval = evaluate_transaction_risk(user_price, stats_result)
            
            alert_ref = None
            if risk_eval['is_anomaly']:
                alert_ref = generate_alert_ref()
                
            st.session_state.current_alert = {
                'profile_tuple': selected_tuple,
                'price': user_price,
                'stats': stats_result,
                'eval': risk_eval,
                'ref_no': alert_ref,
                'timestamp': datetime.datetime.now()
            }
            
            # Store for plotting (Does not pollute historical DataFrame)
            st.session_state.user_inputs.append({
                'Profile_Tuple': selected_tuple,
                'Import_Date': datetime.datetime.now(),
                'Unit_Price_THB_CIF': user_price,
                'is_anomaly': risk_eval['is_anomaly']
            })
            
            # Capping / Pruning: Prevent memory leak and UI lag by keeping only the last 50 inputs
            if len(st.session_state.user_inputs) > 50:
                st.session_state.user_inputs = st.session_state.user_inputs[-50:]
        else:
            st.markdown(f"<span style='color:red'>Cannot assess risk: {stats_result.get('message', '')}</span>", unsafe_allow_html=True)

alert_info = st.session_state.current_alert
action_desc_html = ""
if alert_info and alert_info.get('profile_tuple') == selected_tuple and alert_info.get('eval', {}).get('is_anomaly'):
    eval_res = alert_info.get('eval', {})
    action_code = str(eval_res.get('action_code', ''))
    if action_code == '88':
        action_desc_html = "<strong>รหัสสั่งการตรวจ : 88</strong> (กลุ่มผันผวน (Volatile Group) ตรวจสอบโครงสร้างต้นทุน (Cost Breakdown))"
    elif action_code == '89':
        action_desc_html = "<strong>รหัสสั่งการตรวจ : 89</strong> (กลุ่มเสถียร (Stable Group) ตรวจสอบเอกสารใบแจ้งหนี้ (Invoice) และหลักฐานการโอนเงิน)"
    else:
        safe_action_code = html.escape(action_code) if action_code else "N/A"
        action_desc_html = f"<strong>Action Code:</strong> {safe_action_code}"

tab1, tab2 = st.tabs(["Executive Dashboard", "Officer Risk Terminal"])

with tab1:
    st.header("Executive Dashboard (Visual Evidence)")
    
    # Alert Cross-Reference
    if action_desc_html:
        action_desc_md = action_desc_html.replace("<strong>", "**").replace("</strong>", "**")
        st.markdown(f"### ⚠️ :red[PRAISE Alert Ref No: {alert_info['ref_no']}]")
        st.markdown(f"#### :red[{action_desc_md}]")
        
    # Metrics Panel
    col1, col2 = st.columns(2)
    with col1:
        shipment_count = len(df_hist[(df_hist['Importer_ID'] == importer_id) & (df_hist['Cleaned_Description'] == cleaned_description)])
        st.metric("Total Processed Shipment", f"{shipment_count:,}")
    with col2:
        if alert_info and alert_info.get('profile_tuple') == selected_tuple:
            st.metric("Last User Input Price", f"{alert_info.get('price', 0):,.2f} THB")
        else:
            st.metric("Last User Input Price", "-")

    # Time-Series Plot
    group_df = df_hist[(df_hist['Importer_ID'] == importer_id) & (df_hist['Cleaned_Description'] == cleaned_description)].copy()
    if not group_df.empty and stats_result['status'] == 'SUCCESS':
        # Guard Clause: Fail Gracefully if Import_Date is missing or not datetime
        if 'Import_Date' not in group_df.columns or not pd.api.types.is_datetime64_any_dtype(group_df['Import_Date']):
            st.warning("⚠️ กราฟ Time-Series ถูกปิดใช้งานชั่วคราว: ไม่พบคอลัมน์ Import_Date หรือรูปแบบวันที่ในฐานข้อมูลไม่ถูกต้อง")
        else:
            fig = px.scatter(
                group_df, x='Import_Date', y='Unit_Price_THB_CIF', 
                title=f"Historical Price Trend: {selected_profile}",
                labels={'Unit_Price_THB_CIF': 'Unit Price (THB)', 'Import_Date': 'Import Date'}
            )
            fig.update_traces(marker=dict(color='blue', size=8, opacity=0.6), name='Historical Data')
            
            # Overlay Median (Base Price)
            fig.add_hline(
                y=stats_result['median'], 
                line_dash="solid", 
                line_color="green", 
                annotation_text="Median (ราคาปกติ)", 
                annotation_position="top right"
            )
            
            # Overlay MAD Band (Upper)
            fig.add_hline(
                y=stats_result['median'] + stats_result['mad'], 
                line_dash="dot", 
                line_color="gray", 
                annotation_text="+1 MAD", 
                annotation_position="top left"
            )
            
            # Overlay MAD Band (Lower)
            fig.add_hline(
                y=stats_result['median'] - stats_result['mad'], 
                line_dash="dot", 
                line_color="gray", 
                annotation_text="-1 MAD", 
                annotation_position="bottom left"
            )

            # Overlay Floors (Show Both Lines for Transparency)
            business_floor = stats_result['median'] * BUSINESS_FLOOR_MULTIPLIER
            external_bound = stats_result.get('external_bound', 0.0)

            fig.add_hline(
                y=business_floor, 
                line_dash="dashdot", 
                line_color="orange", 
                annotation_text=f"Floor 1: {int(BUSINESS_FLOOR_MULTIPLIER * 100)}% ของราคาปกติ", 
                annotation_position="bottom left"
            )
            
            if external_bound > 0:
                fig.add_hline(
                    y=external_bound, 
                    line_dash="dashdot", 
                    line_color="purple", 
                    annotation_text=f"Floor 2: {int(EXTERNAL_BENCHMARK_MULTIPLIER * 100)}% ราคาอ้างอิงภายนอก", 
                    annotation_position="bottom right"
                )

            # Overlay Historical Lower Bound
            fig.add_hline(
                y=stats_result['lower_bound'], 
                line_dash="dash", 
                line_color="red", 
                annotation_text="Lower Bound (จุดตัดความเสี่ยง)", 
                annotation_position="bottom right"
            )
                
            # Overlay Temporal User Inputs (Anti-Pollution Rule respected)
            user_points = [p for p in st.session_state.user_inputs if p.get('Profile_Tuple') == selected_tuple]
            if user_points:
                user_df = pd.DataFrame(user_points)
                normal_df = user_df[~user_df['is_anomaly']]
                anomaly_df = user_df[user_df['is_anomaly']]
                
                if not normal_df.empty:
                    fig.add_trace(go.Scatter(
                        x=normal_df['Import_Date'], y=normal_df['Unit_Price_THB_CIF'],
                        mode='markers', marker=dict(color='orange', size=12, symbol='star'),
                        name='User Input (Normal)'
                    ))
                if not anomaly_df.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_df['Import_Date'], y=anomaly_df['Unit_Price_THB_CIF'],
                        mode='markers', marker=dict(color='red', size=12, symbol='circle-x'),
                        name='User Input (Anomaly)'
                    ))

            # Apply Mobile-Friendly Layout Modifications
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            config = {
                'displayModeBar': True,
                'modeBarButtons': [['zoomIn2d', 'zoomOut2d', 'autoScale2d']],
                'displaylogo': False
            }
            st.plotly_chart(fig, width='stretch', config=config)
    else:
        st.markdown("<span style='color:blue'>Insufficient historical data to plot time-series.</span>", unsafe_allow_html=True)

with tab2:
    st.header("Officer Risk Terminal")
    
    # Alert Cross-Reference
    if alert_info and alert_info.get('profile_tuple') == selected_tuple and alert_info.get('eval', {}).get('is_anomaly'):
        st.markdown(f"### ⚠️ <span style='color:red'>PRAISE Alert Ref No: {alert_info['ref_no']}</span>", unsafe_allow_html=True)
        
    if alert_info and alert_info.get('profile_tuple') == selected_tuple:
        decl_no = "SIM-88889999"
        item_no = "1"
        eval_res = alert_info.get('eval', {})
        
        if eval_res.get('is_anomaly'):
            st.markdown(f"""
            <div style="background-color:#ffe6e6;color:black;padding:20px;border-radius:10px;border: 2px solid red;">
                <h3 style="color:red;margin-top:0;">🛑 {eval_res['status_label']} (Price Volatility / Under-valuation detected)</h3>
                <p><strong>Alert Ref No:</strong> {alert_info['ref_no']}</p>
                <p><strong>Target:</strong> Declaration Number: {decl_no} / Item Number: {item_no}</p>
                <p>{action_desc_html}</p>
                <h4 style="color:darkred;">Instruction: ขอเอกสารเพิ่มเติมเพื่อตรวจสอบราคาศุลกากรของสินค้ารายการที่ {item_no}</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
            <div style="background-color:#e6ffe6;color:black;padding:20px;border-radius:10px;border: 2px solid green;">
                <h3 style="color:green;margin-top:0;">✅ {eval_res['status_label']} (Low Risk)</h3>
                <p><strong>Target:</strong> Declaration Number: {decl_no} / Item Number: {item_no}</p>
                <h4 style="color:darkgreen;">Instruction: ระบบไม่พบความเสี่ยงด้านราคาศุลกากร </h4>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Waiting for User Input... Please click the '💻 Run PRAISE Risk Assessment' button to see results.")
