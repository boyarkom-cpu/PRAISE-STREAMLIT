import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import html
from filelock import FileLock
import math

from praise_engine import (
    load_and_validate_csv, 
    prepare_and_filter_data, 
    calculate_statistical_bounds, 
    evaluate_transaction_risk,
    train_praise_anomaly_model,
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

@st.cache_resource
def get_ml_engine(mtime: float):
    _ = mtime
    try:
        model, preprocessor = train_praise_anomaly_model()
        return model, preprocessor
    except Exception as e:
        st.error(f"ML Engine Initialization Error: {e}")
        return None, None

feedback_lake_path = project_root / 'mockup_feedback_lake.csv'
feedback_mtime = feedback_lake_path.stat().st_mtime if feedback_lake_path.exists() else 0.0
ml_model, ml_preprocessor = get_ml_engine(feedback_mtime)
# --- 2. State Management ---
# Anti-Pollution Rule: Store historical data as a Read-Only baseline
# Update baseline automatically if the CSV file was regenerated (mtime changed) and is valid, or if initializing
if 'historical_df' not in st.session_state or (st.session_state.get('csv_mtime', 0.0) != current_mtime and not df_historical.empty):
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

    # Determine defaults for Origin and Transport to use in AI evaluation without requiring user input
    group_df_full = df_hist[(df_hist['Importer_ID'] == importer_id) & (df_hist['Cleaned_Description'] == cleaned_description)]
    default_origin = group_df_full['Origin_Country'].mode().iloc[0] if not group_df_full['Origin_Country'].mode().empty else 'Unknown'
    default_transport = group_df_full['Transport_Mode'].mode().iloc[0] if not group_df_full['Transport_Mode'].mode().empty else 'Unknown'
    default_broker = group_df_full['Broker_ID'].mode().iloc[0] if not group_df_full['Broker_ID'].mode().empty else 'Unknown'
    default_brand = group_df_full['Brand'].mode().iloc[0] if not group_df_full['Brand'].mode().empty else 'Unknown'
    default_year = group_df_full['Product_Year'].mode().iloc[0] if not group_df_full['Product_Year'].mode().empty else 'Unknown'
    default_port = group_df_full['Port_of_Entry'].mode().iloc[0] if not group_df_full['Port_of_Entry'].mode().empty else 'Unknown'
    
    default_quantity = float(group_df_full['Quantity'].median()) if not group_df_full['Quantity'].isna().all() else 1.0
    default_weight = float(group_df_full['Gross_Weight_KG'].median()) if not group_df_full['Gross_Weight_KG'].isna().all() else 1.0
    
    user_price = st.number_input("CIF Unit Price (THB)", min_value=0.01, value=None, step=1.0)

    if st.button("💻 กดเพื่อประเมินความเสี่ยง (Run PRAISE Risk Assessment)", use_container_width=True):
        if user_price is None:
            st.error("⚠️ กรุณาระบุราคาสินค้า (CIF Unit Price) ก่อนทำการประเมิน")
        elif stats_result['status'] in ('SUCCESS', 'INSUFFICIENT_DATA'):
            user_input_row = {
                'Unit_Price_THB_CIF': user_price,
                'Origin_Country': default_origin,
                'Transport_Mode': default_transport,
                'Broker_ID': default_broker,
                'Brand': default_brand,
                'Product_Year': default_year,
                'Port_of_Entry': default_port,
                'Quantity': default_quantity,
                'Gross_Weight_KG': default_weight
            }
            
            risk_eval = evaluate_transaction_risk(
                user_price, 
                stats_result,
                ml_model=ml_model,
                ml_preprocessor=ml_preprocessor,
                user_input_row=user_input_row
            )
            
            alert_ref = None
            if risk_eval['is_anomaly']:
                alert_ref = generate_alert_ref()
                
            st.session_state.current_alert = {
                'profile_tuple': selected_tuple,
                'price': user_price,
                'stats': stats_result,
                'eval': {**risk_eval, 'user_input_row': user_input_row},
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
    elif action_code == '890':
        action_desc_html = "<strong>รหัสสั่งการตรวจ : 890</strong> (Cold Start — ข้อมูลประวัตินำเข้าน้อยกว่าเกณฑ์ขั้นต่ำ บังคับตรวจสอบโดยเจ้าหน้าที่ (Human-in-the-loop))"
    elif action_code == '90':
        action_desc_html = "<strong>รหัสสั่งการตรวจ : 90</strong> (AI Flagged - ตรวจสอบพิกัดศุลกากร (HS Code) และพฤติกรรมความเสี่ยงแฝง)"
    else:
        safe_action_code = html.escape(action_code) if action_code else "N/A"
        action_desc_html = f"<strong>Action Code:</strong> {safe_action_code}"

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Dashboard", 
    "Officer Risk Terminal", 
    "AI Feedback Console", 
    "Data Lake & ML Observability"
])

with tab1:
    st.header("Executive Dashboard (Visual Evidence)")
    
    # Alert Cross-Reference
    if action_desc_html:
        action_desc_md = action_desc_html.replace("<strong>", "**").replace("</strong>", "**")
        st.markdown(f"### ⚠️ :red[PRAISE Alert Ref No: {alert_info['ref_no']}]")
        st.markdown(f"#### :red[{action_desc_md}]")
        
    # Metrics Panel
    col1, col2, col3 = st.columns(3)
    with col1:
        shipment_count = len(df_hist[(df_hist['Importer_ID'] == importer_id) & (df_hist['Cleaned_Description'] == cleaned_description)])
        st.metric("Total Processed Shipment", f"{shipment_count:,}")
    with col2:
        if alert_info and alert_info.get('profile_tuple') == selected_tuple:
            st.metric("Last User Input Price", f"{alert_info.get('price', 0):,.2f} THB")
        else:
            st.metric("Last User Input Price", "-")
    with col3:
        if alert_info and alert_info.get('profile_tuple') == selected_tuple:
            ai_score = alert_info.get('eval', {}).get('ai_score')
            st.metric("AI Anomaly Score", "-" if ai_score is None else f"{ai_score:.1f}%")
        else:
            st.metric("AI Anomaly Score", "-")

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
        
        if eval_res.get('action_code') == '90':
            st.markdown(f"""
            <div style="background-color:#fff3cd;color:black;padding:20px;border-radius:10px;border: 2px solid #ffeeba;">
                <h3 style="color:#856404;margin-top:0;">⚠️ {eval_res['status_label']} (Latent Risk / Behavioral Anomaly)</h3>
                <p><strong>Alert Ref No:</strong> {alert_info['ref_no']}</p>
                <p><strong>Target:</strong> Declaration Number: {decl_no} / Item Number: {item_no}</p>
                <p>{action_desc_html}</p>
                <h4 style="color:#856404;">Instruction: สินค้าราคาผ่านเกณฑ์สถิติ แต่ AI พบพฤติกรรมเสี่ยงด้านประเทศต้นทาง/รูปแบบการขนส่ง ขอเอกสารเพิ่มเติมเพื่อพิจารณาอย่างละเอียด</h4>
            </div>
            """, unsafe_allow_html=True)
        elif eval_res.get('is_anomaly'):
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

with tab3:
    st.header("AI Feedback Console (Human-in-the-loop)")
    st.markdown("เจ้าหน้าที่กรอกผลการตรวจสอบกลับเข้าระบบ เพื่อให้ AI เรียนรู้เพิ่มเติม (Feedback Loop)")
    
    with st.form("feedback_form"):
        ref_input = st.text_input("PRAISE Alert Ref No.", placeholder="e.g., 690101001", key="fb_ref")
        result_choice = st.selectbox("ผลการตรวจสอบ (Result)", ["Confirmed Fraud (พบการกระทำผิดจริง)", "False Alarm (ประเมินผิดพลาด/ปกติ)"], key="fb_result")
        tax_recovered = st.number_input("จำนวนภาษีที่เก็บเพิ่มได้ (Reassessment Duty - THB)", min_value=0.0, value=None, step=100.0, key="fb_tax")
        
        submitted = st.form_submit_button("Submit Feedback to Data Lake")
        
        if submitted:
            if ref_input:
                try:
                    if feedback_lake_path.exists():
                        lock_path = str(feedback_lake_path) + ".lock"
                        with FileLock(lock_path, timeout=10):
                            fb_df = pd.read_csv(str(feedback_lake_path))
                            is_fraud = "Confirmed Fraud" in result_choice
                            
                            alert = st.session_state.get('current_alert')
                            if not alert or str(alert.get('ref_no')) != str(ref_input):
                                st.error(f"ไม่พบข้อมูลต้นฉบับสำหรับ Ref No: {ref_input} ระบบขอปฏิเสธการบันทึกเพื่อป้องกันข้อมูล Training Set ผิดเพี้ยน")
                            else:
                                price = alert.get('price') or 0.0
                                eval_data = alert.get('eval') or {}
                                user_input = eval_data.get('user_input_row') or {}
                                origin = user_input.get('Origin_Country', 'Unknown')
                                transport = user_input.get('Transport_Mode', 'Unknown')
                                broker = user_input.get('Broker_ID', 'Unknown')
                                brand = user_input.get('Brand', 'Unknown')
                                year = user_input.get('Product_Year', 'Unknown')
                                port = user_input.get('Port_of_Entry', 'Unknown')
                                quantity = user_input.get('Quantity', 1.0)
                                weight = user_input.get('Gross_Weight_KG', 1.0)
    
                                new_row = {
                                    "PRAISE_Alert_Ref_No": ref_input,
                                    "Is_Anomaly_Confirmed": is_fraud,
                                    "Recovered_Tax_THB": tax_recovered if tax_recovered is not None else 0.0,
                                    "Review_Timestamp": pd.Timestamp.now(),
                                    "Review_Officer_ID": "OFFICER-UX",
                                    "Unit_Price_THB_CIF": price,
                                    "Origin_Country": origin,
                                    "Transport_Mode": transport,
                                    "Broker_ID": broker,
                                    "Brand": brand,
                                    "Product_Year": year,
                                    "Port_of_Entry": port,
                                    "Quantity": quantity,
                                    "Gross_Weight_KG": weight
                                }
                                
                                fb_df = pd.concat([fb_df, pd.DataFrame([new_row])], ignore_index=True)
                                temp_path = feedback_lake_path.with_name(f"{feedback_lake_path.name}.tmp")
                                fb_df.to_csv(str(temp_path), index=False)
                                temp_path.replace(feedback_lake_path)
                                st.success(f"บันทึกผลการตรวจสอบสำหรับ Ref No: {ref_input} สำเร็จแล้ว ข้อมูลถูกจัดเก็บเข้า Data Lake เรียบร้อย")
                    else:
                        st.error("ไม่พบ Data Lake File")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
            else:
                st.error("กรุณาระบุ PRAISE Alert Ref No.")
                
    st.write("") # Add spacing
    if st.button("🧹 Clear Screen", key="clear_tab3"):
        for key in ["fb_ref", "fb_result", "fb_tax"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

with tab4:
    st.header("Data Lake & ML Observability")
    
    # 1. Gauge Chart Section (Top)
    st.subheader("Model Precision (Simulated)")
    
    # Center the gauge on large screens, stack normally on mobile
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        # Simulated gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 85.0,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AI Precision (Simulated)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"}
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    st.divider()
    
    # 2. Data Lake Table Section (Bottom)
    st.subheader("Data Lake Explorer")
    if feedback_lake_path.exists():
        fb_df = pd.read_csv(str(feedback_lake_path))
        
        rows_per_page = 10
        total_rows = len(fb_df)
        total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1
        
        if 'lake_page_input' not in st.session_state:
            st.session_state.lake_page_input = 1
            
        def go_first():
            st.session_state.lake_page_input = 1
            
        def go_prev():
            st.session_state.lake_page_input -= 1
            
        def go_next():
            st.session_state.lake_page_input += 1
            
        def go_last():
            st.session_state.lake_page_input = total_pages
            
        col_first, col_prev, col_page, col_next, col_last, col_export = st.columns([1, 1, 2, 1, 1, 2])
        with col_first:
            st.button("⏮️ หน้าแรก", disabled=(st.session_state.lake_page_input <= 1), on_click=go_first, use_container_width=True)
        with col_prev:
            st.button("⬅️ ก่อนหน้า", disabled=(st.session_state.lake_page_input <= 1), on_click=go_prev, use_container_width=True)
        with col_page:
            st.number_input(f"หน้า (1 - {total_pages})", min_value=1, max_value=total_pages, key="lake_page_input", label_visibility="collapsed")
        with col_next:
            st.button("ถัดไป ➡️", disabled=(st.session_state.lake_page_input >= total_pages), on_click=go_next, use_container_width=True)
        with col_last:
            st.button("หน้าสุดท้าย ⏭️", disabled=(st.session_state.lake_page_input >= total_pages), on_click=go_last, use_container_width=True)
        with col_export:
            csv = fb_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name='feedback_lake_export.csv',
                mime='text/csv',
                use_container_width=True
            )

        start_idx = (st.session_state.lake_page_input - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        display_df = fb_df.iloc[start_idx:end_idx].copy()
        display_df.index = display_df.index + 1
        
        st.write(f"แสดงข้อมูลแถวที่ {start_idx + 1} ถึง {min(end_idx, total_rows)} จากทั้งหมด {total_rows} แถว (หน้า {st.session_state.lake_page_input}/{total_pages})")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Feedback Lake is empty.")
