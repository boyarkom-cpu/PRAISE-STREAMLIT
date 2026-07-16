import pandas as pd
import numpy as np
import logging
from typing import TypedDict, Optional, Tuple, Any
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- Configuration Constants ---
MAD_SCALE_FACTOR = 0.6745
VOLATILITY_THRESHOLD_CV_PERCENT = 15.0
LOWER_BOUND_Z_SCORE = 3.0
BUSINESS_FLOOR_MULTIPLIER = 0.5
EXTERNAL_BENCHMARK_MULTIPLIER = 0.4
MIN_SAMPLE_SIZE = 20
AI_SCORE_ALERT_THRESHOLD_PERCENT = 70.0

class FilterMetrics(TypedDict, total=False):
    total_input_rows: int
    passed_duty_filter: int
    passed_privilege_filter: int
    passed_incentive_filter: int
    final_output_rows: int

class StatisticalBounds(TypedDict, total=False):
    status: str
    message: str
    cv_percent: float
    is_volatile: bool
    lower_bound: float
    external_bound: float
    median: float
    mad: float
    log_median: float
    log_mad_scaled: float
    sample_size: int

class RiskEvaluation(TypedDict, total=False):
    is_anomaly: bool
    status_label: str
    action_code: Optional[str]
    reason: str
    ai_score: float

def train_praise_anomaly_model() -> Tuple[Optional[IsolationForest], Optional[ColumnTransformer]]:
    """
    Phase 2: Train the Hybrid AI Anomaly Detection Model (Isolation Forest)
    using the Feedback Lake data.
    """
    project_root = Path(__file__).parent.parent
    feedback_lake_path = project_root / 'mockup_feedback_lake.csv'
    
    if not feedback_lake_path.exists():
        return None, None
        
    df = pd.read_csv(str(feedback_lake_path))
    
    # Schema Migration & Validation for existing Feedback Lakes
    if 'Is_Related_Party' not in df.columns:
        df['Is_Related_Party'] = 0
    df['Is_Related_Party'] = df['Is_Related_Party'].astype('int64')
    
    # Feature Selection
    features = ['Unit_Price_THB_CIF', 'Quantity', 'Gross_Weight_KG', 'Origin_Country', 'Transport_Mode', 'Broker_ID', 'Brand', 'Product_Year', 'Port_of_Entry', 'Is_Related_Party']
    X = df[features].copy()
    
    # Preprocessing Pipeline
    numeric_features = ['Unit_Price_THB_CIF', 'Quantity', 'Gross_Weight_KG']
    numeric_transformer = StandardScaler()
    
    categorical_features = ['Origin_Country', 'Transport_Mode', 'Broker_ID', 'Brand', 'Product_Year', 'Port_of_Entry', 'Is_Related_Party']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        sparse_threshold=0
    )
    
    # Calculate empirical contamination from Feedback Lake
    if 'Is_Anomaly_Confirmed' in df.columns:
        empirical_contamination = df['Is_Anomaly_Confirmed'].mean()
        # Cap between 1% and 50% for IsolationForest bounds
        contamination = max(0.01, min(0.5, empirical_contamination))
    else:
        contamination = 0.1 # Fallback
        
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    
    X_processed = preprocessor.fit_transform(X)
    model.fit(X_processed)
    
    return model, preprocessor

def load_and_validate_csv(file_path: str) -> pd.DataFrame:
    """
    Loads the PRAISE mockup CSV and strictly enforces the Pandas dtypes
    defined in the Data Schema to prevent CSV type-inference data corruption
    (e.g., losing leading zeros in HS_Code or categorical values inferring as int).
    """
    dtype_mapping = {
        'Declaration_Number': 'string',
        'Importer_ID': 'string',
        'Broker_ID': 'string',
        'Transport_Mode': 'category',
        'Origin_Country': 'category',
        'Port_of_Entry': 'category',
        'Clearance_Port': 'category',
        'Invoice_No': 'string',
        'Invoice_Details': 'string',
        'Item_Number': 'int64',
        'HS_Code': 'string',
        'Stat_Code': 'string',
        'Product_Code': 'string',
        'Brand': 'category',
        'Model_No': 'string',
        'Product_Year': 'string',
        'Cleaned_Description': 'category',
        'Quantity': 'float64',
        'Net_Weight_KG': 'float64',
        'Gross_Weight_KG': 'float64',
        'Declared_Currency_Code': 'category',
        'Unit_Price_Foreign_CIF': 'float64',
        'Invoice_Amount_Foreign': 'float64',
        'Exchange_Rate': 'float64',
        'Unit_Price_THB_CIF': 'float64',
        'Invoice_Amount_THB': 'float64',
        'Privilege_Code': 'category',
        'Incentive_Scheme': 'category',
        'Duty_Paid_THB': 'float64',
        'VAT_Paid_THB': 'float64',
        'Other_Taxes_THB': 'float64',
        'External_Benchmark_Price_THB': 'float64',
        'Is_Related_Party': 'int64'
    }

    try:
        # Define common broker placeholders that should be safely parsed as NaN instead of crashing the parser
        null_like_values = ['NONE', 'N/A', 'NA', '', '-', 'NULL', 'NAN', '<NA>']
        
        # Use Pandas built-in thousands=',' and na_values to natively handle formatting
        df = pd.read_csv(
            file_path, 
            dtype=dtype_mapping, 
            parse_dates=['Import_Date'], 
            thousands=',',
            na_values=null_like_values
        )

    except Exception as e:
         raise ValueError(f"Failed to load and validate CSV against PRAISE schema: {e}") from e
         
    if not df['Is_Related_Party'].isin([0, 1]).all():
        raise ValueError("Is_Related_Party must contain only 0 or 1.")
    
    return df

def prepare_and_filter_data(df: pd.DataFrame) -> tuple[pd.DataFrame, FilterMetrics]:
    """
    Phase 1 of PRAISE Engine: Data Filtering (Zero-Waste Processing)
    Applies derived columns and filters to isolate High-Risk transactions.
    """
    if df.empty:
         raise ValueError("Input DataFrame is empty.")
    
    required_cols = ['Duty_Paid_THB', 'Invoice_Amount_THB', 'Privilege_Code', 'Incentive_Scheme']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for data filtering: {missing_cols}")

    df_processed = df.copy()

    # Defensive Programming: Data types are strictly enforced at the read_csv level (Phase 0).
    # Any garbage data ('-', 'N/A') was converted to NaN by na_values and will be caught below.

    # Guard Clause: Fail Loudly on Invalid Financial Data (Zero-Silent Bug Policy)
    invalid_invoices = df_processed[pd.isna(df_processed['Invoice_Amount_THB']) | (df_processed['Invoice_Amount_THB'] <= 0)]
    if not invalid_invoices.empty:
        raise ValueError(f"Zero-Silent Bug Policy: Found {len(invalid_invoices)} rows with invalid (<=0 or NaN) Invoice_Amount_THB.")

    invalid_duty = df_processed[pd.isna(df_processed['Duty_Paid_THB']) | (df_processed['Duty_Paid_THB'] < 0)]
    if not invalid_duty.empty:
        raise ValueError(f"Zero-Silent Bug Policy: Found {len(invalid_duty)} rows with invalid (<0 or NaN) Duty_Paid_THB.")

    # Derived Column: Calculate Duty Rate manually
    # Since we explicitly blocked Invoice_Amount_THB <= 0, ZeroDivisionError is impossible here.
    # Round to 2 decimals to fix floating-point precision loss from pre-rounded CSV values
    df_processed['Duty_Rate'] = ((df_processed['Duty_Paid_THB'] / df_processed['Invoice_Amount_THB']) * 100).round(2)

    # Apply Filters Sequentially (Vectorized)
    
    # Business Reason (Filter 1): High Duty Rate transactions (>= 28%) are prioritized as they have a higher probability of duty evasion through under-valuation.
    duty_mask = df_processed['Duty_Rate'] >= 28
    
    # Business Reason (Filter 2): Free Trade Agreements (FTA) often distort natural market prices. We only evaluate standard import duties ('000', '999') to ensure an undistorted baseline.
    privilege_mask = df_processed['Privilege_Code'].isin(['000', '999'])
    
    # Business Reason (Filter 3): Incentive scheme shipments (e.g., BOI, IEAT, Free Zone) are exempt from standard taxation and often exhibit transfer pricing behaviors that skew the mathematical baseline. We strictly isolate and exclude all of them, keeping only 'No Incentive' standard shipments.
    normalized_incentives = df_processed['Incentive_Scheme'].astype(str).str.strip().str.upper()
    null_like_values = ['NONE', 'N/A', 'NA', '', '-', 'NULL', 'NAN', '<NA>']
    incentive_mask = df_processed['Incentive_Scheme'].isna() | normalized_incentives.isin(null_like_values)

    # Business Reason: Sequential tracking of dropped rows enables Executive Observability, allowing analysts to identify if a specific filter is too aggressive.
    step1_mask = duty_mask
    step2_mask = step1_mask & privilege_mask
    final_mask = step2_mask & incentive_mask
    
    # Observability Metrics (Sequential Funnel Counts)
    metrics = {
        'total_input_rows': len(df_processed),
        'passed_duty_filter': int(step1_mask.sum()),
        'passed_privilege_filter': int(step2_mask.sum()),
        'passed_incentive_filter': int(final_mask.sum()),
        'final_output_rows': int(final_mask.sum())
    }
    
    return df_processed[final_mask].reset_index(drop=True), metrics

def _validate_positive_finite_numeric(df: pd.DataFrame, column: str, importer_id: str) -> pd.Series:
    """Helper to validate numeric columns and enforce the Zero-Silent Bug Policy."""
    try:
        series = pd.to_numeric(df[column], errors='raise')
    except (ValueError, TypeError) as e:
        raise ValueError(f"Zero-Silent Bug Policy: Data for {importer_id} contains non-numeric {column} values. Fix source data. Detail: {e}") from e
    
    invalid = series.isna() | ~np.isfinite(series) | (series <= 0)
    if invalid.any():
        raise ValueError(f"Zero-Silent Bug Policy: Data for {importer_id} contains {invalid.sum()} invalid (<=0, NaN, or non-finite) {column} values.")
    
    return series

def calculate_statistical_bounds(df: pd.DataFrame, importer_id: str, cleaned_description: str) -> StatisticalBounds:
    """
    Phase 2 of PRAISE Engine: Algorithmic Self-Benchmarking & Statistical Detection Engine
    Groups by Importer_ID + Cleaned_Description and calculates Robust CV and MAD.
    Uses Robust Statistics (Hampel Filter / Scaled MAD) to natively handle extreme values without trimming.
    """
    if df.empty:
        raise ValueError("Filtered DataFrame is empty. Cannot calculate statistics.")

    required_cols = ['Importer_ID', 'Cleaned_Description', 'Unit_Price_THB_CIF', 'External_Benchmark_Price_THB']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
         raise ValueError(f"Missing required columns for bounds calculation: {missing_cols}")

    # Grouping Strategy
    group_mask = (df['Importer_ID'] == importer_id) & (df['Cleaned_Description'] == cleaned_description)
    group_df = df[group_mask].copy()

    # Defensive Programming & Validation for External_Benchmark_Price_THB
    # BUG-7 Fix: Validate BEFORE coercing — detect non-numeric garbage strings immediately
    group_df['External_Benchmark_Price_THB'] = _validate_positive_finite_numeric(
        group_df, 'External_Benchmark_Price_THB', importer_id
    )

    # Extract the static benchmark price for the group using median for robustness
    ext_price = float(group_df['External_Benchmark_Price_THB'].median())
    B_ext = float(ext_price * EXTERNAL_BENCHMARK_MULTIPLIER)

    # BUG-7 Fix: Validate price column type immediately — detect non-numeric garbage before proceeding
    group_df['Unit_Price_THB_CIF'] = _validate_positive_finite_numeric(
        group_df, 'Unit_Price_THB_CIF', importer_id
    )

    if len(group_df) < MIN_SAMPLE_SIZE:
        return {
            'status': 'INSUFFICIENT_DATA',
            'message': f"Group has {len(group_df)} shipments. Minimum {MIN_SAMPLE_SIZE} required for robust statistical validity (IQR/MAD).",
            'cv_percent': np.nan,
            'is_volatile': False,
            'lower_bound': np.nan,
            'external_bound': B_ext
        }

    prices = group_df['Unit_Price_THB_CIF'].dropna()
    
    if len(prices) < MIN_SAMPLE_SIZE:
        return {
             'status': 'INSUFFICIENT_DATA',
             'message': f"Group has fewer than {MIN_SAMPLE_SIZE} valid Unit_Price_THB_CIF records after dropping NaNs.",
             'cv_percent': np.nan,
             'is_volatile': False,
             'lower_bound': np.nan,
             'external_bound': B_ext
        }

    # Mathematical Refactoring: Log-Space Robust Statistics (Hampel Filter) + Business Floor
    # Zero-Waste Policy: No trimming or dropping of outliers.
    
    # 1. Transform to Log-Space
    # Defensive Guard: Ensure no zero or negative prices before log transform
    if not (prices > 0).all():
        raise ValueError("Critical Error: Found non-positive prices (<= 0) just before Log-Space transformation. This violates the statistical bounds logic.")
    log_prices = np.log(prices)
    
    # 2. Calculate Median (Log and Real space)
    log_median = float(log_prices.median())
    median_price = float(prices.median())
    
    if pd.isna(median_price) or median_price == 0:
        raise ZeroDivisionError(f"Median price is 0 or NaN for {importer_id} - {cleaned_description}. Cannot calculate bounds.")
        
    # 3. Calculate MAD in Log-Space
    log_mad = float((log_prices - log_median).abs().median())
    log_mad_scaled = log_mad / MAD_SCALE_FACTOR
    
    # 4. Calculate Log-Normal Robust CV (%)
    # Mathematically derived from the log-space dispersion (sigma proxy = log_mad_scaled)
    # Formula: CV = sqrt(exp(sigma^2) - 1) * 100
    robust_cv = float(np.sqrt(np.exp(log_mad_scaled**2) - 1)) * 100
    
    # Calculate real-space MAD for context reporting
    real_mad = float((prices - median_price).abs().median())
    real_mad_scaled = real_mad / MAD_SCALE_FACTOR
    
    # Volatility Threshold
    is_volatile = bool(robust_cv > VOLATILITY_THRESHOLD_CV_PERCENT)
    
    # 5. Calculate Lower Bound (Hampel Filter in Log-Space)
    if log_mad_scaled < 1e-6:
        # Zero-variance case: Apply a default 5% tolerance
        B_stat = median_price * 0.95
        clamp_msg = " (Lower bound set to 5% tolerance due to zero historical price variance)"
    else:
        log_lower_bound = float(log_median - (LOWER_BOUND_Z_SCORE * log_mad_scaled))
        calculated_lower_bound = float(np.exp(log_lower_bound))
        
        # 6. Business Logic Floor Clamp
        business_floor = median_price * BUSINESS_FLOOR_MULTIPLIER
        
        if calculated_lower_bound < business_floor:
            B_stat = business_floor
            clamp_msg = f" (Statistical bound clamped to {BUSINESS_FLOOR_MULTIPLIER*100:.0f}% of median due to extreme volatility or undervaluation risk)"
        else:
            B_stat = calculated_lower_bound
            clamp_msg = ""
            
    Effective_Lower_Bound = float(max(B_stat, B_ext))
    if Effective_Lower_Bound == B_ext and B_ext > B_stat:
        clamp_msg += f" (Lower bound clamped to External Benchmark Bound {B_ext:,.2f} THB)"

    return {
        'status': 'SUCCESS',
        'message': f"Log-Space Robust bounds calculated successfully on {len(prices)} shipments.{clamp_msg}",
        'cv_percent': robust_cv,
        'is_volatile': is_volatile,
        'lower_bound': Effective_Lower_Bound,
        'external_bound': B_ext,
        'median': median_price,
        'mad': real_mad_scaled,
        'log_median': log_median,
        'log_mad_scaled': log_mad_scaled,
        'sample_size': len(prices)
    }

def evaluate_transaction_risk(
    user_input_price: float, 
    stats_result: StatisticalBounds,
    ml_model: Optional[IsolationForest] = None,
    ml_preprocessor: Optional[ColumnTransformer] = None,
    user_input_row: Optional[dict] = None
) -> RiskEvaluation:
    """
    Evaluates the Single User Input Row against the calculated statistics and AI Engine.
    Returns the action logic for Tab 2 (Dashboard/Alert Console).
    """
    # Guard Clauses
    if pd.isna(user_input_price) or not np.isfinite(user_input_price) or user_input_price <= 0:
        raise ValueError(f"Invalid user_input_price: {user_input_price}. Must be a finite positive number (> 0).")

    if stats_result.get('status') == 'INSUFFICIENT_DATA':
        B_ext = stats_result.get('external_bound')
        ref_text = f"{B_ext:,.2f}" if isinstance(B_ext, (int, float)) and pd.notna(B_ext) else "Unavailable"
        return {
            'is_anomaly': True,
            'status_label': 'HIGH RISK (COLD START)',
            'action_code': '890',
            'reason': f"Sample size < {MIN_SAMPLE_SIZE}. Forced Alert for Human-in-the-loop review. (Reference External Bound: {ref_text})",
            'ai_score': 0.0
        }

    if stats_result.get('status') != 'SUCCESS':
         return {
            'is_anomaly': False,
            'status_label': 'NOT ASSESSED',
            'action_code': None,
            'reason': stats_result.get('message', 'No valid historical statistics available.'),
            'ai_score': 0.0
        }

    cv = stats_result.get('cv_percent')
    lower_bound = stats_result.get('lower_bound')
    
    if pd.isna(cv) or pd.isna(lower_bound):
        raise ValueError("CV or Lower Bound is NaN despite a SUCCESS status from statistical calculation.")

    # Trigger Condition: Price below Lower Bound (Applies to ALL transactions)
    is_volatile = stats_result.get('is_volatile', False)
    volatility_context = f"Highly volatile group (Robust CV: {cv:.2f}%)" if is_volatile else f"Stable group (Robust CV: {cv:.2f}% <= {VOLATILITY_THRESHOLD_CV_PERCENT}%)"

    external_bound = stats_result.get('external_bound')
    is_clamped_to_ext = (external_bound is not None and lower_bound == external_bound)
    bound_name = "External Benchmark Bound" if is_clamped_to_ext else "robust statistical Lower Bound"

    # Calculate AI Score if model is available
    ai_score_percent = 0.0
    if ml_model is not None and ml_preprocessor is not None and user_input_row is not None:
        try:
            row_df = pd.DataFrame([user_input_row])
            if all(col in row_df.columns for col in ['Unit_Price_THB_CIF', 'Quantity', 'Gross_Weight_KG', 'Origin_Country', 'Transport_Mode', 'Broker_ID', 'Brand', 'Product_Year', 'Port_of_Entry', 'Is_Related_Party']):
                X_input = ml_preprocessor.transform(row_df[['Unit_Price_THB_CIF', 'Quantity', 'Gross_Weight_KG', 'Origin_Country', 'Transport_Mode', 'Broker_ID', 'Brand', 'Product_Year', 'Port_of_Entry', 'Is_Related_Party']])
                score = ml_model.decision_function(X_input)[0]
                # Map decision function to 0-100%. score typically in [-0.5, 0.5]
                # lower score -> higher anomaly risk
                ai_score_percent = max(0.0, min(100.0, 50.0 - (score * 100.0)))
            else:
                logging.warning("Zero-Silent Bug Policy: AI scoring skipped because user_input_row is missing required ML feature columns.")
        except Exception as e:
            logging.warning(f"AI scoring failed, falling back to 0.0: {e}")

    if user_input_price < lower_bound:
        if is_volatile:
            action_code = '89'
            reason_text = f"{volatility_context} - Price ({user_input_price:,.2f}) dropped below {bound_name} ({lower_bound:,.2f}). Reasonable Doubt established. (อนุญาตให้ตรวจปล่อยและส่งข้อมูลให้หน่วยงาน PCA ตรวจสอบโครงสร้างต้นทุน)"
        else:
            action_code = '88'
            reason_text = f"{volatility_context} - Price ({user_input_price:,.2f}) dropped below {bound_name} ({lower_bound:,.2f}). Reasonable Doubt established."
            
        return {
            'is_anomaly': True,
            'status_label': 'HIGH RISK',
            'action_code': action_code,
            'reason': reason_text,
            'ai_score': ai_score_percent
        }
    else:
        if ai_score_percent >= AI_SCORE_ALERT_THRESHOLD_PERCENT:
            return {
                'is_anomaly': True,
                'status_label': 'YELLOW RISK (AI FLAGGED)',
                'action_code': '90',
                'reason': f"Price ({user_input_price:,.2f}) passed {bound_name}, but AI detected behavioral anomalies (Score: {ai_score_percent:.1f}%). Recommend HS Code review. (อนุญาตให้ตรวจปล่อยและส่งข้อมูลให้หน่วยงาน PCA เพื่อประเมินพฤติกรรมความเสี่ยงแฝงพหุมิติ)",
                'ai_score': ai_score_percent
            }
        else:
            return {
                'is_anomaly': False,
                'status_label': 'PASS',
                'action_code': None,
                'reason': f"{volatility_context} - Price ({user_input_price:,.2f}) is above or equal to {bound_name} ({lower_bound:,.2f}). AI Score: {ai_score_percent:.1f}%.",
                'ai_score': ai_score_percent
            }

