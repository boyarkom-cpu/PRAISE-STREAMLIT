import pandas as pd
import numpy as np
import random
import itertools
from pathlib import Path

def generate_random_dates(start_date: pd.Timestamp, end_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Generate n random dates between start_date and end_date."""
    start_u = start_date.value // 10**9
    end_u = end_date.value // 10**9
    # Add 86400 seconds (1 day) to end_u to make the upper bound inclusive
    return pd.to_datetime(np.random.randint(start_u, end_u + 86400, n), unit='s')

def generate_data() -> None:
    """
    Generate exactly 100,000 rows of mock customs data using 8 targeted high-risk profiles.
    No random garbage data generation.
    """
    # Set random seed per generation call for strict reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2025-12-31')
    
    dec_counter = itertools.count(10000)
    data = []
    
    def create_profile_rows(importer_id, desc, hs_code, min_price_foreign, max_price_foreign, n_rows, anomaly_type, duty_rate_fixed, distribution_type='uniform', external_benchmark_price_thb=0.0, origin_weights=None, transport_weights=None, related_party_prob=0.0):
        """Helper function to create structured rows for a specific profile. Base pricing is sampled per-row from a uniform distribution range."""
        if anomaly_type not in ['severe', 'moderate', 'stable']:
            raise ValueError(f"Zero-Silent Bug Policy: Invalid anomaly_type '{anomaly_type}'. Must be 'severe', 'moderate', or 'stable'.")
        
        # BUG-5 Fix: Use profile-specific weighted distributions for Origin/Transport
        # to give Isolation Forest real behavioral signals instead of uniform noise
        if origin_weights is None:
            origin_weights = {'JP': 1, 'US': 1, 'CN': 1, 'DE': 1}
        if transport_weights is None:
            transport_weights = {'Sea': 1, 'Air': 1, 'Land': 1}
            
        profile_data = []
        dates = generate_random_dates(start_date, end_date, n_rows)
        dates = dates.sort_values()
        
        for date in dates:
            qty = float(np.random.randint(1, 100))
            
            # Draw random base price based on distribution type
            mean_price = (min_price_foreign + max_price_foreign) / 2
            
            if distribution_type == 'bimodal':
                if np.random.rand() < 0.6:
                    base_price_foreign = np.random.normal(min_price_foreign + (max_price_foreign - min_price_foreign)*0.2, (max_price_foreign - min_price_foreign)*0.05)
                else:
                    base_price_foreign = np.random.normal(max_price_foreign - (max_price_foreign - min_price_foreign)*0.2, (max_price_foreign - min_price_foreign)*0.05)
            elif distribution_type == 'lognormal':
                mu = np.log(min_price_foreign + (max_price_foreign - min_price_foreign)*0.1)
                sigma = 0.5
                base_price_foreign = np.random.lognormal(mu, sigma)
            elif distribution_type == 'tight_normal':
                base_price_foreign = np.random.normal(mean_price, (max_price_foreign - min_price_foreign)*0.02)
            elif distribution_type == 'left_skewed':
                beta_val = np.random.beta(5, 1.5)
                base_price_foreign = min_price_foreign + beta_val * (max_price_foreign - min_price_foreign)
            elif distribution_type == 'normal':
                base_price_foreign = np.random.normal(mean_price, (max_price_foreign - min_price_foreign)*0.15)
            elif distribution_type == 'step_down':
                time_fraction = (date - start_date).total_seconds() / (end_date - start_date).total_seconds()
                if time_fraction < 0.33:
                    base_price_foreign = max_price_foreign
                elif time_fraction < 0.66:
                    base_price_foreign = max_price_foreign - (max_price_foreign - min_price_foreign) * 0.4
                else:
                    base_price_foreign = min_price_foreign
                base_price_foreign = np.random.normal(base_price_foreign, (max_price_foreign - min_price_foreign)*0.05)
            elif distribution_type == 'quantized':
                steps = np.linspace(min_price_foreign, max_price_foreign, 5)
                base_price_foreign = float(np.random.choice(steps))
            elif distribution_type == 'high_variance':
                base_price_foreign = np.random.uniform(min_price_foreign, max_price_foreign * 1.5)
            elif distribution_type == 'branded_vs_unbranded':
                if np.random.rand() < 0.8:
                    # Unbranded (80%): Highly scattered, lower price (e.g., from China)
                    base_price_foreign = np.random.uniform(min_price_foreign, min_price_foreign + (max_price_foreign - min_price_foreign) * 0.4)
                else:
                    # Branded (20%): Tightly clustered, higher price
                    base_price_foreign = np.random.normal(max_price_foreign * 0.9, (max_price_foreign - min_price_foreign) * 0.02)
            else:
                base_price_foreign = np.random.uniform(min_price_foreign, max_price_foreign)

            # Ensure price doesn't drop to zero or negative
            base_price_foreign = max(0.1, base_price_foreign)
            
            # Apply anomalies
            current_price_foreign = base_price_foreign
            if anomaly_type == 'severe' and (date.year == 2025 and date.month >= 7):
                current_price_foreign = base_price_foreign * np.random.uniform(0.3, 0.5)
            elif anomaly_type == 'moderate':
                if np.random.rand() < 0.2:
                    current_price_foreign = base_price_foreign * np.random.uniform(0.6, 0.8)
                else:
                    current_price_foreign = base_price_foreign * np.random.uniform(0.95, 1.05)
            else:
                current_price_foreign = base_price_foreign * np.random.uniform(0.98, 1.02)
                
            unit_price_foreign = round(current_price_foreign, 2)
            invoice_amount_foreign = round(unit_price_foreign * qty, 2)
            
            # Exchange rate deterministic based on year to eliminate uncorrelated statistical noise
            if date.year == 2023:
                ex_rate = 35.07
            elif date.year == 2024:
                ex_rate = 31.98
            else:
                ex_rate = 32.88
            
            unit_price_thb = round(unit_price_foreign * ex_rate, 2)
            invoice_amount_thb = round(invoice_amount_foreign * ex_rate, 2)
            
            # Use deterministic duty rate strictly tied to HS Code
            duty_rate = duty_rate_fixed
            duty_paid = round(invoice_amount_thb * duty_rate, 2)
            
            net_weight = round(qty * np.random.uniform(1.0, 10.0), 2)
            gross_weight = round(net_weight * np.random.uniform(1.05, 1.20), 2)
            
            row = {
                'Declaration_Number': f"DEC{date.strftime('%y%m%d')}{next(dec_counter)}",
                'Import_Date': date.normalize(),
                'Importer_ID': importer_id,
                'Broker_ID': f"BROK-{random.randint(1, 50):03d}",
                'Transport_Mode': random.choices(list(transport_weights.keys()), weights=list(transport_weights.values()), k=1)[0],
                'Origin_Country': random.choices(list(origin_weights.keys()), weights=list(origin_weights.values()), k=1)[0],
                'Port_of_Entry': random.choice(['BKK', 'LCP', 'SUV']),
                'Clearance_Port': random.choice(['BKK', 'LCP', 'SUV']),
                'Invoice_No': f"INV-{random.randint(10000, 99999)}",
                'Invoice_Details': "Standard Import",
                'Item_Number': 1,
                'HS_Code': hs_code,
                'Stat_Code': '000',
                'Product_Code': f"PROD-{random.randint(100, 999)}",
                'Brand': 'BrandX',
                'Model_No': f"M-{random.randint(10, 99)}",
                'Product_Year': str(random.randint(2021, 2025)),
                'Cleaned_Description': desc,
                'Quantity': qty,
                'Net_Weight_KG': net_weight,
                'Gross_Weight_KG': gross_weight,
                'Declared_Currency_Code': 'USD',
                'Unit_Price_Foreign_CIF': unit_price_foreign,
                'Invoice_Amount_Foreign': invoice_amount_foreign,
                'Exchange_Rate': ex_rate,
                'Unit_Price_THB_CIF': unit_price_thb,
                'Invoice_Amount_THB': invoice_amount_thb,
                'Privilege_Code': random.choice(['000', '999']),
                'Incentive_Scheme': 'None',
                'Duty_Paid_THB': duty_paid,
                'VAT_Paid_THB': round((invoice_amount_thb + duty_paid) * 0.07, 2),
                'Other_Taxes_THB': 0.0,
                'External_Benchmark_Price_THB': external_benchmark_price_thb,
                'Is_Related_Party': int(np.random.choice([1, 0], p=[related_party_prob, 1.0 - related_party_prob]))
            }
            profile_data.append(row)
        return profile_data

    # Generate exact targeted profiles to reach expected rows (100,000 total)
    # Traceability Note: 
    # - IMP-002 (Whisky) and IMP-007 (Passenger Cars) have artificially elevated External Benchmark Prices 
    #   (80,000 and 2,500,000 THB respectively) to ensure their 40% External Bound overrides the 50% Median Floor.
    #   This is explicitly designed for the Executive UI Demonstration.
    # BUG-5 Fix: Each profile now has distinct Origin_Country and Transport_Mode probability
    # distributions reflecting real-world import patterns. This creates genuine behavioral
    # signals for the Isolation Forest AI instead of uniform random noise.
    profiles = [
        ('IMP-001', 'Cake', '19059030', 100.0, 200.0, 10000, 'stable', 0.30, 'normal', 5000.0,
         {'JP': 0.70, 'CN': 0.20, 'US': 0.05, 'DE': 0.05}, {'Sea': 0.60, 'Air': 0.30, 'Land': 0.10}, 0.20),
        ('IMP-002', 'Whisky', '22083090', 500.0, 3000.0, 5500, 'moderate', 0.60, 'lognormal', 80000.0,
         {'US': 0.40, 'JP': 0.40, 'DE': 0.15, 'CN': 0.05}, {'Sea': 0.50, 'Air': 0.40, 'Land': 0.10}, 0.75),
        ('IMP-003', 'Perfumes', '33030000', 3000.0, 3200.0, 20000, 'moderate', 0.30, 'tight_normal', 105400.0,
         {'DE': 0.50, 'US': 0.30, 'JP': 0.15, 'CN': 0.05}, {'Air': 0.60, 'Sea': 0.30, 'Land': 0.10}, 0.80),
        # Note: IMP-004 is intentionally set to exactly 18 rows (< MIN_SAMPLE_SIZE of 20) 
        # to strictly test the Action Code 890 (Force Alert / Cold Start) functionality. Do not increase this number.
        ('IMP-004', 'Leather Belts', '42033000', 1000.0, 2000.0, 18, 'severe', 0.30, 'left_skewed', 50000.0,
         {'CN': 0.60, 'JP': 0.20, 'DE': 0.15, 'US': 0.05}, {'Sea': 0.50, 'Air': 0.30, 'Land': 0.20}, 0.30),
        ('IMP-005', 'Women Dresses', '62044290', 1200.0, 1400.0, 25000, 'stable', 0.30, 'step_down', 44200.0,
         {'CN': 0.80, 'JP': 0.10, 'DE': 0.05, 'US': 0.05}, {'Sea': 0.70, 'Air': 0.20, 'Land': 0.10}, 0.40),
        ('IMP-006', 'Footwear', '64039990', 800.0, 850.0, 22000, 'stable', 0.30, 'normal', 28050.0,
         {'CN': 0.75, 'JP': 0.10, 'DE': 0.10, 'US': 0.05}, {'Sea': 0.70, 'Air': 0.20, 'Land': 0.10}, 0.50),
        ('IMP-007', 'Passenger Cars', '87032324', 30000.0, 80000.0, 2482, 'moderate', 0.80, 'tight_normal', 2500000.0,
         {'JP': 0.50, 'DE': 0.40, 'US': 0.08, 'CN': 0.02}, {'Sea': 0.90, 'Land': 0.08, 'Air': 0.02}, 0.90),
        ('IMP-008', 'Clutches', '87089360', 2000.0, 5000.0, 15000, 'stable', 0.30, 'normal', 100000.0,
         {'CN': 0.50, 'JP': 0.30, 'DE': 0.15, 'US': 0.05}, {'Sea': 0.65, 'Air': 0.25, 'Land': 0.10}, 0.85),
    ]
    expected_total = 100000
    calculated_total = sum(p[5] for p in profiles)
    
    if calculated_total != expected_total:
        raise ValueError(f"Critical Error: Profile rows sum to {calculated_total}, but contract strictly requires exactly {expected_total} rows.")
        
    for profile in profiles:
        data.extend(create_profile_rows(*profile))

    df = pd.DataFrame(data)
    
    # Zero-Silent Bug Policy: Enforce exact row count dynamically
    if len(df) != expected_total:
        raise ValueError(f"Critical Error: Generated dataset has {len(df)} rows, expected exactly {expected_total}.")
    
    # Zero-Silent Bug Policy: Validate downstream filter expectations (Duty Rate >= 28%)
    calculated_duty_rates = (df['Duty_Paid_THB'] / df['Invoice_Amount_THB']).round(2)
    if (calculated_duty_rates < 0.28).any():
        raise ValueError("Critical Error: Some generated rows fail the >= 28% High Duty Rate filter criteria.")
    
    # Enforce strict uniqueness for composite primary key
    if df.duplicated(['Declaration_Number', 'Item_Number']).any():
        raise ValueError("Critical Error: Primary Key constraint violated! Duplicate Declaration_Number + Item_Number found.")
        
    # Save the file to the project root directory
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'mockup_customs_data.csv'
    df.to_csv(str(output_path), index=False, date_format='%Y-%m-%d')
    
    print(f"Phase 6 Complete: Generated exactly {len(df)} rows of targeted customs data at: {output_path.name}")

def generate_feedback_lake() -> None:
    """Generate mock feedback data for ML training."""
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'mockup_customs_data.csv'
    output_path = project_root / 'mockup_feedback_lake.csv'
    
    if not input_path.exists():
        print("Error: mockup_customs_data.csv not found. Please generate data first.")
        return
        
    df = pd.read_csv(str(input_path))
    
    # Sample exactly 1000 rows
    df_sample = df.sample(n=1000, random_state=99).copy()
    
    # Append new columns
    # Is_Anomaly_Confirmed: Boolean, 20% True
    np.random.seed(99)
    df_sample['Is_Anomaly_Confirmed'] = np.random.choice([True, False], size=len(df_sample), p=[0.2, 0.8])
    refs = np.random.choice(range(10000, 99999), size=len(df_sample), replace=False)
    df_sample['PRAISE_Alert_Ref_No'] = [f"REF-{r}" for r in refs]
    
    # Recovered_Tax_THB: Float (if anomaly confirmed, else 0)
    # BUG-2 Fix: Pre-generate random multipliers vectorized for deterministic reproducibility.
    # Using np.random inside lambda/apply depends on Pandas iteration order which may change across versions.
    recovery_multipliers = np.random.uniform(0.1, 0.5, size=len(df_sample))
    df_sample['Recovered_Tax_THB'] = np.where(
        df_sample['Is_Anomaly_Confirmed'],
        (df_sample['Invoice_Amount_THB'] * recovery_multipliers).round(2),
        0.0
    )
    
    # BUG-3 Fix: Pre-generate officer IDs vectorized for deterministic reproducibility.
    # Consistent with BUG-2 fix pattern — avoid np.random inside list comprehension.
    officer_ids = np.random.randint(10, 50, size=len(df_sample))
    df_sample['Review_Officer_ID'] = [f"OFFICER-{oid:02d}" for oid in officer_ids]
    # Review_Timestamp within the last 30 days (deterministic anchor)
    now = pd.Timestamp('2025-01-01')
    df_sample['Review_Timestamp'] = [now - pd.Timedelta(days=np.random.randint(1, 30)) for _ in range(len(df_sample))]
    
    df_sample.to_csv(str(output_path), index=False)
    print(f"Generated Feedback Lake ({len(df_sample)} rows) at: {output_path.name}")

if __name__ == "__main__":
    generate_data()
    generate_feedback_lake()
