"""
Data Augmentation Script
Takes the raw Kaggle dataset and generates realistic dynamic pricing features.
This is standard practice when building pricing models without proprietary competitor data.

FIXES APPLIED:
- Removed synthetic optimal_price formula (was causing data leakage / R²≈1.0)
- Target is now actual_price (real Kaggle ADR = average daily rate paid by guest)
- base_price now comes from room-type lookup table, NOT derived from actual_price
- Added is_test column for chronological train/test split
- Removed s_mult, w_mult, e_imp intermediate columns from saved CSV
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

BASE = Path(__file__).parent.parent

RAW_PATH = BASE / "data" / "kaggle_raw_hotel_bookings.csv"
OUT_PATH = BASE / "data" / "hotel_booking_data.csv"

# ── Room-type base price lookup (rack rates, independent of actual_price) ──
ROOM_BASE_PRICE = {
    "A": (80,  130),
    "B": (80,  130),
    "C": (130, 200),
    "D": (130, 200),
    "E": (160, 250),
    "F": (180, 280),
    "G": (220, 350),
    "H": (180, 280),
    "L": (160, 250),
    "P": (80,  130),
}
DEFAULT_PRICE_RANGE = (100, 180)

print("Loading raw Kaggle dataset...")
df = pd.read_csv(RAW_PATH)

# Drop rows where ADR is 0 or negative (free stays / data errors / cancellations)
df = df[df['adr'] > 5].copy()
print(f"Cleaned Kaggle data: {len(df):,} valid rows")

# ── 1. Map Kaggle columns to pipeline columns ──────────────────────────────
df['actual_price']   = df['adr'].round(2)
df['lead_time_days'] = df['lead_time']
df['length_of_stay'] = (df['stays_in_weekend_nights'] + df['stays_in_week_nights']).clip(lower=1)
df['repeat_guest']   = df['is_repeated_guest']
df['cancellation']   = df['is_canceled']
df['hotel_name']     = df['hotel']
df['room_type']      = df['reserved_room_type']
df['channel']        = df['distribution_channel']
df['guest_type']     = df['customer_type']
df['adults']         = df['adults']
df['children']       = df['children'].fillna(0).astype(int)
df['market_segment'] = df['market_segment']
df['is_canceled']    = df['is_canceled']

# ── 2. Date features ───────────────────────────────────────────────────────
df['checkin_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'].str.strip() + '-' +
    df['arrival_date_day_of_month'].astype(str),
    format='mixed'
)
df['year']            = df['checkin_date'].dt.year
df['month']           = df['checkin_date'].dt.month
df['day_of_week']     = df['checkin_date'].dt.dayofweek
df['day_of_week_name']= df['checkin_date'].dt.day_name()
df['week_of_year']    = df['checkin_date'].dt.isocalendar().week.astype(int)
df['day_of_year']     = df['checkin_date'].dt.dayofyear
df['is_weekend']      = (df['day_of_week'] >= 4).astype(int)
df['is_holiday']      = 0
df['season']          = df['month'].map({
    12:'Winter', 1:'Winter', 2:'Winter',
    3:'Spring',  4:'Spring', 5:'Spring',
    6:'Summer',  7:'Summer', 8:'Summer',
    9:'Fall',   10:'Fall',  11:'Fall'
})

# ── 3. Base price from room-type lookup (INDEPENDENT of actual_price) ─────
# This is the hotel's rack rate — what it would charge without any discount.
# Derived from room category, NOT from actual_price, to avoid leakage.
rng = np.random.default_rng(42)

def get_base_price(room_type, n):
    lo, hi = ROOM_BASE_PRICE.get(str(room_type).upper(), DEFAULT_PRICE_RANGE)
    return rng.uniform(lo, hi, n).round(2)

df['base_price'] = [
    get_base_price(rt, 1)[0] for rt in df['room_type']
]

# ── 4. Competitor prices (clustered around base_price, not actual_price) ──
comp_base              = df['base_price'].values * rng.uniform(0.90, 1.10, len(df))
df['competitor_price_1'] = (comp_base * rng.uniform(0.95, 1.05, len(df))).round(2)
df['competitor_price_2'] = (comp_base * rng.uniform(0.90, 1.10, len(df))).round(2)
df['competitor_price_3'] = (comp_base * rng.uniform(0.85, 1.15, len(df))).round(2)
df['competitor_avg_price'] = df[['competitor_price_1','competitor_price_2',
                                  'competitor_price_3']].mean(axis=1).round(2)
df['competitor_min_price'] = df[['competitor_price_1','competitor_price_2',
                                  'competitor_price_3']].min(axis=1).round(2)
df['competitor_max_price'] = df[['competitor_price_1','competitor_price_2',
                                  'competitor_price_3']].max(axis=1).round(2)

# ── 5. Events & weather (simulated, domain-based) ─────────────────────────
event_types = ["None","Concert","Sports","Conference","Festival","Holiday"]
df['event_type']      = rng.choice(event_types, size=len(df),
                                    p=[0.45, 0.10, 0.10, 0.12, 0.08, 0.15])
df['event_magnitude'] = np.where(df['event_type'] == "None", 0,
                                  rng.integers(1, 6, size=len(df)))
df['weather_score']   = rng.integers(1, 6, size=len(df))
df['reviews_score']   = rng.uniform(3.5, 5.0, size=len(df)).round(1)

# ── 6. Occupancy (seasonal logic, independent of price) ───────────────────
base_occ = 0.60 + (df['month'].map(
    {7:0.2, 8:0.2, 6:0.15, 12:0.15}).fillna(0))
df['occupancy_rate'] = (base_occ + rng.normal(0, 0.08, len(df))).clip(0.20, 0.98).round(4)

# ── 7. Demand score (from occupancy + seasonal signals, NOT from price) ───
df['demand_score'] = ((df['occupancy_rate'] - 0.2) / 0.8).clip(0.1, 1.0).round(4)

# ── 8. Revenue KPIs ────────────────────────────────────────────────────────
# NOTE: actual_price IS the target variable (real ADR from Kaggle).
# We compute revenue metrics from it but do NOT create a synthetic target.
df['revenue_per_night'] = (df['actual_price'] * df['length_of_stay']).round(2)
df['revpar']            = (df['actual_price'] * df['occupancy_rate']).round(2)

# ── 9. Chronological train/test split marker ───────────────────────────────
# Use last ~20% of dates as test set (no data leakage from future to past).
df_sorted   = df.sort_values('checkin_date')
cutoff_idx  = int(len(df_sorted) * 0.80)
cutoff_date = df_sorted['checkin_date'].iloc[cutoff_idx]
df['is_test'] = (df['checkin_date'] >= cutoff_date).astype(int)
print(f"Train cutoff date: {cutoff_date.date()}  "
      f"(train={int((df['is_test']==0).sum()):,}  test={int((df['is_test']==1).sum()):,})")

# ── 10. Final column selection & save ─────────────────────────────────────
keep_cols = [
    "checkin_date","year","month","day_of_week","day_of_week_name",
    "week_of_year","day_of_year","season","is_weekend","is_holiday",
    "hotel_name","room_type","channel","guest_type",
    "lead_time_days","length_of_stay","base_price",
    "competitor_price_1","competitor_price_2","competitor_price_3",
    "competitor_avg_price","competitor_min_price","competitor_max_price",
    "event_type","event_magnitude","occupancy_rate","demand_score",
    "weather_score","reviews_score","repeat_guest","cancellation",
    "actual_price","revenue_per_night","revpar",
    "is_canceled","adults","children","market_segment",
    "is_test",
]
df_final = df[[c for c in keep_cols if c in df.columns]].copy()

df_final.to_csv(OUT_PATH, index=False)
print(f"\nSUCCESS!")
print(f"Augmented dataset saved to: {OUT_PATH}")
print(f"Final shape: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")
print(f"Target variable: actual_price (real Kaggle ADR)")
print(f"\nPrice stats:")
print(df_final['actual_price'].describe().round(2))