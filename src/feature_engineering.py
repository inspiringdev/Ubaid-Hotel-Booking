"""
hotel dynamic pricing - feature engineering module

FIXES APPLIED:
- ref_col in add_competitor_features() and add_demand_features() now always
  uses base_price as reference — NEVER actual_price (which is the target).
  Previously using actual_price to compute price_vs_comp_avg etc. was direct
  data leakage into the feature set.
- price_premium removed (was actual_price / base_price — uses target)
- price_per_person removed (was actual_price / 1.8 — uses target)
- Added base_vs_comp_avg and base_vs_comp_min as safe replacements

functions:
    add_competitor_features(df)   - base price position vs competitors
    add_event_features(df)        - event flags and premium estimates
    add_temporal_features(df)     - cyclical month/day-of-week encoding
    add_demand_features(df)       - high demand flag, lead time buckets
    add_revenue_features(df)      - revenue metrics (EDA only, not model features)
    run_feature_engineering(df)   - runs all of the above in order
"""

import numpy as np
import pandas as pd


def add_competitor_features(df):
    """
    figures out where our BASE PRICE sits relative to competitors.
    uses base_price (rack rate) as reference — NOT actual_price.
    this is safe because base_price is independent of the target.

    a positive base_vs_comp_avg means our rack rate is above competitor average.
    """
    # Always use base_price as reference — never actual_price (that is the target)
    ref_col = "base_price"
    if ref_col not in df.columns:
        print("   warning: base_price not found, skipping competitor features")
        return df

    if "competitor_avg_price" in df.columns:
        df["base_vs_comp_avg"] = (
            (df[ref_col] - df["competitor_avg_price"]) / df["competitor_avg_price"]
        ).round(4)

    if "competitor_min_price" in df.columns:
        df["base_vs_comp_min"] = (
            (df[ref_col] - df["competitor_min_price"]) / df["competitor_min_price"]
        ).round(4)

    if all(c in df.columns for c in ["competitor_max_price", "competitor_min_price"]):
        df["price_comp_spread"] = (
            df["competitor_max_price"] - df["competitor_min_price"]
        ).round(2)

    return df


def add_event_features(df):
    """
    adds a flag for whether there is an event happening,
    and estimates the price premium the event might justify.
    """
    if "event_type" in df.columns:
        df["has_event"] = (df["event_type"] != "None").astype(int)

    if "event_magnitude" in df.columns:
        magnitude_map = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.45, 4: 0.60, 5: 0.80}
        df["event_premium_est"] = df["event_magnitude"].map(magnitude_map).fillna(0)

    return df


def add_temporal_features(df):
    """
    encode month and day of week as sin/cos pairs.
    this ensures month 12 (december) is close to month 1 (january)
    in the model's feature space — which raw numbers do not capture.
    """
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_demand_features(df):
    """
    classify bookings by advance notice and flag high-demand periods.
    NO price column is used here — only occupancy and demand_score.
    """
    def lt_bucket(d):
        if d <= 3:   return "Last_Minute"
        if d <= 14:  return "Short_Term"
        if d <= 60:  return "Medium_Term"
        return "Long_Term"

    if "lead_time_days" in df.columns:
        df["lead_time_bucket"] = df["lead_time_days"].apply(lt_bucket)

    if "occupancy_rate" in df.columns and "demand_score" in df.columns:
        df["high_demand"] = (
            (df["occupancy_rate"] > 0.80) & (df["demand_score"] > 0.85)
        ).astype(int)

    return df


def add_revenue_features(df):
    """
    revenue quality metrics — kept for EDA and analysis but
    NOT included in FEATURE_COLS for model training, because
    they are derived from actual_price (the target variable).
    """
    if "revpar" in df.columns and "actual_price" in df.columns:
        df["revenue_efficiency"] = (
            df["revpar"] / (df["actual_price"] + 1e-9)
        ).round(4)

    if "actual_price" in df.columns and "length_of_stay" in df.columns:
        df["total_stay_revenue"] = (
            df["actual_price"] * df["length_of_stay"]
        ).round(2)

    return df


def run_feature_engineering(df, verbose=True):
    """
    runs all feature engineering steps in order.
    call this on the cleaned dataframe before training models.
    """
    n_before = df.shape[1]
    df = add_competitor_features(df)
    df = add_event_features(df)
    df = add_temporal_features(df)
    df = add_demand_features(df)
    df = add_revenue_features(df)
    n_after = df.shape[1]

    if verbose:
        added = list(df.columns[n_before:])
        print(f"feature engineering: {n_before} -> {n_after} columns "
              f"(+{n_after - n_before} new)")
        for c in added:
            print(f"  + {c}")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/hotel_booking_clean.csv")
    df = run_feature_engineering(df)
    print(f"final shape: {df.shape}")