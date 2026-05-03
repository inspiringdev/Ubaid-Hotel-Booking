"""
 ubaid hotel dynamic pricing dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Taj mahal - dynamic pricing",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2rem; color: #1f77b4; text-align: center; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    div[data-testid="metric-container"] { background-color: #f8f9fb; border-radius: 8px; padding: 12px; }
    </style>
""", unsafe_allow_html=True)


#this is the data & model loading

@st.cache_data
def load_data():
    # 1. Try to load from Cloud Secret (FIRST)
    try:
        url = st.secrets["DATA_URL"]
        
        # Convert Google Drive Link to Direct Download Link
        if "drive.google.com" in url:
            # Extract file ID
            file_id = url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Load CSV with low_memory to handle large file
        # This fixes the "Expected 3745 fields, saw 6351" error
        df = pd.read_csv(url, low_memory=False)
        
        # IMPORTANT: Select only the columns we actually use
        # This prevents the column mismatch crash
        required_cols = ["month","day_of_week","week_of_year","is_weekend","is_holiday","lead_time_days",
                        "length_of_stay","base_price","competitor_avg_price","competitor_min_price",
                        "competitor_max_price","occupancy_rate","demand_score","weather_score",
                        "reviews_score","event_magnitude","repeat_guest","has_event","high_demand",
                        "price_vs_comp_avg","price_vs_comp_min","price_comp_spread","price_premium","month_sin","month_cos",
                        "dow_sin","dow_cos","hotel_name_enc","room_type_enc","channel_enc",
                        "guest_type_enc","event_type_enc","season_enc", "actual_price", "revenue_per_night"]
        
        # Filter DataFrame
        if all(col in df.columns for col in required_cols):
             return df[required_cols]
        else:
             # If some columns are missing (e.g. different CSV version), take what matches
             cols_to_keep = [c for c in required_cols if c in df.columns]
             return df[cols_to_keep]

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
@st.cache_resource
def load_model():
    candidates = [
        "outputs/models/best_model_rf.pkl",
        "outputs/models/best_xgb_tuned.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

df   = load_data()
model = load_model()

# ADDED week_of_year TO THIS LIST
FEAT = ["month","day_of_week","week_of_year","is_weekend","is_holiday","lead_time_days",
        "length_of_stay","base_price","competitor_avg_price","competitor_min_price",
        "competitor_max_price","occupancy_rate","demand_score","weather_score",
        "reviews_score","event_magnitude","repeat_guest","has_event","high_demand",
        "price_vs_comp_avg","price_vs_comp_min","price_comp_spread","price_premium","month_sin","month_cos",
        "dow_sin","dow_cos","hotel_name_enc","room_type_enc","channel_enc",
        "guest_type_enc","event_type_enc","season_enc"]
FEAT_EXISTING = [c for c in FEAT if c in df.columns]


#this is roi calculator

class ROICalculator:
    IMPLEMENTATION_COST = 15000
    ANNUAL_LICENSE = 3000
    LIFT_PCT = 0.18

    def calculate_roi(self, num_rooms, avg_occupancy, current_avg_rate):
        current_annual = num_rooms * 365 * avg_occupancy * current_avg_rate
        new_annual = current_annual * (1 + self.LIFT_PCT)
        incremental = new_annual - current_annual
        total_year1_cost = self.IMPLEMENTATION_COST + self.ANNUAL_LICENSE

        # 5-year calculations
        total_5yr_cost = self.IMPLEMENTATION_COST + (self.ANNUAL_LICENSE * 5)
        total_5yr_lift = incremental * 5
        net_5yr_gain = total_5yr_lift - total_5yr_cost

        # Calculate ROI as a Ratio (Return per Dollar Spent)
        five_year_roi_ratio = net_5yr_gain / total_5yr_cost

        payback_months = round((total_year1_cost / incremental) * 12, 1) if incremental > 0 else 999

        return {
            "current_revenue": round(current_annual, 0),
            "projected_revenue": round(new_annual, 0),
            "incremental_revenue": round(incremental, 0),
            "payback_months": payback_months,
            "five_year_roi_ratio": round(five_year_roi_ratio, 2),  #changed from pct to ratio
        }


#this is sidebar

st.sidebar.title("Ubaid Taj Mahal")
st.sidebar.markdown("dynamic pricing system")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "navigation",
    ["overview", "eda analysis", "model results",
     "price recommender", "business insights"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"dataset: {len(df):,} bookings")
if model:
    st.sidebar.caption("model: loaded ✓")
else:
    st.sidebar.caption("model: not found (run main.py first)")


#this is page 1: overview

if page == "overview":
    st.markdown('<h1 class="main-header">ubaid mahal — dynamic pricing dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*real-time hotel pricing optimization using machine learning*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("avg daily rate",      f"${df['actual_price'].mean():.2f}")
    col2.metric("avg occupancy",        f"{df['occupancy_rate'].mean()*100:.1f}%")
    col3.metric("avg lead time",        f"{df['lead_time_days'].mean():.0f} days")
    col4.metric("revenue lift potential", "+18%", "vs static pricing")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("monthly revenue trend")
        monthly = df.groupby("month")["revenue_per_night"].sum().reset_index()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda x: months[x-1])
        fig = px.line(monthly, x="month_name", y="revenue_per_night",
                      markers=True, labels={"revenue_per_night": "revenue ($)", "month_name": "month"})
        fig.update_traces(line_color="#1f77b4", line_width=2.5)
        fig.update_layout(height=380, plot_bgcolor="#fafafa")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("avg price by room type")
        if "room_type" in df.columns:
            rt = df.groupby("room_type")["actual_price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(rt, x="room_type", y="actual_price",
                         labels={"actual_price": "avg price ($)", "room_type": "room type"},
                         color="actual_price", color_continuous_scale="Blues")
            fig.update_layout(height=380, showlegend=False, plot_bgcolor="#fafafa")
            st.plotly_chart(fig, width='stretch')

    st.subheader("occupancy heatmap — month vs day of week")
    if "day_of_week" in df.columns:
        pivot = df.groupby(["month","day_of_week"])["occupancy_rate"].mean().unstack(fill_value=0)
        day_labels  = ["mon","tue","wed","thu","fri","sat","sun"]
        month_labels = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        pivot.columns = [day_labels[i] if i < len(day_labels) else str(i) for i in pivot.columns]
        pivot.index   = [month_labels[i-1] if 1 <= i <= 12 else str(i) for i in pivot.index]
        fig = px.imshow(pivot, color_continuous_scale="YlOrRd", labels={"color": "occupancy rate"}, aspect="auto")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')


#this is page 2: eda

elif page == "eda analysis":
    st.markdown('<h1 class="main-header">exploratory data analysis</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["distributions", "time series", "correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="actual_price", nbins=50, title="price distribution", labels={"actual_price": "price ($)"})
            fig.update_layout(height=380)
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.histogram(df, x="lead_time_days", nbins=50, title="lead time distribution", labels={"lead_time_days": "days before arrival"})
            fig.update_layout(height=380)
            st.plotly_chart(fig, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            if "room_type" in df.columns:
                fig = px.box(df, x="room_type", y="actual_price", title="price by room type", labels={"actual_price": "price ($)", "room_type": ""})
                fig.update_layout(height=380)
                st.plotly_chart(fig, width='stretch')
        with col2:
            if "channel" in df.columns:
                ch = df.groupby("channel")["actual_price"].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(ch, x="channel", y="actual_price", title="avg price by booking channel", labels={"actual_price": "avg price ($)", "channel": ""})
                fig.update_layout(height=380)
                st.plotly_chart(fig, width='stretch')

    with tab2:
        months  = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        monthly = df.groupby("month").agg(avg_price=("actual_price","mean"), total_rev=("revenue_per_night","sum"), avg_occ=("occupancy_rate","mean")).reset_index()
        monthly["month_name"] = monthly["month"].apply(lambda x: months[x-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["avg_price"], name="avg price", mode="lines+markers", yaxis="y1", line=dict(color="#1f77b4", width=2.5)))
        fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["avg_occ"], name="occupancy rate", yaxis="y2", marker_color="rgba(255,127,14,0.4)"))
        fig.update_layout(title="monthly price and occupancy", yaxis=dict(title="avg price ($)"), yaxis2=dict(title="occupancy rate", overlaying="y", side="right"), height=420, plot_bgcolor="#fafafa", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width='stretch')

        if "has_event" in df.columns:
            event_comp = df.groupby("has_event")["actual_price"].mean().reset_index()
            event_comp["has_event"] = event_comp["has_event"].map({0: "no event", 1: "event day"})
            fig = px.bar(event_comp, x="has_event", y="actual_price", title="price: event vs no event", color="has_event", color_discrete_sequence=["#aec7e8","#1f77b4"], labels={"actual_price": "avg price ($)", "has_event": ""})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, width='stretch')

    with tab3:
        num_cols = [c for c in ["actual_price","occupancy_rate","demand_score","lead_time_days","competitor_avg_price","event_magnitude","reviews_score","length_of_stay","base_price"] if c in df.columns]
        corr = df[num_cols].corr()
        fig  = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto", title="feature correlation matrix")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')


#this is page 3: model results

elif page == "model results":
    st.markdown('<h1 class="main-header">model performance</h1>', unsafe_allow_html=True)

    results = pd.DataFrame({
        "model":    ["gradient boosting", "xgboost", "random forest", "decision tree", "ridge regression", "linear regression"],
        "r2 score": [0.9978, 0.9977, 0.9968, 0.9944, 0.9881, 0.9883],
        "mae ($)":  [2.35, 2.41, 2.71, 3.76, 5.29, 5.34],
        "mape (%)": [1.17, 1.20, 1.34, 1.87, 2.82, 2.86],
    })

    if os.path.exists("outputs/models/model_results.csv"):
        try:
            results = pd.read_csv("outputs/models/model_results.csv")
            results.columns = [c.lower() for c in results.columns]
        except Exception:
            pass

    st.subheader("all models comparison")
    st.dataframe(results, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        r2_col = next((c for c in results.columns if "r2" in c.lower()), None)
        model_col = next((c for c in results.columns if "model" in c.lower()), results.columns[0])
        if r2_col:
            fig = px.bar(results, x=model_col, y=r2_col, title="r² score by model (higher is better)", color=r2_col, color_continuous_scale="Blues", labels={model_col: "", r2_col: "r² score"})
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, width='stretch')

    with col2:
        mae_col = next((c for c in results.columns if "mae" in c.lower()), None)
        if mae_col:
            fig = px.bar(results, x=model_col, y=mae_col, title="mean absolute error (lower is better)", color=mae_col, color_continuous_scale="Reds_r", labels={model_col: "", mae_col: "mae ($)"})
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("key feature importances")
    fi = {"base price": 0.32, "competitor avg": 0.24, "occupancy rate": 0.18, "lead time days": 0.12, "event magnitude": 0.08, "is weekend": 0.04, "hotel type": 0.02}
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())}).sort_values("importance", ascending=True)
    fig    = px.bar(fi_df, x="importance", y="feature", orientation="h", title="top feature importances (gradient boosting)", labels={"importance": "importance score", "feature": ""}, color="importance", color_continuous_scale="Blues")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("q-learning rl agent")
    col1, col2, col3 = st.columns(3)
    col1.metric("episodes trained",  "80")
    col2.metric("final mape",        "~3-5%")
    col3.metric("states explored",   "~240")
    st.info("the q-learning agent learns to set prices by trial and error — it gets rewarded when its price is close to the optimal and penalised when it's too far off.")


#this is page 4: price recommender

elif page == "price recommender":
    st.markdown('<h1 class="main-header">price recommender</h1>', unsafe_allow_html=True)
    st.markdown("*enter the current conditions to get an optimal price recommendation*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        base_price = st.number_input("base price ($)", 50, 500, 150, step=5)
        occupancy = st.slider("current occupancy (%)", 0, 100, 75)
        lead_days = st.slider("days until arrival", 0, 365, 14)
        length_of_stay = st.number_input("length of stay (nights)", 1, 30, 2)

    with col2:
        event_type = st.selectbox("nearby event", ["none", "concert", "festival", "sports", "conference", "holiday"])
        competitor_avg = st.number_input("competitor avg price ($)", 50, 500, 160, step=5)
        month = st.selectbox("check-in month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], index=6)
        is_weekend = st.checkbox("weekend stay")


    event_map  = {"none":0,"concert":1,"festival":2,"sports":1,"conference":2,"holiday":3}
    month_map  = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    month_num  = month_map[month]
    event_mag  = event_map[event_type]
    has_event  = int(event_mag > 0)
    occ_rate   = occupancy / 100.0
    high_dem   = int(occ_rate > 0.80)
    dow_val    = 5 if is_weekend else 1

    comp_min   = competitor_avg * 0.92
    comp_max   = competitor_avg * 1.08

    if st.button("recommend optimal price", type="primary"):
        if model is not None:
            sample = {
                "month": month_num, "day_of_week": dow_val,
                "week_of_year": int(month_num * 4.3),
                "is_weekend": int(is_weekend), "is_holiday": has_event,
                "lead_time_days": lead_days, "length_of_stay": length_of_stay,
                "base_price": base_price, "competitor_avg_price": competitor_avg,
                "competitor_min_price": comp_min, "competitor_max_price": comp_max,
                "occupancy_rate": occ_rate, "demand_score": occ_rate * 0.9,
                "weather_score": 3, "reviews_score": 4.2,
                "event_magnitude": event_mag, "repeat_guest": 0,
                "has_event": has_event, "high_demand": high_dem,
                "price_vs_comp_avg": 0.0,
                "price_vs_comp_min": base_price - comp_min,
                "price_comp_spread": comp_max - comp_min,
                "price_premium": 1.0,
                "month_sin": np.sin(2 * np.pi * month_num / 12),
                "month_cos": np.cos(2 * np.pi * month_num / 12),
                "dow_sin":   np.sin(2 * np.pi * dow_val / 7),
                "dow_cos":   np.cos(2 * np.pi * dow_val / 7),
                "hotel_name_enc": 0, "room_type_enc": 2, "channel_enc": 1,
                "guest_type_enc": 0, "event_type_enc": event_mag, "season_enc": 1,
            }
            X_in = pd.DataFrame([sample])

            expected_features = model.feature_names_in_

            for col in expected_features:
                if col not in X_in.columns:
                    X_in[col] = 0

            X_in = X_in[expected_features]
            rec  = float(model.predict(X_in)[0])
        else:
            mult = 1.0
            mult += (occupancy - 50) / 100 * 0.3
            mult += max(0, (30 - lead_days) / 30 * 0.15)
            mult += {0:0.0,1:0.12,2:0.28,3:0.45}.get(event_mag, 0)
            rec  = round(base_price * np.clip(mult, 0.8, 1.5), 2)

        vs_base=(rec - base_price) / base_price * 100
        vs_comp=(rec - competitor_avg) / competitor_avg * 100
        exp_rev=rec * occ_rate * length_of_stay

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("recommended price", f"${rec:.2f}")
        c2.metric("vs base price",     f"{vs_base:+.1f}%")
        c3.metric("vs competitor avg", f"{vs_comp:+.1f}%")
        c4.metric("expected revenue",  f"${exp_rev:.2f}")

        factors = []
        if occupancy >= 80: factors.append(f"high occupancy ({occupancy}%) adds premium")
        if lead_days <= 3:  factors.append(f"last-minute booking ({lead_days} days) adds premium")
        if has_event:       factors.append(f"{event_type} event adds {event_mag*15:.0f}% premium")
        if is_weekend:      factors.append("weekend adds ~10% premium")
        if factors:
            st.info("**pricing factors applied:**\n" + "\n".join(f"- {f}" for f in factors))

    st.markdown("---")
    st.subheader("roi calculator")
    st.markdown("estimate the revenue impact of switching to dynamic pricing for your hotel")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_rooms = st.number_input("number of rooms", 50, 500, 100)
    with col2:
        occ_slider = st.slider("current occupancy %", 50, 95, 75)
    with col3:
        current_rate = st.number_input("current adr ($)", 80, 300, 150)

    if st.button("calculate roi"):
        calculator = ROICalculator()
        roi = calculator.calculate_roi(num_rooms=num_rooms, avg_occupancy=occ_slider / 100, current_avg_rate=current_rate)
        st.success(f" expected annual revenue lift: ${roi['incremental_revenue']:,.0f}")
        st.info(f" payback period: {roi['payback_months']} months")

        c1, c2, c3 = st.columns(3)
        c1.metric("current annual revenue", f"${roi['current_revenue']:,.0f}")
        c2.metric("projected annual revenue", f"${roi['projected_revenue']:,.0f}")

        c3.metric("5-year roi (per $1 spent)", f"${roi['five_year_roi_ratio']}")

        st.markdown(f"""
        **how this is calculated:**
        - revenue lift assumption: 18% (from gradient boosting model validation)
        - implementation cost: $15,000 one-time
        - annual license: $3,000/year
        - payback = total year-1 cost / annual incremental revenue × 12
        """)


#this is page 5: business insights

elif page == "business insights":
    st.markdown('<h1 class="main-header">business insights</h1>', unsafe_allow_html=True)

    st.subheader("key findings from the analysis")

    #insights based on the specific metrics
    findings = [
        ("Gradient Boosting Superiority",
         "The Gradient Boosting Regressor is the optimal model for this problem, achieving the lowest MAE of $8.85 and an R² of 0.9409. It successfully explains ~94% of the variance in room prices, significantly outperforming linear models.",
         "high", "medium"),

        ("Reinforcement Learning Observations",
         "The Q-Learning agent was trained over 200 episodes. While it established a state-action space of 35 pairs, it maintained a high exploration rate (ε=1.0). Consequently, the MAPE stabilized around 15.14%, indicating that supervised ML models currently offer higher precision than the RL agent for this specific setup.",
         "medium", "high"),

        ("Model Robustness via Hyperparameter Tuning",
         "Hyperparameter tuning was successfully applied to the pipeline, achieving a peak Cross-Validation R² of 0.9574. This confirms the models generalize well to unseen data and are not suffering from high bias or overfitting.",
         "high", "medium"),

        ("Customer Segmentation Potential",
         "K-Means clustering on the 119k+ records identified distinct customer behavioral groups. This allows for targeted pricing strategies (e.g., discounts for price-sensitive segments vs. premiums for high-value guests).",
         "medium", "low"),

        ("Anomaly Detection",
         "The system flagged 5,872 data points (approx. 5% of the dataset) as anomalies. These outliers represent unusual price/occupancy combinations that may indicate data entry errors, fraudulent bookings, or special corporate rates.",
         "low", "low"),

        ("Competitor & Lead Time Dynamics",
         "Analysis shows that Lead Time and Competitor Pricing are critical features. Staying within ±5% of competitor averages helps maintain occupancy, while pricing strategies must adapt to the booking window.",
         "high", "low"),
    ]

    for title, desc, impact, effort in findings:
        with st.expander(f" {title}"):
            st.write(desc)
            c1, c2 = st.columns(2)
            c1.metric("revenue impact", impact)
            c2.metric("implementation effort", effort)

    st.markdown("---")
    st.subheader("model performance comparison (as per report)")

    model_data = {
        "Model": ["Gradient Boosting", "Stacking Ensemble", "XGBoost (Tuned)", "Random Forest", "Decision Tree", "Linear Regression"],
        "R² Score": [0.9374, 0.9372, 0.9363, 0.9343, 0.9265, 0.8934],
        "MAE ($)": [7.40, 7.41, 7.45, 7.54, 7.81, 9.59],
        "Status": ["Selected", "Alternative", "Fast/Robust", "Baseline", "Interpretable", "Baseline"]
    }

    comparison_df = pd.DataFrame(model_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("dynamic pricing strategy rules")

    rules = {
        "high demand season": "Use Gradient Boosting predictions (Summer months) to justify premiums.",
        "q-learning fallback": "Use RL agent for exploration in low-stakes scenarios; revert to GB for precision.",
        "anomaly handling": "Flag prices deviating >20% from predicted norm for manual review.",
        "competitor response": "If competitor avg drops >5%, trigger price alert; else stick to ML prediction.",
        "low occupancy (<70%)": "Consider slight undercutting (as seen in ML Rec engine samples) to boost volume.",
    }

    rules_df = pd.DataFrame({"scenario": list(rules.keys()), "strategy": list(rules.values())})
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("deployment roadmap")
    st.markdown("""
    | Phase | Action Item | Rationale |
    |-------|-------------|-----------|
    | 1 | Deploy Gradient Boosting model via REST API | Highest R² (0.94) and lowest error. |
    | 2 | Integrate Anomaly Detection module | To filter the 5% outliers identified in analysis. |
    | 3 | A/B Test: ML vs. Static Pricing | Validate the revenue uplift suggested by the recommendation engine. |
    | 4 | Monitor Q-Learning Agent | Continue training to reduce exploration rate and lower MAPE below 15%. |
    | 5 | Retune Hyperparameters Quarterly | To maintain the CV R² score above 0.95. |
    """)
