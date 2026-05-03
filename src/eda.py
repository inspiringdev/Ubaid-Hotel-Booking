import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

# ── Paths & Setup ─────────────────────────────────────────────────────────────
import os as _os
_BASE      = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
CLEAN_PATH = _os.path.join(_BASE, "data", "hotel_booking_clean.csv")
FIG_DIR    = _os.path.join(_BASE, "outputs", "figures")
_os.makedirs(FIG_DIR, exist_ok=True)

# ── Style Configuration ───────────────────────────────────────────────────────
PALETTE = ["#1a1a2e","#16213e","#0f3460","#e94560","#533483","#2ec4b6","#ff9f1c","#cbf3f0"]
BG      = "#f8f9fa"
ACCENT  = "#e94560"
DARK    = "#1a1a2e"

plt.rcParams.update({
    "figure.facecolor":  BG, "axes.facecolor": "white",
    "axes.edgecolor":    "#cccccc", "axes.labelcolor": DARK,
    "axes.labelsize":    11, "axes.titlesize": 13,
    "axes.titleweight":  "bold", "xtick.color": DARK,
    "ytick.color":       DARK, "grid.color": "#e8e8e8",
    "grid.linestyle":    "--", "grid.alpha": 0.7,
    "font.family":       "DejaVu Sans", "legend.frameon": False,
})

def savefig(name, fig=None):
    path = os.path.join(FIG_DIR, f"{name}.png")
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=BG, edgecolor="none")
    plt.close("all")
    print(f"   ✓  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
def fig1_summary_statistics(df):
    print("\n[EDA-1] Summary Statistics")
    # We use 'price' instead of 'optimal_price' for historical accuracy
    target_col = "price" if "price" in df.columns else "optimal_price"
    
    key_cols = [c for c in [target_col, "base_price", "competitor_avg_price",
                "occupancy_rate", "lead_time_days", "length_of_stay"] if c in df.columns]
    
    stats_df = df[key_cols].describe().T.round(2)
    stats_df["cv%"] = (stats_df["std"] / stats_df["mean"] * 100).round(1)
    stats_df = stats_df[["mean","std","min","25%","50%","75%","max","cv%"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    tbl = ax.table(cellText=stats_df.values.astype(str),
                   rowLabels=stats_df.index, colLabels=stats_df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef2f7")
            
    ax.set_title("Summary Statistics — Key Numeric Features",
                 pad=20, fontsize=14, fontweight="bold", color=DARK)
    savefig("01_summary_statistics", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig2_price_distributions(df):
    print("\n[EDA-2] Price Distributions")
    # Use 'price' (actual) instead of optimal
    target_col = "price" if "price" in df.columns else "optimal_price"
    
    cols_labels = [
        (target_col,       "Actual Room Price ($)"),
        ("base_price",     "Base Price ($)"),
        ("competitor_avg_price", "Competitor Avg Price ($)")
    ]
    # Filter existing columns
    cols_labels = [(c, l) for c, l in cols_labels if c in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Price Distribution Analysis", fontsize=15,
                 fontweight="bold", color=DARK, y=1.02)
    
    colors = [ACCENT, "#0f3460", "#2ec4b6"]
    for ax, (col, lbl), color in zip(axes.flat, cols_labels, colors):
        data = df[col].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(data.mean(),   color=DARK,      lw=2, ls="--", label=f"Mean ${data.mean():.0f}")
        ax.axvline(data.median(), color="#e74c3c",  lw=2, ls=":",  label=f"Median ${data.median():.0f}")
        ax.set_xlabel(lbl); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8); ax.grid(True)
        
        sk = data.skew()
        ax.text(0.97, 0.92, f"Skew={sk:.2f}", transform=ax.transAxes,
                ha="right", fontsize=8, color=DARK,
                bbox=dict(fc="white", ec="#ccc", alpha=0.8, pad=3))
        
    plt.tight_layout()
    savefig("02_price_distributions", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig3_correlation_heatmap(df):
    print("\n[EDA-3] Correlation Heatmap")
    # IMPORTANT: Removed 'base_price' to avoid the fake 0.99 correlation
    # This allows us to see real relationships like 'lead_time_days' vs 'price'
    
    target_col = "price" if "price" in df.columns else "optimal_price"
    
    want = [target_col, "competitor_avg_price", "occupancy_rate", 
            "lead_time_days", "length_of_stay", "event_magnitude",
            "reviews_score", "is_weekend", "is_holiday"]
    
    # Only keep columns that exist
    num_cols = [c for c in want if c in df.columns]
    
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                center=0, ax=ax, linewidths=0.5, linecolor="#f0f0f0",
                annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14,
                 fontweight="bold", color=DARK, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    savefig("03_correlation_heatmap", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig4_seasonal_trends(df):
    print("\n[EDA-4] Seasonal & Monthly Trends")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Seasonal & Temporal Pricing Trends", fontsize=16,
                 fontweight="bold", color=DARK)
    
    months = range(1, 13)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    target_col = "price" if "price" in df.columns else "optimal_price"

    # Monthly avg price
    ax = axes[0, 0]
    monthly = df.groupby("month")[target_col].mean()
    ax.bar(monthly.index, monthly.values,
           color=[ACCENT if v == monthly.max() else "#0f3460" for v in monthly.values],
           edgecolor="white")
    ax.set_xticks(list(months)); ax.set_xticklabels(month_names, fontsize=8)
    ax.set_title("Avg Actual Price by Month"); ax.set_ylabel("Price ($)")
    ax.grid(True, axis="y")

    # Weekly pattern
    ax = axes[0, 1]
    days   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    # Day of week might be 0-6 or strings, this handles both
    weekly = df.groupby("day_of_week")[target_col].mean()
    ax.plot(days[:len(weekly)], weekly.values, marker="o", color=ACCENT, lw=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(range(len(weekly)), weekly.values, alpha=0.15, color=ACCENT)
    ax.set_title("Avg Actual Price by Day of Week"); ax.set_ylabel("Price ($)")
    ax.grid(True)

    # Season box plots
    ax = axes[1, 0]
    if "season" in df.columns:
        season_order = [s for s in ["Spring","Summer","Fall","Winter"] if s in df["season"].values]
        season_data  = [df[df["season"] == s][target_col].dropna().values for s in season_order]
        bp = ax.boxplot(season_data, labels=season_order, patch_artist=True,
                        medianprops=dict(color="white", lw=2))
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_title("Price Distribution by Season")
    else:
        ax.text(0.5, 0.5, "No 'season' column found", ha='center')
    ax.set_ylabel("Price ($)")
    ax.grid(True, axis="y")

    # Year-over-year
    ax = axes[1, 1]
    if "year" in df.columns:
        for yr, color in zip(sorted(df["year"].unique()), ["#0f3460", ACCENT, "#2ec4b6"]):
            sub = df[df["year"] == yr].groupby("month")[target_col].mean()
            ax.plot(sub.index, sub.values, marker="o", label=str(yr), color=color, lw=2)
        ax.set_xticks(list(months)); ax.set_xticklabels(month_names, fontsize=8)
        ax.set_title("Year-over-Year Price Trend")
        ax.legend()
    ax.set_ylabel("Avg Price ($)")
    ax.grid(True)

    plt.tight_layout()
    savefig("04_seasonal_trends", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig5_event_analysis(df):
    print("\n[EDA-5] Event Impact Analysis")
    # Only run if event columns exist
    if "event_type" not in df.columns or "has_event" not in df.columns:
        print("   (Skipping - Event columns missing)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Impact of Local Events on Pricing", fontsize=16,
                 fontweight="bold", color=DARK)
    target_col = "price" if "price" in df.columns else "optimal_price"

    # Avg price by event type
    ax = axes[0]
    evt_price = df.groupby("event_type")[target_col].mean().sort_values(ascending=True)
    colors    = [ACCENT if et != "None" else "#aaaaaa" for et in evt_price.index]
    ax.barh(evt_price.index, evt_price.values, color=colors, edgecolor="white")
    ax.set_title("Avg Price by Event Type"); ax.set_xlabel("Price ($)")
    ax.grid(True, axis="x")

    # Occupancy: event vs no event
    ax = axes[2] # Swapped index to handle potential missing magnitudes
    if "occupancy_rate" in df.columns:
        has  = df[df["has_event"] == 1]["occupancy_rate"]
        none = df[df["has_event"] == 0]["occupancy_rate"]
        ax.hist(has,  bins=40, alpha=0.6, color=ACCENT,    label="Event",    density=True)
        ax.hist(none, bins=40, alpha=0.6, color="#0f3460",  label="No Event", density=True)
        ax.axvline(has.mean(),  color=ACCENT,   lw=2, ls="--")
        ax.axvline(none.mean(), color="#0f3460", lw=2, ls="--")
        ax.set_title("Occupancy: Event vs No Event")
    ax.set_xlabel("Occupancy Rate")
    ax.set_ylabel("Density"); ax.legend(); ax.grid(True)
    
    # Middle plot: Magnitude
    ax = axes[1]
    if "event_magnitude" in df.columns:
        mag_price = df[df["event_magnitude"] > 0].groupby("event_magnitude")[target_col].mean()
        ax.plot(mag_price.index, mag_price.values, marker="D", color=ACCENT,
                lw=2.5, markersize=10, markeredgecolor=DARK)
        ax.fill_between(mag_price.index, mag_price.values, alpha=0.15, color=ACCENT)
        ax.set_title("Price vs Event Magnitude")
        ax.set_xlabel("Event Magnitude"); ax.set_ylabel("Avg Price ($)")
    else:
        ax.text(0.5, 0.5, "No Magnitude Data", ha='center')
    ax.grid(True)

    plt.tight_layout()
    savefig("05_event_analysis", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig6_competitor_analysis(df):
    print("\n[EDA-6] Competitor Pricing Analysis")
    if "competitor_avg_price" not in df.columns:
        print("   (Skipping - Competitor columns missing)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Competitor Pricing Intelligence", fontsize=16,
                 fontweight="bold", color=DARK)
    target_col = "price" if "price" in df.columns else "optimal_price"

    # Our price vs comp avg scatter
    ax = axes[0, 0]
    sample = df.sample(min(2000, len(df)), random_state=1)
    sc = ax.scatter(sample["competitor_avg_price"], sample[target_col],
                    c=sample["occupancy_rate"] if "occupancy_rate" in df.columns else "blue",
                    cmap="RdYlGn", alpha=0.5, s=20)
    ax.plot([50, 450], [50, 450], "k--", lw=1.5, label="Parity")
    plt.colorbar(sc, ax=ax, label="Occupancy Rate" if "occupancy_rate" in df.columns else "")
    ax.set_xlabel("Competitor Avg Price ($)")
    ax.set_ylabel("Our Actual Price ($)")
    ax.set_title("Our Price vs Competitor Price"); ax.legend(); ax.grid(True)

    # Price positioning
    ax = axes[0, 1]
    if "price_vs_comp_avg" in df.columns:
        pos = df["price_vs_comp_avg"] * 100
        ax.hist(pos, bins=60, color="#0f3460", alpha=0.8, edgecolor="white")
        ax.axvline(0, color=ACCENT, lw=2.5, ls="--", label="At Parity")
        ax.axvline(pos.mean(), color="#2ec4b6", lw=2, ls=":", label=f"Mean {pos.mean():.1f}%")
        ax.set_xlabel("Price vs Competitor (%)")
    else:
        ax.text(0.5, 0.5, "Calculated Column Missing", ha='center')
    ax.set_ylabel("Count")
    ax.set_title("Price Positioning vs Competitors"); ax.legend(); ax.grid(True)

    # Room type range (Simplified)
    ax = axes[1, 0]
    if "room_type" in df.columns:
        rooms = df.groupby("room_type").agg(our_price=(target_col,"mean")).sort_values("our_price")
        ax.barh(rooms.index, rooms["our_price"], color=ACCENT, alpha=0.8, edgecolor="white")
        ax.set_title("Avg Our Price by Room Type")
    else:
        ax.text(0.5, 0.5, "Room Type Missing", ha='center')
    ax.set_xlabel("Price ($)")
    ax.grid(True, axis="x")

    # Scatter spread (Mock if missing)
    ax = axes[1, 1]
    if "price_comp_spread" in df.columns and "price_premium" in df.columns:
        ax.scatter(df["price_comp_spread"], df["price_premium"], alpha=0.2, s=15, color="#533483")
        ax.set_title("Price Spread vs Our Premium")
        ax.set_xlabel("Competitor Spread"); ax.set_ylabel("Premium")
    else:
        ax.text(0.5, 0.5, "Spread Data Missing", ha='center')
    ax.grid(True)

    plt.tight_layout()
    savefig("06_competitor_analysis", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig7_occupancy_demand(df):
    print("\n[EDA-7] Occupancy & Demand Dynamics")
    if "occupancy_rate" not in df.columns:
        print("   (Skipping - Occupancy data missing)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Occupancy Rate & Demand Score Analysis", fontsize=16,
                 fontweight="bold", color=DARK)
    target_col = "price" if "price" in df.columns else "optimal_price"

    ax = axes[0]
    # Scatter with color based on price
    sc = ax.scatter(df["occupancy_rate"], df[target_col],
               c=df[target_col], cmap="plasma", alpha=0.2, s=10)
    ax.set_xlabel("Occupancy Rate"); ax.set_ylabel("Actual Price ($)")
    ax.set_title("Occupancy vs. Actual Price"); ax.grid(True)

    ax = axes[1]
    bins = [0, 0.5, 0.65, 0.75, 0.85, 0.95, 1.01]
    labels = ["<50%","50–65%","65–75%","75–85%","85–95%",">95%"]
    df["occ_bin"] = pd.cut(df["occupancy_rate"], bins=bins, labels=labels)
    occ_price = df.groupby("occ_bin", observed=True)[target_col].mean()
    ax.bar(occ_price.index, occ_price.values,
           color=PALETTE[:len(occ_price)], edgecolor="white")
    ax.set_title("Avg Price by Occupancy Band")
    ax.set_xlabel("Occupancy Rate Band"); ax.set_ylabel("Price ($)")
    ax.grid(True, axis="y")

    ax = axes[2]
    if "demand_score" in df.columns:
        ax.scatter(df["demand_score"], df[target_col], alpha=0.15, s=8, color=ACCENT)
        m2, b2 = np.polyfit(df["demand_score"], df[target_col], 1)
        xs = np.linspace(df["demand_score"].min(), df["demand_score"].max(), 100)
        ax.plot(xs, m2*xs+b2, color=DARK, lw=2.5)
        r2 = df[["demand_score",target_col]].corr().iloc[0, 1] ** 2
        ax.set_title(f"Demand Score vs. Price (r²={r2:.3f})")
        ax.set_xlabel("Demand Score")
    else:
        ax.text(0.5, 0.5, "Demand Score Missing", ha='center')
    ax.set_ylabel("Price ($)"); ax.grid(True)

    plt.tight_layout()
    savefig("07_occupancy_demand", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig8_revenue_analysis(df):
    print("\n[EDA-8] Revenue Analysis")
    # Check if we have revenue data
    rev_col = "revenue_per_night"
    if rev_col not in df.columns and "revenue" in df.columns:
        rev_col = "revenue"
    
    if rev_col not in df.columns:
        print("   (Skipping - Revenue columns missing)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Revenue Analysis & KPIs", fontsize=16,
                 fontweight="bold", color=DARK)

    # Monthly revenue trend
    ax = axes[0, 1]
    df["ym"]  = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    monthly   = df.groupby("ym")[rev_col].mean().reset_index()
    ax.plot(range(len(monthly)), monthly[rev_col], color=ACCENT, lw=2)
    ax.fill_between(range(len(monthly)), monthly[rev_col], alpha=0.15, color=ACCENT)
    ticks = list(range(0, len(monthly), max(1, len(monthly)//6)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([monthly["ym"].iloc[i] for i in ticks], rotation=45, fontsize=8)
    ax.set_title("Monthly Average Revenue per Night"); ax.set_ylabel("Revenue ($)")
    ax.grid(True)

    # Revenue by channel
    ax = axes[1, 0]
    if "channel" in df.columns:
        ch_rev = df.groupby("channel")[rev_col].agg(["mean","sum"])
        ch_rev["sum"] /= 1e6
        ax2  = ax.twinx()
        ax.bar(ch_rev.index, ch_rev["mean"], color=PALETTE[:len(ch_rev)], edgecolor="white", alpha=0.8)
        ax2.plot(ch_rev.index, ch_rev["sum"], color=DARK, marker="o", lw=2)
        ax.set_title("Revenue by Booking Channel")
        ax.set_ylabel("Avg Revenue/Night ($)")
        ax2.set_ylabel("Total Revenue (M$)")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    else:
        ax.text(0.5, 0.5, "Channel Data Missing", ha='center')
    ax.grid(True, axis="y")

    # Revenue Heatmap
    ax = axes[1, 1]
    if "room_type" in df.columns and "season" in df.columns:
        pivot = df.pivot_table(values=rev_col, index="room_type",
                               columns="season", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Revenue ($)"})
        ax.set_title("Revenue Heatmap: Room Type × Season")
    else:
        ax.text(0.5, 0.5, "Room/Season Data Missing", ha='center')
    
    # Simple placeholder for first plot if revpar missing
    ax = axes[0, 0]
    if "revpar" in df.columns:
        hotel_rev = df.groupby("hotel_name")["revpar"].mean().sort_values()
        ax.barh(hotel_rev.index, hotel_rev.values, color=PALETTE[:len(hotel_rev)], edgecolor="white")
        ax.set_title("Avg RevPAR by Hotel")
    else:
        ax.text(0.5, 0.5, "RevPAR Data Missing\n(Showing Total Rev)", ha='center')
        if "hotel_name" in df.columns:
             df.groupby("hotel_name")[rev_col].sum().plot(kind='barh', ax=ax, color=ACCENT)
        ax.set_title("Total Revenue by Hotel")
    ax.set_xlabel("Revenue ($)")
    ax.grid(True, axis="x")

    plt.tight_layout()
    savefig("08_revenue_analysis", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig9_lead_time_channel(df):
    print("\n[EDA-9] Lead-Time & Channel Analysis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lead-Time & Channel Booking Analysis", fontsize=16,
                 fontweight="bold", color=DARK)
    target_col = "price" if "price" in df.columns else "optimal_price"

    ax = axes[0]
    if "lead_time_bucket" in df.columns:
        order     = ["Last_Minute","Short_Term","Medium_Term","Long_Term"]
        lt_price  = df.groupby("lead_time_bucket")[target_col].mean()
        vals      = [lt_price.get(k, 0) for k in order]
        ax.bar(order, vals, color=[ACCENT,"#ff9f1c","#0f3460","#2ec4b6"], edgecolor="white")
        ax.set_title("Avg Price by Lead-Time Bucket")
    else:
        ax.text(0.5, 0.5, "Bucket Data Missing", ha='center')
    ax.set_ylabel("Avg Price ($)")
    ax.grid(True, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    ax = axes[1]
    if "channel" in df.columns and "lead_time_days" in df.columns:
        for ch, color in zip(df["channel"].unique(), PALETTE):
            sub = df[df["channel"] == ch].groupby("lead_time_days")[target_col].mean()
            sub = sub[sub.index <= 120].rolling(5, min_periods=1).mean()
            ax.plot(sub.index, sub.values, label=ch, color=color, lw=1.5)
        ax.set_title("Price Trend by Lead Time × Channel")
        ax.set_xlabel("Lead Time (days)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Channel/LeadTime Data Missing", ha='center')
    ax.set_ylabel("Price ($)")
    ax.grid(True)

    ax = axes[2]
    if "guest_type" in df.columns:
        guest_price = df.groupby("guest_type")[target_col].mean().sort_values(ascending=False)
        ax.bar(guest_price.index, guest_price.values, color=PALETTE[:len(guest_price)], edgecolor="white")
        ax.set_title("Avg Price by Guest Type")
    else:
        ax.text(0.5, 0.5, "Guest Type Missing", ha='center')
    ax.set_ylabel("Price ($)")
    ax.grid(True, axis="y")

    plt.tight_layout()
    savefig("09_lead_time_channel", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig10_room_type_analysis(df):
    print("\n[EDA-10] Room Type Analysis")
    if "room_type" not in df.columns:
        print("   (Skipping - Room type data missing)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Room Type Analysis", fontsize=16,
                 fontweight="bold", color=DARK)
    target_col = "price" if "price" in df.columns else "optimal_price"

    ax = axes[0]
    room_counts = df["room_type"].value_counts()
    ax.pie(room_counts.values, labels=room_counts.index, autopct="%1.1f%%",
           colors=PALETTE[:len(room_counts)], startangle=140,
           wedgeprops=dict(edgecolor="white", lw=1.5))
    ax.set_title("Bookings by Room Type")

    ax = axes[1]
    df.boxplot(column=target_col, by="room_type", ax=ax,
               patch_artist=False, medianprops=dict(color=ACCENT, lw=2))
    ax.set_title("Price Distribution by Room Type")
    ax.set_xlabel("Room Type"); ax.set_ylabel("Actual Price ($)")
    plt.sca(ax); plt.title("")
    ax.grid(True, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    ax = axes[2]
    if "is_weekend" in df.columns and "occupancy_rate" in df.columns:
        pivot2 = df.pivot_table(values="occupancy_rate", index="room_type",
                                columns="is_weekend", aggfunc="mean")
        pivot2.columns = ["Weekday","Weekend"]
        pivot2.plot(kind="bar", ax=ax, color=[ACCENT,"#0f3460"], edgecolor="white", width=0.6)
        ax.set_title("Occupancy: Weekday vs Weekend")
        ax.set_ylabel("Avg Occupancy Rate")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Weekend/Occupancy Missing", ha='center')
    ax.set_xlabel("Room Type")
    ax.grid(True, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    savefig("10_room_type_analysis", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig11_feature_importance_eda(df):
    print("\n[EDA-11] Feature–Target Correlation")
    target_col = "price" if "price" in df.columns else "optimal_price"
    
    # IMPORTANT: Explicitly EXCLUDING 'base_price' to show real drivers
    want = ["competitor_avg_price","occupancy_rate","demand_score",
            "event_magnitude","lead_time_days","length_of_stay",
            "reviews_score","is_weekend","is_holiday","has_event",
            "repeat_guest"]
            
    feat_cols = [c for c in want if c in df.columns and c != target_col]
    
    # Only run if we have features
    if not feat_cols:
        print("   (Skipping - Not enough feature columns)")
        return

    corrs = df[feat_cols + [target_col]].corr()[target_col].drop(target_col)
    corrs = corrs.sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = [ACCENT if c > 0 else "#0f3460" for c in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white")
    ax.axvline(0, color=DARK, lw=1)
    ax.set_title(f"Feature Correlation with {target_col.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold", color=DARK)
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.grid(True, axis="x")
    for i, (name, val) in enumerate(corrs.items()):
        ax.text(val + (0.005 if val >= 0 else -0.005), i,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    savefig("11_feature_target_correlation", fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig12_pair_plots(df):
    print("\n[EDA-12] Pairplot")
    target_col = "price" if "price" in df.columns else "optimal_price"
    cols = [c for c in [target_col, "occupancy_rate", "lead_time_days",
                         "competitor_avg_price"] if c in df.columns]
    
    if len(cols) < 2:
        print("   (Skipping - Not enough numeric columns)")
        return
        
    if "room_type" in df.columns:
        g = sns.pairplot(df[cols + ["room_type"]].sample(min(1500, len(df)), random_state=1),
                         hue="room_type", diag_kind="kde",
                         plot_kws=dict(alpha=0.3, s=10),
                         palette=PALETTE[:df["room_type"].nunique()])
        g.fig.suptitle("Pairplot — Key Features by Room Type", y=1.01, fontsize=14, fontweight="bold")
        savefig("12_pairplot", g.fig)
    else:
        print("   (Skipping - 'room_type' missing for hue)")


# ─────────────────────────────────────────────────────────────────────────────
def run_eda():
    print("=" * 65)
    print("EXPLORATORY DATA ANALYSIS — HOTEL DYNAMIC PRICING")
    print("=" * 65)
    
    try:
        df = pd.read_csv(CLEAN_PATH, parse_dates=["checkin_date"])
        print(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

        # List of functions to run
        figs = [
            fig1_summary_statistics,
            fig2_price_distributions,
            fig3_correlation_heatmap,
            fig4_seasonal_trends,
            fig5_event_analysis,
            fig6_competitor_analysis,
            fig7_occupancy_demand,
            fig8_revenue_analysis,
            fig9_lead_time_channel,
            fig10_room_type_analysis,
            fig11_feature_importance_eda,
            fig12_pair_plots,
        ]

        for fig_func in figs:
            try:
                fig_func(df)
            except Exception as e:
                print(f"   ⚠ Error generating {fig_func.__name__}: {e}")

        print("\n" + "=" * 65)
        print(f"✓  EDA Complete. Figures saved to {FIG_DIR}")
        print("=" * 65)
        
    except FileNotFoundError:
        print(f"   ✗ Error: File not found at {CLEAN_PATH}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

if __name__ == "__main__":
    run_eda()