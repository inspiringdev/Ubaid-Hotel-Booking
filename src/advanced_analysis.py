"""
advanced analysis module

analyses included:
- k-means customer segmentation
- pca dimensionality reduction
- time-series revenue forecasting
- hyperparameter tuning (randomizedsearchcv)
- stacking ensemble model
- learning curves
- elasticity analysis
- permutation feature importance
- anomaly detection (isolationforest)
- revenue scenario analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RandomizedSearchCV, learning_curve)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor, IsolationForest)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import pickle, os, warnings, json
warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN  = os.path.join(BASE, "data", "hotel_booking_clean.csv")
ADVFIG = os.path.join(BASE, "outputs", "adv_figures")
MODELS = os.path.join(BASE, "outputs", "models")

C = dict(
    surface="#111827", card="#1a2035", border="#252d42",
    accent="#3b82f6", gold="#f59e0b", teal="#10b981", rose="#f43f5e",
    violet="#8b5cf6", orange="#f97316", sky="#0ea5e9", lime="#84cc16",
    text="#f1f5f9", muted="#64748b", white="#ffffff"
)
PAL6 = [C["accent"], C["gold"], C["teal"], C["rose"], C["violet"], C["orange"]]

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["card"],
    "axes.edgecolor": C["border"], "axes.labelcolor": C["text"],
    "axes.labelsize": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": C["white"], "xtick.color": C["muted"], "ytick.color": C["muted"],
    "grid.color": C["border"], "grid.linestyle": "--", "grid.alpha": 0.5,
    "text.color": C["text"], "font.family": "DejaVu Sans",
    "legend.facecolor": C["card"], "legend.edgecolor": C["border"], "legend.fontsize": 9,
})

def _save(name, fig=None):
    p = os.path.join(ADVFIG, f"{name}.png")
    (fig or plt).savefig(p, dpi=150, bbox_inches="tight",
                         facecolor=C["surface"], edgecolor="none")
    plt.close("all")
    print(f"  saved {name}.png")

FEAT = [
    "month", "day_of_week", "week_of_year", "is_weekend", "is_holiday",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lead_time_days", "length_of_stay",
    "competitor_avg_price", "competitor_min_price", "competitor_max_price",
    "base_vs_comp_avg", "base_vs_comp_min", "price_comp_spread",
    "occupancy_rate", "demand_score", "high_demand",
    "event_magnitude", "has_event", "weather_score",
    "reviews_score", "repeat_guest",
    "hotel_name_enc", "room_type_enc", "channel_enc",
    "guest_type_enc", "event_type_enc", "season_enc", "lead_time_bucket_enc",
]
TARGET = "actual_price"


def _load_split(df):
    """chronological split"""
    existing = [c for c in FEAT if c in df.columns]
    X = df[existing].fillna(0)
    y = df[TARGET]
    if "is_test" in df.columns:
        tr = df["is_test"] == 0
        te = df["is_test"] == 1
        return (X[tr].reset_index(drop=True), X[te].reset_index(drop=True),
                y[tr].reset_index(drop=True), y[te].reset_index(drop=True))
    split = int(len(X) * 0.80)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def run_advanced_analysis():
    os.makedirs(ADVFIG, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    print("loading data ...")
    df = pd.read_csv(CLEAN)
    existing = [c for c in FEAT if c in df.columns]
    X_tr, X_te, y_tr, y_te = _load_split(df)
    print(f"  {len(df):,} records  {len(existing)} features  target={TARGET}")

    #this is k-means segmentation
    print("\n[1] k-means customer segmentation")
    seg_cols = [c for c in ["lead_time_days","length_of_stay","actual_price",
                             "occupancy_rate","reviews_score","repeat_guest",
                             "event_magnitude","demand_score"] if c in df.columns]
    Xs = StandardScaler().fit_transform(df[seg_cols].fillna(0))

    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)

    best_k = 5
    km5 = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["segment"] = km5.fit_predict(Xs)
    pca2 = PCA(n_components=2, random_state=42)
    X2d  = pca2.fit_transform(Xs)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(C["surface"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("customer segmentation — k-means (k=5)",
                 fontsize=14, fontweight="bold", color=C["white"], y=1.01)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(range(2, 11), inertias, "o-", color=C["accent"], lw=2.5, ms=7)
    ax.axvline(best_k, color=C["gold"], ls="--", lw=1.5, label="k=5")
    ax.set_title("elbow method"); ax.set_xlabel("k"); ax.set_ylabel("inertia")
    ax.legend(); ax.grid(True)

    ax = fig.add_subplot(gs[0, 1:])
    for i, col in enumerate(PAL6[:best_k]):
        mask = df["segment"] == i
        ax.scatter(X2d[mask, 0], X2d[mask, 1], c=col, alpha=.35, s=8, label=f"seg {i+1}")
        cx, cy = X2d[mask, 0].mean(), X2d[mask, 1].mean()
        ax.scatter(cx, cy, c=col, s=200, marker="*", edgecolors="white", zorder=5)
    ax.set_title("pca 2d segment map"); ax.legend(markerscale=2); ax.grid(True)

    seg_profile = df.groupby("segment")[["actual_price","occupancy_rate","lead_time_days"]].mean()
    for col_i, col_name in enumerate(seg_profile.columns[:3]):
        ax = fig.add_subplot(gs[1, col_i])
        vals = seg_profile[col_name].values
        ax.bar([f"s{i+1}" for i in range(best_k)], vals,
               color=[PAL6[s] for s in range(best_k)], edgecolor="white")
        ax.set_title(col_name.replace("_"," ")); ax.grid(True, axis="y")
    _save("A01_kmeans_segmentation", fig)

    seg_rev = df.groupby("segment").agg(
        count=("actual_price","count"), avg_price=("actual_price","mean"),
        avg_occ=("occupancy_rate","mean"),
    ).round(2)
    seg_rev.to_csv(os.path.join(MODELS, "segment_profiles.csv"))

    #this is pca explained variance
    print("\n[2] pca explained variance")
    X_all    = df[existing].fillna(0)
    Xsc      = StandardScaler().fit_transform(X_all)
    pca_full = PCA(random_state=42)
    pca_full.fit(Xsc)
    evr   = pca_full.explained_variance_ratio_
    cumev = np.cumsum(evr)
    n90   = int(np.argmax(cumev >= 0.90) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("pca — dimensionality reduction", fontsize=13, color=C["white"])
    ax = axes[0]
    ax.bar(range(1, 21), evr[:20]*100, color=C["accent"], alpha=.75, edgecolor="white")
    ax.plot(range(1, 21), cumev[:20]*100, "o-", color=C["gold"], lw=2, ms=5, label="cumulative")
    ax.axhline(90, color=C["teal"], ls="--", lw=1.5, label="90% threshold")
    ax.axvline(n90, color=C["rose"], ls=":", lw=1.5, label=f"n={n90}")
    ax.set_title("scree plot"); ax.legend(); ax.grid(True)

    loadings  = pd.DataFrame(pca_full.components_[:2].T,
                              index=X_all.columns, columns=["PC1","PC2"])
    top_feats = loadings["PC1"].abs().nlargest(15).index
    ldf       = loadings.loc[top_feats]
    axes[1].scatter(ldf["PC1"], ldf["PC2"], c=range(len(ldf)), cmap="plasma", s=80)
    for feat, row in ldf.iterrows():
        axes[1].annotate(feat[:12], (row["PC1"]+.005, row["PC2"]+.005),
                          fontsize=7.5, color=C["text"])
    axes[1].axhline(0, color=C["border"], lw=1)
    axes[1].axvline(0, color=C["border"], lw=1)
    axes[1].set_title("pc1 vs pc2 feature loadings"); axes[1].grid(True)
    plt.tight_layout()
    _save("A02_pca_analysis", fig)

    #this is time-series forecasting
    print("\n[3] time-series forecasting")
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    rev_col = "revenue_per_night" if "revenue_per_night" in df.columns else "actual_price"
    ts = (df.groupby("checkin_date")[rev_col]
            .mean().reset_index()
            .rename(columns={"checkin_date":"ds", rev_col:"y"})
            .sort_values("ds").reset_index(drop=True))
    ts["rolling_7"]  = ts["y"].rolling(7,  min_periods=1).mean()
    ts["rolling_30"] = ts["y"].rolling(30, min_periods=1).mean()
    ts["t"]          = (ts["ds"] - ts["ds"].min()).dt.days
    coefs            = np.polyfit(ts["t"], ts["y"], deg=3)
    t_future         = np.arange(ts["t"].max()+1, ts["t"].max()+31)
    dates_future     = pd.date_range(ts["ds"].max() + pd.Timedelta(1,"d"), periods=30)
    y_future         = np.poly1d(coefs)(t_future)
    ts["std30"]      = ts["y"].rolling(30, min_periods=1).std().fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("time-series revenue analysis and 30-day forecast",
                 fontsize=13, color=C["white"])
    ax = axes[0]
    ax.fill_between(ts["ds"], ts["rolling_30"]-1.5*ts["std30"],
                    ts["rolling_30"]+1.5*ts["std30"], alpha=.15, color=C["accent"])
    ax.plot(ts["ds"], ts["y"],           color=C["muted"],  lw=.6,  alpha=.5, label="daily")
    ax.plot(ts["ds"], ts["rolling_7"],   color=C["teal"],   lw=1.2, label="7d ma")
    ax.plot(ts["ds"], ts["rolling_30"],  color=C["accent"], lw=2,   label="30d ma")
    ax.plot(dates_future, y_future,      color=C["rose"],   lw=2.5, ls="--", label="forecast")
    ax.set_title("revenue per night + forecast"); ax.legend(ncol=4); ax.grid(True)

    df["year_col"]  = df["checkin_date"].dt.year
    df["month_col"] = df["checkin_date"].dt.month
    heat = df.pivot_table(values=rev_col, index="year_col",
                           columns="month_col", aggfunc="mean")
    heat.columns = ["jan","feb","mar","apr","may","jun",
                    "jul","aug","sep","oct","nov","dec"][:len(heat.columns)]
    sns.heatmap(heat, ax=axes[1], annot=True, fmt=".0f",
                cmap="YlOrRd", linewidths=.5, cbar_kws={"shrink":.7})
    axes[1].set_title("monthly revenue heatmap by year")
    plt.tight_layout()
    _save("A03_time_series_forecast", fig)

    #this is hyperparameter tuning
    print("\n[4] hyperparameter tuning")
    param_dist = {
        "n_estimators":     [100, 150, 200, 300],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.03, 0.05, 0.08, 0.10, 0.15],
        "subsample":        [0.7, 0.8, 0.85, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    }
    rs = RandomizedSearchCV(XGBRegressor(random_state=42, verbosity=0),
                             param_dist, n_iter=20, cv=3, scoring="r2",
                             n_jobs=-1, random_state=42, verbose=0)
    rs.fit(X_tr, y_tr)
    best_xgb = rs.best_estimator_
    print(f"  best cv r²: {rs.best_score_:.4f}")
    pd.DataFrame(rs.cv_results_).to_csv(
        os.path.join(MODELS, "hyperparameter_tuning.csv"), index=False)
    with open(os.path.join(MODELS, "best_xgb_tuned.pkl"), "wb") as f:
        pickle.dump(best_xgb, f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("hyperparameter tuning — randomizedsearchcv",
                 fontsize=13, color=C["white"])
    for ax, param in zip(axes, ["param_learning_rate","param_max_depth","param_n_estimators"]):
        sub     = pd.DataFrame(rs.cv_results_)[[param, "mean_test_score"]].dropna()
        sub[param] = sub[param].astype(str)
        grouped = sub.groupby(param)["mean_test_score"].mean().sort_index()
        ax.bar(grouped.index, grouped.values, color=C["accent"], edgecolor="white")
        ax.set_title(param.replace("param_","").replace("_"," "))
        ax.set_ylabel("mean cv r²"); ax.grid(True, axis="y")
    plt.tight_layout()
    _save("A04_hyperparameter_tuning", fig)

    #this is stacking ensemble
    print("\n[5] stacking ensemble")
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=100, learning_rate=.08,
                                          max_depth=5, random_state=42)),
        ("dt", DecisionTreeRegressor(max_depth=10, random_state=42)),
    ]
    stack = StackingRegressor(estimators=base_models,
                               final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    stack.fit(X_tr, y_tr)
    stack_preds = stack.predict(X_te)
    stack_r2    = r2_score(y_te, stack_preds)
    stack_mae   = mean_absolute_error(y_te, stack_preds)
    stack_rmse  = float(np.sqrt(mean_squared_error(y_te, stack_preds)))
    print(f"  stacking r²={stack_r2:.4f}  mae={stack_mae:.2f}")
    with open(os.path.join(MODELS, "stacking_model.pkl"), "wb") as f:
        pickle.dump(stack, f)

    #this is learning curves
    print("\n[6] learning curves")
    X_all = df[existing].fillna(0)
    y_all = df[TARGET]
    gb_lc = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        gb_lc, X_all, y_all, cv=5, n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 10), scoring="r2")
    tr_m, tr_s = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_m, va_s = val_scores.mean(axis=1),   val_scores.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("learning curves — gradient boosting", fontsize=13, color=C["white"])
    ax = axes[0]
    ax.fill_between(train_sizes, tr_m-tr_s, tr_m+tr_s, alpha=.15, color=C["accent"])
    ax.fill_between(train_sizes, va_m-va_s, va_m+va_s, alpha=.15, color=C["teal"])
    ax.plot(train_sizes, tr_m, "o-", color=C["accent"], lw=2, label="train r²")
    ax.plot(train_sizes, va_m, "s-", color=C["teal"],   lw=2, label="validation r²")
    ax.set_title("r² learning curve"); ax.legend(); ax.grid(True)
    ax = axes[1]
    bias     = 1 - va_m
    variance = tr_m - va_m
    ax.fill_between(train_sizes, 0, bias,             alpha=.5, color=C["rose"],   label="bias")
    ax.fill_between(train_sizes, bias, bias+variance, alpha=.5, color=C["violet"], label="variance")
    ax.set_title("bias-variance decomposition"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    _save("A05_learning_curves", fig)

    #this is permutation importance
    print("\n[7] permutation importance")
    gb_perm = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
    gb_perm.fit(X_tr, y_tr)
    perm = permutation_importance(gb_perm, X_te, y_te, n_repeats=15,
                                   random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature":    X_tr.columns,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("permutation feature importance", fontsize=13, color=C["white"])
    colors = [C["accent"] if i < 5 else C["gold"] if i < 10 else C["muted"]
              for i in range(len(imp_df))]
    axes[0].barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                 xerr=imp_df["std"][::-1], color=colors[::-1],
                 edgecolor="none", capsize=3)
    axes[0].set_title("top 20 features")
    axes[0].set_xlabel("mean decrease in r²"); axes[0].grid(True, axis="x")

    top10 = imp_df["feature"].head(10).tolist()
    x     = range(len(top10))
    builtin = pd.Series(gb_perm.feature_importances_, index=X_tr.columns)
    axes[1].bar([i-.2 for i in x], builtin[top10].values, .35,
                label="built-in", color=C["accent"], alpha=.8)
    axes[1].bar([i+.2 for i in x],
                imp_df.set_index("feature").loc[top10,"importance"].values, .35,
                label="permutation", color=C["teal"], alpha=.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([t[:12] for t in top10], rotation=35, ha="right", fontsize=8)
    axes[1].set_title("built-in vs permutation (top 10)")
    axes[1].legend(); axes[1].grid(True, axis="y")
    plt.tight_layout()
    _save("A06_permutation_importance", fig)

    #this is price elasticity
    print("\n[8] price elasticity")
    # Uses actual_price vs base_price to compute elasticity — this is analysis,
    # not a model feature, so using actual_price here is fine
    if "base_price" in df.columns:
        df["price_change_pct"]  = ((df["actual_price"] - df["base_price"])
                                    / df["base_price"] * 100)
    else:
        df["price_change_pct"]  = ((df["actual_price"] - df["actual_price"].mean())
                                    / df["actual_price"].mean() * 100)
    df["demand_change_pct"] = ((df["occupancy_rate"] - df["occupancy_rate"].mean())
                                / df["occupancy_rate"].mean() * 100)
    df["elasticity"] = (df["demand_change_pct"]
                        / df["price_change_pct"].replace(0, np.nan)).clip(-5, 5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("price elasticity of demand", fontsize=13, color=C["white"])
    sample = df.sample(min(3000, len(df)), random_state=1)
    sc = axes[0,0].scatter(sample["price_change_pct"], sample["demand_change_pct"],
                            c=sample["occupancy_rate"], cmap="RdYlGn", alpha=.4, s=12)
    m, b = np.polyfit(sample["price_change_pct"].dropna(),
                      sample["demand_change_pct"].dropna(), 1)
    xf = np.linspace(sample["price_change_pct"].min(),
                     sample["price_change_pct"].max(), 100)
    axes[0,0].plot(xf, m*xf+b, color=C["rose"], lw=2.5, label=f"slope={m:.3f}")
    plt.colorbar(sc, ax=axes[0,0], label="occupancy")
    axes[0,0].set_title("price change vs demand change")
    axes[0,0].legend(); axes[0,0].grid(True)

    if "room_type" in df.columns:
        elas_room = df.groupby("room_type")["elasticity"].median().sort_values()
        axes[0,1].barh(elas_room.index, elas_room.values,
                       color=[C["rose"] if v < -1 else C["gold"] if v < 0
                               else C["teal"] for v in elas_room.values])
        axes[0,1].axvline(-1, color=C["rose"], ls="--", lw=1.5, label="elastic")
        axes[0,1].set_title("elasticity by room type")
        axes[0,1].legend(); axes[0,1].grid(True, axis="x")

    elas_month = df.groupby("month")["elasticity"].median()
    axes[1,0].bar(range(1,13), elas_month.values,
                  color=[C["rose"] if v < -0.5 else C["gold"] if v < 0
                          else C["teal"] for v in elas_month.values])
    axes[1,0].set_xticks(range(1,13))
    axes[1,0].set_xticklabels(["j","f","m","a","m","j","j","a","s","o","n","d"])
    axes[1,0].set_title("elasticity by month"); axes[1,0].grid(True, axis="y")

    axes[1,1].hist(df["elasticity"].dropna(), bins=80, color=C["accent"], alpha=.8)
    axes[1,1].axvline(-1, color=C["rose"], lw=2, ls="--", label="elastic")
    axes[1,1].axvline(df["elasticity"].median(), color=C["teal"], lw=2,
                      label=f"median={df['elasticity'].median():.2f}")
    axes[1,1].set_title("elasticity distribution")
    axes[1,1].legend(); axes[1,1].grid(True)
    plt.tight_layout()
    _save("A07_price_elasticity", fig)

    # ── 9. anomaly detection ──────────────────────────────────────────────
    print("\n[9] anomaly detection")
    iso_cols = [c for c in ["actual_price","occupancy_rate","revenue_per_night","demand_score"]
                if c in df.columns]
    iso_X = StandardScaler().fit_transform(df[iso_cols].fillna(0))
    iso   = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"]       = iso.fit_predict(iso_X)
    df["anomaly_score"] = iso.score_samples(iso_X)
    n_anom = int((df["anomaly"] == -1).sum())
    print(f"  {n_anom} anomalies ({n_anom/len(df)*100:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("anomaly detection — isolation forest", fontsize=13, color=C["white"])
    normal = df[df["anomaly"] ==  1]
    anom   = df[df["anomaly"] == -1]
    axes[0].scatter(normal["actual_price"], normal["occupancy_rate"],
                    c=C["accent"], alpha=.2, s=6, label=f"normal ({len(normal):,})")
    axes[0].scatter(anom["actual_price"],   anom["occupancy_rate"],
                    c=C["rose"], alpha=.8, s=25, marker="x",
                    label=f"anomaly ({len(anom):,})")
    axes[0].set_title("price vs occupancy"); axes[0].legend(); axes[0].grid(True)
    axes[1].hist(df[df["anomaly"]==1]["anomaly_score"],  bins=50, color=C["accent"],
                 alpha=.7, label="normal", density=True)
    axes[1].hist(df[df["anomaly"]==-1]["anomaly_score"], bins=20, color=C["rose"],
                 alpha=.9, label="anomaly", density=True)
    axes[1].set_title("anomaly score distribution"); axes[1].legend(); axes[1].grid(True)
    if "room_type" in df.columns:
        anom_rt = df[df["anomaly"]==-1]["room_type"].value_counts()
        axes[2].bar(anom_rt.index, anom_rt.values, color=C["rose"], edgecolor="white")
        axes[2].set_title("anomalies by room type"); axes[2].grid(True, axis="y")
    plt.tight_layout()
    _save("A08_anomaly_detection", fig)

    # ── 10. advanced model comparison ────────────────────────────────────
    print("\n[10] advanced model comparison")
    models_final = {
        "linear regression": LinearRegression(),
        "ridge":             Ridge(alpha=1.0),
        "decision tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "random forest":     RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42),
        "gradient boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                        learning_rate=.08, random_state=42),
        "xgboost (tuned)":   best_xgb,
        "stacking ensemble": stack,
    }
    results_all = []
    for name, mdl in models_final.items():
        try:
            if name not in ["xgboost (tuned)", "stacking ensemble"]:
                mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
            mae   = mean_absolute_error(y_te, preds)
            rmse  = float(np.sqrt(mean_squared_error(y_te, preds)))
            r2    = r2_score(y_te, preds)
            mp    = np.mean(np.abs((np.array(y_te)-preds)
                            / np.maximum(np.abs(np.array(y_te)),1))) * 100
            results_all.append({"Model":name,"MAE":round(mae,3),"RMSE":round(rmse,3),
                                 "R2":round(r2,4),"MAPE":round(mp,3)})
            print(f"  {name:<25}  r²={r2:.4f}  mae={mae:.2f}")
        except Exception as e:
            print(f"  {name} skipped: {e}")

    results_df = pd.DataFrame(results_all).sort_values("R2", ascending=False)
    results_df.to_csv(os.path.join(MODELS, "advanced_model_results.csv"), index=False)

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("advanced model comparison", fontsize=13, color=C["white"])
    gs2 = gridspec.GridSpec(1, 2, figure=fig, wspace=.35)
    ax  = fig.add_subplot(gs2[0])
    ax.barh(results_df["Model"], results_df["R2"],
            color=[PAL6[i % len(PAL6)] for i in range(len(results_df))], edgecolor="white")
    ax.set_title("r² score"); ax.invert_yaxis(); ax.grid(True, axis="x")
    ax = fig.add_subplot(gs2[1])
    x2 = range(len(results_df))
    ax.bar([i-.28 for i in x2], results_df["MAE"],  .28, label="mae",   color=C["accent"], alpha=.85)
    ax.bar([i     for i in x2], results_df["RMSE"], .28, label="rmse",  color=C["gold"],   alpha=.85)
    ax.bar([i+.28 for i in x2], results_df["MAPE"], .28, label="mape%", color=C["teal"],   alpha=.85)
    ax.set_xticks(list(x2))
    ax.set_xticklabels(results_df["Model"], rotation=30, ha="right", fontsize=8)
    ax.set_title("error metrics (lower = better)"); ax.legend(); ax.grid(True, axis="y")
    plt.tight_layout()
    _save("A09_advanced_model_comparison", fig)

    # ── 11. revenue scenarios ─────────────────────────────────────────────
    print("\n[11] revenue scenarios")
    avg_price = df["actual_price"].mean()
    avg_occ   = df["occupancy_rate"].mean()
    occ_rates   = np.linspace(.40, 1.00, 50)
    price_range = np.linspace(50, 400, 50)
    OCC, PRICE  = np.meshgrid(occ_rates, price_range)
    REVENUE     = OCC * PRICE

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor(C["surface"])
    fig.suptitle("revenue optimisation scenarios", fontsize=13, color=C["white"])
    contour = axes[0].contourf(OCC, PRICE, REVENUE, levels=25, cmap="RdYlGn")
    plt.colorbar(contour, ax=axes[0], label="revenue/night ($)")
    axes[0].scatter([avg_occ],[avg_price], c="white", s=200, marker="*",
                    zorder=5, label="current avg")
    axes[0].set_title("revenue surface"); axes[0].legend()

    scenarios = {
        "conservative\n(-10%)": avg_price * 0.90,
        "current":              avg_price,
        "optimised\n(+10%)":   avg_price * 1.10,
        "peak\n(+25%)":        avg_price * 1.25,
        "event\n(+40%)":       avg_price * 1.40,
    }
    revs = {k: v * avg_occ for k, v in scenarios.items()}
    bars = axes[1].bar(scenarios.keys(), revs.values(),
                       color=[PAL6[i] for i in range(len(scenarios))], edgecolor="white")
    axes[1].set_title(f"scenario revpar (occ={avg_occ:.0%})"); axes[1].grid(True, axis="y")
    for bar, v in zip(bars, revs.values()):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+.5, f"${v:.0f}",
                     ha="center", va="bottom", fontsize=9, color=C["text"])

    price_levels = np.linspace(50, 450, 100)
    for occ_level, color in [(0.50, C["rose"]), (0.70, C["gold"]),
                              (0.85, C["accent"]), (0.95, C["teal"])]:
        axes[2].plot(price_levels, price_levels*occ_level, color=color, lw=2,
                     label=f"occ={occ_level:.0%}")
    axes[2].set_title("revpar by price x occupancy"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    _save("A10_revenue_scenarios", fig)

    # ── save summary ──────────────────────────────────────────────────────
    summary = {
        "target":           TARGET,
        "stacking_r2":      round(stack_r2, 4),
        "stacking_mae":     round(stack_mae, 2),
        "n_anomalies":      n_anom,
        "pca_components_90": n90,
        "best_xgb_cv_r2":   round(float(rs.best_score_), 4),
        "elasticity_median": round(float(df["elasticity"].median()), 3),
        "model_results":    results_df.to_dict("records"),
    }
    with open(os.path.join(MODELS, "advanced_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*55)
    print("advanced analysis complete")
    print(f"  figures  -> {ADVFIG}")
    print(f"  summary  -> {os.path.join(MODELS,'advanced_summary.json')}")
    print("="*55)


if __name__ == "__main__":
    run_advanced_analysis()