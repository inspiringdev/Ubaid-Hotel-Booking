
import numpy as np
import pandas as pd
import pickle, os, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model   import LinearRegression, Ridge, Lasso
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm            import SVR
from sklearn.neighbors      import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics        import (mean_absolute_error,
                                    mean_squared_error,
                                    r2_score)
from sklearn.preprocessing  import StandardScaler
from xgboost                import XGBRegressor

import os as _os
_BASE     = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
CLEAN_PATH = _os.path.join(_BASE, "data", "hotel_booking_clean.csv")
MODEL_DIR  = _os.path.join(_BASE, "outputs", "models")
FIG_DIR    = _os.path.join(_BASE, "outputs", "figures")
_os.makedirs(MODEL_DIR, exist_ok=True)
_os.makedirs(FIG_DIR,   exist_ok=True)

ACCENT  = "#e94560"
DARK    = "#1a1a2e"
BG      = "#f8f9fa"
PALETTE = ["#1a1a2e","#16213e","#0f3460","#e94560","#533483",
           "#2ec4b6","#ff9f1c","#cbf3f0","#4ecdc4","#95e1d3","#f38181"]

# ── Feature columns ────────────────────────────────────────────────────────
# Rules:
#   actual_price  = TARGET — excluded from features
#   base_price    = excluded (was derived from actual_price in old pipeline)
#   price_premium = excluded (actual/base — uses target)
#   price_vs_comp = excluded (actual vs comp — uses target)
#   revpar        = excluded (actual * occupancy — uses target)
#   revenue_per_night = excluded (actual * length_of_stay — uses target)
#   is_test       = excluded (split marker)
# ──────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # temporal
    "month", "day_of_week", "week_of_year", "is_weekend", "is_holiday",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # booking behaviour
    "lead_time_days", "length_of_stay",
    # competitor pricing (safe — base_price vs competitors)
    "competitor_avg_price", "competitor_min_price", "competitor_max_price",
    "base_vs_comp_avg", "base_vs_comp_min", "price_comp_spread",
    # demand signals
    "occupancy_rate", "demand_score", "high_demand",
    # event & external
    "event_magnitude", "has_event", "weather_score",
    # guest signals
    "reviews_score", "repeat_guest",
    # encoded categoricals
    "hotel_name_enc", "room_type_enc", "channel_enc",
    "guest_type_enc", "event_type_enc", "season_enc", "lead_time_bucket_enc",
]

TARGET_COL = "actual_price"   # real Kaggle ADR — what the guest actually paid


# ── Helpers ────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100


def evaluate(name, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    return {"Model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "MAPE%": round(mp, 3), "R²": round(r2, 4)}


# ── Chronological data loading ─────────────────────────────────────────────
def load_data():
    """
    Chronological split using the is_test flag created in data_augmentation.py.
    This ensures the model is trained on past bookings and evaluated on
    future bookings — the correct setup for time-series pricing data.

    If is_test flag is missing (e.g. running on old data), falls back to
    sorting by checkin_date and splitting at 80th percentile.
    """
    df       = pd.read_csv(CLEAN_PATH)
    existing = [c for c in FEATURE_COLS if c in df.columns]
    X        = df[existing].fillna(0)
    y        = df[TARGET_COL]

    if "is_test" in df.columns:
        print("   using chronological split (is_test flag)")
        train_mask = df["is_test"] == 0
        test_mask  = df["is_test"] == 1
        return (X[train_mask].reset_index(drop=True),
                X[test_mask].reset_index(drop=True),
                y[train_mask].reset_index(drop=True),
                y[test_mask].reset_index(drop=True))
    else:
        # Fallback: sort by checkin_date and take last 20% as test
        print("   is_test flag not found — falling back to date-sorted split")
        if "checkin_date" in df.columns:
            df_sorted  = df.sort_values("checkin_date").reset_index(drop=True)
            X          = df_sorted[[c for c in existing if c in df_sorted.columns]].fillna(0)
            y          = df_sorted[TARGET_COL]
        split = int(len(X) * 0.80)
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ── Model training ─────────────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 65)
    print("TRAINING 10 REGRESSION MODELS")
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target: {TARGET_COL}  (real ADR from Kaggle)")
    print("=" * 65)

    scaler   = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    models = {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Lasso Regression":   Lasso(alpha=0.5, max_iter=5000),
        "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":      RandomForestRegressor(n_estimators=200,
                                                     max_depth=None,
                                                     min_samples_leaf=2,
                                                     n_jobs=-1, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200,
                                                          learning_rate=0.08,
                                                          max_depth=5,
                                                          random_state=42),
        "XGBoost":            XGBRegressor(n_estimators=250,
                                            learning_rate=0.07,
                                            max_depth=6,
                                            subsample=0.85,
                                            colsample_bytree=0.80,
                                            random_state=42,
                                            verbosity=0),
        "SVR":                SVR(kernel="rbf", C=100, epsilon=5),
        "KNN":                KNeighborsRegressor(n_neighbors=7,
                                                   weights="distance",
                                                   n_jobs=-1),
        "MLP Neural Network": MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                                            activation="relu",
                                            max_iter=500,
                                            learning_rate_init=0.001,
                                            random_state=42,
                                            early_stopping=True),
    }

    scaled_models = {"SVR", "KNN", "MLP Neural Network",
                     "Linear Regression", "Ridge Regression", "Lasso Regression"}

    results = []
    trained = {}

    for name, mdl in models.items():
        t0   = time.time()
        Xtr  = Xs_train if name in scaled_models else X_train
        Xte  = Xs_test  if name in scaled_models else X_test

        mdl.fit(Xtr, y_train)
        preds   = mdl.predict(Xte)
        elapsed = time.time() - t0

        m        = evaluate(name, y_test, preds)
        m["Train_Time_s"] = round(elapsed, 2)

        cv_xtr = Xs_train if name in scaled_models else X_train
        cv = cross_val_score(mdl, cv_xtr, y_train, cv=KFold(5, shuffle=True, random_state=1), scoring="r2", n_jobs=-1)
        m["CV_R²_mean"] = round(cv.mean(), 4)
        m["CV_R²_std"]  = round(cv.std(),  4)

        results.append(m)
        trained[name] = (mdl, preds)

        print(f"   {name:<25}  MAE={m['MAE']:>7.2f}  "
              f"RMSE={m['RMSE']:>7.2f}  "
              f"R²={m['R²']:.4f}  "
              f"MAPE={m['MAPE%']:.2f}%  "
              f"({m['Train_Time_s']}s)")

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    print(f"\n   Best model : {results_df.iloc[0]['Model']}")
    print(f"   Best R²    : {results_df.iloc[0]['R²']:.4f}  "
          f"(realistic for real hotel pricing data)")
    return results_df, trained, scaler


# ── Q-Learning RL Agent ────────────────────────────────────────────────────

class HotelPricingEnv:
    """
    Gym-style environment for hotel dynamic pricing.
    State  : (demand_bin, occupancy_bin, competitor_bin, event_flag)
    Action : price_multiplier applied to base_price
    Reward : Revenue = price × occupancy (penalised if far from competitor avg)
    """
    ACTIONS = np.array([0.80, 0.85, 0.90, 0.95, 1.00,
                        1.05, 1.10, 1.15, 1.20, 1.25, 1.30])

    def __init__(self, df: pd.DataFrame):
        self.df    = df.reset_index(drop=True)
        self.n     = len(df)
        self.idx   = 0
        self.dem_bins = pd.qcut(df["demand_score"],        q=4, labels=False, duplicates="drop")
        self.occ_bins = pd.qcut(df["occupancy_rate"],      q=4, labels=False, duplicates="drop")
        self.cmp_bins = pd.qcut(df["competitor_avg_price"],q=4, labels=False, duplicates="drop")

    def _state(self, idx):
        d = int(self.dem_bins.iloc[idx] or 0)
        o = int(self.occ_bins.iloc[idx] or 0)
        c = int(self.cmp_bins.iloc[idx] or 0)
        e = int(self.df["has_event"].iloc[idx])
        return (d, o, c, e)

    def reset(self):
        self.idx = 0
        return self._state(0)

    def step(self, action_idx: int):
        row        = self.df.iloc[self.idx]
        multiplier = self.ACTIONS[action_idx]
        price      = row["base_price"] * multiplier
        comp_avg   = row["competitor_avg_price"]
        occupancy  = row["occupancy_rate"]

        # Revenue-based reward: maximise revenue but penalise being too far
        # above competitor average (risk of losing bookings)
        revenue      = price * occupancy
        comp_penalty = max(0, (price - comp_avg * 1.20)) * occupancy * 0.5
        reward       = revenue - comp_penalty

        self.idx = (self.idx + 1) % self.n
        done     = (self.idx == 0)
        return self._state(self.idx), reward, done


class QLearningAgent:
    def __init__(self, n_actions: int, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.Q             = {}

    def _qvals(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._qvals(state)))

    def learn(self, s, a, r, s_next, done):
        q   = self._qvals(s)
        qn  = self._qvals(s_next)
        tgt = r + (0 if done else self.gamma * qn.max())
        q[a] += self.alpha * (tgt - q[a])
        if done:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)


def train_rl_agent(df: pd.DataFrame, episodes=200):
    """
    Train Q-learning agent for 200 episodes (increased from 40 for convergence).
    Uses a fixed sample for reproducibility.
    """
    print("\n" + "=" * 65)
    print("REINFORCEMENT LEARNING — Q-LEARNING AGENT")
    print(f"Episodes: {episodes}  (increased from 40 for convergence)")
    print("=" * 65)

    required = ["base_price","competitor_avg_price","occupancy_rate",
                "demand_score","has_event"]
    available = [c for c in required if c in df.columns]
    if len(available) < len(required):
        missing = set(required) - set(available)
        print(f"   warning: missing columns {missing} — skipping RL agent")
        return None, [], []

    # Fixed random_state ensures same sample across runs (reproducible)
    sample = df.sample(min(3000, len(df)), random_state=7).reset_index(drop=True)
    env    = HotelPricingEnv(sample)
    agent  = QLearningAgent(n_actions=len(HotelPricingEnv.ACTIONS))

    ep_rewards = []
    ep_mape_   = []

    for ep in range(episodes):
        state   = env.reset()
        total_r = 0.0
        prices  = []
        refs    = []   # competitor avg as pricing benchmark

        for step in range(min(2000, len(env.df))):
            a               = agent.choose_action(state)
            s_next, r, done = env.step(a)
            agent.learn(state, a, r, s_next, done)
            state           = s_next
            total_r        += r

            row = env.df.iloc[env.idx % len(env.df)]
            prices.append(row["base_price"] * HotelPricingEnv.ACTIONS[a])
            refs.append(row["competitor_avg_price"])
            if done:
                break

        ep_rewards.append(total_r)
        ep_mape_.append(mape(refs, prices))

        if (ep + 1) % 20 == 0:
            print(f"   Episode {ep+1:>3}  "
                  f"Total Reward={total_r:>12,.0f}  "
                  f"vs-comp MAPE={ep_mape_[-1]:.2f}%  "
                  f"ε={agent.epsilon:.3f}")

    print(f"\n   Final Q-table  : {len(agent.Q):,} state-action pairs")
    print(f"   Final MAPE     : {ep_mape_[-1]:.2f}%")
    print(f"   Best MAPE      : {min(ep_mape_):.2f}%")
    return agent, ep_rewards, ep_mape_


# ── Visualisations ─────────────────────────────────────────────────────────

def plot_model_comparison(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Model Performance Comparison", fontsize=16,
                 fontweight="bold", color=DARK)
    metrics = [("R²", True), ("RMSE", False), ("MAE", False), ("MAPE%", False)]
    for ax, (metric, higher) in zip(axes.flat, metrics):
        df_s   = results_df.sort_values(metric, ascending=not higher)
        colors = [ACCENT if i == 0 else "#0f3460" for i in range(len(df_s))]
        ax.barh(df_s["Model"], df_s[metric], color=colors, edgecolor="white")
        ax.set_title(metric)
        ax.grid(True, axis="x")
        ax.invert_yaxis()
        for i, (_, row) in enumerate(df_s.iterrows()):
            ax.text(row[metric] + ax.get_xlim()[1] * 0.005, i,
                    f"{row[metric]:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    p = f"{FIG_DIR}/13_model_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   saved -> {p}")


def plot_predictions(y_test, preds_dict):
    best_name = max(preds_dict.items(),
                    key=lambda x: r2_score(y_test, x[1][1]))[0]
    preds     = preds_dict[best_name][1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Best Model Predictions — {best_name}", fontsize=14,
                 fontweight="bold", color=DARK)

    ax = axes[0]
    ax.scatter(y_test, preds, alpha=0.25, s=10, color=ACCENT)
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=2, label="Perfect Fit")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs. Predicted")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    residuals = np.array(y_test) - preds
    ax.scatter(preds, residuals, alpha=0.2, s=10, color="#0f3460")
    ax.axhline(0, color=ACCENT, lw=2, ls="--")
    ax.set_xlabel("Predicted ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residual Plot"); ax.grid(True)

    ax = axes[2]
    ax.hist(residuals, bins=60, color=ACCENT, edgecolor="white", alpha=0.8)
    ax.axvline(residuals.mean(), color=DARK, lw=2, ls="--",
               label=f"Mean={residuals.mean():.2f}")
    ax.set_xlabel("Residual ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    p = f"{FIG_DIR}/14_best_model_predictions.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   saved -> {p}")


def plot_feature_importance(trained, X_train):
    rf_model = trained.get("Random Forest")
    if rf_model is None:
        return
    mdl, _ = rf_model
    imps   = pd.Series(mdl.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)[:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    colors = [ACCENT if i < 5 else "#0f3460" for i in range(len(imps))]
    ax.barh(imps.index[::-1], imps.values[::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_title("Random Forest — Top 20 Feature Importances",
                 fontsize=13, fontweight="bold", color=DARK)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x")
    plt.tight_layout()
    p = f"{FIG_DIR}/15_feature_importance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   saved -> {p}")


def plot_rl_training(ep_rewards, ep_mape_):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Q-Learning Agent Training Curves (200 episodes)",
                 fontsize=14, fontweight="bold", color=DARK)

    ax = axes[0]
    ax.plot(ep_rewards, color=ACCENT, lw=1.5)
    # Rolling mean to show trend
    if len(ep_rewards) >= 10:
        rolling = pd.Series(ep_rewards).rolling(10).mean()
        ax.plot(rolling, color=DARK, lw=2.5, label="10-ep rolling mean")
        ax.legend()
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.grid(True)

    ax = axes[1]
    ax.plot(ep_mape_, color="#0f3460", lw=1.5)
    if len(ep_mape_) >= 10:
        rolling = pd.Series(ep_mape_).rolling(10).mean()
        ax.plot(rolling, color=ACCENT, lw=2.5, label="10-ep rolling mean")
        ax.legend()
    ax.set_title("Price vs Competitor MAPE (%) — Lower is Better")
    ax.set_xlabel("Episode"); ax.set_ylabel("MAPE (%)")
    ax.grid(True)

    plt.tight_layout()
    p = f"{FIG_DIR}/16_rl_training.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   saved -> {p}")


def plot_cross_val(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    df_s = results_df.sort_values("CV_R²_mean", ascending=False)
    x    = range(len(df_s))
    ax.bar(x, df_s["CV_R²_mean"], color=ACCENT, alpha=0.8,
           edgecolor="white", label="CV Mean R²")
    ax.errorbar(x, df_s["CV_R²_mean"], yerr=df_s["CV_R²_std"],
                fmt="none", color=DARK, capsize=5, lw=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_s["Model"], rotation=30, ha="right", fontsize=9)
    ax.set_title("5-Fold Cross-Validation R² (Mean ± Std)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("CV R²")
    ax.legend(); ax.grid(True, axis="y")
    plt.tight_layout()
    p = f"{FIG_DIR}/17_cross_validation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   saved -> {p}")


def price_recommendation_sample(model, feature_cols, df_sample):
    """
    Show price recommendations for a sample — compare ML suggestion
    vs actual price paid vs competitor average.
    """
    print("\n" + "=" * 65)
    print("PRICE RECOMMENDATION ENGINE — SAMPLE OUTPUT")
    print("=" * 65)
    sample   = df_sample.sample(min(10, len(df_sample)), random_state=5)
    existing = [c for c in feature_cols if c in sample.columns]
    X_samp   = sample[existing].fillna(0)
    rec      = model.predict(X_samp)

    print(f"\n  {'Room':<6} {'CompAvg $':>10} {'Actual $':>10} "
          f"{'ML Rec $':>10} {'Diff %':>8}")
    print("  " + "-" * 50)
    for i, (_, row) in enumerate(sample.iterrows()):
        act  = row.get("actual_price", 0)
        diff = (rec[i] - act) / max(act, 1) * 100
        print(f"  {str(row.get('room_type','?')):<6} "
              f"${row.get('competitor_avg_price', 0):>9.0f} "
              f"${act:>9.0f} "
              f"${rec[i]:>9.0f} "
              f"{diff:>+7.1f}%")

    uplift = (rec.sum() - sample["actual_price"].values.sum())
    print(f"\n  Potential revenue uplift on sample: ${uplift:+,.2f}")


# ── Main pipeline ──────────────────────────────────────────────────────────
def run_modeling():
    print("=" * 65)
    print("MACHINE LEARNING — HOTEL DYNAMIC PRICING")
    print(f"Target variable : {TARGET_COL}  (real Kaggle ADR)")
    print("=" * 65)

    df = pd.read_csv(CLEAN_PATH)
    X_train, X_test, y_train, y_test = load_data()
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"   Features: {X_train.shape[1]}")

    results_df, trained, scaler = train_all_models(X_train, X_test, y_train, y_test)

    results_df.to_csv(f"{MODEL_DIR}/model_results.csv", index=False)
    print(f"\n   Results saved -> {MODEL_DIR}/model_results.csv")

    best_mdl = trained["Random Forest"][0]
    with open(f"{MODEL_DIR}/best_model_rf.pkl", "wb") as f:
        pickle.dump(best_mdl, f)
    with open(f"{MODEL_DIR}/scaler_model.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 65)
    print("GENERATING MODEL VISUALISATIONS")
    print("=" * 65)
    plot_model_comparison(results_df)
    plot_predictions(y_test, trained)
    plot_feature_importance(trained, X_train)
    plot_cross_val(results_df)

    agent, ep_rewards, ep_mape_ = train_rl_agent(df, episodes=200)
    if agent is not None:
        plot_rl_training(ep_rewards, ep_mape_)
        with open(f"{MODEL_DIR}/rl_agent.pkl", "wb") as f:
            pickle.dump(agent, f)

    price_recommendation_sample(best_mdl, FEATURE_COLS, df)

    print("\n" + "=" * 65)
    print("FINAL LEADERBOARD (Test Set — Chronological Split)")
    print("=" * 65)
    print(results_df[["Model","MAE","RMSE","MAPE%","R²","CV_R²_mean"]].to_string(index=False))

    return results_df, trained, agent


if __name__ == "__main__":
    run_modeling()