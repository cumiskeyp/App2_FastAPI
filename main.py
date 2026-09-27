"""
Phase 6 – App 2 (FastAPI): Full Analytics Dashboard
S&P 500 Tech-5 Open-Gap Direction Predictor
=====================================================
Run locally:  uvicorn main:app --reload --port 8002
Deploy:       Render (see render.yaml)
"""

import io, base64, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
CONTINUOUS_COLS = [
    "daily_return","rsi_14","adx_14","macd","macd_hist","obv_trend",
    "spy_daily_return","vix_close","rsi_delta_3d","adx_delta_3d",
    "atr_14","bb_width","bb_pctb","realized_vol_20d",
    "stoch_k","stoch_d","roc_10","mom_10","cci_14",
    "vol_rel_20d","ad_line","adosc",
    "hl_range_pct","dist_from_20d_high","dist_from_20d_low",
    "gap_pct_lag1","vix_change",
]
CAT_COLS   = ["vix_bin","rsi_bin","adx_bin"]
TARGET_COL = "open_dir"
TICKERS    = ["MSFT","GOOGL","NVDA","AAPL","AMZN"]

SAMPLE_SENTIMENT = {
    "MSFT":  {"Positive":40,"Neutral":52,"Negative": 8,"signal":"Bullish ↑","score":+40},
    "NVDA":  {"Positive":40,"Neutral":44,"Negative":16,"signal":"Bullish ↑","score":+24},
    "AAPL":  {"Positive":34,"Neutral":55,"Negative":11,"signal":"Bullish ↑","score":+22},
    "AMZN":  {"Positive":22,"Neutral":39,"Negative":39,"signal":"Bearish ↓","score":-17},
    "GOOGL": {"Positive":20,"Neutral":50,"Negative":30,"signal":"Bearish ↓","score":-30},
}

app = FastAPI(title="Phase 6 – App 2: Analytics Dashboard")
templates = Jinja2Templates(directory="templates")
templates.env.filters["enumerate"] = enumerate
templates.env.filters["abs"] = abs


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_models():
    models = {
        "Naive Bayes":       GaussianNB(),
        "Decision Tree":     DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=6,
                                                    random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                         learning_rate=0.05,
                                                         max_depth=4, random_state=42),
        "SVM":               SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
    }
    if XGBOOST_OK:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    return models


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def prepare_features(df, mode):
    y = df[TARGET_COL].astype(int)
    if mode == "groomed":
        if "adosc_t1" not in df.columns:
            df = df.copy(); df["adosc_t1"] = df["adosc"].shift(1)
        X = df[["adosc_t1"]].dropna(); y = y.loc[X.index]
    else:
        available = [c for c in CONTINUOUS_COLS if c in df.columns]
        if "adosc_t1" not in df.columns and "adosc" in df.columns:
            df = df.copy(); df["adosc_t1"] = df["adosc"].shift(1)
        if "adosc_t1" in df.columns: available.append("adosc_t1")
        parts = [df[available]]
        cats = [c for c in CAT_COLS if c in df.columns]
        if cats: parts.append(pd.get_dummies(df[cats], drop_first=True))
        X = pd.concat(parts, axis=1).dropna(); y = y.loc[X.index]
    return X, y, list(X.columns)


def run_model(model, X, y, train_frac):
    n = int(len(X) * train_frac)
    X_tr, X_te = X.iloc[:n], X.iloc[n:]
    y_tr, y_te = y.iloc[:n], y.iloc[n:]
    sc = StandardScaler()
    model.fit(sc.fit_transform(X_tr), y_tr)
    p_tr = model.predict_proba(sc.transform(X_tr))[:,1]
    p_te = model.predict_proba(sc.transform(X_te))[:,1]
    pred = (p_te >= 0.5).astype(int)
    return {
        "auc_tr": float(roc_auc_score(y_tr, p_tr)),
        "auc_te": float(roc_auc_score(y_te, p_te)),
        "fpr_tr": roc_curve(y_tr, p_tr)[0], "tpr_tr": roc_curve(y_tr, p_tr)[1],
        "fpr_te": roc_curve(y_te, p_te)[0], "tpr_te": roc_curve(y_te, p_te)[1],
        "cm": confusion_matrix(y_te, pred),
        "report": classification_report(y_te, pred, output_dict=True),
        "n_train": len(X_tr), "n_test": len(X_te),
        "p_te": p_te, "pred": pred, "y_te": y_te.values,
    }


# ── Chart generators ──────────────────────────────────────────────────────────
def chart_price(df):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    dc = next((c for c in df.columns if "date" in c.lower()), None)
    if dc and "Close" in df.columns:
        ax.plot(pd.to_datetime(df[dc], errors="coerce"), df["Close"], lw=1, color="#4c8bf5")
        ax.set_xlabel("Date")
    elif "Close" in df.columns:
        ax.plot(df["Close"].values, lw=1, color="#4c8bf5")
    ax.set_title("Close Price History"); ax.set_ylabel("Price ($)"); ax.grid(alpha=0.3)
    fig.tight_layout(); return fig_to_b64(fig)


def chart_returns(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    if "daily_return" in df.columns:
        r = df["daily_return"].dropna()
        axes[0].hist(r, bins=60, color="#4c8bf5", alpha=0.85, edgecolor="white")
        axes[0].axvline(0, color="red", lw=1, ls="--")
        axes[0].set_title("Daily Return Distribution"); axes[0].set_xlabel("Daily Return")
        axes[0].grid(alpha=0.3)
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts().sort_index()
        axes[1].bar(["Down (0)","Up (1)"], vc.values,
                    color=["#d62728","#21c55d"], alpha=0.85)
        axes[1].set_title("Target Class Balance"); axes[1].grid(alpha=0.3, axis="y")
        for i,v in enumerate(vc.values):
            axes[1].text(i, v+1, str(v), ha="center", fontsize=10)
    fig.tight_layout(); return fig_to_b64(fig)


def chart_rsi_adx(df):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    if "rsi_14" in df.columns:
        axes[0].plot(df["rsi_14"].values, lw=1, color="#9467bd")
        axes[0].axhline(70, color="red", ls="--", lw=0.8, alpha=0.7)
        axes[0].axhline(30, color="green", ls="--", lw=0.8, alpha=0.7)
        axes[0].set_ylabel("RSI-14"); axes[0].set_ylim(0,100); axes[0].grid(alpha=0.3)
        axes[0].set_title("RSI & ADX Indicators")
    if "adx_14" in df.columns:
        axes[1].plot(df["adx_14"].values, lw=1, color="#8c564b")
        axes[1].axhline(25, color="darkorange", ls="--", lw=0.8, alpha=0.7)
        axes[1].set_ylabel("ADX-14"); axes[1].set_xlabel("Row index"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); return fig_to_b64(fig)


def chart_vol(df):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if "realized_vol_20d" in df.columns:
        ax.plot(df["realized_vol_20d"].values, lw=1, color="darkorange", label="20d Realised Vol")
    elif "daily_return" in df.columns:
        ax.plot(df["daily_return"].rolling(20).std().values, lw=1, color="darkorange",
                label="Rolling 20d Std")
    ax.set_title("Realised Volatility"); ax.set_xlabel("Row index")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); return fig_to_b64(fig)


def chart_corr(df):
    avail = [c for c in CONTINUOUS_COLS if c in df.columns][:15]
    fig, ax = plt.subplots(figsize=(9, 7))
    if avail:
        corr = df[avail].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.4,
                    cmap="RdBu_r", center=0, ax=ax, annot_kws={"size":7}, square=True)
    ax.set_title("Feature Correlation Matrix"); fig.tight_layout(); return fig_to_b64(fig)


def chart_sent_bar(sd):
    tickers = list(sd.keys())
    pos = [sd[t]["Positive"] for t in tickers]
    neu = [sd[t]["Neutral"]  for t in tickers]
    neg = [sd[t]["Negative"] for t in tickers]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(tickers)); h = 0.55
    ax.barh(y, pos, h, label="Positive", color="#21c55d", alpha=0.85)
    ax.barh(y, neu, h, left=pos, label="Neutral", color="#aec7e8", alpha=0.85)
    ax.barh(y, neg, h, left=[p+n for p,n in zip(pos,neu)], label="Negative",
            color="#d62728", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(tickers, fontsize=12)
    ax.set_xlabel("% of Headlines"); ax.set_title("FinBERT Sentiment Distribution")
    ax.legend(loc="lower right"); ax.set_xlim(0,100); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); return fig_to_b64(fig)


def chart_sent_net(sd):
    tickers = list(sd.keys()); scores = [sd[t]["score"] for t in tickers]
    colors = ["#21c55d" if s>=0 else "#d62728" for s in scores]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(tickers, scores, color=colors, alpha=0.85, width=0.55)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Net Sentiment (Pos% − Neg%)"); ax.set_title("Net Sentiment Signal")
    for bar,s in zip(bars,scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{s:+d}%", ha="center", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); return fig_to_b64(fig)


def chart_roc(res, name):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(res["fpr_tr"], res["tpr_tr"],
            label=f"Train AUC={res['auc_tr']:.4f}", ls="--", alpha=0.7)
    ax.plot(res["fpr_te"], res["tpr_te"],
            label=f"Test  AUC={res['auc_te']:.4f}", lw=2, color="orange")
    ax.plot([0,1],[0,1],"k:",alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {name}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); return fig_to_b64(fig)


def chart_cm(cm, name):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=["Down","Up"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"CM — {name}"); fig.tight_layout(); return fig_to_b64(fig)


def chart_prob_hist(p_te):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(p_te, bins=40, color="#4c8bf5", alpha=0.85, edgecolor="white")
    ax.axvline(0.5, color="red", ls="--", lw=1.2, label="Threshold (0.50)")
    ax.set_xlabel("Predicted P(Up)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Probabilities")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); return fig_to_b64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": list(build_models().keys()),
        "tickers": TICKERS,
    })


@app.post("/analyse", response_class=HTMLResponse)
async def analyse(
    request: Request,
    file: UploadFile = File(...),
    model_choice: str = Form(...),
    feature_mode: str = Form(...),
    train_frac: float = Form(...),
    ticker_filter: list[str] = Form(default=TICKERS),
    model_choice_pred: str = Form(default="Naive Bayes"),
):
    models_list = list(build_models().keys())
    error = None
    eda = sent = pred = None

    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
        rows_raw = len(df_raw)

        # Normalise ticker column
        if "Ticker" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Ticker": "ticker"})

        df = df_raw.copy()
        if "ticker" in df.columns and ticker_filter:
            df = df[df["ticker"].isin(ticker_filter)].copy()

        rows_filtered = len(df)

        if TARGET_COL not in df.columns:
            raise ValueError(f"Column '{TARGET_COL}' not found.")

        # ── EDA charts ────────────────────────────────────────────────────────
        num_cols = [c for c in CONTINUOUS_COLS if c in df.columns]
        desc_rows = []
        if num_cols:
            desc = df[num_cols[:8]].describe().T.round(4)
            for col in desc.index:
                r = desc.loc[col]
                desc_rows.append({
                    "col": col,
                    "count": f"{r['count']:.0f}",
                    "mean":  f"{r['mean']:.4f}",
                    "std":   f"{r['std']:.4f}",
                    "min":   f"{r['min']:.4f}",
                    "max":   f"{r['max']:.4f}",
                })

        vc = df[TARGET_COL].value_counts(normalize=True).mul(100).round(1)
        eda = {
            "rows_raw": f"{rows_raw:,}",
            "rows_filtered": f"{rows_filtered:,}",
            "cols": len(df_raw.columns),
            "price_img":   chart_price(df),
            "returns_img": chart_returns(df),
            "rsi_img":     chart_rsi_adx(df),
            "vol_img":     chart_vol(df),
            "corr_img":    chart_corr(df),
            "up_pct":      f"{vc.get(1,0):.1f}",
            "dn_pct":      f"{vc.get(0,0):.1f}",
            "desc_rows":   desc_rows,
        }

        # ── Sentiment ─────────────────────────────────────────────────────────
        sent = {
            "data": SAMPLE_SENTIMENT,
            "bar_img": chart_sent_bar(SAMPLE_SENTIMENT),
            "net_img": chart_sent_net(SAMPLE_SENTIMENT),
            "summary": pd.DataFrame.from_dict(SAMPLE_SENTIMENT, orient="index")
                         .drop(columns=["signal"])
                         .rename(columns={"score":"Net Score"})
                         .to_dict(orient="records"),
            "summary_keys": ["Ticker","Positive","Neutral","Negative","Net Score"],
        }

        # ── Predictions ───────────────────────────────────────────────────────
        X, y, feat_names = prepare_features(df, feature_mode)
        model = build_models()[model_choice]
        res = run_model(model, X, y, train_frac)

        delta = res["auc_te"] - res["auc_tr"]
        rep = res["report"]
        rep_rows = []
        for k, label in [("0","Down (0)"),("1","Up (1)"),
                          ("macro avg","Macro Avg"),("weighted avg","Weighted Avg")]:
            if k in rep:
                r = rep[k]
                rep_rows.append({
                    "label":label,
                    "precision": f"{r.get('precision',0):.3f}",
                    "recall":    f"{r.get('recall',0):.3f}",
                    "f1":        f"{r.get('f1-score',0):.3f}",
                    "support":   f"{r.get('support',0):.0f}",
                })

        pred_table = []
        for i in range(min(50, len(res["y_te"]))):
            pred_table.append({
                "actual": int(res["y_te"][i]),
                "prob_up": float(round(res["p_te"][i], 4)),
                "predicted": int(res["pred"][i]),
                "correct": int(res["pred"][i] == res["y_te"][i]),
            })

        pred = {
            "model": model_choice,
            "auc_tr": f"{res['auc_tr']:.4f}",
            "auc_te": f"{res['auc_te']:.4f}",
            "delta":  f"{delta:+.4f}",
            "delta_class": "positive" if delta >= 0 else "negative",
            "n_train": f"{res['n_train']:,}",
            "n_test":  f"{res['n_test']:,}",
            "auc_level": "high" if res["auc_te"] > 0.55 else ("ok" if res["auc_te"] > 0.50 else "low"),
            "roc_img":  chart_roc(res, model_choice),
            "cm_img":   chart_cm(res["cm"], model_choice),
            "prob_img": chart_prob_hist(res["p_te"]),
            "rep_rows": rep_rows,
            "pred_table": pred_table,
            "feat_names": feat_names,
        }

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models_list,
        "tickers": TICKERS,
        "selected_model": model_choice,
        "selected_feature": feature_mode,
        "selected_frac": train_frac,
        "selected_tickers": ticker_filter,
        "eda": eda,
        "sent": sent,
        "pred": pred,
        "error": error,
    })
