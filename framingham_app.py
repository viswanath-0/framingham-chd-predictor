# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Framingham Heart Study — CHD Risk Predictor                               ║
# ║  End-to-End Classification Pipeline · Dark Theme · Professional Portfolio  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, f_classif,
                                       mutual_info_classif, RFE)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHD Risk Predictor",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;-webkit-font-smoothing:antialiased;}
.stApp{background:#080D1A !important;}
.main .block-container{padding:1.5rem 2rem 4rem;max-width:1380px;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"]{background:#0A0F20 !important;border-right:1px solid rgba(255,255,255,0.06) !important;}
[data-testid="stSidebar"] .stMarkdown p{color:#64748B !important;}
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong{color:#E6EDF3 !important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:5px;gap:2px;margin-bottom:1.75rem;box-shadow:0 4px 20px rgba(0,0,0,0.5);}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:9px 18px;font-size:0.85rem;font-weight:500;color:#64748B !important;background:transparent;border:none !important;}
.stTabs [aria-selected="true"]{background:#EF4444 !important;color:#FFFFFF !important;box-shadow:0 2px 10px rgba(239,68,68,0.5),0 4px 0 rgba(120,20,20,0.8) !important;transform:translateY(-1px);}

/* Metric cards */
[data-testid="metric-container"]{background:#0F1728 !important;border:1px solid rgba(255,255,255,0.08) !important;border-radius:14px !important;padding:1.25rem 1.5rem !important;box-shadow:0 2px 8px rgba(0,0,0,0.4),0 12px 28px rgba(0,0,0,0.3),0 1px 0 rgba(255,255,255,0.05) inset !important;transition:transform 0.2s,box-shadow 0.2s !important;}
[data-testid="metric-container"]:hover{transform:translateY(-3px) !important;box-shadow:0 6px 20px rgba(239,68,68,0.15),0 20px 40px rgba(0,0,0,0.4) !important;}
[data-testid="stMetricValue"]{font-size:1.7rem !important;font-weight:800 !important;color:#E6EDF3 !important;letter-spacing:-0.04em !important;}
[data-testid="stMetricLabel"]{font-size:0.68rem !important;font-weight:600 !important;color:#EF4444 !important;text-transform:uppercase !important;letter-spacing:0.1em !important;}
[data-testid="stMetricDelta"] svg{display:none;}
[data-testid="stMetricDelta"]{color:#34D399 !important;}

/* Cards */
.card{background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:1.5rem;box-shadow:0 2px 8px rgba(0,0,0,0.4),0 12px 28px rgba(0,0,0,0.3),0 1px 0 rgba(255,255,255,0.05) inset;transition:transform 0.2s,box-shadow 0.2s;height:100%;}
.card:hover{transform:translateY(-4px);border-color:rgba(239,68,68,0.3);box-shadow:0 6px 24px rgba(239,68,68,0.1),0 24px 48px rgba(0,0,0,0.4);}
.card-sm{background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem 1.25rem;box-shadow:0 2px 8px rgba(0,0,0,0.35),0 6px 16px rgba(0,0,0,0.25);transition:transform 0.15s,box-shadow 0.15s;height:100%;}
.card-sm:hover{transform:translateY(-2px);border-color:rgba(239,68,68,0.2);box-shadow:0 4px 16px rgba(239,68,68,0.08),0 12px 28px rgba(0,0,0,0.35);}

/* Typography */
.eyebrow{font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#EF4444;margin-bottom:4px;}
.section-head{font-size:1.5rem;font-weight:700;color:#E6EDF3 !important;letter-spacing:-0.03em;margin-bottom:0.25rem;}
.section-sub{font-size:0.9rem;color:#64748B !important;line-height:1.6;margin-bottom:1.5rem;}
h1,h2,h3,h4,h5,h6{color:#E6EDF3 !important;}
.stMarkdown p{color:#94A3B8;}
label{color:#94A3B8 !important;}

/* Badges */
.badge{display:inline-block;padding:2px 10px;border-radius:100px;font-size:0.7rem;font-weight:600;}
.b-red   {background:rgba(239,68,68,0.15); color:#F87171; border:1px solid rgba(239,68,68,0.3);}
.b-green {background:rgba(52,211,153,0.12);color:#34D399; border:1px solid rgba(52,211,153,0.25);}
.b-amber {background:rgba(251,191,36,0.12);color:#FBBF24; border:1px solid rgba(251,191,36,0.25);}
.b-blue  {background:rgba(79,148,255,0.15);color:#4F94FF; border:1px solid rgba(79,148,255,0.3);}
.b-slate {background:rgba(148,163,184,0.08);color:#94A3B8;border:1px solid rgba(148,163,184,0.2);}

/* Hero */
.app-hero{background:linear-gradient(140deg,#2D0A0A 0%,#1A0505 60%,#2D0A0A 100%);border:1px solid rgba(239,68,68,0.2);border-radius:20px;padding:2.5rem;margin-bottom:1.75rem;position:relative;overflow:hidden;transform:perspective(1000px) rotateX(1deg);box-shadow:0 8px 32px rgba(0,0,0,0.6),0 24px 56px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.07);}
.app-hero::after{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 80% 50%,rgba(239,68,68,0.12) 0%,transparent 60%);}
.hero-tag{font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:rgba(239,68,68,0.7) !important;margin-bottom:0.75rem;position:relative;z-index:1;}
.hero-title{font-size:2rem;font-weight:800;letter-spacing:-0.04em;line-height:1.15;color:#E6EDF3 !important;position:relative;z-index:1;margin-bottom:0.5rem;}
.hero-sub{font-size:0.9rem;color:#64748B !important;line-height:1.6;position:relative;z-index:1;}

/* Risk prediction card */
.risk-card{border-radius:24px;padding:2.75rem 2rem;text-align:center;position:relative;overflow:hidden;transform:perspective(1200px) rotateX(2deg);transform-origin:center bottom;}
.risk-low   {background:linear-gradient(145deg,#0A2D1A 0%,#080D1A 55%,#0A2D1A 100%);border:1px solid rgba(52,211,153,0.25);box-shadow:0 8px 24px rgba(0,0,0,0.6),0 24px 56px rgba(0,0,0,0.5),inset 0 1px 0 rgba(52,211,153,0.1);}
.risk-medium{background:linear-gradient(145deg,#2D1A0A 0%,#080D1A 55%,#2D1A0A 100%);border:1px solid rgba(251,191,36,0.25);box-shadow:0 8px 24px rgba(0,0,0,0.6),0 24px 56px rgba(0,0,0,0.5),inset 0 1px 0 rgba(251,191,36,0.1);}
.risk-high  {background:linear-gradient(145deg,#2D0A0A 0%,#080D1A 55%,#2D0A0A 100%);border:1px solid rgba(239,68,68,0.25);box-shadow:0 8px 24px rgba(0,0,0,0.6),0 24px 56px rgba(0,0,0,0.5),inset 0 1px 0 rgba(239,68,68,0.1);}
.risk-label{font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:rgba(255,255,255,0.4) !important;margin-bottom:10px;position:relative;z-index:1;}
.risk-pct  {font-size:4rem;font-weight:800;letter-spacing:-0.06em;color:#FFFFFF !important;line-height:1;position:relative;z-index:1;}
.risk-level{font-size:1rem;font-weight:600;margin-top:10px;position:relative;z-index:1;}

/* Model rows */
.mrow{display:flex;align-items:center;justify-content:space-between;background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.875rem 1.25rem;margin-bottom:6px;box-shadow:0 2px 8px rgba(0,0,0,0.35);transition:transform 0.15s,box-shadow 0.15s,border-color 0.15s;}
.mrow:hover{transform:translateX(5px);border-color:rgba(239,68,68,0.3);box-shadow:0 4px 16px rgba(239,68,68,0.08),-4px 0 0 #EF4444;}
.mrow-best{background:rgba(239,68,68,0.07);border-color:rgba(239,68,68,0.3) !important;box-shadow:0 4px 16px rgba(239,68,68,0.12),-4px 0 0 #EF4444,0 1px 0 rgba(255,255,255,0.05) inset !important;}
.mname{font-weight:600;font-size:0.88rem;color:#E6EDF3 !important;}
.mtype{font-size:0.72rem;color:#64748B !important;margin-top:2px;}
.mval{font-size:0.85rem;font-weight:600;color:#E6EDF3 !important;text-align:right;}
.mval-sub{font-size:0.72rem;color:#64748B !important;text-align:right;}

/* Info boxes */
.note-box{background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2);border-left:4px solid #0EA5E9;border-radius:8px;padding:0.875rem 1.1rem;}
.note-box p{font-size:0.82rem;color:#7DD3FC !important;margin:0;line-height:1.65;}
.warn-box{background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);border-left:4px solid #FBBF24;border-radius:8px;padding:0.875rem 1.1rem;}
.warn-box p{font-size:0.82rem;color:#FDE68A !important;margin:0;line-height:1.65;}

/* Pipeline steps */
.pipeline-step{background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.1rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.4),0 8px 20px rgba(0,0,0,0.3);transition:transform 0.2s,box-shadow 0.2s;}
.pipeline-step:hover{transform:translateY(-4px) scale(1.02);box-shadow:0 10px 28px rgba(239,68,68,0.15),0 24px 48px rgba(0,0,0,0.45);}

/* Why-chosen cards */
.why-card{background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.25rem;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.35);transition:transform 0.2s,box-shadow 0.2s,border-color 0.2s;}
.why-card:hover{transform:translateY(-3px);border-color:rgba(239,68,68,0.2);box-shadow:0 8px 24px rgba(239,68,68,0.1),0 20px 40px rgba(0,0,0,0.4);}
.why-icon{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;font-weight:800;margin-bottom:10px;}
.why-title{font-size:0.88rem;font-weight:700;color:#E6EDF3 !important;margin-bottom:6px;}
.why-desc{font-size:0.78rem;color:#64748B !important;line-height:1.6;}

/* Input section */
.input-group{background:#0F1728;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.25rem;margin-bottom:0.75rem;box-shadow:0 2px 8px rgba(0,0,0,0.3);}
.input-group-title{font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#EF4444;margin-bottom:0.75rem;}

/* Button */
.stButton>button{background:#EF4444 !important;color:white !important;border:none !important;border-radius:10px !important;font-weight:700 !important;font-size:0.9rem !important;padding:0.75rem 1.5rem !important;width:100% !important;box-shadow:0 4px 0 #7f1d1d,0 4px 16px rgba(239,68,68,0.4) !important;transform:translateY(0) !important;transition:transform 0.1s,box-shadow 0.1s !important;}
.stButton>button:hover{background:#F87171 !important;transform:translateY(-2px) !important;box-shadow:0 6px 0 #7f1d1d,0 10px 24px rgba(239,68,68,0.5) !important;}
.stButton>button:active{transform:translateY(3px) !important;box-shadow:0 1px 0 #7f1d1d,0 2px 8px rgba(239,68,68,0.3) !important;}

/* Inputs */
.stSelectbox>div>div{background:#0F1728 !important;border:1.5px solid rgba(255,255,255,0.1) !important;border-radius:9px !important;}
.stNumberInput>div>div>input{background:#0F1728 !important;border:1.5px solid rgba(255,255,255,0.1) !important;border-radius:9px !important;color:#E6EDF3 !important;}
.stSlider [data-baseweb="slider"]{margin:0.5rem 0;}
div[data-baseweb="popover"]{background:#0F1728 !important;border:1px solid rgba(255,255,255,0.1) !important;}
div[data-baseweb="menu"]{background:#0F1728 !important;}
div[data-baseweb="menu"] li{color:#E6EDF3 !important;}
div[data-baseweb="menu"] li:hover{background:#162040 !important;}

/* Misc */
.divider{height:1px;background:rgba(255,255,255,0.07);margin:1.5rem 0;}
hr{border-color:rgba(255,255,255,0.07) !important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#080D1A;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.12);border-radius:99px;}
[data-testid="stPlotlyChart"]{border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.4);}
[data-testid="stPlotlyChart"]>div{background:#0F1728 !important;border-radius:14px;}
[data-testid="stDataFrame"]{background:#0F1728 !important;border-radius:10px !important;box-shadow:0 2px 12px rgba(0,0,0,0.4) !important;}
</style>
""", unsafe_allow_html=True)

# ── COLORS ─────────────────────────────────────────────────────────────────────
BG_DEEP  = "#080D1A"
BG_CARD  = "#0F1728"
T_PRI    = "#E6EDF3"
T_SEC    = "#94A3B8"
A_RED    = "#EF4444"
A_GREEN  = "#34D399"
A_AMBER  = "#FBBF24"
A_BLUE   = "#4F94FF"
A_PURPLE = "#A78BFA"
COLORS   = [A_RED, A_BLUE, A_GREEN, A_AMBER, A_PURPLE, "#38BDF8", "#FB923C"]

PLOTLY_LAYOUT = dict(
    font_family      = "Inter",
    font_color       = T_PRI,
    paper_bgcolor    = BG_CARD,
    plot_bgcolor     = BG_DEEP,
    margin           = dict(l=20, r=20, t=45, b=20),
    title_font_size  = 14,
    title_font_color = T_PRI,
    colorway         = COLORS,
    xaxis = dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.08)",
                 tickfont=dict(size=11, color="#8B9EB7"), title_font=dict(color="#8B9EB7"),
                 zerolinecolor="rgba(255,255,255,0.04)", color="#8B9EB7"),
    yaxis = dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.08)",
                 tickfont=dict(size=11, color="#8B9EB7"), title_font=dict(color="#8B9EB7"),
                 zerolinecolor="rgba(255,255,255,0.04)", color="#8B9EB7"),
    legend = dict(font=dict(color=T_PRI, size=12), bgcolor="rgba(0,0,0,0)",
                  bordercolor="rgba(255,255,255,0.08)"),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(font=dict(color=T_PRI, family="Inter", size=12),
                      title_font=dict(color=T_PRI, size=14, family="Inter"),
                      legend=dict(font=dict(color=T_PRI, size=12, family="Inter"),
                                  bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(tickfont=dict(color="#8B9EB7", size=11), title_font=dict(color="#8B9EB7"),
                     gridcolor="rgba(255,255,255,0.06)", color="#8B9EB7")
    fig.update_yaxes(tickfont=dict(color="#8B9EB7", size=11), title_font=dict(color="#8B9EB7"),
                     gridcolor="rgba(255,255,255,0.06)", color="#8B9EB7")
    return fig

# ── DATA & PIPELINE ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path_or_bytes):
    if hasattr(path_or_bytes, 'read'):
        return pd.read_csv(path_or_bytes)
    return pd.read_csv(path_or_bytes)

@st.cache_resource
def build_pipeline(_df):
    df = _df.copy()
    ALL_FEATURES = ['male','age','education','currentSmoker','cigsPerDay','BPMeds',
                    'prevalentStroke','prevalentHyp','diabetes','totChol',
                    'sysBP','diaBP','BMI','heartRate','glucose']
    TARGET = 'TenYearCHD'

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y)

    # Impute
    median_cols = ['glucose','totChol','BMI','heartRate','cigsPerDay']
    mode_cols   = ['BPMeds','education']

    imp_med  = SimpleImputer(strategy='median')
    imp_mode = SimpleImputer(strategy='most_frequent')

    X_train_imp = X_train.copy()
    X_test_imp  = X_test.copy()
    X_train_imp[median_cols] = imp_med.fit_transform(X_train[median_cols])
    X_test_imp[median_cols]  = imp_med.transform(X_test[median_cols])
    X_train_imp[mode_cols]   = imp_mode.fit_transform(X_train[mode_cols])
    X_test_imp[mode_cols]    = imp_mode.transform(X_test[mode_cols])

    # Feature selection
    vt = VarianceThreshold(threshold=0.1)
    vt.fit(X_train_imp)
    vt_kept = list(vt.get_feature_names_out())

    kb = SelectKBest(f_classif, k=10)
    kb.fit(X_train_imp, y_train)
    kb_feats = list(kb.get_feature_names_out())
    kb_scores = dict(zip(X_train_imp.columns, kb.scores_))

    mi_scores_arr = mutual_info_classif(X_train_imp, y_train, random_state=42)
    mi_scores     = dict(zip(X_train_imp.columns, mi_scores_arr))
    mi_feats = [f for f,_ in sorted(mi_scores.items(), key=lambda x:-x[1])[:10]]

    rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=10)
    rfe.fit(X_train_imp, y_train)
    rfe_feats = X_train_imp.columns[rfe.support_].tolist()

    dt_fi = DecisionTreeClassifier(random_state=42)
    dt_fi.fit(X_train_imp, y_train)
    fi_scores  = dict(zip(X_train_imp.columns, dt_fi.feature_importances_))
    fi_feats   = [f for f,_ in sorted(fi_scores.items(), key=lambda x:-x[1])[:10]]

    selected_features = rfe_feats

    X_train_sel = X_train_imp[selected_features]
    X_test_sel  = X_test_imp[selected_features]

    # Models (hardcoded best params from tuning)
    models_spec = {
        'DT Baseline'    : DecisionTreeClassifier(random_state=42),
        'DT GridSearch'  : DecisionTreeClassifier(
            criterion='entropy', max_depth=5,
            min_samples_leaf=4, min_samples_split=10, random_state=42),
        'DT RandomSearch': DecisionTreeClassifier(
            criterion='gini', max_depth=7, max_features='sqrt',
            min_samples_leaf=4, min_samples_split=10, random_state=42),
        'DT Bayesian'    : DecisionTreeClassifier(
            criterion='entropy', max_depth=6,
            min_samples_split=12, min_samples_leaf=5,
            max_features='sqrt', random_state=42),
    }

    results   = {}
    fitted    = {}

    for name, mdl in models_spec.items():
        mdl.fit(X_train_sel, y_train)
        yp   = mdl.predict(X_test_sel)
        yprob= mdl.predict_proba(X_test_sel)[:, 1]
        results[name] = {
            'Accuracy' : accuracy_score(y_test, yp),
            'Precision': precision_score(y_test, yp, zero_division=0),
            'Recall'   : recall_score(y_test, yp, zero_division=0),
            'F1'       : f1_score(y_test, yp, zero_division=0),
            'ROC-AUC'  : roc_auc_score(y_test, yprob),
        }
        fitted[name] = mdl

    best_key = max(results, key=lambda k: results[k]['ROC-AUC'])
    best_mdl = fitted[best_key]

    # CV on best model
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    cv_sc = cross_val_score(
        DecisionTreeClassifier(criterion='entropy', max_depth=6,
                               min_samples_split=12, min_samples_leaf=5,
                               max_features='sqrt', random_state=42),
        X_train_sel, y_train, cv=skf, scoring='roc_auc')

    # Confusion matrix
    y_pred_best = best_mdl.predict(X_test_sel)
    cm = confusion_matrix(y_test, y_pred_best)

    # ROC curves
    roc_data = {}
    for name, mdl in fitted.items():
        yprob = mdl.predict_proba(X_test_sel)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    return dict(
        fitted=fitted, results=results, best_key=best_key, best_mdl=best_mdl,
        selected_features=selected_features,
        X_train_imp=X_train_imp, X_test_imp=X_test_imp,
        X_train_sel=X_train_sel, X_test_sel=X_test_sel,
        y_train=y_train, y_test=y_test,
        cv_sc=cv_sc, cm=cm, roc_data=roc_data,
        imp_med=imp_med, imp_mode=imp_mode,
        median_cols=median_cols, mode_cols=mode_cols,
        kb_scores=kb_scores, mi_scores=mi_scores, fi_scores=fi_scores,
        vt_kept=vt_kept, kb_feats=kb_feats, mi_feats=mi_feats,
        rfe_feats=rfe_feats, fi_feats=fi_feats,
    )

def predict_risk(inputs, pipe):
    sel  = pipe['selected_features']
    mdl  = pipe['best_mdl']
    med_c= pipe['median_cols']
    mod_c= pipe['mode_cols']

    inp_df = pd.DataFrame([inputs])
    # No NaN expected from form but impute for safety
    train_imp = pipe['X_train_imp']
    for col in med_c:
        if col in inp_df.columns and pd.isna(inp_df[col].iloc[0]):
            inp_df[col] = train_imp[col].median()
    X_in = inp_df[sel]
    prob = mdl.predict_proba(X_in)[0][1]
    return prob

# ── CHART HELPERS ──────────────────────────────────────────────────────────────
def fig_target_dist(df):
    vc = df['TenYearCHD'].value_counts().reset_index()
    vc.columns = ['CHD', 'Count']
    vc['Label'] = vc['CHD'].map({0: 'No CHD (85%)', 1: 'CHD Risk (15%)'})
    fig = go.Figure(go.Pie(
        labels=vc['Label'], values=vc['Count'], hole=0.6,
        marker=dict(colors=[A_BLUE, A_RED], line=dict(color=BG_DEEP, width=2)),
        textfont=dict(color='#FFFFFF', size=13),
        textinfo='percent',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='Target Distribution', height=250,
                      showlegend=True)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    fig.update_layout(legend=dict(font=dict(color=T_PRI, size=12), bgcolor='rgba(0,0,0,0)'))
    fig.update_layout(font=dict(color=T_PRI, family='Inter'))
    return fig

def fig_age_dist(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df['TenYearCHD']==0]['age'], name='No CHD',
                                marker_color=A_BLUE, opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=df[df['TenYearCHD']==1]['age'], name='CHD Risk',
                                marker_color=A_RED, opacity=0.7, nbinsx=20))
    fig.update_layout(**PLOTLY_LAYOUT, title='Age Distribution by CHD Status',
                      xaxis_title='Age', yaxis_title='Count', barmode='overlay')
    return apply_theme(fig)

def fig_feature_dist(df, col, title):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[df['TenYearCHD']==0][col], name='No CHD',
                          marker_color=A_BLUE, boxmean=True))
    fig.add_trace(go.Box(y=df[df['TenYearCHD']==1][col], name='CHD Risk',
                          marker_color=A_RED, boxmean=True))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, yaxis_title=col)
    return apply_theme(fig)

def fig_corr(df):
    nums = ['age','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']
    c = df[nums].corr().round(2)
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.columns,
                                colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                                text=c.values, texttemplate='%{text}', textfont_size=10))
    fig.update_layout(**PLOTLY_LAYOUT, title='Feature Correlation Heatmap', height=380)
    return apply_theme(fig)

def fig_fs_scores(kb_scores, mi_scores, fi_scores):
    feats = list(kb_scores.keys())
    kb_max = max(kb_scores.values()); mi_max = max(mi_scores.values()); fi_max = max(fi_scores.values())
    fig = go.Figure()
    fig.add_trace(go.Bar(name='SelectKBest (F)', x=feats,
                          y=[kb_scores[f]/kb_max for f in feats], marker_color=A_BLUE))
    fig.add_trace(go.Bar(name='Mutual Info', x=feats,
                          y=[mi_scores[f]/mi_max for f in feats], marker_color=A_GREEN))
    fig.add_trace(go.Bar(name='DT Importance', x=feats,
                          y=[fi_scores[f]/fi_max for f in feats], marker_color=A_AMBER))
    fig.update_layout(**PLOTLY_LAYOUT, title='Feature Selection Scores (normalised)',
                      barmode='group', xaxis_tickangle=-45)
    return apply_theme(fig)

def fig_roc_curves(roc_data, best_key):
    fig = go.Figure()
    clrs = [A_RED, A_BLUE, A_GREEN, A_AMBER]
    for (name,(fpr,tpr,rauc)), clr in zip(roc_data.items(), clrs):
        dash = 'solid' if name == best_key else 'dash'
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={rauc:.3f})',
                                  line=dict(color=clr, width=2.5 if name==best_key else 1.5, dash=dash)))
    fig.add_shape(type='line', x0=0,y0=0,x1=1,y1=1, line=dict(color='rgba(255,255,255,0.2)', dash='dot'))
    fig.update_layout(**PLOTLY_LAYOUT, title='ROC Curves — All Models',
                      xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return apply_theme(fig)

def fig_confusion(cm):
    labels = ['No CHD', 'CHD Risk']
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels,
                                colorscale=[[0,'#0F1728'],[1,'#EF4444']],
                                text=cm, texttemplate='<b>%{text}</b>',
                                textfont=dict(size=22, color='white')))
    fig.update_layout(**PLOTLY_LAYOUT, title='Confusion Matrix — Best Model',
                      xaxis_title='Predicted', yaxis_title='Actual', height=320)
    return apply_theme(fig)

def fig_model_roc_bar(results):
    df2 = pd.DataFrame([{'Model':k,'ROC-AUC':v['ROC-AUC']} for k,v in results.items()])
    df2 = df2.sort_values('ROC-AUC', ascending=True)
    colors = [A_RED if 'Bayesian' in m else A_PURPLE if 'Grid' in m
              else A_AMBER if 'Random' in m else '#64748B' for m in df2['Model']]
    fig = go.Figure(go.Bar(x=df2['ROC-AUC'], y=df2['Model'], orientation='h',
                            marker_color=colors, text=df2['ROC-AUC'].round(3),
                            textposition='outside', textfont=dict(size=11, color=T_PRI)))
    fig.update_layout(**PLOTLY_LAYOUT, title='ROC-AUC by Model', xaxis_range=[0,1.05])
    return apply_theme(fig)

def fig_cv(cv_sc):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f'Fold {i+1}' for i in range(len(cv_sc))], y=cv_sc,
                          marker_color=A_RED, text=cv_sc.round(3),
                          textposition='outside', textfont=dict(size=11, color=T_PRI)))
    fig.add_hline(y=cv_sc.mean(), line_dash='dash', line_color=A_GREEN,
                  annotation_text=f'Mean={cv_sc.mean():.3f}', annotation_font_color=A_GREEN)
    fig.update_layout(**PLOTLY_LAYOUT, title='5-Fold Stratified CV — ROC-AUC', yaxis_range=[0,1.05])
    return apply_theme(fig)

def fig_fi_best(selected_features, best_mdl):
    fi = dict(zip(selected_features, best_mdl.feature_importances_))
    df2= pd.DataFrame(fi.items(),columns=['Feature','Importance']).sort_values('Importance')
    fig= px.bar(df2, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale=[[0,'#1a0505'],[1,'#EF4444']],
                title='Feature Importances — Best Model')
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    return apply_theme(fig)

def risk_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(prob*100, 1),
        title={'text':'10-Year CHD Risk', 'font':{'size':13,'family':'Inter','color':T_PRI}},
        number={'suffix':'%','font':{'size':32,'family':'Inter','color':T_PRI}},
        gauge={
            'axis':{'range':[0,100],'tickfont':{'size':10,'color':'#8B9EB7'}},
            'bar':{'color':A_RED if prob>0.2 else A_AMBER if prob>0.1 else A_GREEN},
            'steps':[
                {'range':[0,10],  'color':'rgba(52,211,153,0.12)'},
                {'range':[10,20], 'color':'rgba(251,191,36,0.12)'},
                {'range':[20,100],'color':'rgba(239,68,68,0.12)'},
            ],
            'threshold':{'line':{'color':'white','width':3},'value':round(prob*100,1)}
        }
    ))
    fig.update_layout(height=230, margin=dict(l=20,r=20,t=50,b=10),
                      paper_bgcolor=BG_CARD, font_family='Inter', font_color=T_PRI)
    return fig

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### CHD Risk Predictor")
        st.divider()
        uploaded = st.file_uploader("Upload dataset (framingham.csv)", type=['csv'])
        st.divider()
        st.markdown("""
**About this app**
Framingham Heart Study — predicts 10-year coronary heart disease risk using a Decision Tree classifier.

**Dataset:** 4,240 patients · 15 features

**Best model:** Decision Tree (Bayesian-tuned)

**Primary metric:** ROC-AUC (handles class imbalance)
        """)

    # Load data — try multiple paths then fall back to sidebar upload
    import os
    df = None
    if uploaded:
        df = load_data(uploaded)
    else:
        for path in ['framingham.csv',
                     'data/framingham.csv',
                     os.path.join(os.path.dirname(__file__), 'framingham.csv')]:
            try:
                df = load_data(path)
                break
            except Exception:
                continue

    if df is None:
        st.markdown("""
        <div class="warn-box">
          <p><strong>Dataset not found.</strong><br><br>
          Place <code>framingham.csv</code> in the same folder as <code>app.py</code>
          and restart the app — OR upload it using the sidebar file uploader.</p>
        </div>""", unsafe_allow_html=True)
        with st.sidebar:
            st.markdown("### Quick Upload")
            direct_upload = st.file_uploader("framingham.csv", type=['csv'], key='direct')
            if direct_upload:
                df = load_data(direct_upload)
        if df is None:
            st.stop()

    # Train models
    with st.spinner("Training models — runs once, then cached..."):
        pipe = build_pipeline(df)

    results  = pipe['results']
    best_key = pipe['best_key']

    # TABS
    t1, t2, t3, t4 = st.tabs([
        "Overview",
        "EDA & Statistics",
        "Model Performance",
        "CHD Risk Predictor",
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    with t1:
        st.markdown("""
        <div class="app-hero">
          <div class="hero-tag">Framingham Heart Study — Portfolio Project</div>
          <div class="hero-title">10-Year Coronary Heart<br>Disease Risk Prediction</div>
          <div class="hero-sub">
            Classification pipeline on 4,240 patient records. Decision Tree classifier tuned
            via GridSearch, RandomSearch, and Bayesian Optuna — evaluated by ROC-AUC to handle 15% class imbalance.
          </div>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Patients",    f"{len(df):,}")
        c2.metric("Features",          "15")
        c3.metric("CHD Positive Rate", f"{df['TenYearCHD'].mean()*100:.1f}%")
        c4.metric("Best ROC-AUC",      f"{results[best_key]['ROC-AUC']:.3f}")
        c5.metric("Best Recall",       f"{results[best_key]['Recall']:.3f}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("<div class='section-head'>Target Distribution</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Strong class imbalance — 85% negative, 15% CHD positive. ROC-AUC chosen as primary metric because accuracy would be misleadingly high.</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_target_dist(df), use_container_width=True, theme=None)
            st.markdown("""
            <div class="warn-box">
              <p>Class imbalance alert: A naive model predicting "No CHD" always scores 85% accuracy — making accuracy a poor metric. ROC-AUC measures true discrimination ability regardless of class ratio.</p>
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-head'>Feature Overview</div>", unsafe_allow_html=True)
            feat_info = [
                ("male","Binary","Gender (1=Male, 0=Female)"),
                ("age","Continuous","Age in years (32–70)"),
                ("education","Ordinal","Education level (1–4)"),
                ("currentSmoker","Binary","Current smoker"),
                ("cigsPerDay","Continuous","Cigarettes per day"),
                ("BPMeds","Binary","On BP medication"),
                ("prevalentStroke","Binary","History of stroke"),
                ("prevalentHyp","Binary","Hypertension history"),
                ("diabetes","Binary","Diabetic"),
                ("totChol","Continuous","Total cholesterol (mg/dL)"),
                ("sysBP","Continuous","Systolic BP (mmHg)"),
                ("diaBP","Continuous","Diastolic BP (mmHg)"),
                ("BMI","Continuous","Body mass index"),
                ("heartRate","Continuous","Heart rate (bpm)"),
                ("glucose","Continuous","Blood glucose (mg/dL)"),
            ]
            for feat, ftype, desc in feat_info:
                badge = 'b-blue' if ftype=='Binary' else 'b-amber' if ftype=='Ordinal' else 'b-slate'
                st.markdown(f"""<div style='display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
                  <span style='width:130px;font-size:0.82rem;font-weight:600;color:{T_PRI}'>{feat}</span>
                  <span class='badge {badge}' style='min-width:70px;text-align:center'>{ftype}</span>
                  <span style='font-size:0.78rem;color:#64748B'>{desc}</span>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — EDA
    # ══════════════════════════════════════════════════════════════════
    with t2:
        st.markdown("<div class='section-head'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Distribution of features split by CHD outcome. Key risk factors are older age, higher systolic BP, and elevated glucose.</div>", unsafe_allow_html=True)

        r1a, r1b = st.columns(2)
        with r1a: st.plotly_chart(fig_age_dist(df), use_container_width=True, theme=None)
        with r1b: st.plotly_chart(fig_feature_dist(df,'sysBP','Systolic BP by CHD Status'), use_container_width=True, theme=None)

        r2a, r2b = st.columns(2)
        with r2a: st.plotly_chart(fig_feature_dist(df,'glucose','Glucose by CHD Status'), use_container_width=True, theme=None)
        with r2b: st.plotly_chart(fig_feature_dist(df,'BMI','BMI by CHD Status'), use_container_width=True, theme=None)

        r3a, r3b = st.columns(2)
        with r3a: st.plotly_chart(fig_feature_dist(df,'totChol','Total Cholesterol by CHD Status'), use_container_width=True, theme=None)
        with r3b: st.plotly_chart(fig_corr(df), use_container_width=True, theme=None)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-head'>Missing Value Summary</div>", unsafe_allow_html=True)

        null_df = df.isnull().sum().reset_index()
        null_df.columns = ['Feature','Missing Count']
        null_df['Missing %'] = (null_df['Missing Count']/len(df)*100).round(1)
        null_df = null_df[null_df['Missing Count']>0].sort_values('Missing Count', ascending=False)

        na, nb = st.columns(2)
        with na:
            st.dataframe(null_df, use_container_width=True)
        with nb:
            fig_null = px.bar(null_df, x='Feature', y='Missing %',
                              color='Missing %', color_continuous_scale=[[0,'#1a0505'],[1,'#EF4444']],
                              title='Missing % per Feature')
            fig_null.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(apply_theme(fig_null), use_container_width=True, theme=None)

        st.markdown("""
        <div class="note-box">
          <p>Imputation strategy: Median for right-skewed continuous features (glucose, totChol, BMI, heartRate, cigsPerDay). Mode for categorical/binary features (BPMeds, education). Fit on training set only to prevent data leakage.</p>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════
    with t3:
        st.markdown("<div class='section-head'>ML Pipeline & Model Performance</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Decision Tree classifier tuned with 3 hyperparameter strategies. Primary metric: ROC-AUC (handles 15% class imbalance).</div>", unsafe_allow_html=True)

        # Pipeline steps
        steps = [
            ("1","Imputation","Median for skewed cols, Mode for binary","#EF4444"),
            ("2","VarianceThreshold","Remove near-zero variance features","#F59E0B"),
            ("3","Feature Selection","SelectKBest, MI, RFE, DT Importances","#A78BFA"),
            ("4","Apply RFE Features","10 features selected by RFE wrapper","#4F94FF"),
            ("5","Train & Tune","Baseline, Grid, Random, Bayesian DT","#34D399"),
        ]
        cols = st.columns(5)
        for col, (num,title,sub,clr) in zip(cols, steps):
            col.markdown(f"""
            <div class="pipeline-step" style="border-top:3px solid {clr}">
              <div style="font-size:1.5rem;font-weight:800;color:{clr};margin-bottom:6px">{num}</div>
              <div style="font-weight:700;font-size:0.82rem;color:{T_PRI};margin-bottom:4px">{title}</div>
              <div style="font-size:0.72rem;color:#64748B">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Feature selection scores
        fa, fb = st.columns(2)
        with fa:
            st.plotly_chart(fig_fs_scores(pipe['kb_scores'], pipe['mi_scores'], pipe['fi_scores']),
                            use_container_width=True, theme=None)
        with fb:
            st.plotly_chart(fig_fi_best(pipe['selected_features'], pipe['best_mdl']),
                            use_container_width=True, theme=None)

        st.markdown(f"""
        <div class="note-box">
          <p><strong>Selected features (RFE, 10 of 15):</strong> {', '.join(pipe['selected_features'])} — chosen by recursive wrapper that directly evaluates Decision Tree classification performance.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # All models ranked
        st.markdown("<div class='section-head'>All Models — Comparison</div>", unsafe_allow_html=True)
        for name, v in sorted(results.items(), key=lambda x:-x[1]['ROC-AUC']):
            is_best = name == best_key
            border  = 'mrow-best' if is_best else ''
            crown   = ' (Best)' if is_best else ''
            clr     = A_RED if is_best else T_PRI
            st.markdown(f"""
            <div class="mrow {border}">
              <div>
                <div class="mname">{name}{crown}</div>
                <div class="mtype">Decision Tree Classifier</div>
              </div>
              <div style="display:flex;gap:1.5rem;align-items:center">
                <div><div class="mval" style="color:{clr}">{v['ROC-AUC']:.4f}</div><div class="mval-sub">ROC-AUC</div></div>
                <div><div class="mval">{v['Accuracy']:.4f}</div><div class="mval-sub">Accuracy</div></div>
                <div><div class="mval">{v['Recall']:.4f}</div><div class="mval-sub">Recall</div></div>
                <div><div class="mval">{v['F1']:.4f}</div><div class="mval-sub">F1</div></div>
                <div><div class="mval">{v['Precision']:.4f}</div><div class="mval-sub">Precision</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1: st.plotly_chart(fig_model_roc_bar(results), use_container_width=True, theme=None)
        with g2: st.plotly_chart(fig_roc_curves(pipe['roc_data'], best_key), use_container_width=True, theme=None)

        g3, g4 = st.columns(2)
        with g3: st.plotly_chart(fig_confusion(pipe['cm']), use_container_width=True, theme=None)
        with g4: st.plotly_chart(fig_cv(pipe['cv_sc']), use_container_width=True, theme=None)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # CV metrics
        cv_sc = pipe['cv_sc']
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("CV Mean ROC-AUC", f"{cv_sc.mean():.4f}")
        m2.metric("CV Std",          f"{cv_sc.std():.4f}")
        m3.metric("CV Min",          f"{cv_sc.min():.4f}")
        m4.metric("CV Max",          f"{cv_sc.max():.4f}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Why DT
        st.markdown("<div class='section-head'>Why Decision Tree?</div>", unsafe_allow_html=True)
        w1,w2,w3,w4 = st.columns(4)
        for col,num,title,text,clr in [
            (w1,'01','Medical Interpretability','Clinicians can follow the exact splits — age > 55 AND sysBP > 140 — to understand why a patient is flagged as high risk.',A_RED),
            (w2,'02','No Scaling Required','All 15 features are numerical but DT splits on thresholds, not distances. No StandardScaler or normalisation needed.',A_BLUE),
            (w3,'03','Handles Imbalance Signal','With Bayesian tuning of min_samples_leaf and max_depth, DT avoids memorising the majority class.',A_GREEN),
            (w4,'04','Bayesian Tuning Edge','Optuna finds optimal depth/leaf constraints that balance sensitivity (recall) with overall ROC-AUC.',A_PURPLE),
        ]:
            col.markdown(f"""
            <div class="why-card">
              <div class="why-icon" style="background:{clr}1A;color:{clr}">{num}</div>
              <div class="why-title">{title}</div>
              <div class="why-desc">{text}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — CHD RISK PREDICTOR
    # ══════════════════════════════════════════════════════════════════
    with t4:
        st.markdown("<div class='section-head'>10-Year CHD Risk Predictor</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-sub'>Enter patient details below. The Bayesian-tuned Decision Tree estimates the probability of developing coronary heart disease within 10 years. ROC-AUC = {results[best_key]['ROC-AUC']:.3f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        inp_col, res_col = st.columns([1, 1.3])

        with inp_col:
            st.markdown("#### Patient Information")

            st.markdown("<div class='input-group-title' style='margin-top:1rem'>Demographics</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            gender  = c1.selectbox("Gender", ["Male","Female"])
            age_val = c2.number_input("Age", min_value=32, max_value=70, value=50)
            edu_val = st.select_slider("Education Level",
                options=[1,2,3,4],
                format_func=lambda x: {1:"Some High School",2:"HS/GED",3:"Some College",4:"College"}[x],
                value=2)

            st.markdown("<div class='input-group-title' style='margin-top:1rem'>Smoking</div>", unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            smoker   = c3.selectbox("Current Smoker", ["No","Yes"])
            cigs_day = c4.number_input("Cigarettes per Day", min_value=0, max_value=70, value=0)

            st.markdown("<div class='input-group-title' style='margin-top:1rem'>Medical History</div>", unsafe_allow_html=True)
            c5,c6,c7,c8 = st.columns(4)
            bp_meds  = c5.selectbox("BP Medication", ["No","Yes"])
            stroke   = c6.selectbox("Stroke History", ["No","Yes"])
            hyp      = c7.selectbox("Hypertension", ["No","Yes"])
            diabetes = c8.selectbox("Diabetes", ["No","Yes"])

            st.markdown("<div class='input-group-title' style='margin-top:1rem'>Clinical Measurements</div>", unsafe_allow_html=True)
            c9, c10 = st.columns(2)
            tot_chol  = c9.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=700, value=235)
            sys_bp    = c10.number_input("Systolic BP (mmHg)", min_value=80, max_value=300, value=130)
            c11, c12 = st.columns(2)
            dia_bp    = c11.number_input("Diastolic BP (mmHg)", min_value=48, max_value=145, value=82)
            bmi_val   = c12.number_input("BMI", min_value=15.0, max_value=57.0, value=25.5, step=0.1)
            c13, c14 = st.columns(2)
            hr_val    = c13.number_input("Heart Rate (bpm)", min_value=44, max_value=143, value=75)
            gluc_val  = c14.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=80)

            predict_btn = st.button("Predict CHD Risk")

        with res_col:
            if predict_btn:
                inputs = {
                    'male':           1 if gender=="Male" else 0,
                    'age':            age_val,
                    'education':      edu_val,
                    'currentSmoker':  1 if smoker=="Yes" else 0,
                    'cigsPerDay':     cigs_day,
                    'BPMeds':         1 if bp_meds=="Yes" else 0,
                    'prevalentStroke':1 if stroke=="Yes" else 0,
                    'prevalentHyp':   1 if hyp=="Yes" else 0,
                    'diabetes':       1 if diabetes=="Yes" else 0,
                    'totChol':        tot_chol,
                    'sysBP':          sys_bp,
                    'diaBP':          dia_bp,
                    'BMI':            bmi_val,
                    'heartRate':      hr_val,
                    'glucose':        gluc_val,
                }
                prob  = predict_risk(inputs, pipe)
                pct   = prob * 100
                level = "High" if prob>0.2 else "Medium" if prob>0.1 else "Low"
                clr_map = {'Low':A_GREEN, 'Medium':A_AMBER, 'High':A_RED}
                cls_map = {'Low':'risk-low','Medium':'risk-medium','High':'risk-high'}
                clr   = clr_map[level]

                st.session_state['chd_result'] = {'prob':prob,'pct':pct,'level':level,'inputs':inputs}

            if 'chd_result' in st.session_state:
                r = st.session_state['chd_result']
                prob=r['prob']; pct=r['pct']; level=r['level']
                clr = clr_map[level] if 'clr_map' in dir() else ({'Low':A_GREEN,'Medium':A_AMBER,'High':A_RED}[level])
                cls = {'Low':'risk-low','Medium':'risk-medium','High':'risk-high'}[level]

                # Big risk card
                st.markdown(f"""
                <div class="risk-card {cls}">
                  <div class="risk-label">10-Year CHD Risk Estimate</div>
                  <div class="risk-pct">{pct:.1f}%</div>
                  <div class="risk-level" style="color:{'#34D399' if level=='Low' else '#FBBF24' if level=='Medium' else '#F87171'}">{level} Risk</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                # Gauge
                st.plotly_chart(risk_gauge(prob), use_container_width=True, theme=None)

                # Key metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level",    level)
                m2.metric("Probability",   f"{pct:.1f}%")
                m3.metric("Model",         "DT Bayesian", f"AUC={results[best_key]['ROC-AUC']:.3f}")

                # Risk factor comparison vs population average
                st.markdown("#### How This Patient Compares to Population")
                pop_means = df[r['inputs'].keys()].mean()
                factors   = []
                for feat, val in r['inputs'].items():
                    pop_val = pop_means[feat]
                    factors.append({'Feature':feat, 'Patient':val, 'Population Avg':round(pop_val,2)})
                factors_df = pd.DataFrame(factors)
                st.dataframe(factors_df, use_container_width=True)

                # Clinical interpretation
                interpretation = []
                inp = r['inputs']
                if inp['age'] > 55:      interpretation.append(("Age > 55", "High risk age group"))
                if inp['sysBP'] > 140:   interpretation.append(("Systolic BP > 140", "Stage 2 hypertension range"))
                if inp['glucose'] > 126: interpretation.append(("Glucose > 126", "Diabetic range"))
                if inp['totChol'] > 240: interpretation.append(("Cholesterol > 240", "High cholesterol"))
                if inp['currentSmoker']: interpretation.append(("Current smoker", "Major CHD risk factor"))
                if inp['BMI'] > 30:      interpretation.append(("BMI > 30", "Obese range"))

                if interpretation:
                    st.markdown("#### Elevated Risk Factors Detected")
                    for flag, note in interpretation:
                        st.markdown(f"""<div style='display:flex;gap:10px;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
                          <span class='badge b-red'>{flag}</span>
                          <span style='font-size:0.8rem;color:#94A3B8'>{note}</span>
                        </div>""", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="note-box" style="margin-top:1rem">
                  <p>Fill in the patient details on the left and click <strong>Predict CHD Risk</strong> to get the 10-year coronary heart disease probability estimate.</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("#### Key Risk Factors (from model feature importances)")
                fi = dict(zip(pipe['selected_features'], pipe['best_mdl'].feature_importances_))
                fi_sorted = sorted(fi.items(), key=lambda x:-x[1])[:6]
                for feat, imp in fi_sorted:
                    bar_w = int(imp/max(fi.values())*100)
                    st.markdown(f"""<div style='margin-bottom:8px'>
                      <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                        <span style='font-size:0.82rem;font-weight:600;color:{T_PRI}'>{feat}</span>
                        <span style='font-size:0.75rem;color:#64748B'>{imp*100:.1f}%</span>
                      </div>
                      <div style='height:6px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden'>
                        <div style='height:100%;width:{bar_w}%;background:linear-gradient(90deg,{A_RED},{A_PURPLE});border-radius:99px'></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="note-box">
          <p><strong>Model:</strong> Decision Tree Classifier — Bayesian-tuned (Optuna) &nbsp;|&nbsp;
          ROC-AUC = {results[best_key]['ROC-AUC']:.4f} &nbsp;|&nbsp;
          Recall = {results[best_key]['Recall']:.4f} &nbsp;|&nbsp;
          F1 = {results[best_key]['F1']:.4f}<br><br>
          This prediction is for educational and portfolio purposes only. It is not a substitute for clinical diagnosis. Actual CHD risk assessment requires full clinical evaluation by a licensed physician.</p>
        </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()