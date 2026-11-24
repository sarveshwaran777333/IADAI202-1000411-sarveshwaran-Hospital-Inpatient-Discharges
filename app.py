import re
import html
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# Dark base theme for Streamlit's own widgets (applied on the next start if the
# file did not exist yet). The light/dark mix is handled by the CSS layer below.
try:
    _config_file = Path(__file__).resolve().parent / ".streamlit" / "config.toml"
    if not _config_file.exists():
        _config_file.parent.mkdir(exist_ok=True)
        _config_file.write_text(
            '[theme]\nbase = "dark"\nprimaryColor = "#00e5ff"\n'
            'backgroundColor = "#07111f"\nsecondaryBackgroundColor = "#0d1a2b"\n'
            'textColor = "#eef7ff"\n',
            encoding="utf-8",
        )
except Exception:
    pass

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, classification_report

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediScope | Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "MediScope"

DATASET_FILENAME = (
    "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"
)

SAMPLE_ROWS = 250_000

AGE_ORDER = [
    "0 to 17",
    "18 to 29",
    "30 to 49",
    "50 to 69",
    "70 or Older",
    "Unknown",
]

SEVERITY_ORDER = [
    "Minor",
    "Moderate",
    "Major",
    "Extreme",
    "Unknown",
]

PAYMENT_ORDER = [
    "Medicare",
    "Medicaid",
    "Private Health Insurance",
    "Self-Pay",
    "Other",
    "Unknown",
]

COLORS = [
    "#00E5FF",
    "#8B5CF6",
    "#00E676",
    "#FFB300",
    "#FF4081",
    "#448AFF",
    "#FF7043",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --cyan: #00e5ff;
        --purple: #8b5cf6;
        --green: #00e676;
        --text: #eef7ff;
        --muted: #9aacc4;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(0,229,255,.075), transparent 27%),
            radial-gradient(circle at 90% 5%, rgba(139,92,246,.09), transparent 30%),
            #07111f;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0b192b,#081321);
        border-right: 1px solid rgba(0,229,255,.20);
    }

    h1, h2, h3 {
        color: #f4faff !important;
        letter-spacing: -.5px;
    }

    p, label, li {
        color: #c8d7e9;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg,#12243a,#0d1a2b);
        border: 1px solid rgba(0,229,255,.18);
        border-radius: 16px;
        padding: 18px;
        transition: transform .25s ease,border-color .25s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0,229,255,.65);
    }

    [data-testid="stMetricLabel"] {
        color: #a7bed8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-weight: 800 !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px;
        margin-bottom: 22px;
        border-radius: 22px;
        background:
            linear-gradient(115deg,rgba(0,229,255,.11),transparent 48%),
            linear-gradient(145deg,#132740,#0d192a);
        border: 1px solid rgba(0,229,255,.25);
        box-shadow: 0 12px 45px rgba(0,0,0,.18);
    }

    .hero-title {
        font-size: clamp(28px,4vw,43px);
        font-weight: 800;
        color: #f4fbff;
        letter-spacing: -1.5px;
        line-height: 1.15;
    }

    .hero-title span {
        color: #00e5ff;
        text-shadow: 0 0 22px rgba(0,229,255,.35);
    }

    .hero-subtitle {
        color: #b4c9df;
        margin-top: 12px;
        font-size: 14px;
        line-height: 1.7;
        max-width: 750px;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }

    .hero-tag {
        border: 1px solid rgba(0,229,255,.24);
        background: rgba(0,229,255,.06);
        color: #bdefff;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
    }

    .hospital-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 13px 8px 18px;
    }

    .hospital-icon {
        position: relative;
        flex: 0 0 58px;
        width: 58px;
        height: 58px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .hospital-building {
        position: relative;
        z-index: 3;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(145deg,#00e5ff,#0088ff);
        color: #07111f;
        font-size: 31px;
        font-weight: 800;
        line-height: 1;
        box-shadow: 0 0 25px rgba(0,229,255,.35);
        animation: hospital-glow 2.2s ease-in-out infinite;
    }

    .hospital-ring {
        position: absolute;
        inset: 0;
        border: 2px solid rgba(0,229,255,.7);
        border-radius: 17px;
        opacity: 0;
        animation: hospital-expand 2.4s ease-out infinite;
    }

    .hospital-ring:nth-child(2) {
        animation-delay: .8s;
    }

    .hospital-ring:nth-child(3) {
        animation-delay: 1.6s;
    }

    @keyframes hospital-expand {
        0% {
            transform: scale(.72);
            opacity: .8;
        }
        100% {
            transform: scale(1.75);
            opacity: 0;
        }
    }

    @keyframes hospital-glow {
        0%,100% {
            box-shadow: 0 0 13px rgba(0,229,255,.25);
        }
        50% {
            box-shadow: 0 0 28px rgba(0,229,255,.65);
        }
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        color: #f4fbff;
    }

    .brand-name span {
        color: #00e5ff;
    }

    .brand-caption {
        font-size: 10px;
        color: #91a9c4;
        margin-top: 3px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .mascot-wrap {
        position: relative;
        overflow: hidden;
        margin: 16px 0 18px;
        padding: 18px 12px;
        text-align: center;
        border-radius: 18px;
        background:
            radial-gradient(circle at 50% 35%,rgba(0,229,255,.12),transparent 65%),
            linear-gradient(145deg,#12263c,#0b1727);
        border: 1px solid rgba(0,229,255,.18);
    }

    .robot {
        position: relative;
        width: 94px;
        height: 84px;
        margin: 3px auto 12px;
        animation: robot-float 3.2s ease-in-out infinite;
    }

    .robot-head {
        position: absolute;
        left: 12px;
        top: 8px;
        width: 70px;
        height: 57px;
        border-radius: 20px;
        background: linear-gradient(145deg,#e9fbff,#8bdcf4);
        border: 3px solid #00c8ee;
        box-shadow: 0 0 25px rgba(0,229,255,.18);
    }

    .robot-ear {
        position: absolute;
        top: 26px;
        width: 10px;
        height: 20px;
        border-radius: 5px;
        background: #00c8ee;
    }

    .robot-ear.left {
        left: 3px;
    }

    .robot-ear.right {
        right: 3px;
    }

    .robot-antenna {
        position: absolute;
        top: 0;
        left: 45px;
        height: 12px;
        width: 4px;
        background: #00e5ff;
        border-radius: 4px;
    }

    .robot-antenna:after {
        content: "";
        position: absolute;
        top: -5px;
        left: -4px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #00e5ff;
        box-shadow: 0 0 12px #00e5ff;
    }

    .robot-eyes {
        display: flex;
        justify-content: center;
        gap: 17px;
        padding-top: 17px;
    }

    .robot-eye {
        width: 9px;
        height: 12px;
        border-radius: 50%;
        background: #12334c;
        animation: robot-blink 4.5s infinite;
    }

    .robot-mouth {
        width: 20px;
        height: 6px;
        margin: 7px auto 0;
        border-bottom: 3px solid #12334c;
        border-radius: 0 0 12px 12px;
    }

    .robot-body {
        position: absolute;
        left: 31px;
        bottom: 0;
        width: 32px;
        height: 14px;
        border-radius: 8px 8px 5px 5px;
        background: #00c8ee;
    }

    .robot-name {
        color: #00e5ff;
        font-size: 14px;
        font-weight: 800;
    }

    .robot-description {
        color: #a7bed8;
        font-size: 10px;
        line-height: 1.6;
        margin-top: 5px;
    }

    @keyframes robot-float {
        0%,100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-7px);
        }
    }

    @keyframes robot-blink {
        0%,46%,50%,100% {
            transform: scaleY(1);
        }
        48% {
            transform: scaleY(.12);
        }
    }

    .section-label {
        color: #00e5ff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin: 5px 0 12px;
    }

    .insight-card {
        height: 100%;
        min-height: 115px;
        padding: 17px;
        border-radius: 15px;
        border: 1px solid rgba(0,229,255,.17);
        background: linear-gradient(145deg,#12243a,#0d1929);
        transition: transform .2s ease,border-color .2s ease;
    }

    .insight-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0,229,255,.5);
    }

    .insight-title {
        color: #9db5d0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .insight-value {
        color: #00e5ff;
        font-size: 23px;
        font-weight: 800;
        margin-top: 12px;
    }

    .insight-description {
        color: #9fb3ca;
        font-size: 11px;
        margin-top: 5px;
        line-height: 1.5;
    }

    .footer {
        margin-top: 35px;
        padding: 18px 0 8px;
        text-align: center;
        border-top: 1px solid rgba(0,229,255,.14);
        color: #7189a5;
        font-size: 11px;
        line-height: 1.8;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(0,229,255,.32);
        background: rgba(0,229,255,.08);
        color: #dffaff;
        font-weight: 600;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        border-color: #00e5ff;
        background: rgba(0,229,255,.17);
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0,229,255,.16);
        border-radius: 12px;
        overflow: hidden;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3D VISUAL LAYER (CSS + THREE.JS)
# ============================================================

MS_CSS = r"""<style>
/* ===== MediScope 3D layer: canvas, theme, depth ===== */
@property --ms-ang { syntax: '<angle>'; inherits: false; initial-value: 0deg; }
@keyframes msSpin { to { --ms-ang: 360deg; } }
@keyframes msSheen { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
@keyframes msPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,229,255,.45); } 50% { box-shadow: 0 0 0 9px rgba(0,229,255,0); } }

html, body { background: #060f1c; }
#ms-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
#ms-bg canvas { display: block; width: 100% !important; height: 100% !important; }
#root, .stApp { position: relative; z-index: 1; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main,
[data-testid="stBottom"] > div { background: transparent !important; }
[data-testid="stHeader"] { background: rgba(7,17,31,.55) !important; backdrop-filter: blur(14px); }
iframe[height="0"] { display: none; }
[data-testid="stElementContainer"]:has(> iframe[height="0"]),
.element-container:has(> iframe[height="0"]) { height: 0 !important; margin: 0 !important; overflow: hidden; position: absolute; }

#ms-progress { position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 999999; transform-origin: 0 50%; transform: scaleX(0);
  background: linear-gradient(90deg, #00e5ff, #8b5cf6, #00e676); box-shadow: 0 0 14px rgba(0,229,255,.7); }
#ms-cursor { position: fixed; top: 0; left: 0; width: 340px; height: 340px; z-index: 2; pointer-events: none; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,229,255,.16), rgba(139,92,246,.07) 45%, transparent 70%); mix-blend-mode: screen; will-change: transform; }
body.ms-mix #ms-cursor { mix-blend-mode: normal; opacity: .55; }

#ms-theme-btn { position: fixed; right: 22px; bottom: 22px; z-index: 999998; display: flex; align-items: center; gap: 9px; cursor: pointer;
  padding: 11px 18px; border-radius: 999px; font: 700 12px 'Inter', sans-serif; color: #e8fbff; letter-spacing: .2px;
  background: linear-gradient(145deg, rgba(14,34,58,.92), rgba(8,20,36,.92)); border: 1px solid rgba(0,229,255,.45);
  box-shadow: 0 8px 0 rgba(0,120,160,.45), 0 18px 34px rgba(0,0,0,.35); transition: transform .15s, box-shadow .15s; backdrop-filter: blur(10px); }
#ms-theme-btn:hover { transform: translateY(-2px); }
#ms-theme-btn:active { transform: translateY(6px); box-shadow: 0 2px 0 rgba(0,120,160,.45), 0 8px 16px rgba(0,0,0,.3); }
#ms-theme-btn i { width: 14px; height: 14px; border-radius: 50%; background: linear-gradient(90deg, #eaf4ff 50%, #07111f 50%); border: 1px solid #00e5ff; animation: msPulse 2.4s infinite; }

/* ----- light + dark mix: the content column is flipped to light, depth cards stay dark ----- */
body.ms-mix [data-testid="stMain"] { filter: invert(1) hue-rotate(180deg); }
body.ms-mix .hero, body.ms-mix [data-testid="stMetric"], body.ms-mix .insight-card,
body.ms-mix [data-testid="stPlotlyChart"], body.ms-mix .mascot-wrap { filter: invert(1) hue-rotate(180deg); }
body.ms-mix [data-testid="stMain"] [data-testid="stMetric"] { box-shadow: 0 22px 40px -18px rgba(8,30,60,.55); }

/* ----- sidebar ----- */
[data-testid="stSidebar"] { background: linear-gradient(190deg, rgba(11,25,43,.94), rgba(6,14,26,.96)) !important;
  backdrop-filter: blur(18px); border-right: 1px solid rgba(0,229,255,.28) !important; box-shadow: 18px 0 60px -20px rgba(0,229,255,.25); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
[data-testid="stSidebar"] [role="radiogroup"] > label { padding: 9px 14px; border-radius: 12px; border: 1px solid rgba(0,229,255,.10);
  background: rgba(255,255,255,.02); transition: transform .2s, background .2s, border-color .2s, box-shadow .2s; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { transform: translateX(7px) translateZ(0); background: rgba(0,229,255,.09); border-color: rgba(0,229,255,.45); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { background: linear-gradient(100deg, rgba(0,229,255,.22), rgba(139,92,246,.18));
  border-color: rgba(0,229,255,.7); box-shadow: 0 0 22px rgba(0,229,255,.25), inset 0 0 18px rgba(0,229,255,.08); }

/* ----- hero ----- */
.hero { transform-style: preserve-3d; transition: transform .18s ease-out; isolation: isolate; }
.hero::before { content: ""; position: absolute; inset: 0; z-index: 0; pointer-events: none; opacity: .55;
  background: linear-gradient(115deg, transparent 30%, rgba(0,229,255,.16) 48%, transparent 66%); background-size: 220% 100%; animation: msSheen 6s linear infinite; }
.hero > * { position: relative; z-index: 1; }
.hero .ms-hero3d { position: absolute !important; z-index: 0 !important; top: 0; right: 0; bottom: 0; width: min(42%, 430px); pointer-events: none; }
.hero .ms-hero3d canvas { width: 100% !important; height: 100% !important; display: block; }
.hero-title { text-shadow: 0 1px 0 rgba(0,170,200,.55), 0 2px 0 rgba(0,150,185,.45), 0 3px 0 rgba(0,130,170,.35), 0 14px 28px rgba(0,229,255,.28); }
.hero-tag { transition: transform .2s, background .2s; }
.hero-tag:hover { transform: translateY(-3px) translateZ(20px); background: rgba(0,229,255,.18); }
@media (max-width: 900px) { .hero .ms-hero3d { display: none; } }

/* ----- depth cards ----- */
[data-testid="stMetric"], .insight-card, .mascot-wrap { position: relative; transform-style: preserve-3d; transition: transform .16s ease-out, border-color .25s, box-shadow .25s; will-change: transform; }
[data-testid="stMetric"]::after, .insight-card::after, .hero::after { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; opacity: 0; transition: opacity .25s;
  background: radial-gradient(260px circle at var(--mx, 50%) var(--my, 50%), rgba(0,229,255,.20), transparent 60%); }
[data-testid="stMetric"]:hover::after, .insight-card:hover::after, .hero:hover::after { opacity: 1; }
[data-testid="stMetric"] { box-shadow: 0 18px 38px -20px rgba(0,0,0,.7), inset 0 1px 0 rgba(255,255,255,.07); }
[data-testid="stMetric"]:hover, .insight-card:hover { box-shadow: 0 26px 50px -18px rgba(0,229,255,.38), inset 0 1px 0 rgba(255,255,255,.12); }

/* ----- chart cards: rotating neon border ----- */
[data-testid="stPlotlyChart"] { position: relative; border-radius: 22px; padding: 6px; border: 1.5px solid transparent;
  background: linear-gradient(160deg, #0d1c30, #07121f) padding-box,
              conic-gradient(from var(--ms-ang), rgba(0,229,255,.9), rgba(139,92,246,.75), rgba(0,230,118,.7), rgba(0,229,255,.9)) border-box;
  animation: msSpin 9s linear infinite; box-shadow: 0 30px 60px -28px rgba(0,229,255,.35); overflow: hidden; }
[data-testid="stPlotlyChart"] .modebar { background: rgba(8,20,36,.7) !important; border-radius: 10px; }

/* ----- entrance (set only by script, so content never stays hidden without JS) ----- */
.ms-pre { opacity: 0; transform: translateY(34px) scale(.96) rotateX(8deg); }
.ms-in { opacity: 1; transition: opacity .8s cubic-bezier(.2,.8,.2,1), transform .9s cubic-bezier(.2,.8,.2,1); }

/* ----- headings, buttons, tables ----- */
h1 { text-shadow: 0 2px 0 rgba(0,160,200,.35), 0 12px 26px rgba(0,229,255,.18); }
h2, h3 { position: relative; padding-bottom: 6px; }
h2::after, h3::after { content: ""; position: absolute; left: 0; bottom: 0; width: 64px; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, #00e5ff, #8b5cf6, #00e676, #00e5ff); background-size: 200% 100%; animation: msSheen 3s linear infinite; }
.stButton > button, .stDownloadButton > button { box-shadow: 0 6px 0 rgba(0,140,180,.55), 0 14px 22px rgba(0,0,0,.3); transition: transform .12s, box-shadow .12s, background .2s, border-color .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(5px); box-shadow: 0 1px 0 rgba(0,140,180,.55), 0 6px 10px rgba(0,0,0,.3); }
[data-testid="stDataFrame"] { box-shadow: 0 24px 48px -26px rgba(0,229,255,.3); }
[data-testid="stExpander"] { border-radius: 14px; border: 1px solid rgba(0,229,255,.16); backdrop-filter: blur(6px); }

/* ----- math page ----- */
.math-step { position: relative; padding: 18px 22px; margin: 6px 0 14px; border-radius: 18px; border: 1px solid rgba(0,229,255,.2);
  background: linear-gradient(150deg, rgba(18,38,62,.92), rgba(9,20,34,.92)); color: #c8d7e9; line-height: 1.7; font-size: 14px; box-shadow: 0 20px 40px -24px rgba(0,229,255,.35); }
.math-step b { color: #00e5ff; }
.math-num { display: inline-flex; width: 30px; height: 30px; align-items: center; justify-content: center; margin-right: 10px; border-radius: 10px;
  font-weight: 800; color: #07111f; background: linear-gradient(145deg, #00e5ff, #8b5cf6); box-shadow: 0 5px 0 rgba(0,100,150,.5); }
body.ms-mix .math-step { filter: invert(1) hue-rotate(180deg); }

/* ----- Dr. MediBot: floating 3D doctor + chat dialog ----- */
.st-key-doc_launch { position: fixed !important; left: -9999px; top: 0; width: 1px; height: 1px; overflow: hidden; opacity: 0; }
#ms-doc-launch { position: fixed; right: 18px; bottom: 82px; z-index: 999997; width: 132px; height: 158px; cursor: pointer; outline: none;
  filter: drop-shadow(0 14px 22px rgba(0,229,255,.38)); transition: transform .25s cubic-bezier(.2,.8,.2,1); }
#ms-doc-launch:hover, #ms-doc-launch:focus-visible { transform: translateY(-6px) scale(1.06); }
#ms-doc-canvas { width: 100%; height: 100%; }
#ms-doc-canvas canvas { width: 100% !important; height: 100% !important; display: block; }
#ms-doc-bubble { position: absolute; right: 8px; bottom: 100%; margin-bottom: -4px; white-space: nowrap; padding: 9px 14px; border-radius: 16px 16px 4px 16px;
  font: 700 12px 'Inter', sans-serif; color: #e8fbff; background: linear-gradient(145deg, rgba(14,34,58,.96), rgba(8,20,36,.96));
  border: 1px solid rgba(0,229,255,.55); box-shadow: 0 10px 26px rgba(0,0,0,.35), 0 0 18px rgba(0,229,255,.25);
  opacity: 0; transform: translateY(8px) scale(.94); transform-origin: 100% 100%; pointer-events: none; transition: opacity .3s, transform .3s; }
#ms-doc-launch.ms-show #ms-doc-bubble { opacity: 1; transform: none; }
@media (max-width: 640px) { #ms-doc-launch { width: 96px; height: 116px; bottom: 76px; right: 10px; } }

[data-testid="stDialog"] div[role="dialog"] { background: linear-gradient(160deg, rgba(12,26,44,.97), rgba(6,14,26,.98)) !important;
  border: 1.5px solid rgba(0,229,255,.45); border-radius: 24px; box-shadow: 0 40px 90px -20px rgba(0,229,255,.35), 0 0 0 1px rgba(139,92,246,.2); }
[data-testid="stDialog"] div[role="dialog"] h2, [data-testid="stDialog"] div[role="dialog"] p,
[data-testid="stDialog"] div[role="dialog"] li, [data-testid="stDialog"] div[role="dialog"] label { color: #e4f1ff; }
[data-testid="stDialog"] div[role="dialog"] h2::after { display: none; }
.ms-doc-stage { position: relative; height: 190px; margin: 0 0 10px; border-radius: 18px; overflow: hidden; border: 1px solid rgba(0,229,255,.25);
  background: radial-gradient(circle at 50% 80%, rgba(0,229,255,.22), transparent 62%), radial-gradient(circle at 15% 15%, rgba(139,92,246,.2), transparent 45%), linear-gradient(160deg, #0d1c30, #070f1b); }
.ms-doc-stage canvas { width: 100% !important; height: 100% !important; display: block; }
[data-testid="stDialog"] [data-testid="stChatMessage"] { background: rgba(0,229,255,.05); border: 1px solid rgba(0,229,255,.14); border-radius: 14px; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
  .ms-pre { opacity: 1; transform: none; }
}
</style>
"""

MS_JS = r"""(function () {
  'use strict';
  var W = window, D = document, B = D.body;
  var reduce = !!(W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var THREE_URL = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  var state = { theme: 'mix', mx: 0.5, my: 0.5, px: 0, py: 0, scroll: 0, hero: null, bg: null, doc: null, stage: null, docHover: false, lastMut: 0 };

  try { state.theme = W.localStorage.getItem('ms-theme') || 'mix'; } catch (e) {}

  /* ---------- theme ---------- */
  function applyTheme(t) {
    state.theme = t;
    B.classList.remove('ms-mix', 'ms-dark');
    B.classList.add(t === 'dark' ? 'ms-dark' : 'ms-mix');
    try { W.localStorage.setItem('ms-theme', t); } catch (e) {}
    var btn = D.getElementById('ms-theme-btn');
    if (btn) btn.innerHTML = t === 'dark' ? '<i></i>Full dark' : '<i></i>Light + dark mix';
    if (state.bg) state.bg.setTheme(t);
  }

  /* ---------- static chrome ---------- */
  function ensureChrome() {
    if (!D.getElementById('ms-bg')) {
      var bg = D.createElement('div'); bg.id = 'ms-bg'; B.insertBefore(bg, B.firstChild);
    }
    if (!D.getElementById('ms-progress')) {
      var p = D.createElement('div'); p.id = 'ms-progress'; B.appendChild(p);
    }
    if (!D.getElementById('ms-cursor')) {
      var c = D.createElement('div'); c.id = 'ms-cursor'; B.appendChild(c);
    }
    if (!D.getElementById('ms-theme-btn')) {
      var b = D.createElement('button'); b.id = 'ms-theme-btn'; b.type = 'button';
      b.setAttribute('aria-label', 'Toggle theme');
      b.addEventListener('click', function () { applyTheme(state.theme === 'dark' ? 'mix' : 'dark'); });
      B.appendChild(b);
    }
    applyTheme(state.theme);
  }

  /* ---------- background scene ---------- */
  function crossGeometry(THREE) {
    var s = new THREE.Shape(), a = 1, b = 0.34;
    s.moveTo(-b, -a); s.lineTo(b, -a); s.lineTo(b, -b); s.lineTo(a, -b); s.lineTo(a, b); s.lineTo(b, b);
    s.lineTo(b, a); s.lineTo(-b, a); s.lineTo(-b, b); s.lineTo(-a, b); s.lineTo(-a, -b); s.lineTo(-b, -b);
    s.closePath();
    return new THREE.ExtrudeGeometry(s, { depth: 0.5, bevelEnabled: true, bevelSize: 0.08, bevelThickness: 0.08, bevelSegments: 2 });
  }

  var PALETTE = {
    dark: { c: '#060f1c', a: '#00e5ff', b: '#8b5cf6', p: '#8ff3ff', cross: '#17c9ee', glow: 0.1, pa: 1.0 },
    mix:  { c: '#e7f1fa', a: '#6fd3ff', b: '#b8a6ff', p: '#1c7fb8', cross: '#2aa7d6', glow: 0.22, pa: 0.9 }
  };

  function initBackground(THREE) {
    var host = D.getElementById('ms-bg');
    if (!host || state.bg) return;
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch (e) { return; }
    renderer.setPixelRatio(Math.min(W.devicePixelRatio || 1, 1.5));
    renderer.setSize(W.innerWidth, W.innerHeight);
    host.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    var cam = new THREE.PerspectiveCamera(58, W.innerWidth / W.innerHeight, 0.1, 300);
    cam.position.set(0, 0, 32);

    var U = {
      uTime: { value: 0 }, uMouse: { value: new THREE.Vector2(0.5, 0.5) },
      uA: { value: new THREE.Color(PALETTE.dark.a) }, uB: { value: new THREE.Color(PALETTE.dark.b) },
      uC: { value: new THREE.Color(PALETTE.dark.c) }, uGlow: { value: PALETTE.dark.glow }
    };
    var gradient = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        uniforms: U, depthTest: false, depthWrite: false,
        vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.9999, 1.0); }',
        fragmentShader: [
          'precision highp float; varying vec2 vUv;',
          'uniform float uTime, uGlow; uniform vec3 uA, uB, uC; uniform vec2 uMouse;',
          'float h(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
          'float n(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);',
          '  return mix(mix(h(i), h(i+vec2(1.0,0.0)), f.x), mix(h(i+vec2(0.0,1.0)), h(i+vec2(1.0,1.0)), f.x), f.y); }',
          'float fbm(vec2 p){ float v = 0.0, a = 0.5; for(int i=0;i<5;i++){ v += a*n(p); p *= 2.0; a *= 0.5; } return v; }',
          'void main(){',
          '  vec2 uv = vUv; vec2 p = uv * vec2(3.2, 2.2); float t = uTime * 0.05;',
          '  float q = fbm(p + t); float r = fbm(p + 2.0*q + vec2(1.7, 9.2) - t);',
          '  vec3 col = mix(uC, uA, smoothstep(0.35, 0.95, q) * 0.34);',
          '  col = mix(col, uB, smoothstep(0.40, 0.98, r) * 0.30);',
          '  float d = distance(uv, uMouse);',
          '  col = mix(col, uA, uGlow * exp(-d * 5.5));',
          '  gl_FragColor = vec4(col, 1.0);',
          '}'
        ].join('\n')
      })
    );
    gradient.frustumCulled = false; gradient.renderOrder = -100; scene.add(gradient);

    /* particles */
    var N = reduce ? 260 : 950;
    var pos = new Float32Array(N * 3), seed = new Float32Array(N);
    for (var i = 0; i < N; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 90; pos[i * 3 + 1] = (Math.random() - 0.5) * 60;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 60; seed[i] = Math.random();
    }
    var pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    pg.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    var PU = { uTime: { value: 0 }, uColor: { value: new THREE.Color(PALETTE.dark.p) }, uAlpha: { value: 1 } };
    var points = new THREE.Points(pg, new THREE.ShaderMaterial({
      uniforms: PU, transparent: true, depthWrite: false,
      vertexShader: [
        'attribute float aSeed; uniform float uTime; varying float vA;',
        'void main(){ vec3 p = position;',
        '  p.y += sin(uTime*0.35 + aSeed*6.283)*1.4; p.x += cos(uTime*0.27 + aSeed*12.0)*1.1;',
        '  vec4 mv = modelViewMatrix * vec4(p, 1.0);',
        '  gl_PointSize = (70.0 + aSeed*130.0) / -mv.z; vA = 0.30 + 0.70 * fract(aSeed*7.0);',
        '  gl_Position = projectionMatrix * mv; }'
      ].join('\n'),
      fragmentShader: [
        'uniform vec3 uColor; uniform float uAlpha; varying float vA;',
        'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;',
        '  gl_FragColor = vec4(uColor, vA * uAlpha * smoothstep(0.5, 0.0, d)); }'
      ].join('\n')
    }));
    scene.add(points);

    /* lights + floating medical crosses and wire orbs */
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var l1 = new THREE.PointLight(0x00e5ff, 1.6, 120); l1.position.set(-25, 18, 20); scene.add(l1);
    var l2 = new THREE.PointLight(0x8b5cf6, 1.4, 120); l2.position.set(28, -14, 18); scene.add(l2);

    var cg = crossGeometry(THREE), crossMat = new THREE.MeshStandardMaterial({
      color: PALETTE.dark.cross, emissive: 0x053a4a, metalness: 0.35, roughness: 0.25, transparent: true, opacity: 0.55
    });
    var crosses = [];
    for (var k = 0; k < (reduce ? 6 : 15); k++) {
      var m = new THREE.Mesh(cg, crossMat), sc = 0.7 + Math.random() * 1.9;
      m.scale.set(sc, sc, sc);
      m.position.set((Math.random() - 0.5) * 70, (Math.random() - 0.5) * 42, -8 - Math.random() * 26);
      m.rotation.set(Math.random() * 3, Math.random() * 3, Math.random() * 3);
      m.userData = { s: 0.2 + Math.random() * 0.5, f: Math.random() * 6, y: m.position.y };
      scene.add(m); crosses.push(m);
    }
    var orbMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true, transparent: true, opacity: 0.2 });
    var orbs = [];
    for (var o = 0; o < (reduce ? 3 : 7); o++) {
      var orb = new THREE.Mesh(new THREE.IcosahedronGeometry(1 + Math.random() * 2.2, 1), orbMat);
      orb.position.set((Math.random() - 0.5) * 80, (Math.random() - 0.5) * 46, -14 - Math.random() * 20);
      orb.userData = { s: 0.1 + Math.random() * 0.25 }; scene.add(orb); orbs.push(orb);
    }

    function setTheme(t) {
      var p = PALETTE[t === 'dark' ? 'dark' : 'mix'];
      U.uC.value.set(p.c); U.uA.value.set(p.a); U.uB.value.set(p.b); U.uGlow.value = p.glow;
      PU.uColor.value.set(p.p); PU.uAlpha.value = p.pa;
      crossMat.color.set(p.cross); orbMat.color.set(t === 'dark' ? 0x00e5ff : 0x1c7fb8);
      B.style.background = p.c;
    }

    var clock = new THREE.Clock(), camY = 0;
    function frame() {
      W.requestAnimationFrame(frame);
      if (D.hidden) return;
      var t = clock.getElapsedTime();
      U.uTime.value = t; PU.uTime.value = t;
      U.uMouse.value.set(state.mx, 1 - state.my);
      crosses.forEach(function (c) {
        c.rotation.x += 0.0016 * c.userData.s * 6; c.rotation.y += 0.0022 * c.userData.s * 6;
        c.position.y = c.userData.y + Math.sin(t * c.userData.s + c.userData.f) * 1.6;
      });
      orbs.forEach(function (o) { o.rotation.y += 0.002 * o.userData.s * 10; o.rotation.x += 0.0012 * o.userData.s * 10; });
      camY += ((-state.scroll * 0.012) - camY) * 0.06;
      cam.position.x += ((state.px * 3.2) - cam.position.x) * 0.04;
      cam.position.y += ((camY - state.py * 2.2) - cam.position.y) * 0.04;
      cam.lookAt(0, camY * 0.6, 0);
      if (state.hero) state.hero.render(t);
      if (state.doc || state.stage) { var ds = doctorState(t); if (state.doc) state.doc.render(t, { wave: state.docHover || (t % 14) < 2.2, talking: false, lx: state.px, ly: state.py }); if (state.stage) state.stage.render(t, { wave: !ds.talking && (t % 9) < 1.6, talking: ds.talking, lx: state.px, ly: state.py }); }
      renderer.render(scene, cam);
    }
    W.addEventListener('resize', function () {
      cam.aspect = W.innerWidth / W.innerHeight; cam.updateProjectionMatrix();
      renderer.setSize(W.innerWidth, W.innerHeight);
    });
    state.bg = { setTheme: setTheme };
    setTheme(state.theme);
    if (!reduce) frame(); else { renderer.render(scene, cam); }
  }

  /* ---------- hero DNA scene ---------- */
  function attachHero() {
    var THREE = W.THREE; if (!THREE) return;
    var hero = D.querySelector('.hero');
    if (!hero) return;
    if (state.hero && state.hero.box.isConnected) return;
    if (state.hero) { state.hero.dispose(); state.hero = null; }

    var box = D.createElement('div'); box.className = 'ms-hero3d'; hero.appendChild(box);
    var renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); } catch (e) { return; }
    renderer.setPixelRatio(Math.min(W.devicePixelRatio || 1, 2));
    box.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    var cam = new THREE.PerspectiveCamera(42, 1, 0.1, 100); cam.position.set(0, 0, 15);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    var pl = new THREE.PointLight(0x00e5ff, 2.2, 60); pl.position.set(6, 6, 10); scene.add(pl);
    var pl2 = new THREE.PointLight(0x8b5cf6, 1.8, 60); pl2.position.set(-7, -5, 8); scene.add(pl2);

    var group = new THREE.Group(); group.rotation.z = 0.42; scene.add(group);
    var sg = new THREE.SphereGeometry(0.2, 16, 16);
    var mA = new THREE.MeshStandardMaterial({ color: 0x00e5ff, emissive: 0x007c99, metalness: 0.4, roughness: 0.25 });
    var mB = new THREE.MeshStandardMaterial({ color: 0x8b5cf6, emissive: 0x3d1d99, metalness: 0.4, roughness: 0.25 });
    var rungs = [], turns = 3.2, steps = 46, R = 1.55, H = 9.2;
    var pa = [], pb = [];
    for (var i = 0; i < steps; i++) {
      var f = i / (steps - 1), a = f * Math.PI * 2 * turns, y = (f - 0.5) * H;
      var x1 = Math.cos(a) * R, z1 = Math.sin(a) * R;
      var s1 = new THREE.Mesh(sg, mA); s1.position.set(x1, y, z1); group.add(s1);
      var s2 = new THREE.Mesh(sg, mB); s2.position.set(-x1, y, -z1); group.add(s2);
      pa.push(new THREE.Vector3(x1, y, z1)); pb.push(new THREE.Vector3(-x1, y, -z1));
      if (i % 2 === 0) { rungs.push(x1, y, z1, -x1, y, -z1); }
    }
    var rg = new THREE.BufferGeometry();
    rg.setAttribute('position', new THREE.Float32BufferAttribute(rungs, 3));
    group.add(new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: 0x9be9ff, transparent: true, opacity: 0.55 })));
    [pa, pb].forEach(function (arr, idx) {
      var lg = new THREE.BufferGeometry().setFromPoints(arr);
      group.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: idx ? 0xb69bff : 0x4df0ff, transparent: true, opacity: 0.9 })));
    });
    var ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.35 });
    var r1 = new THREE.Mesh(new THREE.TorusGeometry(4.6, 0.03, 8, 120), ringMat); r1.rotation.x = 1.2; scene.add(r1);
    var r2 = new THREE.Mesh(new THREE.TorusGeometry(5.4, 0.025, 8, 120), new THREE.MeshBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.35 })); r2.rotation.y = 1.1; scene.add(r2);

    var sp = new Float32Array(150 * 3);
    for (var j = 0; j < 150; j++) { var rr = 5 + Math.random() * 3, th = Math.random() * 6.283, ph = Math.acos(2 * Math.random() - 1);
      sp[j * 3] = rr * Math.sin(ph) * Math.cos(th); sp[j * 3 + 1] = rr * Math.cos(ph); sp[j * 3 + 2] = rr * Math.sin(ph) * Math.sin(th); }
    var sgm = new THREE.BufferGeometry(); sgm.setAttribute('position', new THREE.BufferAttribute(sp, 3));
    var stars = new THREE.Points(sgm, new THREE.PointsMaterial({ color: 0xbff6ff, size: 0.07, transparent: true, opacity: 0.8 }));
    scene.add(stars);

    function resize() {
      var w = box.clientWidth || 320, h = box.clientHeight || 300;
      renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
    }
    resize();
    var ro = (W.ResizeObserver ? new W.ResizeObserver(resize) : null); if (ro) ro.observe(box);

    var api = {
      box: box,
      render: function (t) {
        if (!box.isConnected) return;
        group.rotation.y = t * 0.55 + state.px * 0.8; group.rotation.x = state.py * 0.35;
        r1.rotation.z = t * 0.25; r2.rotation.x = t * 0.2; stars.rotation.y = t * 0.05;
        renderer.render(scene, cam);
      },
      dispose: function () { try { ro && ro.disconnect(); renderer.dispose(); renderer.forceContextLoss(); } catch (e) {} }
    };
    state.hero = api;
    if (reduce) api.render(0);
  }


  /* ---------- Dr. MediBot 3D doctor robot ---------- */
  function buildDoctor(THREE) {
    var g = new THREE.Group();
    function mat(c, e, m, r) { return new THREE.MeshStandardMaterial({ color: c, emissive: e || 0x000000, metalness: m === undefined ? 0.2 : m, roughness: r === undefined ? 0.3 : r }); }
    var white = mat(0xf4fbff, 0x1a2530, 0.15, 0.28), teal = mat(0x00c8ee, 0x006f88, 0.5, 0.25),
        dark = mat(0x0a1626, 0x000000, 0.7, 0.15), red = mat(0xff3d6e, 0x8a1030, 0.2, 0.4),
        silver = mat(0xdfe8f0, 0x000000, 0.95, 0.12);
    var glow = new THREE.MeshBasicMaterial({ color: 0x7ffcff });

    var body = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.5, 1.1, 32), white); body.position.y = -0.55; g.add(body);
    var belly = new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), white); belly.position.y = -1.1; g.add(belly);
    var bh = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.1, 0.06), red); bh.position.set(0, -0.36, 0.56); g.add(bh);
    var bv = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.34, 0.06), red); bv.position.set(0, -0.36, 0.56); g.add(bv);
    var collar = new THREE.Mesh(new THREE.TorusGeometry(0.46, 0.05, 10, 40), teal); collar.rotation.x = Math.PI / 2; collar.position.y = -0.02; g.add(collar);

    var steth = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.032, 10, 36, Math.PI * 1.35), dark); steth.position.set(0, -0.12, 0.3); steth.rotation.set(-0.35, 0, Math.PI * 0.82); g.add(steth);
    var chestPiece = new THREE.Mesh(new THREE.SphereGeometry(0.085, 16, 16), silver); chestPiece.position.set(0.06, -0.62, 0.55); g.add(chestPiece);

    var headG = new THREE.Group(); headG.position.y = 0.6; g.add(headG);
    var head = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 24), white); head.scale.set(1, 0.9, 0.95); headG.add(head);
    var visor = new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 20), dark); visor.scale.set(1.05, 0.62, 0.62); visor.position.set(0, -0.02, 0.2); headG.add(visor);
    var eyes = [];
    [-0.2, 0.2].forEach(function (x) {
      var e = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 16), glow); e.scale.set(1, 1.3, 0.5); e.position.set(x, 0.02, 0.5); headG.add(e); eyes.push(e);
    });
    var mouth = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.035, 0.03), glow); mouth.position.set(0, -0.14, 0.49); headG.add(mouth);
    var band = new THREE.Mesh(new THREE.TorusGeometry(0.58, 0.028, 10, 48), teal); band.rotation.x = Math.PI / 2; band.position.y = 0.3; band.scale.set(1, 0.95, 1); headG.add(band);
    var mirror = new THREE.Mesh(new THREE.CircleGeometry(0.17, 28), silver); mirror.position.set(0, 0.34, 0.56); mirror.rotation.x = -0.45; headG.add(mirror);
    var mrim = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.022, 10, 28), teal); mrim.position.copy(mirror.position); mrim.rotation.copy(mirror.rotation); headG.add(mrim);
    [-1, 1].forEach(function (s) {
      var ear = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 16), teal); ear.scale.set(0.45, 0.8, 0.8); ear.position.set(s * 0.62, 0, 0); headG.add(ear);
    });
    var ant = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.28, 8), silver); ant.position.set(0, 0.7, -0.05); headG.add(ant);
    var tip = new THREE.Mesh(new THREE.SphereGeometry(0.07, 14, 14), glow); tip.position.set(0, 0.88, -0.05); headG.add(tip);

    function arm(side) {
      var pivot = new THREE.Group(); pivot.position.set(side * 0.72, -0.2, 0);
      var upper = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.09, 0.62, 16), white); upper.position.y = -0.31; pivot.add(upper);
      var hand = new THREE.Mesh(new THREE.SphereGeometry(0.135, 16, 16), teal); hand.position.y = -0.68; pivot.add(hand);
      var shoulder = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 16), teal); pivot.add(shoulder);
      g.add(pivot); return pivot;
    }
    var armL = arm(-1), armR = arm(1);
    var ring = new THREE.Mesh(new THREE.TorusGeometry(0.62, 0.02, 8, 60), new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.7 }));
    ring.rotation.x = Math.PI / 2; ring.position.y = -1.55; g.add(ring);
    var ring2 = ring.clone(); ring2.scale.set(1.35, 1.35, 1.35); ring2.material = new THREE.MeshBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.45 }); g.add(ring2);

    var wave = 0, talk = 0, look = { x: 0, y: 0 };
    return {
      group: g,
      update: function (t, o) {
        o = o || {};
        wave += ((o.wave ? 1 : 0) - wave) * 0.08;
        talk += ((o.talking ? 1 : 0) - talk) * 0.25;
        g.position.y = Math.sin(t * 1.6) * 0.07;
        look.x += ((o.lx || 0) - look.x) * 0.08; look.y += ((o.ly || 0) - look.y) * 0.08;
        g.rotation.y = Math.sin(t * 0.6) * 0.18 + look.x * 0.45;
        headG.rotation.y = look.x * 0.3; headG.rotation.x = look.y * 0.22 + Math.sin(t * 1.1) * 0.03;
        armR.rotation.z = -0.22 + wave * (-2.2 + Math.sin(t * 7) * 0.45);
        armL.rotation.z = 0.22 + Math.sin(t * 1.4) * 0.05 - talk * (0.5 + Math.sin(t * 5) * 0.2);
        mouth.scale.y = 1 + talk * Math.abs(Math.sin(t * 15)) * 5; mouth.scale.x = 1 - talk * 0.25 * Math.abs(Math.sin(t * 15));
        var blink = (t % 3.8) < 0.13 ? 0.1 : 1.3; eyes.forEach(function (e) { e.scale.y = blink; });
        tip.scale.setScalar(1 + Math.sin(t * 4) * 0.25 + talk * 0.5);
        ring.rotation.z = t * 0.8; ring2.rotation.z = -t * 0.5;
      }
    };
  }

  function doctorScene(THREE, box, opts) {
    var renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); } catch (e) { return null; }
    renderer.setPixelRatio(Math.min(W.devicePixelRatio || 1, 2));
    box.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    var cam = new THREE.PerspectiveCamera(opts.fov || 38, 1, 0.1, 60); cam.position.set(0, opts.camY || 0, opts.camZ || 6.4);
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    var k = new THREE.PointLight(0x00e5ff, 2.4, 40); k.position.set(3, 3, 5); scene.add(k);
    var f = new THREE.PointLight(0x8b5cf6, 1.8, 40); f.position.set(-4, -1, 4); scene.add(f);
    var doc = buildDoctor(THREE); doc.group.scale.setScalar(opts.scale || 1); doc.group.position.y = opts.y || 0; scene.add(doc.group);
    function resize() { var w = box.clientWidth || 120, h = box.clientHeight || 150; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
    resize();
    var ro = W.ResizeObserver ? new W.ResizeObserver(resize) : null; if (ro) ro.observe(box);
    return {
      box: box,
      render: function (t, o) { if (!box.isConnected) return; doc.update(t, o); renderer.render(scene, cam); },
      dispose: function () { try { ro && ro.disconnect(); renderer.dispose(); renderer.forceContextLoss(); } catch (e) {} }
    };
  }

  function openDoctor() {
    var btn = D.querySelector('.st-key-doc_launch button') || D.querySelector('.st-key-doc_launch_sidebar button');
    if (!btn) {
      var all = D.querySelectorAll('button');
      for (var i = 0; i < all.length; i++) { if (/Ask Dr\. MediBot/.test(all[i].textContent || '')) { btn = all[i]; break; } }
    }
    if (btn) btn.click();
  }

  function attachDoctorLauncher() {
    var THREE = W.THREE; if (!THREE) return;
    var wrap = D.getElementById('ms-doc-launch');
    if (!wrap) {
      wrap = D.createElement('div'); wrap.id = 'ms-doc-launch'; wrap.setAttribute('role', 'button'); wrap.tabIndex = 0;
      wrap.setAttribute('aria-label', 'Ask Dr. MediBot, the 3D hospital assistant');
      wrap.innerHTML = '<div id="ms-doc-bubble">Need help? Ask Dr. MediBot!</div><div id="ms-doc-canvas"></div>';
      wrap.addEventListener('click', openDoctor);
      wrap.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDoctor(); } });
      wrap.addEventListener('pointerenter', function () { state.docHover = true; wrap.classList.add('ms-show'); });
      wrap.addEventListener('pointerleave', function () { state.docHover = false; wrap.classList.remove('ms-show'); });
      B.appendChild(wrap);
      W.setTimeout(function () { wrap.classList.add('ms-show'); }, 1800);
      W.setTimeout(function () { if (!state.docHover) wrap.classList.remove('ms-show'); }, 11000);
    }
    if (state.doc && state.doc.box.isConnected) return;
    if (state.doc) state.doc.dispose();
    state.doc = doctorScene(THREE, D.getElementById('ms-doc-canvas'), { scale: 1, camZ: 6.6, y: 0.1 });
    if (state.doc && reduce) state.doc.render(0, {});
  }

  function attachDoctorStage() {
    var THREE = W.THREE; if (!THREE) return;
    var el = D.querySelector('.ms-doc-stage');
    if (!el) { if (state.stage) { state.stage.dispose(); state.stage = null; } return; }
    if (state.stage && state.stage.box === el && el.querySelector('canvas')) return;
    if (state.stage) { state.stage.dispose(); state.stage = null; }
    var s = doctorScene(THREE, el, { fov: 34, camZ: 6.2, scale: 1.0, y: 0.05 });
    if (s) { state.stage = s; if (reduce) s.render(0, {}); }
  }

  function doctorState(t) {
    return { wave: state.docHover, talking: (W.performance.now() - state.lastMut) < 650 && !!D.querySelector('[role="dialog"]'), lx: state.px, ly: state.py };
  }

  /* ---------- tilt / spotlight / reveal ---------- */
  var TILT = '.hero, [data-testid="stMetric"], .insight-card, .mascot-wrap';
  function bindTilt() {
    D.querySelectorAll(TILT).forEach(function (el) {
      if (el.getAttribute('data-ms-tilt')) return;
      el.setAttribute('data-ms-tilt', '1');
      var max = el.classList.contains('hero') ? 2.2 : 9;
      el.addEventListener('pointermove', function (e) {
        var r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
        el.style.setProperty('--mx', (x * 100).toFixed(1) + '%'); el.style.setProperty('--my', (y * 100).toFixed(1) + '%');
        if (reduce) return;
        el.style.transform = 'perspective(900px) rotateX(' + ((0.5 - y) * max).toFixed(2) + 'deg) rotateY(' + ((x - 0.5) * max * 1.6).toFixed(2) + 'deg) translateZ(14px)';
      });
      el.addEventListener('pointerleave', function () { el.style.transform = ''; });
    });
  }

  var io = W.IntersectionObserver ? new W.IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) { en.target.classList.remove('ms-pre'); en.target.classList.add('ms-in'); io.unobserve(en.target); }
    });
  }, { threshold: 0.08 }) : null;
  function bindReveal() {
    if (!io || reduce) return;
    D.querySelectorAll('[data-testid="stPlotlyChart"], [data-testid="stMetric"], .insight-card, [data-testid="stDataFrame"]').forEach(function (el) {
      if (el.getAttribute('data-ms-rv')) return;
      el.setAttribute('data-ms-rv', '1'); el.classList.add('ms-pre'); io.observe(el);
    });
  }

  /* ---------- global events ---------- */
  var cursor = null, cx = 0, cy = 0, tx = 0, ty = 0;
  D.addEventListener('pointermove', function (e) {
    state.mx = e.clientX / W.innerWidth; state.my = e.clientY / W.innerHeight;
    state.px = state.mx * 2 - 1; state.py = state.my * 2 - 1; tx = e.clientX; ty = e.clientY;
  }, { passive: true });
  function glow() {
    W.requestAnimationFrame(glow);
    if (!cursor) cursor = D.getElementById('ms-cursor'); if (!cursor) return;
    cx += (tx - cx) * 0.14; cy += (ty - cy) * 0.14;
    cursor.style.transform = 'translate3d(' + (cx - 170) + 'px,' + (cy - 170) + 'px,0)';
  }
  D.addEventListener('scroll', function (e) {
    var t = e.target; if (!t || t === D) t = D.scrollingElement || D.documentElement;
    var max = (t.scrollHeight || 0) - (t.clientHeight || 0);
    if (max < 200) return;
    state.scroll = t.scrollTop || 0;
    var bar = D.getElementById('ms-progress'); if (bar) bar.style.transform = 'scaleX(' + Math.min(1, state.scroll / max).toFixed(4) + ')';
  }, true);

  /* ---------- boot ---------- */
  function refresh() { ensureChrome(); attachHero(); attachDoctorLauncher(); attachDoctorStage(); bindTilt(); bindReveal(); }
  var pending = false;
  new W.MutationObserver(function (muts) {
    for (var mi = 0; mi < muts.length; mi++) {
      var tg = muts[mi].target, nd = tg && tg.nodeType === 3 ? tg.parentNode : tg;
      if (nd && nd.closest && nd.closest('[role="dialog"]')) { state.lastMut = W.performance.now(); break; }
    }
    if (pending) return; pending = true;
    W.requestAnimationFrame(function () { pending = false; refresh(); });
  }).observe(B, { childList: true, subtree: true, characterData: true });

  ensureChrome(); bindTilt(); bindReveal(); if (!reduce) glow();
  function ready() { initBackground(W.THREE); attachHero(); attachDoctorLauncher(); attachDoctorStage(); }
  if (W.THREE) { ready(); } else {
    var s = D.createElement('script'); s.src = THREE_URL; s.async = true; s.onload = ready;
    s.onerror = function () { B.style.background = PALETTE.dark.c; };
    D.head.appendChild(s);
  }
  W.__msRefresh = refresh;
})();
"""

st.markdown(MS_CSS, unsafe_allow_html=True)

components.html(
    "<script>(function(){var P=window.parent,D=P.document;"
    "if(P.__msBooted){if(P.__msRefresh){P.__msRefresh();}return;}"
    "P.__msBooted=true;var s=D.createElement('script');s.id='ms-main';"
    "s.textContent=" + json.dumps(MS_JS).replace("</", "<\\/") + ";"
    "D.head.appendChild(s);})();</script>",
    height=0,
)


# ============================================================
# BRANDING
# ============================================================

def render_hospital_brand():
    st.markdown(
        """
        <div class="hospital-brand">
            <div class="hospital-icon">
                <div class="hospital-ring"></div>
                <div class="hospital-ring"></div>
                <div class="hospital-ring"></div>
                <div class="hospital-building">+</div>
            </div>
            <div>
                <div class="brand-name">Medi<span>Scope</span></div>
                <div class="brand-caption">Health analytics platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mascot():
    st.markdown(
        """
        <div class="mascot-wrap">
            <div class="robot">
                <div class="robot-antenna"></div>
                <div class="robot-ear left"></div>
                <div class="robot-ear right"></div>
                <div class="robot-head">
                    <div class="robot-eyes">
                        <div class="robot-eye"></div>
                        <div class="robot-eye"></div>
                    </div>
                    <div class="robot-mouth"></div>
                </div>
                <div class="robot-body"></div>
            </div>
            <div class="robot-name">MediBot</div>
            <div class="robot-description">
                Your hospital data assistant<br>
                Turning data into useful insights
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET LOADING
# ============================================================

def find_dataset():
    base = Path(__file__).resolve().parent

    candidates = [
        base / DATASET_FILENAME,
        base / "data" / DATASET_FILENAME,
        base / "dataset" / DATASET_FILENAME,
        Path.cwd() / DATASET_FILENAME,
        Path.cwd() / "data" / DATASET_FILENAME,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for directory in [base, base / "data", base / "dataset"]:
        if directory.is_dir():
            try:
                matches = [
                    path for path in directory.glob("*.csv")
                    if "hospital" in path.name.lower()
                    and "discharge" in path.name.lower()
                ]
                if matches:
                    return matches[0]
            except OSError:
                pass

    return None


@st.cache_data(show_spinner=False, max_entries=2)
def load_dataset(path_string, file_size, modified_time, sample_rows):
    path = Path(path_string)

    try:
        data = pd.read_csv(
            path,
            nrows=sample_rows,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            path,
            nrows=sample_rows,
            low_memory=False,
            encoding="latin-1",
            on_bad_lines="skip",
        )

    data.columns = [
        re.sub(r"\s+", " ", str(column)).strip()
        for column in data.columns
    ]

    return data.loc[:, ~pd.Index(data.columns).duplicated()].copy()


# ============================================================
# COLUMN DETECTION
# ============================================================

def normalize_column_name(column):
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def find_column(data, possible_names):
    normalized = {
        normalize_column_name(column): column
        for column in data.columns
    }

    for name in possible_names:
        key = normalize_column_name(name)
        if key in normalized:
            return normalized[key]

    for name in possible_names:
        key = normalize_column_name(name)

        if len(key) < 5:
            continue

        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    return None


# ============================================================
# DATA CLEANING
# ============================================================

def parse_numeric(series):
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text_series(series):
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
                "N/A": pd.NA,
            }
        )
        .fillna("Unknown")
        .astype(str)
    )


def standardize_severity(value):
    value = str(value).strip().lower()

    if value in {"1", "minor", "low", "mild"} or "minor" in value:
        return "Minor"

    if value in {"2", "moderate", "medium"} or "moderate" in value:
        return "Moderate"

    if value in {"3", "major", "high", "severe"} or "major" in value:
        return "Major"

    if value in {"4", "extreme", "critical"} or "extreme" in value:
        return "Extreme"

    return "Unknown"


def standardize_payment(value):
    value = str(value).strip().lower()

    if value in {"", "nan", "none", "unknown"}:
        return "Unknown"

    if "medicare" in value:
        return "Medicare"

    if "medicaid" in value:
        return "Medicaid"

    if "private" in value or "commercial" in value:
        return "Private Health Insurance"

    if "self" in value:
        return "Self-Pay"

    if any(
        token in value
        for token in [
            "blue cross",
            "aetna",
            "united healthcare",
            "unitedhealthcare",
            "cigna",
            "healthfirst",
            "empire",
        ]
    ):
        return "Private Health Insurance"

    return "Other"


def standardize_gender(value):
    value = str(value).strip().lower()

    if value in {"m", "male", "man"}:
        return "Male"

    if value in {"f", "female", "woman"}:
        return "Female"

    if value in {"u", "unknown", "other", "nan", "none", ""}:
        return "Unknown"

    return "Other"


def categorize_age(value):
    value = str(value).strip().lower()

    if value in {"", "nan", "none", "unknown", "not available"}:
        return "Unknown"

    if "0 to 17" in value or "0-17" in value:
        return "0 to 17"

    if "18 to 29" in value or "18-29" in value:
        return "18 to 29"

    if "30 to 49" in value or "30-49" in value:
        return "30 to 49"

    if "50 to 69" in value or "50-69" in value:
        return "50 to 69"

    if "70 or older" in value or "70+" in value:
        return "70 or Older"

    match = re.search(r"\d+", value)

    if match:
        age = int(match.group())

        if age <= 17:
            return "0 to 17"
        if age <= 29:
            return "18 to 29"
        if age <= 49:
            return "30 to 49"
        if age <= 69:
            return "50 to 69"
        return "70 or Older"

    return "Unknown"


def categorize_diagnosis(value):
    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "unknown"}:
        return "Unknown"

    value = re.sub(r"\s+", " ", value)

    return value[:62] + "..." if len(value) > 65 else value


def prepare_data(raw_data):
    data = raw_data.copy()

    aliases = {
        "length_of_stay": [
            "Length of Stay",
            "LOS",
            "Length of Stay (Days)",
        ],
        "total_charges": [
            "Total Charges",
            "Charges",
        ],
        "age": [
            "Age Group",
            "Age",
            "Patient Age",
        ],
        "gender": [
            "Gender",
            "Sex",
        ],
        "race": [
            "Race",
            "Patient Race",
        ],
        "severity": [
            "APR Severity of Illness Description",
            "Severity of Illness Description",
            "Severity of Illness",
            "Severity",
        ],
        "diagnosis": [
            "APR DRG Description",
            "APR DRG Code",
            "Diagnosis Description",
            "Diagnosis",
            "Principal Diagnosis Description",
        ],
        "payment": [
            "Payment Typology 1",
            "Payment Type",
            "Payment Typology",
            "Primary Payer",
        ],
        "facility": [
            "Facility Name",
            "Hospital Name",
            "Facility",
        ],
        "county": [
            "Facility County",
            "County",
        ],
        "region": [
            "Health Service Area",
            "Region",
        ],
        "year": [
            "Discharge Year",
            "Year",
        ],
        "date": [
            "Discharge Date",
            "Discharge Date/Time",
        ],
        "record_id": [
            "Record ID",
            "Patient ID",
            "Case ID",
        ],
    }

    source_columns = {
        key: find_column(data, names)
        for key, names in aliases.items()
    }

    original_rows = len(data)

    if source_columns["length_of_stay"]:
        data["LOS_Days"] = parse_numeric(
            data[source_columns["length_of_stay"]]
        )
    else:
        data["LOS_Days"] = np.nan

    if source_columns["total_charges"]:
        data["Total_Charges"] = parse_numeric(
            data[source_columns["total_charges"]]
        )
    else:
        data["Total_Charges"] = np.nan

    if source_columns["age"]:
        data["Age_Group"] = data[
            source_columns["age"]
        ].map(categorize_age)
    else:
        data["Age_Group"] = "Unknown"

    if source_columns["gender"]:
        data["Gender"] = data[
            source_columns["gender"]
        ].map(standardize_gender)
    else:
        data["Gender"] = "Unknown"

    if source_columns["race"]:
        data["Race"] = clean_text_series(
            data[source_columns["race"]]
        )
    else:
        data["Race"] = "Unknown"

    if source_columns["severity"]:
        data["Severity"] = data[
            source_columns["severity"]
        ].map(standardize_severity)
    else:
        data["Severity"] = "Unknown"

    if source_columns["diagnosis"]:
        data["Diagnosis"] = data[
            source_columns["diagnosis"]
        ].map(categorize_diagnosis)
    else:
        data["Diagnosis"] = "Unknown"

    if source_columns["payment"]:
        data["Payment_Type"] = data[
            source_columns["payment"]
        ].map(standardize_payment)
    else:
        data["Payment_Type"] = "Unknown"

    if source_columns["facility"]:
        data["Facility"] = clean_text_series(
            data[source_columns["facility"]]
        )
    else:
        data["Facility"] = "Unknown"

    if source_columns["county"]:
        data["County"] = clean_text_series(
            data[source_columns["county"]]
        )
    else:
        data["County"] = "Unknown"

    if source_columns["region"]:
        data["Region"] = clean_text_series(
            data[source_columns["region"]]
        )
    else:
        data["Region"] = "Unknown"

    if source_columns["year"]:
        data["Discharge_Year"] = parse_numeric(
            data[source_columns["year"]]
        )
    elif source_columns["date"]:
        dates = pd.to_datetime(
            data[source_columns["date"]],
            errors="coerce",
        )
        data["Discharge_Year"] = dates.dt.year
    else:
        data["Discharge_Year"] = np.nan

    if source_columns["record_id"]:
        data["Record_ID"] = clean_text_series(
            data[source_columns["record_id"]]
        )
    else:
        data["Record_ID"] = np.arange(
            1,
            len(data) + 1,
        ).astype(str)

    invalid_los = (
        data["LOS_Days"].isna()
        | (data["LOS_Days"] <= 0)
    )

    invalid_charges = data["Total_Charges"] < 0

    cleaned = data.loc[
        ~invalid_los & ~invalid_charges
    ].copy()

    cleaned["LOS_Days"] = cleaned["LOS_Days"].clip(
        upper=3650
    )

    cleaned["Total_Charges"] = cleaned[
        "Total_Charges"
    ].replace([np.inf, -np.inf], np.nan)

    cleaned["Total_Charges"] = cleaned[
        "Total_Charges"
    ].clip(lower=0)

    cleaned["Age_Group"] = pd.Categorical(
        cleaned["Age_Group"],
        categories=AGE_ORDER,
        ordered=True,
    )

    cleaned["Severity"] = pd.Categorical(
        cleaned["Severity"],
        categories=SEVERITY_ORDER,
        ordered=True,
    )

    cleaned["Payment_Type"] = pd.Categorical(
        cleaned["Payment_Type"],
        categories=PAYMENT_ORDER,
        ordered=True,
    )

    stats = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned),
        "removed_rows": original_rows - len(cleaned),
        "invalid_los_rows": int(invalid_los.sum()),
        "invalid_charge_rows": int(invalid_charges.sum()),
        "source_columns": source_columns,
    }

    return cleaned, stats


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def format_currency(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_days(value):
    try:
        return f"{float(value):,.2f} days"
    except (TypeError, ValueError):
        return "N/A"


def style_plot(fig, height=390):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#dcecff",
            size=11,
        ),
        margin=dict(l=18, r=18, t=45, b=35),
        title_font=dict(size=15, color="#eff9ff"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(
            bgcolor="#10253a",
            font_color="#ffffff",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(140,180,220,.2)",
    )

    fig.update_yaxes(
        gridcolor="rgba(140,180,220,.10)",
        zeroline=False,
        linecolor="rgba(140,180,220,.2)",
    )

    return fig


GRAPH_EXPLANATIONS = {
    "overview_age_distribution": "Shows the number of inpatient discharge records in each patient age group. Each bar represents an age group; taller bars indicate more records. Use it to compare which age groups are most represented in the loaded dataset sample.",
    "overview_payment_pie": "Shows how discharge records are distributed across primary payment categories. Each slice represents a payment category, and its size indicates its share of the displayed records.",
    "overview_los_histogram": "Shows the frequency distribution of length of stay in days. The horizontal axis represents stay duration and the vertical axis represents the number of records in each interval. Taller bars indicate more common stay durations.",
    "overview_severity_los": "Compares the average length of stay across severity-of-illness categories. Each bar represents the mean stay in days for one category. A taller bar indicates a higher average stay in the filtered records.",
    "diagnosis_average_stay": "Compares average length of stay across the displayed diagnosis categories. Each horizontal bar represents the mean number of inpatient days for a diagnosis category. The chart is limited to the selected number of high-volume diagnosis categories.",
    "diagnosis_severity_box": "Shows the distribution of individual length-of-stay values within each severity category. The line inside each box is the median, the box represents the middle 50% of values, and points beyond the whiskers may be outliers. A sample may be used when there are many records.",
    "charges_severity_box": "Compares the distribution of recorded total charges across severity categories. The median, middle 50% of values, spread, and potential outliers can be compared. Charges are recorded amounts and should not be interpreted as the cost of treatment or as a measure of quality.",
    "charges_age_gender": "Compares average recorded healthcare charges across age groups, with separate bars for gender categories. Bar height represents mean charges. Compare bars within and between age groups while considering the number of records in each group.",
    "charges_age_violin": "Shows the distribution and density of recorded charges across age groups. Wider parts indicate charge values where more observations are concentrated. The embedded box helps show the median and spread. A sample may be used for display performance.",
    "facility_discharge_volume": "Ranks the displayed high-volume facilities by the number of discharge records in the filtered dataset. Longer bars indicate more records. This measures volume in the loaded sample, not overall hospital performance or quality.",
    "facility_average_stay": "Compares average length of stay at the facilities selected by discharge volume. Each bar represents the mean stay in days. Differences are descriptive and do not establish differences in hospital quality or case complexity.",
    "county_facility_heatmap": "Shows average length of stay for combinations of county and facility. Each cell represents the mean stay in days; the colour scale indicates its magnitude. Blank cells mean there are no matching records for that combination in the filtered data.",
    "quality_missing_values": "Shows the percentage of missing values in selected dataset columns. Longer bars indicate a larger share of missing entries. This helps identify fields that may need special care during preprocessing.",
    "ai_confusion_matrix": "Compares actual severity categories with the categories predicted by the demonstration machine-learning model. Rows represent actual classes and columns represent predicted classes. Larger values on the diagonal indicate correct classifications; off-diagonal values show misclassifications. This experiment is not validated for clinical use.",
}


# ============================================================
# 3D CHART ENGINE
# Converts every 2D Plotly figure used by MediScope into a
# real 3D scene (extruded bars, solid donuts, 3D box plots,
# KDE violins, voxel heatmaps) with rotate / zoom / hover.
# ============================================================

_CI = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
_CJ = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
_CK = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]

_3D_LIGHT = dict(ambient=0.55, diffuse=0.95, specular=0.45, roughness=0.35, fresnel=0.25)
_3D_LIGHTPOS = dict(x=2500, y=-3500, z=4500)
_CHART_FONT = dict(family="Inter, sans-serif", color="#dcecff", size=11)


def _short(text, limit=24):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _finite(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _fmt3d(value):
    value = _finite(value)
    if abs(value) >= 100000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _trace_color(trace, index):
    for owner in ("marker", "line"):
        part = getattr(trace, owner, None)
        color = getattr(part, "color", None) if part is not None else None
        if isinstance(color, str):
            return color
    return COLORS[index % len(COLORS)]


def cuboid_mesh(boxes, labels=None):
    """boxes: list of (x0, x1, y0, y1, z0, z1). Returns arrays for go.Mesh3d."""
    X, Y, Z, I, J, K, T, B = [], [], [], [], [], [], [], []
    for n, (x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        X += [x0, x1, x1, x0, x0, x1, x1, x0]
        Y += [y0, y0, y1, y1, y0, y0, y1, y1]
        Z += [z0, z0, z0, z0, z1, z1, z1, z1]
        base = 8 * n
        I += [base + a for a in _CI]
        J += [base + a for a in _CJ]
        K += [base + a for a in _CK]
        B += [n] * 8
        T += [labels[n] if labels else ""] * 8
    return dict(x=X, y=Y, z=Z, i=I, j=J, k=K, text=T, box=B)


def donut_slice_mesh(a0, a1, r_in, r_out, z0, z1, offset=(0.0, 0.0)):
    steps = max(3, int(math.ceil(abs(a1 - a0) / (2 * math.pi / 72))) + 1)
    angles = np.linspace(a0, a1, steps)
    ox, oy = offset
    X, Y, Z = [], [], []
    for a in angles:
        c, s = math.cos(a), math.sin(a)
        X += [r_out * c + ox, r_out * c + ox, r_in * c + ox, r_in * c + ox]
        Y += [r_out * s + oy, r_out * s + oy, r_in * s + oy, r_in * s + oy]
        Z += [z1, z0, z1, z0]
    I, J, K = [], [], []

    def tri(a, b, c):
        I.append(a)
        J.append(b)
        K.append(c)

    for m in range(steps - 1):
        ot, ob, it, ib = 4 * m, 4 * m + 1, 4 * m + 2, 4 * m + 3
        ot2, ob2, it2, ib2 = ot + 4, ob + 4, it + 4, ib + 4
        tri(ot, ot2, it2)
        tri(ot, it2, it)
        tri(ob, ib2, ob2)
        tri(ob, ib, ib2)
        tri(ot, ob, ob2)
        tri(ot, ob2, ot2)
        tri(it, it2, ib2)
        tri(it, ib2, ib)
    last = 4 * (steps - 1)
    for base in (0, last):
        tri(base, base + 2, base + 3)
        tri(base, base + 3, base + 1)
    return dict(x=X, y=Y, z=Z, i=I, j=J, k=K)


def kde_curve(values, points=64, clip=(0.5, 99.5)):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 1.0
    if len(v) > 3000:
        v = np.random.default_rng(42).choice(v, 3000, replace=False)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    bandwidth = 0.9 * spread * len(v) ** (-0.2) if spread > 0 else 1.0
    lo, hi = np.percentile(v, clip)
    if hi <= lo:
        lo, hi = v.min() - 1.0, v.max() + 1.0
    grid = np.linspace(lo, hi, points)
    z = (grid[:, None] - v[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (len(v) * bandwidth * math.sqrt(2 * math.pi))
    return grid, density, bandwidth


def box_stats(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= low_fence) & (v <= high_fence)]
    low = inside.min() if len(inside) else q1
    high = inside.max() if len(inside) else q3
    outliers = v[(v < low_fence) | (v > high_fence)]
    return dict(q1=q1, median=med, q3=q3, iqr=iqr, low=low, high=high,
                low_fence=low_fence, high_fence=high_fence,
                outliers=outliers, n=len(v))


def _axis3d(title=None, tickvals=None, ticktext=None, labels=True, rng=None):
    axis = dict(
        showbackground=True,
        backgroundcolor="rgba(8,22,40,0.55)",
        gridcolor="rgba(120,185,235,0.18)",
        zerolinecolor="rgba(120,185,235,0.35)",
        linecolor="rgba(120,185,235,0.35)",
        tickfont=dict(size=9, color="#a9c3de"),
        showticklabels=labels,
        title=dict(text=title or "", font=dict(size=11, color="#8fe3f7")),
    )
    if tickvals is not None:
        axis["tickvals"] = list(tickvals)
        axis["ticktext"] = list(ticktext)
    if rng is not None:
        axis["range"] = list(rng)
    return axis


def _rotation_frames(eye, count=90):
    radius = math.hypot(eye[0], eye[1])
    start = math.atan2(eye[1], eye[0])
    frames = []
    for k in range(count + 1):
        angle = start + 2 * math.pi * k / count
        frames.append(
            go.Frame(
                layout=dict(
                    scene=dict(
                        camera=dict(
                            eye=dict(x=radius * math.cos(angle), y=radius * math.sin(angle), z=eye[2])
                        )
                    )
                )
            )
        )
    return frames


def _finish_3d(fig3, source, scene, eye=(1.7, 1.5, 0.95), height=560):
    title = None
    try:
        title = source.layout.title.text
    except Exception:
        title = None
    scene = dict(scene)
    scene["bgcolor"] = "rgba(0,0,0,0)"
    scene["camera"] = dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=0, r=0, t=52, b=0),
        height=height,
        title=dict(text=title, font=dict(size=15, color="#eff9ff"), x=0.02),
        scene=scene,
        showlegend=False,
        hoverlabel=dict(bgcolor="#10253a", font_color="#ffffff"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.0,
                xanchor="left",
                yanchor="bottom",
                pad=dict(r=6, t=6, b=6),
                showactive=False,
                bgcolor="rgba(0,229,255,0.12)",
                bordercolor="rgba(0,229,255,0.45)",
                font=dict(color="#dffaff", size=11),
                buttons=[
                    dict(
                        label="⟳ Rotate 360°",
                        method="animate",
                        args=[None, dict(frame=dict(duration=45, redraw=True),
                                         transition=dict(duration=0),
                                         fromcurrent=True, mode="immediate")],
                    ),
                    dict(
                        label="❚❚ Stop",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           transition=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
    )
    fig3.frames = _rotation_frames(eye)
    return fig3


def _axis_title(source, which):
    try:
        axis = source.layout.xaxis if which == "x" else source.layout.yaxis
        text = axis.title.text
        return text if text else None
    except Exception:
        return None


def _continuous_scale(source):
    try:
        scale = source.layout.coloraxis.colorscale
        if scale:
            return [list(item) for item in scale]
    except Exception:
        pass
    return "Viridis"


def _mesh_trace(mesh, color=None, intensity=None, colorscale=None, cbar_title=None,
                opacity=0.96, cmin=None, cmax=None):
    kwargs = dict(
        x=mesh["x"], y=mesh["y"], z=mesh["z"], i=mesh["i"], j=mesh["j"], k=mesh["k"],
        flatshading=True, lighting=_3D_LIGHT, lightposition=_3D_LIGHTPOS, opacity=opacity,
    )
    if "text" in mesh:
        kwargs["hovertext"] = mesh["text"]
        kwargs["hoverinfo"] = "text"
    else:
        kwargs["hoverinfo"] = "skip"
    if intensity is not None:
        kwargs.update(
            intensity=intensity, colorscale=colorscale or "Viridis", showscale=cbar_title is not None,
            cmin=cmin, cmax=cmax,
        )
        if cbar_title is not None:
            kwargs["colorbar"] = dict(
                title=dict(text=cbar_title, font=dict(size=10, color="#bcd4ea")),
                thickness=11, len=0.55, x=1.0,
                tickfont=dict(size=9, color="#bcd4ea"),
            )
    else:
        kwargs["color"] = color or COLORS[0]
    return go.Mesh3d(**kwargs)


def _bars_3d(source):
    traces = list(source.data)
    horizontal = getattr(traces[0], "orientation", None) == "h"

    def pairs(trace):
        cats = list(trace.y if horizontal else trace.x)
        vals = list(trace.x if horizontal else trace.y)
        return [str(c) for c in cats], vals

    axis = source.layout.yaxis if horizontal else source.layout.xaxis
    seen = []
    for trace in traces:
        for c in pairs(trace)[0]:
            if c not in seen:
                seen.append(c)
    order = [str(c) for c in axis.categoryarray] if axis.categoryarray is not None else []
    cats = [c for c in order if c in seen] + [c for c in seen if c not in order]
    index = {c: n for n, c in enumerate(cats)}

    grouped = len(traces) > 1 and max(len(pairs(t)[0]) for t in traces) > 1
    width = 0.62

    marker_color = traces[0].marker.color if len(traces) == 1 else None
    continuous = False
    if marker_color is not None and not isinstance(marker_color, str):
        try:
            numeric = np.asarray(marker_color, dtype=float)
            continuous = numeric.ndim == 1 and len(numeric) == len(pairs(traces[0])[0])
        except (TypeError, ValueError):
            continuous = False

    fig3 = go.Figure()
    all_boxes, all_labels, all_vals = [], [], []
    label_x, label_y, label_z, label_t = [], [], [], []
    for ti, trace in enumerate(traces):
        names, values = pairs(trace)
        boxes, labels, vals = [], [], []
        for name, value in zip(names, values):
            height = _finite(value)
            cat_pos = index[name]
            depth = ti if grouped else 0
            c0, c1 = cat_pos - width / 2, cat_pos + width / 2
            d0, d1 = depth - width / 2, depth + width / 2
            boxes.append((d0, d1, c0, c1, 0, height) if horizontal else (c0, c1, d0, d1, 0, height))
            series = f"{trace.name}: " if grouped and trace.name else ""
            labels.append(f"<b>{name}</b><br>{series}{_fmt3d(height)}")
            vals.append(height)
            label_x.append(depth if horizontal else cat_pos)
            label_y.append(cat_pos if horizontal else depth)
            label_z.append(height)
            label_t.append(_fmt3d(height))
        if continuous:
            all_boxes += boxes
            all_labels += labels
            all_vals += vals
        else:
            mesh = cuboid_mesh(boxes, labels)
            fig3.add_trace(_mesh_trace(mesh, color=_trace_color(trace, ti)))

    if continuous:
        mesh = cuboid_mesh(all_boxes, all_labels)
        intensity = [all_vals[b] for b in mesh["box"]]
        title = _axis_title(source, "x" if horizontal else "y") or "Value"
        fig3.add_trace(
            _mesh_trace(mesh, intensity=intensity, colorscale=_continuous_scale(source),
                        cbar_title=title, cmin=min(all_vals), cmax=max(all_vals))
        )

    if len(label_t) <= 40:
        fig3.add_trace(
            go.Scatter3d(
                x=label_x, y=label_y, z=label_z, mode="text", text=label_t,
                textposition="top center", textfont=dict(size=10, color="#e8f6ff"),
                hoverinfo="skip",
            )
        )

    cat_title = _axis_title(source, "y" if horizontal else "x")
    val_title = _axis_title(source, "x" if horizontal else "y")
    tick_text = [_short(c) for c in cats]
    series_names = [str(t.name) for t in traces]
    depth_count = len(traces) if grouped else 1
    depth_axis = (
        _axis3d(None, range(depth_count), series_names, True, [-0.8, depth_count - 0.2])
        if grouped
        else _axis3d(None, [], [], False, [-1, 1])
    )
    cat_axis = _axis3d(cat_title, range(len(cats)), tick_text, True, [-0.8, len(cats) - 0.2])
    z_axis = _axis3d(val_title or "Value")
    cat_len = min(max(len(cats) * 0.42, 1.5), 3.4)
    depth_len = max(depth_count * 0.55, 0.9)
    if horizontal:
        scene = dict(xaxis=depth_axis, yaxis=cat_axis, zaxis=z_axis,
                     aspectmode="manual", aspectratio=dict(x=depth_len, y=cat_len, z=1.0))
    else:
        scene = dict(xaxis=cat_axis, yaxis=depth_axis, zaxis=z_axis,
                     aspectmode="manual", aspectratio=dict(x=cat_len, y=depth_len, z=1.0))
    return _finish_3d(fig3, source, scene, height=max(560, int(_finite(source.layout.height, 420)) + 120))


def _hist_3d(source):
    trace = source.data[0]
    values = np.asarray(trace.x, dtype=float)
    values = values[np.isfinite(values)]
    bins = int(trace.nbinsx) if getattr(trace, "nbinsx", None) else 30
    counts, edges = np.histogram(values, bins=bins)
    gap = (edges[1] - edges[0]) * 0.08
    boxes, labels = [], []
    for n, count in enumerate(counts):
        boxes.append((edges[n] + gap, edges[n + 1] - gap, -0.5, 0.5, 0, float(count)))
        labels.append(f"<b>{edges[n]:,.1f} – {edges[n + 1]:,.1f}</b><br>Records: {int(count):,}")
    mesh = cuboid_mesh(boxes, labels)
    intensity = [float(counts[b]) for b in mesh["box"]]
    scale = [[0, "#1b2a6b"], [0.45, "#0088d9"], [0.75, "#00E5FF"], [1, "#9dfff0"]]
    fig3 = go.Figure(
        _mesh_trace(mesh, intensity=intensity, colorscale=scale, cbar_title="Records",
                    cmin=0, cmax=float(max(counts.max(), 1)))
    )
    span = float(edges[-1] - edges[0]) or 1.0
    scene = dict(
        xaxis=_axis3d(_axis_title(source, "x") or "Value"),
        yaxis=_axis3d(None, [], [], False, [-1.5, 1.5]),
        zaxis=_axis3d(_axis_title(source, "y") or "Records"),
        aspectmode="manual", aspectratio=dict(x=3.0, y=0.8, z=1.0),
    )
    return _finish_3d(fig3, source, scene, eye=(1.5, 1.9, 0.85), height=560)


def _pie_3d(source):
    trace = source.data[0]
    labels = [str(v) for v in trace.labels]
    values = np.array([_finite(v) for v in trace.values], dtype=float)
    total = values.sum() or 1.0
    hole = _finite(getattr(trace, "hole", 0.0))
    r_in = max(hole, 0.05)
    colors = list(trace.marker.colors) if trace.marker and trace.marker.colors is not None else []
    height = 0.32
    fig3 = go.Figure()
    cumulative = 0.0
    lx, ly, lz, lt = [], [], [], []
    for n, (label, value) in enumerate(zip(labels, values)):
        fraction = value / total
        if fraction <= 0:
            continue
        a0 = math.pi / 2 + cumulative * 2 * math.pi
        a1 = math.pi / 2 + (cumulative + fraction) * 2 * math.pi
        mid = (a0 + a1) / 2
        offset = (0.05 * math.cos(mid), 0.05 * math.sin(mid))
        mesh = donut_slice_mesh(a0, a1, r_in, 1.0, 0.0, height, offset)
        hover = f"<b>{label}</b><br>{int(value):,} records<br>{fraction * 100:.1f}%"
        mesh["text"] = [hover] * len(mesh["x"])
        color = colors[n] if n < len(colors) and isinstance(colors[n], str) else COLORS[n % len(COLORS)]
        fig3.add_trace(_mesh_trace(mesh, color=color, opacity=0.97))
        radius = 1.0 + 0.2
        lx.append(radius * math.cos(mid) + offset[0])
        ly.append(radius * math.sin(mid) + offset[1])
        lz.append(height + 0.05)
        lt.append(f"{_short(label, 18)}<br>{fraction * 100:.1f}%" if fraction > 0.025 else "")
        cumulative += fraction
    fig3.add_trace(
        go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt, textfont=dict(size=10, color="#e8f6ff"),
                     hoverinfo="skip")
    )
    hidden = dict(visible=False, showbackground=False)
    scene = dict(
        xaxis=dict(hidden, range=[-1.6, 1.6]), yaxis=dict(hidden, range=[-1.6, 1.6]),
        zaxis=dict(hidden, range=[-0.1, 0.6]),
        aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.42),
    )
    return _finish_3d(fig3, source, scene, eye=(0.2, -1.55, 1.25), height=560)


def _box_3d(source):
    traces = list(source.data)
    fig3 = go.Figure()
    names = []
    wx, wy, wz = [], [], []
    ox, oy, oz = [], [], []
    rng = np.random.default_rng(7)
    all_max = 0.0
    for ti, trace in enumerate(traces):
        stats = box_stats(trace.y)
        name = str(trace.name) if trace.name else (str(trace.x[0]) if trace.x is not None and len(trace.x) else str(ti))
        names.append(name)
        if stats is None:
            continue
        color = _trace_color(trace, ti)
        w = 0.34
        hover = (f"<b>{name}</b><br>Q1: {_fmt3d(stats['q1'])}<br>Median: {_fmt3d(stats['median'])}"
                 f"<br>Q3: {_fmt3d(stats['q3'])}<br>IQR: {_fmt3d(stats['iqr'])}<br>n = {stats['n']:,}")
        body = cuboid_mesh([(ti - w, ti + w, -w, w, stats["q1"], stats["q3"])], [hover])
        fig3.add_trace(_mesh_trace(body, color=color, opacity=0.88))
        eps = max(stats["iqr"], 1e-9) * 0.015
        med = cuboid_mesh([(ti - w - 0.04, ti + w + 0.04, -w - 0.04, w + 0.04,
                            stats["median"] - eps, stats["median"] + eps)], [hover])
        fig3.add_trace(_mesh_trace(med, color="#ffffff", opacity=1.0))
        wx += [ti, ti, None, ti, ti, None, ti - 0.14, ti + 0.14, None, ti - 0.14, ti + 0.14, None]
        wy += [0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None]
        wz += [stats["low"], stats["q1"], None, stats["q3"], stats["high"], None,
               stats["low"], stats["low"], None, stats["high"], stats["high"], None]
        out = stats["outliers"]
        if len(out) > 120:
            out = rng.choice(out, 120, replace=False)
        ox += [ti + rng.uniform(-0.18, 0.18) for _ in out]
        oy += [rng.uniform(-0.18, 0.18) for _ in out]
        oz += [float(v) for v in out]
        all_max = max(all_max, stats["high"], float(np.max(out)) if len(out) else 0.0)
    fig3.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="lines", line=dict(color="#cfe9ff", width=4),
                                hoverinfo="skip"))
    if ox:
        fig3.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode="markers",
                                    marker=dict(size=2.4, color="#ffb300", opacity=0.75),
                                    hovertemplate="Outlier: %{z:,.2f}<extra></extra>"))
    scene = dict(
        xaxis=_axis3d(_axis_title(source, "x"), range(len(names)), [_short(n) for n in names], True,
                      [-0.7, len(names) - 0.3]),
        yaxis=_axis3d(None, [], [], False, [-1, 1]),
        zaxis=_axis3d(_axis_title(source, "y") or "Value"),
        aspectmode="manual", aspectratio=dict(x=min(max(len(names) * 0.55, 1.6), 3.4), y=0.9, z=1.0),
    )
    return _finish_3d(fig3, source, scene, height=580)


def _violin_3d(source):
    traces = list(source.data)
    fig3 = go.Figure()
    names = []
    angles = np.linspace(0, 2 * math.pi, 19)[:-1]
    for ti, trace in enumerate(traces):
        name = str(trace.name) if trace.name else str(ti)
        names.append(name)
        grid, density, bandwidth = kde_curve(trace.y)
        peak = density.max() if density.max() > 0 else 1.0
        half = 0.42 * density / peak
        rings = len(grid)
        X, Y, Z = [], [], []
        for w, g in zip(half, grid):
            X += [ti + w * math.cos(a) for a in angles]
            Y += [w * math.sin(a) for a in angles]
            Z += [float(g)] * len(angles)
        n_ang = len(angles)
        I, J, K = [], [], []
        for r in range(rings - 1):
            for a in range(n_ang):
                a2 = (a + 1) % n_ang
                p, q = r * n_ang + a, r * n_ang + a2
                p2, q2 = (r + 1) * n_ang + a, (r + 1) * n_ang + a2
                I += [p, p]
                J += [q, q2]
                K += [q2, p2]
        for ring_index in (0, rings - 1):
            center = len(X)
            X.append(float(ti))
            Y.append(0.0)
            Z.append(float(grid[ring_index]))
            for a in range(n_ang):
                a2 = (a + 1) % n_ang
                I.append(center)
                J.append(ring_index * n_ang + a)
                K.append(ring_index * n_ang + a2)
        hover = f"<b>{name}</b><br>KDE bandwidth h = {bandwidth:,.3f}"
        mesh = dict(x=X, y=Y, z=Z, i=I, j=J, k=K, text=[hover] * len(X))
        fig3.add_trace(_mesh_trace(mesh, color=_trace_color(trace, ti), opacity=0.92))
        values = np.asarray(trace.y, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            fig3.add_trace(go.Scatter3d(x=[ti], y=[0], z=[float(np.median(values))], mode="markers",
                                        marker=dict(size=5, color="#ffffff"),
                                        hovertemplate=f"{name} median: %{{z:,.2f}}<extra></extra>"))
    scene = dict(
        xaxis=_axis3d(_axis_title(source, "x"), range(len(names)), [_short(n) for n in names], True,
                      [-0.7, len(names) - 0.3]),
        yaxis=_axis3d(None, [], [], False, [-1, 1]),
        zaxis=_axis3d(_axis_title(source, "y") or "Value"),
        aspectmode="manual", aspectratio=dict(x=min(max(len(names) * 0.55, 1.6), 3.4), y=0.9, z=1.0),
    )
    return _finish_3d(fig3, source, scene, height=580)


def _heat_3d(source):
    trace = source.data[0]
    z = np.array(trace.z, dtype=float)
    if z.ndim != 2:
        raise ValueError("heatmap needs a 2D matrix")
    rows, cols = z.shape
    xs = [str(v) for v in trace.x] if trace.x is not None else [str(c) for c in range(cols)]
    ys = [str(v) for v in trace.y] if trace.y is not None else [str(r) for r in range(rows)]
    boxes, labels, vals = [], [], []
    tx, ty, tz, tt = [], [], [], []
    for r in range(rows):
        for c in range(cols):
            value = z[r, c]
            if not np.isfinite(value):
                continue
            boxes.append((c - 0.4, c + 0.4, r - 0.4, r + 0.4, 0, float(value)))
            labels.append(f"<b>{ys[r]}</b><br>{xs[c]}<br>Value: {_fmt3d(value)}")
            vals.append(float(value))
            tx.append(c)
            ty.append(r)
            tz.append(float(value))
            tt.append(_fmt3d(value))
    if not boxes:
        raise ValueError("empty heatmap")
    mesh = cuboid_mesh(boxes, labels)
    intensity = [vals[b] for b in mesh["box"]]
    title = None
    try:
        title = source.layout.coloraxis.colorbar.title.text
    except Exception:
        title = None
    fig3 = go.Figure(
        _mesh_trace(mesh, intensity=intensity, colorscale=_continuous_scale(source),
                    cbar_title=title or "Value", cmin=min(vals), cmax=max(vals))
    )
    if len(tt) <= 36:
        fig3.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="text", text=tt, textposition="top center",
                                    textfont=dict(size=10, color="#ffffff"), hoverinfo="skip"))
    scene = dict(
        xaxis=_axis3d(_axis_title(source, "x"), range(cols), [_short(x, 16) for x in xs], True,
                      [-0.7, cols - 0.3]),
        yaxis=_axis3d(_axis_title(source, "y"), range(rows), [_short(y, 16) for y in ys], True,
                      [-0.7, rows - 0.3]),
        zaxis=_axis3d(title or "Value"),
        aspectmode="manual",
        aspectratio=dict(x=min(max(cols * 0.5, 1.5), 3.4), y=min(max(rows * 0.5, 1.2), 3.0), z=1.0),
    )
    return _finish_3d(fig3, source, scene, eye=(1.6, -1.7, 1.0), height=600)


def to_3d(fig):
    """Return a 3D version of a supported 2D figure, or the original if unsupported."""
    try:
        kinds = {trace.type for trace in fig.data}
        if not kinds:
            return fig
        if kinds <= {"bar"}:
            return _bars_3d(fig)
        if kinds <= {"histogram"}:
            return _hist_3d(fig)
        if kinds == {"pie"}:
            return _pie_3d(fig)
        if kinds <= {"box"}:
            return _box_3d(fig)
        if kinds <= {"violin"}:
            return _violin_3d(fig)
        if kinds <= {"heatmap"}:
            return _heat_3d(fig)
    except Exception:
        return fig
    return fig


CHART_3D = True


def show_chart(fig, key, force_3d=False):
    explanation = GRAPH_EXPLANATIONS.get(
        key,
        "This chart visualizes a summary of the currently filtered hospital discharge records. Read the axis labels, units, legend, and hover information to understand the variables and values being displayed.",
    )

    shown = to_3d(fig) if (force_3d or CHART_3D) else fig
    if shown is not fig:
        explanation += (
            " In 3D view: drag to rotate, scroll to zoom, hover any shape for its exact "
            "value, and press Rotate 360 for an automatic turn."
        )

    _, help_column = st.columns([12, 1])
    with help_column:
        with st.popover("?", help="Explain this graph"):
            st.markdown("#### Graph explanation")
            st.write(explanation)

    st.plotly_chart(
        shown,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def insight_card(title, value, description):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{html.escape(str(title))}</div>
            <div class="insight-value">{html.escape(str(value))}</div>
            <div class="insight-description">{html.escape(str(description))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_values(data, column, limit=15):
    return (
        data[column]
        .astype(str)
        .value_counts()
        .head(limit)
        .index.tolist()
    )


# ============================================================
# DR. MEDIBOT - 3D DOCTOR ROBOT ASSISTANT (chat engine)
# Answers questions about the app, the charts and the live data.
# It runs fully offline: no API key or internet connection needed.
# ============================================================

import difflib

DOC_GREETING = (
    "Hello, I'm **Dr. MediBot** 🩺, your MediScope assistant.\n\n"
    "Ask me anything about this web app: what a page or chart shows, how to use the filters "
    "or the 3D controls, how a number is calculated, or what the data says right now "
    "(for example *\"What is the average length of stay?\"*)."
)

DOC_QUICK_QUESTIONS = [
    "What does this page show?",
    "What is the average length of stay?",
    "Which age group has the most discharges?",
    "How do I rotate the 3D charts?",
    "Explain the box plot",
    "How is accuracy calculated?",
]

PAGE_INFO = {
    "Overview": "The **Overview** page shows the headline numbers (records, average stay, average charges, facilities, median stay, total charges, counties) and four graphs: patients by age group, payment type, the length-of-stay distribution and average stay by severity, plus quick findings.",
    "Diagnosis & Stay": "The **Diagnosis & Stay** page compares average length of stay across diagnosis categories (use the slider to choose how many), shows how stay varies by severity in a box plot, and gives a downloadable diagnosis summary table.",
    "Charges & Demographics": "The **Charges & Demographics** page shows average, median and total charges, charges by severity (box plot), average charges by age group and gender, the charge distribution by age (violin) and a demographic summary table.",
    "Facilities & Counties": "The **Facilities & Counties** page ranks hospitals by discharge volume and by average stay, shows a county-by-facility heatmap of average stay, and gives a downloadable facility summary table.",
    "Data Quality": "The **Data Quality** page shows how many rows were cleaned away, the preprocessing steps, which source columns were detected, a missing-value graph, length-of-stay outliers and a cleaned data preview.",
    "Mathematical Analysis": "The **Mathematical Analysis** page recalculates mean, median, standard deviation, payment percentages, severity comparisons and the Pearson correlation from your current filters.",
    "Math Behind the Graphs": "The **Math Behind the Graphs** page walks through 10 steps showing the formula, the live numbers and the 3D graph for every calculation behind the dashboard.",
    "AI Experiment": "The **AI Experiment** page trains a demonstration model (Random Forest or a baseline) to predict severity, then shows accuracy, a confusion matrix and a classification report. It is educational only, not for clinical use.",
    "Data Explorer": "The **Data Explorer** page lets you search records, narrow them by length of stay and charges, view the table and a numeric summary, and download the result as CSV.",
}

_MEDICAL_PATTERN = re.compile(
    r"\b(should i take|what medicine|which medicine|which drug|dosage|dose of|i have (a )?(pain|fever|cough|headache)|"
    r"my symptoms?|diagnose me|am i sick|treat my|cure for|prescri(be|ption) for|is it safe to take)\b",
    re.IGNORECASE,
)


def _doc_frame(ctx):
    return ctx["f"] if ctx.get("f") is not None and len(ctx["f"]) else None


def _a_records(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    return (f"There are **{len(f):,} discharge records** in the current filtered view, out of "
            f"{len(ctx['df']):,} cleaned records in the loaded sample.")


def _a_avg_los(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    return (f"The **average length of stay is {f['LOS_Days'].mean():,.2f} days** and the **median is "
            f"{f['LOS_Days'].median():,.2f} days** for the current filters. When the mean is higher "
            "than the median, a few long stays are pulling the average up.")


def _a_avg_charges(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    c = f["Total_Charges"].dropna()
    if c.empty:
        return "There are no valid charge values for the current filters."
    return (f"Average recorded charges are **${c.mean():,.2f}** (median ${c.median():,.2f}), and the total of "
            f"all recorded charges is **${c.sum():,.2f}**. Charges are recorded amounts, not the cost of treatment.")


def _a_top_age(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    vc = f["Age_Group"].astype(str).value_counts()
    return (f"**{vc.index[0]}** is the most represented age group with **{int(vc.iloc[0]):,} records** "
            f"({vc.iloc[0] / len(f) * 100:.1f}% of the filtered records).")


def _a_top_payment(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    vc = f["Payment_Type"].astype(str).value_counts()
    return (f"The most frequent payment category is **{vc.index[0]}** ({vc.iloc[0] / len(f) * 100:.1f}% "
            f"of records), followed by **{vc.index[1]}**." if len(vc) > 1 else
            f"The only payment category is **{vc.index[0]}**.")


def _a_top_facility(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    vc = f["Facility"].astype(str).value_counts()
    return (f"**{vc.index[0]}** has the most discharge records (**{int(vc.iloc[0]):,}**) in the loaded sample. "
            "This measures volume only, not quality.")


def _a_long_facility(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    top = f["Facility"].astype(str).value_counts().head(15).index
    g = f[f["Facility"].astype(str).isin(top)].groupby("Facility", observed=False)["LOS_Days"].mean().sort_values()
    return (f"Among the 15 highest-volume facilities, **{g.index[-1]}** has the longest average stay "
            f"(**{g.iloc[-1]:.2f} days**) and **{g.index[0]}** the shortest (**{g.iloc[0]:.2f} days**). "
            "Differences are descriptive and do not show hospital quality.")


def _a_severity_los(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    g = f.groupby("Severity", observed=False)["LOS_Days"].mean().dropna()
    if g.empty:
        return "There is no severity data for the current filters."
    parts = ", ".join(f"{str(k)}: {v:.2f} days" for k, v in g.items())
    return f"Average stay by severity, from your current filters: {parts}. The longest is **{g.idxmax()}**."


def _a_severity_charges(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    g = f.groupby("Severity", observed=False)["Total_Charges"].mean().dropna()
    if g.empty:
        return "There are no valid charge values for the current filters."
    parts = ", ".join(f"{str(k)}: ${v:,.0f}" for k, v in g.items())
    return f"Average charges by severity: {parts}. The highest is **{g.idxmax()}**."


def _a_top_diagnosis(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    d = f[f["Diagnosis"] != "Unknown"]
    if d.empty:
        return "No diagnosis categories are available for these filters."
    top = d["Diagnosis"].value_counts().head(15).index
    g = d[d["Diagnosis"].isin(top)].groupby("Diagnosis")["LOS_Days"].mean().sort_values()
    return (f"Among the 15 most common diagnosis categories, **{g.index[-1]}** has the longest average stay "
            f"(**{g.iloc[-1]:.2f} days**). The most common category overall is **{d['Diagnosis'].value_counts().index[0]}**.")


def _a_gender(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    vc = f["Gender"].astype(str).value_counts()
    return "Gender split in the filtered records: " + ", ".join(
        f"**{k}** {v / len(f) * 100:.1f}% ({int(v):,})" for k, v in vc.items()) + "."


def _a_counties(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    return (f"The filtered data covers **{f['Facility'].nunique():,} facilities** in "
            f"**{f['County'].nunique():,} counties**.")


def _a_corr(ctx):
    f = _doc_frame(ctx)
    if f is None:
        return "No records match the current filters."
    p = f[["LOS_Days", "Total_Charges"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(p) < 3 or p["LOS_Days"].nunique() < 2 or p["Total_Charges"].nunique() < 2:
        return "There are not enough paired values to calculate a correlation."
    r = p["LOS_Days"].corr(p["Total_Charges"])
    strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
    direction = "positive" if r > 0 else "negative"
    return (f"The Pearson correlation between stay and charges is **r = {r:.3f}**, a {strength} {direction} linear "
            "association. Correlation does not prove causation.")


def _a_cleaning(ctx):
    s = ctx["stats"]
    return (f"The loaded sample had **{s['original_rows']:,} rows**. **{s['removed_rows']:,}** "
            f"({s['removed_rows'] / max(s['original_rows'], 1) * 100:.2f}%) were removed during cleaning "
            f"(missing or non-positive length of stay: {s['invalid_los_rows']:,}; negative charges: "
            f"{s['invalid_charge_rows']:,}), leaving **{s['cleaned_rows']:,}**.")


def _a_missing(ctx):
    m = ctx["df"].isna().sum()
    m = (m / max(len(ctx["df"]), 1) * 100).sort_values(ascending=False)
    if m.iloc[0] == 0:
        return "No empty cells were found in the processed columns."
    return "The columns with the most missing values are " + ", ".join(
        f"**{k}** ({v:.1f}%)" for k, v in m.head(3).items()) + ". See the Data Quality page for the full graph."


def _a_this_page(ctx):
    return PAGE_INFO.get(ctx["page"], "Use the sidebar to choose a page.")


def _a_all_pages(ctx):
    return "MediScope has these pages:\n\n" + "\n".join(f"- {v}" for v in PAGE_INFO.values())


def _a_ai(ctx):
    r = ctx.get("ai")
    base = ("The **AI Experiment** page trains a model to predict the severity of illness from age group, gender, "
            "payment type, diagnosis and other features. Pick features, a model and a test percentage, then press "
            "*Train and evaluate model*. It is educational and **not validated for clinical use**.")
    if r:
        base += (f"\n\nYour last trained model ({r['model_type']}) reached **{r['accuracy'] * 100:.2f}% test "
                 f"accuracy** on {r['testing_rows']:,} test records.")
    return base


def _graph(key):
    return lambda ctx: GRAPH_EXPLANATIONS[key] + "\n\nIn 3D, drag to rotate and hover for exact values."


DOC_KB = [
    {"k": ["hello", "hi there", "hey", "good morning", "good evening"], "a": "Hello! How can I help you with MediScope today? Ask about a chart, a number, a page or the 3D controls."},
    {"k": ["thank", "thanks", "great job", "awesome"], "a": "You're welcome! Ask me anything else about the dashboard."},
    {"k": ["who are you", "your name", "what are you", "are you a real doctor", "are you ai"],
     "a": "I'm **Dr. MediBot**, the 3D assistant of MediScope. I'm a data helper, **not a medical doctor**: I explain this dashboard and its numbers, and I can't give medical advice."},
    {"k": ["what can you do", "help me", "how can you help", "what can i ask", "assist"],
     "a": "I can:\n- explain any **page** or **graph**\n- tell you **live numbers** from your filtered data (records, average stay, charges, top facility, correlation...)\n- explain **terms** like length of stay, median, IQR, severity, correlation\n- show **how to use** filters, downloads, 3D controls and the theme button\n- explain the **maths** and the **AI experiment**"},
    {"k": ["this page", "current page", "which page", "where am i", "what does this page", "what is on this page"], "a": _a_this_page},
    {"k": ["all pages", "list pages", "what pages", "navigation", "menu", "sections"], "a": _a_all_pages},
    {"k": ["overview page"], "a": lambda c: PAGE_INFO["Overview"]},
    {"k": ["diagnosis page", "diagnosis & stay"], "a": lambda c: PAGE_INFO["Diagnosis & Stay"]},
    {"k": ["charges page", "demographics page"], "a": lambda c: PAGE_INFO["Charges & Demographics"]},
    {"k": ["facilities page", "counties page"], "a": lambda c: PAGE_INFO["Facilities & Counties"]},
    {"k": ["data explorer", "explorer page", "search records", "search the data"],
     "a": lambda c: PAGE_INFO["Data Explorer"] + "\n\nType a facility, diagnosis, county, region or record ID in the search box, then use the stay slider and the maximum-charges box."},
    {"k": ["math behind", "how are graphs calculated", "how things are calculated", "calculation steps", "maths page"], "a": lambda c: PAGE_INFO["Math Behind the Graphs"]},
    {"k": ["mathematical analysis"], "a": lambda c: PAGE_INFO["Mathematical Analysis"]},
    {"k": ["how many records", "number of records", "total records", "how many discharges", "record count", "discharge records"], "a": _a_records},
    {"k": ["average length of stay", "average stay", "mean stay", "avg los", "average los", "how long do patients stay", "median stay", "mean length"], "a": _a_avg_los},
    {"k": ["average charges", "average cost", "mean charges", "total charges", "how much do patients pay", "average bill", "total cost"], "a": _a_avg_charges},
    {"k": ["most discharges age", "age group most", "which age group", "most common age", "age distribution", "oldest patients", "age with most"], "a": _a_top_age},
    {"k": ["payment type", "most common payment", "who pays", "insurance most", "medicare share", "payer"], "a": _a_top_payment},
    {"k": ["facility most", "busiest hospital", "which hospital", "most discharge records", "highest volume", "largest hospital", "top facility", "top hospital"], "a": _a_top_facility},
    {"k": ["longest average stay facility", "facility longest stay", "hospital longest stay", "shortest stay facility"], "a": _a_long_facility},
    {"k": ["severity longest", "stay by severity", "severity stay", "which severity", "severity average stay"], "a": _a_severity_los},
    {"k": ["severity charges", "charges by severity", "most expensive severity"], "a": _a_severity_charges},
    {"k": ["diagnosis longest", "which diagnosis", "top diagnosis", "common diagnosis", "diagnosis stay"], "a": _a_top_diagnosis},
    {"k": ["gender split", "male female", "how many men", "how many women", "gender share"], "a": _a_gender},
    {"k": ["how many counties", "how many facilities", "counties represented", "facilities represented"], "a": _a_counties},
    {"k": ["correlation value", "correlation between stay and charges", "pearson", "are charges related to stay", "relationship stay charges"], "a": _a_corr},
    {"k": ["rows removed", "cleaned rows", "how much data was removed", "removed during cleaning", "invalid records", "data cleaning"], "a": _a_cleaning},
    {"k": ["missing values", "missing data", "empty cells", "null values", "incomplete data"], "a": _a_missing},
    {"k": ["age bar", "age graph", "age chart", "patients by age"], "a": _graph("overview_age_distribution")},
    {"k": ["payment pie", "payment donut", "payment chart", "donut", "pie chart"], "a": _graph("overview_payment_pie")},
    {"k": ["histogram", "length of stay distribution", "stay distribution"], "a": _graph("overview_los_histogram")},
    {"k": ["box plot", "boxplot", "whisker", "quartile", "iqr", "interquartile", "outlier"],
     "a": lambda c: _graph("diagnosis_severity_box")(c) + "\n\nMath: IQR = Q3 − Q1; whiskers stop at the last value within 1.5 × IQR of the box. See *Math Behind the Graphs*, step 5."},
    {"k": ["charges box", "charges by severity graph"], "a": _graph("charges_severity_box")},
    {"k": ["age and gender", "gender chart", "grouped bar"], "a": _graph("charges_age_gender")},
    {"k": ["violin", "density plot", "kde"],
     "a": lambda c: _graph("charges_age_violin")(c) + "\n\nMath: each violin is a kernel density estimate (see step 6 of *Math Behind the Graphs*)."},
    {"k": ["facility volume", "facility discharge graph", "discharge volume"], "a": _graph("facility_discharge_volume")},
    {"k": ["facility average stay graph", "average stay by facility"], "a": _graph("facility_average_stay")},
    {"k": ["heatmap", "heat map", "county facility"], "a": _graph("county_facility_heatmap")},
    {"k": ["missing value graph", "missing value chart"], "a": _graph("quality_missing_values")},
    {"k": ["confusion matrix"], "a": _graph("ai_confusion_matrix")},
    {"k": ["diagnosis graph", "diagnosis chart", "average stay per diagnosis"], "a": _graph("diagnosis_average_stay")},
    {"k": ["accuracy", "precision", "recall", "f1", "how is accuracy calculated", "classification report"],
     "a": "**Accuracy** = correct predictions ÷ all predictions (the diagonal of the confusion matrix divided by its total). **Precision** for a class = correct predictions of that class ÷ all predictions of it. **Recall** = correct predictions of that class ÷ all real cases of it. Always read them with the confusion matrix, because class imbalance can hide errors. See step 9 of *Math Behind the Graphs*."},
    {"k": ["ai experiment", "machine learning", "random forest", "train model", "predict severity", "dummy baseline", "model training"], "a": _a_ai},
    {"k": ["rotate", "3d chart", "3d graph", "zoom chart", "move the chart", "drag", "spin", "rotate 360"],
     "a": "In every 3D chart: **drag** to rotate, **scroll** to zoom, **hover** a shape for its exact value, and press **⟳ Rotate 360°** for an automatic spin (**❚❚ Stop** pauses it). Use the camera icons in the chart's top-right corner to reset the view or download an image."},
    {"k": ["2d chart", "flat chart", "classic chart", "turn off 3d", "disable 3d", "switch to 2d"],
     "a": "Turn the **3D charts** switch in the sidebar off to see the classic flat charts. The *Math Behind the Graphs* page always stays in 3D."},
    {"k": ["theme", "dark mode", "light mode", "light theme", "dark theme", "change colour", "change color"],
     "a": "Use the round **theme button at the bottom-right** to switch between the *light + dark mix* and *full dark*. Your choice is remembered."},
    {"k": ["filter", "filters", "sidebar", "select age", "narrow the data"],
     "a": "The **sidebar filters** (age group, gender, severity of illness, payment type) apply to every page: KPIs, graphs, tables, the explorer and the downloads all update together. If nothing matches, a warning asks you to change the filters."},
    {"k": ["download", "export", "save csv", "csv file", "save data"],
     "a": "Download buttons appear on **Diagnosis & Stay** (diagnosis summary), **Facilities & Counties** (facility summary), **Data Quality** (cleaned sample) and **Data Explorer** (the records currently displayed). Each saves a CSV file."},
    {"k": ["question mark", "explain this graph", "explain graph", "help icon", "what is the ? button"],
     "a": "The small **?** button above each chart opens a plain-language explanation of that graph."},
    {"k": ["background", "animation", "particles", "robot", "who is the robot", "the doctor"],
     "a": "The animated background, the DNA helix in the banner and the doctor robot are drawn live with **Three.js**. They are decoration only and never change the data. If your device is slow, the *prefers reduced motion* system setting calms the animations."},
    {"k": ["length of stay", "los", "what is los"], "a": "**Length of stay (LOS)** is the number of days a patient spends in hospital between admission and discharge. Records with missing or non-positive LOS are removed, and values are capped at 3,650 days."},
    {"k": ["severity of illness", "what is severity", "apr severity", "minor moderate major extreme"],
     "a": "**Severity of illness** is a standard 4-level classification (**Minor, Moderate, Major, Extreme**) of how sick a patient was, taken from the APR severity field. Unrecognised values are labelled *Unknown*."},
    {"k": ["sparcs", "dataset", "data source", "where does the data come from", "which data"],
     "a": "The data is the **2021 Hospital Inpatient Discharges (SPARCS, de-identified)** public dataset from New York State. To keep the app fast, up to **250,000 records** are loaded as a sample."},
    {"k": ["mean", "average formula", "how is mean calculated"], "a": "**Mean** = sum of all values ÷ number of values. Example: stays of 2, 4 and 9 days give (2 + 4 + 9) ÷ 3 = 5 days."},
    {"k": ["median"], "a": "**Median** is the middle value after sorting. It is less affected by a few extreme stays than the mean."},
    {"k": ["standard deviation", "spread", "variance"], "a": "**Standard deviation** measures how far values typically sit from the mean: s = √(Σ(x − x̄)² ÷ (n − 1)). A bigger value means more spread-out stays."},
    {"k": ["correlation", "causation"], "a": "**Correlation** (Pearson's r) runs from −1 to +1 and measures a *linear* association. It does **not** show that one thing causes the other. Ask *\"What is the correlation between stay and charges?\"* for your live value."},
    {"k": ["limitations", "limits", "can i trust", "reliable", "accuracy of the dashboard", "sample"],
     "a": "Key limitations: results use a **sample** (up to 250,000 rows), patterns show association **not causation**, charges are recorded amounts not treatment cost, differences between hospitals are not quality measures, and the AI model is **educational only**."},
    {"k": ["slow", "loading", "takes long", "performance", "lag"],
     "a": "The first load reads a large CSV, so it can take a moment; afterwards it is cached. If 3D feels slow, switch the **3D charts** toggle off, or choose *Full dark* and enable your system's reduced-motion setting."},
    {"k": ["who made", "creator", "author", "built by", "credits", "student", "school"],
     "a": "MediScope was created by **Sarveshwaran K** (No. 1000411, Jain Vidyalaya IB World School) as a Data Analytics project."},
    {"k": ["technology", "built with", "libraries", "tools used", "framework", "python"],
     "a": "MediScope uses **Python, Pandas, NumPy, Plotly, scikit-learn and Streamlit**, plus **Three.js** for the 3D background, banner and robot."},
]

DOC_FALLBACK_TIPS = [
    "What does this page show?",
    "What is the average length of stay?",
    "Which facility has the most discharges?",
    "Explain the heatmap",
    "How do I download the data?",
]


_DOC_VOCAB = []


def _doc_vocab():
    if not _DOC_VOCAB:
        found = set()
        for entry in DOC_KB:
            for kw in entry["k"]:
                found.update(w for w in re.findall(r"[a-z0-9&']+", kw) if len(w) >= 4)
        _DOC_VOCAB.extend(sorted(found))
    return _DOC_VOCAB


def doctor_reply(question, ctx):
    text = str(question).strip()
    if not text:
        return "Please type a question and I'll do my best to help."
    if _MEDICAL_PATTERN.search(text):
        return ("I'm a data assistant, **not a clinician**, so I can't give medical advice, diagnoses or "
                "treatment suggestions. Please speak to a qualified healthcare professional. I'm happy to "
                "explain anything about this dashboard instead.")

    words = re.findall(r"[a-z0-9&']+", text.lower())
    if " ".join(words) in {"hi", "hey", "hello", "hii", "yo", "hola", "good day"}:
        return DOC_KB[0]["a"]
    vocab = _doc_vocab()
    fixed = []
    for w in words:
        if len(w) >= 5 and w not in vocab:
            close = difflib.get_close_matches(w, vocab, n=1, cutoff=0.8)
            fixed.append(close[0] if close else w)
        else:
            fixed.append(w)
    norm = " " + " ".join(fixed) + " "
    tokens = [t for t in norm.split() if len(t) >= 5]
    best, best_score = None, 0.0
    for entry in DOC_KB:
        score = 0.0
        for kw in entry["k"]:
            if (" " + kw) in norm:
                score += 3 * len(kw.split())
            elif " " not in kw and len(kw) >= 5:
                if difflib.get_close_matches(kw, tokens, n=1, cutoff=0.85):
                    score += 1.5
        if score > best_score:
            best, best_score = entry, score

    if best is not None and best_score >= 3:
        answer = best["a"]
        try:
            return answer(ctx) if callable(answer) else answer
        except Exception:
            return "I couldn't calculate that from the current data. Try changing the filters and asking again."

    tips = "\n".join(f"- {t}" for t in DOC_FALLBACK_TIPS)
    return ("I'm not sure I understood that. I can answer questions about the pages, graphs, filters, the 3D "
            f"controls, the maths, the AI experiment and the live numbers. Try one of these:\n\n{tips}")


def _doc_stream(text):
    for chunk in re.findall(r"\S+\s*", text):
        yield chunk
        time.sleep(0.018)


def _doctor_panel():
    history = st.session_state.setdefault("doc_chat", [{"role": "assistant", "content": DOC_GREETING}])

    stage = st.container()
    box = st.container(height=300)

    pending = None
    cols = st.columns(3)
    for index, label in enumerate(DOC_QUICK_QUESTIONS):
        if cols[index % 3].button(label, key=f"doc_q_{index}", use_container_width=True):
            pending = label

    with st.form("doc_form", clear_on_submit=True):
        typed = st.text_input(
            "Ask Dr. MediBot",
            placeholder="Ask any question about this web app...",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("Send to Dr. MediBot", use_container_width=True)

    if st.button("Clear conversation", key="doc_clear"):
        st.session_state["doc_chat"] = [{"role": "assistant", "content": DOC_GREETING}]
        history = st.session_state["doc_chat"]
        pending = None
        sent = False

    question = pending or (typed.strip() if sent and typed and typed.strip() else None)

    with stage:
        st.markdown('<div class="ms-doc-stage"></div>', unsafe_allow_html=True)

    ctx = {
        "page": page,
        "df": df,
        "f": filtered_df,
        "stats": cleaning_stats,
        "ai": st.session_state.get("ai_results"),
    }

    with box:
        for message in history:
            with st.chat_message(message["role"], avatar="🩺" if message["role"] == "assistant" else None):
                st.markdown(message["content"])
        if question:
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            reply = doctor_reply(question, ctx)
            with st.chat_message("assistant", avatar="🩺"):
                st.write_stream(_doc_stream(reply))
            history.append({"role": "assistant", "content": reply})


if hasattr(st, "dialog"):
    @st.dialog("Dr. MediBot · your 3D hospital assistant", width="large")
    def doctor_dialog():
        _doctor_panel()
else:
    def doctor_dialog():
        with st.sidebar.expander("Dr. MediBot · ask me anything", expanded=True):
            _doctor_panel()


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

dataset_path = find_dataset()

if dataset_path is None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Medi<span>Scope</span></div>
            <div class="hero-subtitle">
                Hospital inpatient discharge analytics dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.error(
        "The hospital discharge CSV file was not found. "
        "Place the dataset in the same folder as app.py or in a data/ folder."
    )

    st.code(
        "Project folder/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv",
        language="text",
    )

    st.stop()


file_size = dataset_path.stat().st_size
modified_time = dataset_path.stat().st_mtime

try:
    with st.spinner("Loading hospital discharge data..."):
        raw_df = load_dataset(
            str(dataset_path),
            file_size,
            modified_time,
            SAMPLE_ROWS,
        )
        df, cleaning_stats = prepare_data(raw_df)

except Exception as exc:
    st.error("The dataset could not be loaded or processed.")
    st.code(str(exc))
    st.stop()


if df.empty:
    st.error(
        "No valid records remain after preprocessing. "
        "Check the source dataset and its column names."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_hospital_brand()
    render_mascot()

    st.markdown(
        '<div class="section-label">Navigation</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Dashboard page",
        [
            "Overview",
            "Diagnosis & Stay",
            "Charges & Demographics",
            "Facilities & Counties",
            "Data Quality",
            "Mathematical Analysis",
            "Math Behind the Graphs",
            "AI Experiment",
            "Data Explorer",
        ],
        label_visibility="collapsed",
    )

    side_open = st.button(
        "💬 Ask Dr. MediBot",
        key="doc_launch_sidebar",
        use_container_width=True,
        help="Chat with the 3D doctor robot about anything in this web app.",
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Filters</div>',
        unsafe_allow_html=True,
    )

    available_ages = [
        value for value in AGE_ORDER
        if value in df["Age_Group"].astype(str).unique()
    ]

    selected_ages = st.multiselect(
        "Age group",
        available_ages,
        default=available_ages,
    )

    available_genders = sorted(
        df["Gender"].astype(str).dropna().unique().tolist()
    )

    selected_genders = st.multiselect(
        "Gender",
        available_genders,
        default=available_genders,
    )

    available_severities = [
        value for value in SEVERITY_ORDER
        if value in df["Severity"].astype(str).unique()
    ]

    selected_severities = st.multiselect(
        "Severity of illness",
        available_severities,
        default=available_severities,
    )

    available_payments = sorted(
        df["Payment_Type"].astype(str).dropna().unique().tolist()
    )

    selected_payments = st.multiselect(
        "Payment type",
        available_payments,
        default=available_payments,
    )

    st.caption(
        "Selections apply to the dashboard charts and data explorer."
    )

    st.divider()

    CHART_3D = st.toggle(
        "3D charts",
        value=True,
        help="Turn off to see the classic flat charts. The Math Behind the Graphs page is always 3D.",
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["Age_Group"].astype(str).isin(selected_ages)
    & filtered_df["Gender"].astype(str).isin(selected_genders)
    & filtered_df["Severity"].astype(str).isin(selected_severities)
    & filtered_df["Payment_Type"].astype(str).isin(selected_payments)
]


# ============================================================
# DR. MEDIBOT TRIGGER (hidden button used by the floating 3D robot)
# ============================================================

open_chat = st.button("Ask Dr. MediBot", key="doc_launch")

if open_chat or side_open:
    doctor_dialog()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Medi<span>Scope</span> Health Intelligence
        </div>
        <div class="hero-subtitle">
            Explore hospital inpatient discharges, length of stay,
            financial patterns, patient demographics, and facility
            statistics through an interactive analytics dashboard.
        </div>
        <div class="hero-tags">
            <div class="hero-tag">SPARCS Inpatient Data</div>
            <div class="hero-tag">Interactive Analytics</div>
            <div class="hero-tag">Data Quality Monitoring</div>
            <div class="hero-tag">Machine Learning Demonstration</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if filtered_df.empty:
    st.warning(
        "No records match the selected filters. Change the filters in the sidebar."
    )
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-label">System overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Discharge records",
        format_number(len(filtered_df)),
    )

    c2.metric(
        "Average length of stay",
        format_days(filtered_df["LOS_Days"].mean()),
    )

    c3.metric(
        "Average total charges",
        format_currency(filtered_df["Total_Charges"].mean()),
    )

    c4.metric(
        "Facilities represented",
        format_number(filtered_df["Facility"].nunique()),
    )

    c5, c6, c7 = st.columns(3)

    with c5:
        insight_card(
            "Median stay",
            format_days(filtered_df["LOS_Days"].median()),
            "The midpoint of recorded inpatient stays.",
        )

    with c6:
        insight_card(
            "Total recorded charges",
            format_currency(filtered_df["Total_Charges"].sum()),
            "Sum of charges in the current filtered records.",
        )

    with c7:
        insight_card(
            "Counties represented",
            format_number(filtered_df["County"].nunique()),
            "Distinct facility counties in the filtered data.",
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Patient records by age group")

        age_counts = (
            filtered_df["Age_Group"]
            .astype(str)
            .value_counts()
            .reindex(AGE_ORDER, fill_value=0)
            .rename_axis("Age Group")
            .reset_index(name="Records")
        )

        fig = px.bar(
            age_counts,
            x="Age Group",
            y="Records",
            color="Age Group",
            color_discrete_sequence=COLORS,
            text="Records",
            title="Distribution of inpatient discharges",
        )

        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
        )

        show_chart(
            style_plot(fig),
            "overview_age_distribution",
        )

    with right:
        st.subheader("Primary payment type")

        payment_counts = (
            filtered_df["Payment_Type"]
            .astype(str)
            .value_counts()
            .rename_axis("Payment Type")
            .reset_index(name="Records")
        )

        fig = px.pie(
            payment_counts,
            names="Payment Type",
            values="Records",
            hole=0.60,
            color_discrete_sequence=COLORS,
            title="Discharges by payment category",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        show_chart(
            style_plot(fig),
            "overview_payment_pie",
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Length-of-stay distribution")

        los_data = filtered_df["LOS_Days"].dropna()

        if len(los_data) > 20_000:
            los_data = los_data.sample(
                20_000,
                random_state=42,
            )

        fig = px.histogram(
            x=los_data,
            nbins=45,
            color_discrete_sequence=["#00E5FF"],
            title="How long do patients stay?",
            labels={"x": "Length of stay (days)", "count": "Records"},
        )

        show_chart(
            style_plot(fig),
            "overview_los_histogram",
        )

    with right:
        st.subheader("Average stay by severity")

        severity_summary = (
            filtered_df.groupby("Severity", observed=False)["LOS_Days"]
            .mean()
            .reset_index()
        )

        severity_summary["Severity"] = (
            severity_summary["Severity"].astype(str)
        )

        fig = px.bar(
            severity_summary,
            x="Severity",
            y="LOS_Days",
            color="Severity",
            category_orders={"Severity": SEVERITY_ORDER},
            color_discrete_sequence=COLORS,
            title="Average inpatient stay by severity",
            labels={
                "LOS_Days": "Average stay (days)",
                "Severity": "Severity category",
            },
        )

        show_chart(
            style_plot(fig),
            "overview_severity_los",
        )

    st.subheader("Quick findings")

    severity_summary = severity_summary.dropna(
        subset=["LOS_Days"]
    )

    if not severity_summary.empty:
        longest = severity_summary.loc[
            severity_summary["LOS_Days"].idxmax()
        ]

        st.markdown(
            f"- The highest average length of stay among displayed "
            f"severity categories is **{longest['Severity']}**, at "
            f"**{longest['LOS_Days']:.2f} days**."
        )

    age_counts_series = (
        filtered_df["Age_Group"].astype(str).value_counts()
    )

    if not age_counts_series.empty:
        st.markdown(
            f"- The most represented age group is "
            f"**{age_counts_series.index[0]}**, with "
            f"**{age_counts_series.iloc[0]:,} records**."
        )

    payment_series = (
        filtered_df["Payment_Type"].astype(str).value_counts()
    )

    if not payment_series.empty:
        st.markdown(
            f"- The most frequent payment category is "
            f"**{payment_series.index[0]}**."
        )

    st.caption(
        "These findings are descriptive and do not establish causation "
        "or measure hospital quality."
    )


# ============================================================
# DIAGNOSIS AND STAY
# ============================================================

elif page == "Diagnosis & Stay":

    st.title("Diagnosis & Length of Stay")

    st.write(
        "Compare inpatient stay across diagnosis categories and examine "
        "how length of stay varies with illness severity."
    )

    diagnosis_limit = st.slider(
        "Number of diagnosis categories",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    diagnosis_summary = (
        filtered_df.groupby("Diagnosis", observed=False)
        .agg(
            Records=("Record_ID", "count"),
            Average_Stay=("LOS_Days", "mean"),
            Median_Stay=("LOS_Days", "median"),
            Average_Charges=("Total_Charges", "mean"),
        )
        .reset_index()
    )

    diagnosis_summary = diagnosis_summary[
        diagnosis_summary["Diagnosis"] != "Unknown"
    ]

    diagnosis_summary = diagnosis_summary.sort_values(
        "Records",
        ascending=False,
    ).head(diagnosis_limit)

    st.subheader("Average stay per diagnosis")

    if not diagnosis_summary.empty:
        fig = px.bar(
            diagnosis_summary.sort_values(
                "Average_Stay",
                ascending=True,
            ),
            x="Average_Stay",
            y="Diagnosis",
            orientation="h",
            color="Average_Stay",
            color_continuous_scale=[
                "#14263b",
                "#007cba",
                "#00e5ff",
            ],
            title="Average length of stay by diagnosis category",
            labels={
                "Average_Stay": "Average stay (days)",
                "Diagnosis": "Diagnosis category",
            },
        )

        show_chart(
            style_plot(
                fig,
                height=max(400, diagnosis_limit * 28),
            ),
            "diagnosis_average_stay",
        )

    else:
        st.info("No diagnosis categories are available for these filters.")

    st.subheader("Length of stay by severity")

    box_data = filtered_df.copy()

    if len(box_data) > 15_000:
        box_data = box_data.sample(
            15_000,
            random_state=42,
        )

    fig = px.box(
        box_data,
        x="Severity",
        y="LOS_Days",
        color="Severity",
        category_orders={"Severity": SEVERITY_ORDER},
        color_discrete_sequence=COLORS,
        points="outliers",
        title="Variation in length of stay by severity",
        labels={
            "LOS_Days": "Length of stay (days)",
            "Severity": "Severity category",
        },
    )

    show_chart(
        style_plot(fig),
        "diagnosis_severity_box",
    )

    st.subheader("Diagnosis summary table")

    display_summary = diagnosis_summary.rename(
        columns={
            "Records": "Number of records",
            "Average_Stay": "Average stay (days)",
            "Median_Stay": "Median stay (days)",
            "Average_Charges": "Average charges ($)",
        }
    ).copy()

    for column in [
        "Average stay (days)",
        "Median stay (days)",
        "Average charges ($)",
    ]:
        if column in display_summary.columns:
            display_summary[column] = display_summary[column].round(2)

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download diagnosis summary",
        data=display_summary.to_csv(index=False).encode("utf-8"),
        file_name="diagnosis_stay_summary.csv",
        mime="text/csv",
    )


# ============================================================
# CHARGES AND DEMOGRAPHICS
# ============================================================

elif page == "Charges & Demographics":

    st.title("Charges & Patient Demographics")

    st.write(
        "Explore recorded charges across age groups, gender, "
        "and severity categories."
    )

    charge_data = filtered_df[
        filtered_df["Total_Charges"].notna()
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average charges",
        format_currency(charge_data["Total_Charges"].mean()),
    )

    c2.metric(
        "Median charges",
        format_currency(charge_data["Total_Charges"].median()),
    )

    c3.metric(
        "Total charges",
        format_currency(charge_data["Total_Charges"].sum()),
    )

    st.subheader("Charges by severity")

    box_charge_data = charge_data.copy()

    if len(box_charge_data) > 15_000:
        box_charge_data = box_charge_data.sample(
            15_000,
            random_state=42,
        )

    fig = px.box(
        box_charge_data,
        x="Severity",
        y="Total_Charges",
        color="Severity",
        category_orders={"Severity": SEVERITY_ORDER},
        color_discrete_sequence=COLORS,
        points="outliers",
        title="Distribution of total charges by severity",
        labels={
            "Total_Charges": "Total charges ($)",
            "Severity": "Severity category",
        },
    )

    fig.update_yaxes(tickprefix="$")

    show_chart(
        style_plot(fig),
        "charges_severity_box",
    )

    st.subheader("Average charges by age group and gender")

    demographic_summary = (
        charge_data.groupby(
            ["Age_Group", "Gender"],
            observed=False,
        )["Total_Charges"]
        .mean()
        .reset_index()
    )

    demographic_summary["Age_Group"] = (
        demographic_summary["Age_Group"].astype(str)
    )

    demographic_summary["Gender"] = (
        demographic_summary["Gender"].astype(str)
    )

    fig = px.bar(
        demographic_summary,
        x="Age_Group",
        y="Total_Charges",
        color="Gender",
        barmode="group",
        category_orders={"Age_Group": AGE_ORDER},
        color_discrete_sequence=COLORS,
        title="Average charges by age group and gender",
        labels={
            "Age_Group": "Age group",
            "Total_Charges": "Average charges ($)",
            "Gender": "Gender",
        },
    )

    fig.update_yaxes(tickprefix="$")

    show_chart(
        style_plot(fig),
        "charges_age_gender",
    )

    st.subheader("Charge distribution by age group")

    if not charge_data.empty:
        violin_data = charge_data.sample(
            min(len(charge_data), 12_000),
            random_state=42,
        )

        fig = px.violin(
            violin_data,
            x="Age_Group",
            y="Total_Charges",
            color="Age_Group",
            category_orders={"Age_Group": AGE_ORDER},
            color_discrete_sequence=COLORS,
            box=True,
            points=False,
            title="Spread of recorded charges across age groups",
            labels={
                "Age_Group": "Age group",
                "Total_Charges": "Total charges ($)",
            },
        )

        fig.update_yaxes(tickprefix="$")

        show_chart(
            style_plot(fig),
            "charges_age_violin",
        )

    demographic_table = (
        charge_data.groupby(
            ["Age_Group", "Gender"],
            observed=False,
        )
        .agg(
            Records=("Record_ID", "count"),
            Average_Charges=("Total_Charges", "mean"),
            Median_Charges=("Total_Charges", "median"),
            Average_Stay=("LOS_Days", "mean"),
        )
        .reset_index()
    )

    demographic_table["Age_Group"] = (
        demographic_table["Age_Group"].astype(str)
    )

    for column in [
        "Average_Charges",
        "Median_Charges",
        "Average_Stay",
    ]:
        demographic_table[column] = demographic_table[column].round(2)

    st.subheader("Demographic summary")

    st.dataframe(
        demographic_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FACILITIES AND COUNTIES
# ============================================================

elif page == "Facilities & Counties":

    st.title("Facilities & County Analysis")

    st.write(
        "Compare discharge volume, average stay, and recorded charges "
        "across hospitals and counties."
    )

    facility_summary = (
        filtered_df.groupby("Facility", observed=False)
        .agg(
            Discharges=("Record_ID", "count"),
            Average_Stay=("LOS_Days", "mean"),
            Average_Charges=("Total_Charges", "mean"),
            Total_Charges=("Total_Charges", "sum"),
            County=("County", "first"),
        )
        .reset_index()
    )

    facility_limit = st.slider(
        "Number of facilities to display",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    top_facilities = facility_summary.sort_values(
        "Discharges",
        ascending=False,
    ).head(facility_limit)

    st.subheader("Facilities with the most discharge records")

    fig = px.bar(
        top_facilities.sort_values("Discharges"),
        x="Discharges",
        y="Facility",
        orientation="h",
        color="Discharges",
        color_continuous_scale=[
            "#14263b",
            "#007cba",
            "#00e5ff",
        ],
        title="Discharge volume by facility",
        labels={
            "Discharges": "Number of records",
            "Facility": "Facility",
        },
    )

    show_chart(
        style_plot(
            fig,
            height=max(420, facility_limit * 28),
        ),
        "facility_discharge_volume",
    )

    st.subheader("Average length of stay by facility")

    fig = px.bar(
        top_facilities.sort_values("Average_Stay"),
        x="Average_Stay",
        y="Facility",
        orientation="h",
        color="Average_Stay",
        color_continuous_scale=[
            "#14263b",
            "#7C4DFF",
            "#00E5FF",
        ],
        title="Average stay among highest-volume facilities",
        labels={
            "Average_Stay": "Average stay (days)",
            "Facility": "Facility",
        },
    )

    show_chart(
        style_plot(
            fig,
            height=max(420, facility_limit * 28),
        ),
        "facility_average_stay",
    )

    st.subheader("County and facility heatmap")

    county_limit = st.slider(
        "Number of counties in heatmap",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
    )

    top_counties = top_values(
        filtered_df,
        "County",
        county_limit,
    )

    heat_facilities = top_values(
        filtered_df,
        "Facility",
        15,
    )

    heat_data = filtered_df[
        filtered_df["County"].astype(str).isin(top_counties)
        & filtered_df["Facility"].astype(str).isin(heat_facilities)
    ]

    if not heat_data.empty:
        heat_matrix = heat_data.pivot_table(
            index="County",
            columns="Facility",
            values="LOS_Days",
            aggfunc="mean",
            observed=False,
        )

        fig = px.imshow(
            heat_matrix,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Average length of stay by county and facility",
            labels={
                "x": "Facility",
                "y": "County",
                "color": "Average stay (days)",
            },
        )

        fig.update_layout(xaxis_tickangle=-35)

        show_chart(
            style_plot(
                fig,
                height=max(420, len(heat_matrix.index) * 38),
            ),
            "county_facility_heatmap",
        )
    else:
        st.info("There is not enough county and facility data for this heatmap.")

    st.subheader("Facility summary table")

    facility_display = facility_summary.sort_values(
        "Discharges",
        ascending=False,
    ).head(100).copy()

    facility_display = facility_display.rename(
        columns={
            "Discharges": "Discharge records",
            "Average_Stay": "Average stay (days)",
            "Average_Charges": "Average charges ($)",
            "Total_Charges": "Total charges ($)",
        }
    )

    for column in [
        "Average stay (days)",
        "Average charges ($)",
        "Total charges ($)",
    ]:
        facility_display[column] = facility_display[column].round(2)

    st.dataframe(
        facility_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download facility summary",
        data=facility_display.to_csv(index=False).encode("utf-8"),
        file_name="facility_summary.csv",
        mime="text/csv",
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.title("Data Quality & Preprocessing")

    st.write(
        "Review the data-cleaning process, detected source columns, "
        "missing values, and potential outliers."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original loaded rows",
        format_number(cleaning_stats["original_rows"]),
    )

    c2.metric(
        "Rows after cleaning",
        format_number(cleaning_stats["cleaned_rows"]),
    )

    c3.metric(
        "Rows removed",
        format_number(cleaning_stats["removed_rows"]),
    )

    c4.metric(
        "Removal percentage",
        f"{cleaning_stats['removed_rows'] / max(cleaning_stats['original_rows'], 1) * 100:.2f}%",
    )

    st.subheader("Preprocessing steps")

    steps = [
        (
            "Column identification",
            "Identify relevant source columns for length of stay, charges, "
            "age, gender, race, severity, diagnosis, payment, facility, and county.",
        ),
        (
            "Numeric conversion",
            "Convert length of stay and total charges into numeric values.",
        ),
        (
            "Invalid length of stay",
            "Remove rows with missing, nonnumeric, or nonpositive length of stay.",
        ),
        (
            "Invalid charges",
            "Remove rows with negative total charges and preserve missing charges as missing.",
        ),
        (
            "Category standardization",
            "Standardize age group, gender, severity, and payment categories.",
        ),
        (
            "Missing categorical values",
            "Represent missing categorical values with an Unknown label.",
        ),
        (
            "Analysis fields",
            "Create consistent column names for charts, filtering, and summaries.",
        ),
    ]

    for name, description in steps:
        with st.expander(name):
            st.write(description)

    st.subheader("Detected source columns")

    detected_rows = []

    for logical_name, source_name in cleaning_stats[
        "source_columns"
    ].items():
        detected_rows.append(
            {
                "Analysis field": logical_name.replace("_", " ").title(),
                "Detected source column": source_name or "Not found",
                "Status": "Detected" if source_name else "Unavailable",
            }
        )

    st.dataframe(
        pd.DataFrame(detected_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Missing-value profile")

    missing_counts = df.isna().sum()

    missing_df = pd.DataFrame(
        {
            "Column": missing_counts.index,
            "Missing values": missing_counts.values,
        }
    )

    missing_df["Missing percentage"] = (
        missing_df["Missing values"] / max(len(df), 1) * 100
    ).round(2)

    missing_df = missing_df.sort_values(
        "Missing percentage",
        ascending=False,
    ).head(25)

    fig = px.bar(
        missing_df.sort_values("Missing percentage"),
        x="Missing percentage",
        y="Column",
        orientation="h",
        color="Missing percentage",
        color_continuous_scale=[
            "#14263b",
            "#FFB300",
            "#FF4081",
        ],
        title="Columns with the highest missing-value percentages",
        labels={
            "Missing percentage": "Missing values (%)",
            "Column": "Column name",
        },
    )

    show_chart(
        style_plot(
            fig,
            height=max(420, len(missing_df) * 25),
        ),
        "quality_missing_values",
    )

    st.subheader("Length-of-stay outliers")

    q1 = df["LOS_Days"].quantile(0.25)
    q3 = df["LOS_Days"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = max(0, q1 - 1.5 * iqr)

    outlier_count = int(
        (
            (df["LOS_Days"] < lower_fence)
            | (df["LOS_Days"] > upper_fence)
        ).sum()
    )

    left, right = st.columns(2)

    left.metric(
        "Potential LOS outliers",
        format_number(outlier_count),
    )

    right.metric(
        "IQR upper fence",
        format_days(upper_fence),
    )

    st.caption(
        "IQR outliers are statistical flags, not automatically invalid records. "
        "Long stays may be legitimate. LOS values are capped at 3,650 days "
        "to limit extreme values."
    )

    st.subheader("Cleaned dataset preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download cleaned dataset sample",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_cleaned_sample.csv",
        mime="text/csv",
    )


# ============================================================
# MATHEMATICAL ANALYSIS
# ============================================================

elif page == "Mathematical Analysis":

    st.title("Mathematical Analysis")
    st.write(
        "Explore the statistical calculations used by MediScope. "
        "All results on this page are recalculated from the currently filtered records."
    )

    los_values = pd.to_numeric(
        filtered_df["LOS_Days"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    charge_values = pd.to_numeric(
        filtered_df["Total_Charges"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    st.subheader("1. Descriptive statistics: length of stay")
    st.markdown(
        "The mean gives the arithmetic average, the median gives the middle value, "
        "and the standard deviation measures how spread out stays are around the mean."
    )
    st.latex(r"\bar{x}=\frac{\sum_{i=1}^{n}x_i}{n}")
    st.caption("xᵢ = an individual length of stay in days; n = number of valid stay records.")

    if not los_values.empty:
        mean_los = los_values.mean()
        median_los = los_values.median()
        std_los = los_values.std(ddof=1) if len(los_values) > 1 else np.nan
        min_los = los_values.min()
        max_los = los_values.max()
        a, b, c, d, e = st.columns(5)
        a.metric("Mean stay", format_days(mean_los))
        b.metric("Median stay", format_days(median_los))
        c.metric("Standard deviation", format_days(std_los))
        d.metric("Minimum stay", format_days(min_los))
        e.metric("Maximum stay", format_days(max_los))
        st.code(
            f"Mean = {los_values.sum():,.2f} / {len(los_values):,} = {mean_los:,.2f} days"
        )
    else:
        st.info("No valid length-of-stay values are available for the selected filters.")

    st.divider()
    st.subheader("2. Descriptive statistics: healthcare charges")
    st.markdown(
        "The following calculations use records with a valid numeric total-charge value. "
        "Missing charge values are excluded from these calculations."
    )
    st.latex(r"\bar{C}=\frac{\sum_{i=1}^{n}C_i}{n}")
    st.caption("Cᵢ = recorded total charges for one discharge record; n = number of records with valid charges.")

    if not charge_values.empty:
        mean_charge = charge_values.mean()
        median_charge = charge_values.median()
        std_charge = charge_values.std(ddof=1) if len(charge_values) > 1 else np.nan
        a, b, c, d = st.columns(4)
        a.metric("Mean charges", format_currency(mean_charge))
        b.metric("Median charges", format_currency(median_charge))
        c.metric("Standard deviation", format_currency(std_charge))
        d.metric("Total recorded charges", format_currency(charge_values.sum()))
        st.code(
            f"Mean charges = ${charge_values.sum():,.2f} / {len(charge_values):,} = ${mean_charge:,.2f}"
        )
    else:
        st.info("No valid charge values are available for the selected filters.")

    st.divider()
    st.subheader("3. Percentage distribution by payment type")
    st.markdown(
        "This calculation finds the share of filtered discharge records in each payment category."
    )
    st.latex(r"\text{Percentage}=\frac{\text{Category count}}{\text{Total records}}\times100")

    payment_math = (
        filtered_df["Payment_Type"].astype(str).value_counts(dropna=False)
        .rename_axis("Payment type").reset_index(name="Records")
    )
    payment_math["Percentage"] = (
        payment_math["Records"] / max(len(filtered_df), 1) * 100
    ).round(2)
    if not payment_math.empty:
        st.dataframe(payment_math, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4. Severity comparison")
    st.markdown(
        "The table groups the filtered records by severity and calculates the number "
        "of records, average stay, median stay, and average recorded charges for each group."
    )
    severity_math = (
        filtered_df.groupby("Severity", observed=False)
        .agg(
            Records=("Record_ID", "count"),
            Average_Stay=("LOS_Days", "mean"),
            Median_Stay=("LOS_Days", "median"),
            Average_Charges=("Total_Charges", "mean"),
        )
        .reset_index()
    )
    severity_math["Severity"] = severity_math["Severity"].astype(str)
    severity_math = severity_math[severity_math["Records"] > 0]
    for column in ["Average_Stay", "Median_Stay", "Average_Charges"]:
        severity_math[column] = severity_math[column].round(2)
    st.dataframe(severity_math, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("5. Correlation: length of stay and charges")
    st.markdown(
        "Pearson's correlation coefficient measures the direction and strength of a "
        "linear association between two numerical variables. Only records containing "
        "both a valid length of stay and valid charges are included."
    )
    st.latex(r"r=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}")
    paired = filtered_df[["LOS_Days", "Total_Charges"]].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) >= 2 and paired["LOS_Days"].nunique() > 1 and paired["Total_Charges"].nunique() > 1:
        correlation = paired["LOS_Days"].corr(paired["Total_Charges"])
        st.metric("Pearson correlation (r)", f"{correlation:.4f}")
        if correlation > 0:
            st.write("The filtered records show a positive linear association: higher stay values tend to occur with higher charge values.")
        elif correlation < 0:
            st.write("The filtered records show a negative linear association: higher stay values tend to occur with lower charge values.")
        else:
            st.write("The calculated linear correlation is approximately zero.")
        st.caption(f"Calculated using {len(paired):,} records containing both values. Correlation does not prove causation.")
    else:
        st.info("At least two paired records with variation in both variables are required to calculate correlation.")

    st.divider()
    st.subheader("6. How the dashboard uses these calculations")
    st.markdown(
        "- **KPI cards:** use count, mean, median, and sum calculations.\n"
        "- **Bar charts:** use grouped counts or averages to compare categories.\n"
        "- **Pie chart:** uses category counts to show payment-type proportions.\n"
        "- **Histograms:** group numerical values into intervals and count records in each interval.\n"
        "- **Boxplots:** summarize distributions using quartiles, medians, and potential outliers.\n"
        "- **Heatmap:** uses grouped mean length of stay for each county–facility combination.\n"
        "- **AI experiment:** uses a classification model and evaluates predictions against held-out test records."
    )
    st.warning(
        "These are descriptive calculations on the loaded dataset sample (up to 250,000 rows) "
        "after the selected filters. They do not establish causation, diagnose patients, "
        "or measure hospital quality."
    )


# ============================================================
# MATH BEHIND THE GRAPHS (new page - every graph in 3D)
# ============================================================

elif page == "Math Behind the Graphs":

    def step(number, title, body):
        st.markdown(
            f'<div class="math-step"><span class="math-num">{number}</span>'
            f'<b>{html.escape(title)}</b><br>{body}</div>',
            unsafe_allow_html=True,
        )

    st.title("Math Behind the Graphs")
    st.write(
        "Follow every graph in MediScope from raw records to the final 3D shape. "
        "Each step shows the formula, the live numbers calculated from your current "
        "filters, and the 3D graph those numbers produce. Drag to rotate, scroll to zoom, "
        "hover for exact values."
    )

    n_records = len(filtered_df)
    los_all = pd.to_numeric(filtered_df["LOS_Days"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    chg_all = pd.to_numeric(filtered_df["Total_Charges"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()

    # ---- Step 1: counting -------------------------------------------------
    step(1, "Counting records: bar charts, KPI cards and the age graph",
         "Every bar is a <b>count</b>. For a category <i>g</i> the height is the number of "
         "records whose label equals <i>g</i>. The KPI card <i>Discharge records</i> is the "
         "same count over all categories.")
    st.latex(r"n_g=\sum_{i=1}^{n}\mathbf{1}\,[\,g_i=g\,]\qquad\qquad n=\sum_g n_g")

    age_counts_m = (
        filtered_df["Age_Group"].astype(str).value_counts()
        .reindex(AGE_ORDER, fill_value=0).rename_axis("Age group").reset_index(name="Count n_g")
    )
    age_counts_m["Share of n (%)"] = (age_counts_m["Count n_g"] / max(n_records, 1) * 100).round(2)
    st.dataframe(age_counts_m, use_container_width=True, hide_index=True)
    st.code(
        f"n = {' + '.join(f'{int(v):,}' for v in age_counts_m['Count n_g'])} = {n_records:,}"
    )
    fig = px.bar(age_counts_m, x="Age group", y="Count n_g", color="Age group",
                 color_discrete_sequence=COLORS, title="Step 1 · counts n_g per age group",
                 category_orders={"Age group": AGE_ORDER},
                 labels={"Count n_g": "Records (n_g)"})
    show_chart(style_plot(fig), "math_step1_counts", force_3d=True)

    # ---- Step 2: percentages / donut angles ------------------------------
    step(2, "Shares and angles: the payment donut",
         "A donut slice is a <b>share</b> of the total. Its size is the share multiplied by "
         "360°, so the angle of every slice is calculated from the same counts as Step 1.")
    st.latex(r"p_g=\frac{n_g}{n}\times100\%\qquad\qquad \theta_g=\frac{n_g}{n}\times360^\circ")
    pay_m = (
        filtered_df["Payment_Type"].astype(str).value_counts()
        .rename_axis("Payment type").reset_index(name="Count n_g")
    )
    pay_m["Share p_g (%)"] = (pay_m["Count n_g"] / max(n_records, 1) * 100).round(2)
    pay_m["Angle θ_g (°)"] = (pay_m["Count n_g"] / max(n_records, 1) * 360).round(1)
    st.dataframe(pay_m, use_container_width=True, hide_index=True)
    st.caption(f"Check: the angles add up to {pay_m['Angle θ_g (°)'].sum():.1f}° (a full turn).")
    fig = px.pie(pay_m, names="Payment type", values="Count n_g", hole=0.6,
                 color_discrete_sequence=COLORS, title="Step 2 · slice size = n_g / n")
    show_chart(style_plot(fig), "math_step2_donut", force_3d=True)

    # ---- Step 3: mean, median, std, histogram bins -----------------------
    step(3, "Centre, spread and bins: the length-of-stay histogram",
         "The histogram cuts the stay axis into <i>k</i> equal-width bins and counts how many "
         "stays fall in each. The mean, median and standard deviation describe the same data.")
    st.latex(r"w=\frac{x_{\max}-x_{\min}}{k}\qquad "
             r"\bar{x}=\frac{\sum x_i}{n}\qquad "
             r"s=\sqrt{\frac{\sum (x_i-\bar{x})^2}{n-1}}")
    if len(los_all) > 1:
        k_bins = st.slider("Number of bins k", 10, 80, 45, 5, key="math_bins")
        x_min, x_max = float(los_all.min()), float(los_all.max())
        width_w = (x_max - x_min) / k_bins
        mean_v, med_v, std_v = float(los_all.mean()), float(los_all.median()), float(los_all.std(ddof=1))
        a, b, c = st.columns(3)
        a.metric("Bin width w", f"{width_w:,.3f} days")
        b.metric("Mean x̄", format_days(mean_v))
        c.metric("Std. deviation s", format_days(std_v))
        st.code(
            f"w = ({x_max:,.1f} - {x_min:,.1f}) / {k_bins} = {width_w:,.3f}\n"
            f"mean = {los_all.sum():,.1f} / {len(los_all):,} = {mean_v:,.3f}\n"
            f"median = middle value of the sorted stays = {med_v:,.3f}\n"
            f"skew hint: mean - median = {mean_v - med_v:,.3f} "
            f"({'right-skewed: a few long stays pull the mean up' if mean_v > med_v else 'not right-skewed'})"
        )
        sample_los = los_all.sample(min(len(los_all), 20_000), random_state=42)
        fig = px.histogram(x=sample_los, nbins=k_bins, title="Step 3 · stays counted into k equal bins",
                           labels={"x": "Length of stay (days)", "count": "Records"})
        show_chart(style_plot(fig), "math_step3_hist", force_3d=True)
    else:
        st.info("Not enough length-of-stay values for these filters.")

    # ---- Step 4: grouped means -------------------------------------------
    step(4, "Group averages: average stay and charges by severity",
         "A grouped average adds up the values of one group and divides by that group's own "
         "count. Each bar in the severity graphs is one of these divisions.")
    st.latex(r"\bar{x}_g=\frac{1}{n_g}\sum_{i\in g}x_i")
    grp = (
        filtered_df.groupby("Severity", observed=False)
        .agg(n_g=("LOS_Days", "count"), sum_los=("LOS_Days", "sum"), mean_los=("LOS_Days", "mean"),
             mean_charge=("Total_Charges", "mean"))
        .reset_index()
    )
    grp["Severity"] = grp["Severity"].astype(str)
    grp = grp[grp["n_g"] > 0].copy()
    show_grp = grp.rename(columns={"n_g": "Records n_g", "sum_los": "Sum of stays Σx",
                                   "mean_los": "Mean stay Σx / n_g", "mean_charge": "Mean charges ($)"}).round(2)
    st.dataframe(show_grp, use_container_width=True, hide_index=True)
    if not grp.empty:
        top = grp.iloc[0]
        st.code(f"{top['Severity']}: {top['sum_los']:,.1f} / {int(top['n_g']):,} = {top['mean_los']:,.3f} days")
        fig = px.bar(grp, x="Severity", y="mean_los", color="Severity",
                     category_orders={"Severity": SEVERITY_ORDER}, color_discrete_sequence=COLORS,
                     title="Step 4 · mean stay x̄_g per severity",
                     labels={"mean_los": "Mean stay (days)"})
        show_chart(style_plot(fig), "math_step4_means", force_3d=True)

    # ---- Step 5: quartiles / box plot ------------------------------------
    step(5, "Quartiles and fences: how a box plot is built",
         "Sort the values. <b>Q1</b> is the 25th percentile, the <b>median</b> the 50th and "
         "<b>Q3</b> the 75th. The box spans Q1 to Q3. Whiskers reach the furthest values still "
         "inside the fences, and anything beyond a fence is drawn as an outlier dot.")
    st.latex(r"\mathrm{IQR}=Q_3-Q_1\qquad "
             r"\text{lower fence}=Q_1-1.5\,\mathrm{IQR}\qquad "
             r"\text{upper fence}=Q_3+1.5\,\mathrm{IQR}")
    box_rows, box_sample = [], filtered_df.sample(min(len(filtered_df), 15_000), random_state=42)
    for sev in [s for s in SEVERITY_ORDER if s in set(box_sample["Severity"].astype(str))]:
        vals = box_sample.loc[box_sample["Severity"].astype(str) == sev, "LOS_Days"]
        stats = box_stats(vals)
        if stats:
            box_rows.append({"Severity": sev, "n": stats["n"], "Q1": round(stats["q1"], 2),
                             "Median": round(stats["median"], 2), "Q3": round(stats["q3"], 2),
                             "IQR": round(stats["iqr"], 2), "Upper fence": round(stats["high_fence"], 2),
                             "Outliers": len(stats["outliers"])})
    if box_rows:
        st.dataframe(pd.DataFrame(box_rows), use_container_width=True, hide_index=True)
        r0 = box_rows[0]
        st.code(f"{r0['Severity']}: IQR = {r0['Q3']} - {r0['Q1']} = {r0['IQR']};  "
                f"upper fence = {r0['Q3']} + 1.5 x {r0['IQR']} = {r0['Upper fence']}")
        fig = px.box(box_sample, x="Severity", y="LOS_Days", color="Severity",
                     category_orders={"Severity": SEVERITY_ORDER}, color_discrete_sequence=COLORS,
                     title="Step 5 · quartile boxes of length of stay",
                     labels={"LOS_Days": "Length of stay (days)"})
        show_chart(style_plot(fig), "math_step5_box", force_3d=True)

    # ---- Step 6: KDE ------------------------------------------------------
    step(6, "Smooth density: how a violin is drawn",
         "A violin replaces the histogram's blocks with a smooth curve. At every height <i>y</i> "
         "a small bell curve (kernel) is placed on each data point and all bells are added. "
         "The result is the width of the violin at that height. The bell width <i>h</i> comes "
         "from Silverman's rule.")
    st.latex(r"\hat f(y)=\frac{1}{nh}\sum_{i=1}^{n}K\!\left(\frac{y-y_i}{h}\right)\quad "
             r"K(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}\quad "
             r"h=0.9\,\min\!\left(s,\frac{\mathrm{IQR}}{1.34}\right)n^{-1/5}")
    if len(chg_all) > 2:
        ridge_sample = filtered_df[filtered_df["Total_Charges"].notna()]
        ridge_sample = ridge_sample.sample(min(len(ridge_sample), 12_000), random_state=42)
        groups = [g for g in AGE_ORDER if g in set(ridge_sample["Age_Group"].astype(str))]
        surf_z, surf_grid, h_rows = [], None, []
        lo = float(np.percentile(ridge_sample["Total_Charges"], 0.5))
        hi = float(np.percentile(ridge_sample["Total_Charges"], 99.5))
        common = np.linspace(lo, hi, 60)
        for g in groups:
            v = ridge_sample.loc[ridge_sample["Age_Group"].astype(str) == g, "Total_Charges"].to_numpy(float)
            if len(v) < 3:
                continue
            grid_g, dens_g, h_g = kde_curve(v)
            surf_z.append(np.interp(common, grid_g, dens_g, left=0.0, right=0.0))
            h_rows.append({"Age group": g, "n": len(v), "Bandwidth h ($)": round(h_g, 2)})
        if surf_z:
            st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
            names = [r["Age group"] for r in h_rows]
            surface = go.Figure(
                go.Surface(
                    x=list(range(len(names))), y=common, z=np.array(surf_z).T,
                    colorscale=[[0, "#0b1d3a"], [0.35, "#007cba"], [0.7, "#00E5FF"], [1, "#B6FFF0"]],
                    showscale=True, colorbar=dict(title="Density f(y)", thickness=11, len=0.55),
                    hovertemplate="Age group index %{x}<br>Charges $%{y:,.0f}<br>Density %{z:.2e}<extra></extra>",
                )
            )
            surface.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=600,
                margin=dict(l=0, r=0, t=52, b=0), font=_CHART_FONT,
                title=dict(text="Step 6 · density surface of charges, one ridge per age group",
                           font=dict(size=15, color="#eff9ff"), x=0.02),
                scene=dict(
                    bgcolor="rgba(0,0,0,0)",
                    xaxis=_axis3d("Age group", range(len(names)), names, True),
                    yaxis=_axis3d("Total charges ($)"),
                    zaxis=_axis3d("Density f(y)"),
                    aspectmode="manual", aspectratio=dict(x=1.8, y=1.8, z=0.8),
                    camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0)),
                ),
            )
            st.plotly_chart(surface, use_container_width=True, key="math_step6_surface",
                            config={"displaylogo": False, "responsive": True})
            violin_df = ridge_sample.copy()
            violin_df["Age_Group"] = violin_df["Age_Group"].astype(str)
            fig = px.violin(violin_df, x="Age_Group", y="Total_Charges", color="Age_Group",
                            category_orders={"Age_Group": AGE_ORDER}, color_discrete_sequence=COLORS,
                            title="Step 6 · the same densities as solid 3D violins",
                            labels={"Age_Group": "Age group", "Total_Charges": "Total charges ($)"})
            show_chart(style_plot(fig), "math_step6_violin", force_3d=True)
    else:
        st.info("Not enough charge values for these filters.")

    # ---- Step 7: Pearson --------------------------------------------------
    step(7, "Correlation: do longer stays go with higher charges?",
         "Pearson's <i>r</i> multiplies how far each stay and each charge sit from their own "
         "averages. If both tend to be above (or below) average together, the sum is positive.")
    st.latex(r"r=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}"
             r"{\sqrt{\sum (x_i-\bar{x})^2}\,\sqrt{\sum (y_i-\bar{y})^2}}")
    paired_m = filtered_df[["LOS_Days", "Total_Charges", "Severity"]].copy()
    paired_m["LOS_Days"] = pd.to_numeric(paired_m["LOS_Days"], errors="coerce")
    paired_m["Total_Charges"] = pd.to_numeric(paired_m["Total_Charges"], errors="coerce")
    paired_m = paired_m.replace([np.inf, -np.inf], np.nan).dropna(subset=["LOS_Days", "Total_Charges"])
    if len(paired_m) >= 3 and paired_m["LOS_Days"].nunique() > 1 and paired_m["Total_Charges"].nunique() > 1:
        xs_p, ys_p = paired_m["LOS_Days"].to_numpy(float), paired_m["Total_Charges"].to_numpy(float)
        dx, dy = xs_p - xs_p.mean(), ys_p - ys_p.mean()
        sxy, sxx, syy = float((dx * dy).sum()), float((dx * dx).sum()), float((dy * dy).sum())
        r_val = sxy / math.sqrt(sxx * syy)
        slope = sxy / sxx
        intercept = ys_p.mean() - slope * xs_p.mean()
        st.code(
            f"Σ(x-x̄)(y-ȳ) = {sxy:,.4e}\nΣ(x-x̄)²     = {sxx:,.4e}\nΣ(y-ȳ)²     = {syy:,.4e}\n"
            f"r = {sxy:,.4e} / sqrt({sxx:,.4e} x {syy:,.4e}) = {r_val:.4f}\n"
            f"best-fit line: charges = {intercept:,.2f} + {slope:,.2f} x stay"
        )
        sev_index = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        scatter = paired_m.sample(min(len(paired_m), 4000), random_state=42)
        zs = scatter["Severity"].astype(str).map(sev_index).fillna(4).to_numpy(float)
        line_x = np.linspace(xs_p.min(), np.percentile(xs_p, 99), 40)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter3d(
            x=scatter["LOS_Days"], y=scatter["Total_Charges"], z=zs, mode="markers",
            marker=dict(size=2.6, color=scatter["Total_Charges"], colorscale="Viridis", opacity=0.75,
                        colorbar=dict(title="Charges ($)", thickness=11, len=0.55)),
            hovertemplate="Stay %{x:.0f} days<br>Charges $%{y:,.0f}<extra></extra>"))
        fig_sc.add_trace(go.Scatter3d(
            x=line_x, y=intercept + slope * line_x, z=[2.0] * len(line_x), mode="lines",
            line=dict(color="#FF4081", width=6), hoverinfo="skip"))
        sev_names = [s for s in SEVERITY_ORDER]
        fig_sc.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=600, showlegend=False,
            margin=dict(l=0, r=0, t=52, b=0), font=_CHART_FONT,
            title=dict(text=f"Step 7 · stay vs charges vs severity (r = {r_val:.3f})",
                       font=dict(size=15, color="#eff9ff"), x=0.02),
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=_axis3d("Length of stay (days)"), yaxis=_axis3d("Total charges ($)"),
                zaxis=_axis3d("Severity", range(len(sev_names)), sev_names, True),
                aspectmode="manual", aspectratio=dict(x=1.5, y=1.5, z=0.9),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            ),
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="math_step7_scatter",
                        config={"displaylogo": False, "responsive": True})
        st.caption(f"{len(paired_m):,} paired records used. The pink line is the least-squares line; "
                   "correlation does not prove causation.")
    else:
        st.info("At least three paired records with variation are needed for a correlation.")

    # ---- Step 8: heatmap cells -------------------------------------------
    step(8, "Heatmap cells: one average per county and facility",
         "Each cell of the facility heatmap is the average stay of the records that share one "
         "county and one facility. Cells with no records stay empty. In 3D the cell value is also "
         "the height of the tower.")
    st.latex(r"\text{cell}_{c,f}=\frac{1}{n_{c,f}}\sum_{i\in(c,f)}x_i")
    top_c = top_values(filtered_df, "County", 8)
    top_f = top_values(filtered_df, "Facility", 10)
    heat_m = filtered_df[filtered_df["County"].astype(str).isin(top_c)
                         & filtered_df["Facility"].astype(str).isin(top_f)]
    if not heat_m.empty:
        cell_counts = heat_m.groupby(["County", "Facility"], observed=False)["LOS_Days"].agg(["count", "sum", "mean"]).reset_index()
        cell_counts = cell_counts[cell_counts["count"] > 0]
        if not cell_counts.empty:
            ex = cell_counts.sort_values("count", ascending=False).iloc[0]
            st.code(f"{ex['County']} / {_short(ex['Facility'], 40)}: "
                    f"{ex['sum']:,.1f} / {int(ex['count']):,} = {ex['mean']:,.3f} days")
        matrix = heat_m.pivot_table(index="County", columns="Facility", values="LOS_Days",
                                    aggfunc="mean", observed=False)
        fig = px.imshow(matrix, aspect="auto", color_continuous_scale="Viridis",
                        title="Step 8 · each tower = one county × facility average",
                        labels={"x": "Facility", "y": "County", "color": "Average stay (days)"})
        show_chart(style_plot(fig), "math_step8_heat", force_3d=True)
    else:
        st.info("Not enough county and facility data for these filters.")

    # ---- Step 9: accuracy / confusion ------------------------------------
    step(9, "Model scoring: accuracy, precision and recall",
         "The AI Experiment is scored by comparing predicted and actual severity. A correct "
         "prediction lands on the diagonal of the confusion matrix. Precision asks how many "
         "predictions of a class were right; recall asks how many real cases of a class were found.")
    st.latex(r"\text{accuracy}=\frac{\sum_k C_{kk}}{\sum_{j,k}C_{jk}}\qquad "
             r"\text{precision}_k=\frac{C_{kk}}{\sum_j C_{jk}}\qquad "
             r"\text{recall}_k=\frac{C_{kk}}{\sum_j C_{kj}}")
    saved = st.session_state.get("ai_results")
    if saved:
        cm = pd.crosstab(pd.Series(saved["y_test"], name="Actual severity"),
                         pd.Series(saved["predictions"], name="Predicted severity"))
        diag = int(sum(cm.loc[c, c] for c in cm.index if c in cm.columns))
        total_cm = int(cm.to_numpy().sum())
        st.code(f"accuracy = {diag:,} / {total_cm:,} = {diag / max(total_cm, 1):.4f}")
        fig = px.imshow(cm, color_continuous_scale="Viridis", aspect="auto",
                        title="Step 9 · confusion matrix from your last trained model",
                        labels={"x": "Predicted severity", "y": "Actual severity", "color": "Records"})
        show_chart(style_plot(fig), "math_step9_confusion", force_3d=True)
    else:
        st.info("Train a model on the AI Experiment page and its confusion matrix will appear here "
                "with the accuracy worked out step by step.")

    # ---- Step 10: missing values ------------------------------------------
    step(10, "Data quality: missing-value percentages",
         "The data-quality graph divides the number of empty cells in a column by the number of rows.")
    st.latex(r"\text{missing}_j=\frac{\text{empty cells in column }j}{n}\times100\%")
    miss = df.isna().sum()
    miss_df = pd.DataFrame({"Column": miss.index, "Empty cells": miss.values})
    miss_df["Missing (%)"] = (miss_df["Empty cells"] / max(len(df), 1) * 100).round(2)
    miss_df = miss_df.sort_values("Missing (%)", ascending=False).head(12)
    st.dataframe(miss_df, use_container_width=True, hide_index=True)
    if not miss_df.empty and miss_df["Missing (%)"].max() > 0:
        r0 = miss_df.iloc[0]
        st.code(f"{r0['Column']}: {int(r0['Empty cells']):,} / {len(df):,} x 100 = {r0['Missing (%)']:.2f}%")
        fig = px.bar(miss_df.sort_values("Missing (%)"), x="Missing (%)", y="Column", orientation="h",
                     color="Missing (%)", color_continuous_scale=["#14263b", "#FFB300", "#FF4081"],
                     title="Step 10 · missing values per column")
        show_chart(style_plot(fig), "math_step10_missing", force_3d=True)
    else:
        st.success("No empty cells were found in the processed columns.")

    st.warning(
        "All calculations are descriptive and use the loaded sample after your filters. They "
        "do not prove causation, diagnose patients or measure hospital quality."
    )


# ============================================================
# AI EXPERIMENT
# ============================================================

elif page == "AI Experiment":

    st.title("AI Experiment: Predicting Severity")

    st.write(
        "Train a demonstration machine-learning model to predict standardized "
        "severity categories using demographic and administrative features. "
        "This experiment is educational and is not intended for clinical use."
    )

    if not SKLEARN_AVAILABLE:
        st.error(
            "scikit-learn is not installed. Add scikit-learn to requirements.txt "
            "and redeploy the application."
        )
        st.stop()

    st.warning(
        "The model uses the loaded dataset sample. Evaluation results may not "
        "represent performance on other hospitals or years."
    )

    ai_data = filtered_df[
        filtered_df["Severity"].astype(str) != "Unknown"
    ].copy()

    ai_data["Target"] = ai_data["Severity"].astype(str)

    features_available = [
        "Age_Group",
        "Gender",
        "Race",
        "Payment_Type",
        "Diagnosis",
        "County",
    ]

    selected_features = st.multiselect(
        "Select model input features",
        features_available,
        default=[
            "Age_Group",
            "Gender",
            "Payment_Type",
            "Diagnosis",
        ],
    )

    model_type = st.selectbox(
        "Choose a model",
        [
            "Random Forest",
            "Dummy baseline",
        ],
    )

    test_percentage = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    if st.button(
        "Train and evaluate model",
        type="primary",
        use_container_width=True,
    ):
        if not selected_features:
            st.error("Select at least one input feature.")
            st.stop()

        model_data = ai_data[
            selected_features + ["Target"]
        ].copy()

        class_counts = model_data["Target"].value_counts()
        eligible_classes = class_counts[
            class_counts >= 2
        ].index.tolist()

        model_data = model_data[
            model_data["Target"].isin(eligible_classes)
        ]

        if len(model_data) < 100:
            st.error(
                "There are not enough records for this model demonstration."
            )
            st.stop()

        X = model_data[selected_features].copy()
        y = model_data["Target"].astype(str)

        class_counts = y.value_counts()

        if len(class_counts) < 2:
            st.error("At least two severity classes are required.")
            st.stop()

        test_count = max(
            len(class_counts),
            int(np.ceil(len(model_data) * test_percentage / 100)),
        )

        if len(model_data) - test_count < len(class_counts):
            st.error(
                "There are too few training records for the number of classes."
            )
            st.stop()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_count,
                random_state=42,
                stratify=y,
            )

            preprocessing = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        Pipeline(
                            steps=[
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="most_frequent"
                                    ),
                                ),
                                (
                                    "encoder",
                                    OneHotEncoder(
                                        handle_unknown="ignore"
                                    ),
                                ),
                            ]
                        ),
                        selected_features,
                    )
                ],
                remainder="drop",
            )

            if model_type == "Random Forest":
                classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            else:
                classifier = DummyClassifier(
                    strategy="most_frequent"
                )

            model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("classifier", classifier),
                ]
            )

            with st.spinner("Training the model..."):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            st.session_state["ai_results"] = {
                "accuracy": accuracy,
                "model_type": model_type,
                "y_test": y_test.tolist(),
                "predictions": predictions.tolist(),
                "features": selected_features,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
            }

        except Exception as exc:
            st.error("The model could not be trained.")
            st.code(str(exc))

    if "ai_results" in st.session_state:
        results = st.session_state["ai_results"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test accuracy",
            f"{results['accuracy'] * 100:.2f}%",
        )

        c2.metric(
            "Training records",
            format_number(results["training_rows"]),
        )

        c3.metric(
            "Testing records",
            format_number(results["testing_rows"]),
        )

        st.subheader("Actual versus predicted severity")

        comparison = pd.DataFrame(
            {
                "Actual severity": results["y_test"],
                "Predicted severity": results["predictions"],
            }
        )

        confusion = pd.crosstab(
            comparison["Actual severity"],
            comparison["Predicted severity"],
        )

        fig = px.imshow(
            confusion,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Confusion matrix",
            labels={
                "x": "Predicted severity",
                "y": "Actual severity",
                "color": "Records",
            },
        )

        show_chart(
            style_plot(fig),
            "ai_confusion_matrix",
        )

        st.subheader("Classification report")

        report = classification_report(
            results["y_test"],
            results["predictions"],
            output_dict=True,
            zero_division=0,
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            use_container_width=True,
        )

        st.write(f"**Model:** {results['model_type']}")
        st.write(
            "**Features:** " + ", ".join(results["features"])
        )

        st.caption(
            "Accuracy alone does not describe all model errors. "
            "Review precision, recall, class imbalance, and the confusion matrix. "
            "This model is not validated for patient care."
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("Data Explorer")

    st.write(
        "Search processed records, narrow the data by length of stay and "
        "charges, inspect the results, and download a CSV."
    )

    search_text = st.text_input(
        "Search facility, diagnosis, county, region, or record ID",
        placeholder="Enter a search term...",
    )

    explorer_df = filtered_df.copy()

    if search_text.strip():
        search_mask = pd.Series(
            False,
            index=explorer_df.index,
        )

        for column in [
            "Facility",
            "Diagnosis",
            "County",
            "Record_ID",
            "Region",
        ]:
            search_mask |= explorer_df[column].astype(str).str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )

        explorer_df = explorer_df[search_mask]

    if explorer_df.empty:
        st.info("No records match the current search.")
    else:
        minimum_los = float(
            max(0, np.floor(explorer_df["LOS_Days"].min()))
        )

        maximum_los = float(
            np.ceil(explorer_df["LOS_Days"].max())
        )

        if maximum_los <= minimum_los:
            maximum_los = minimum_los + 1

        los_range = st.slider(
            "Length-of-stay range (days)",
            min_value=minimum_los,
            max_value=maximum_los,
            value=(minimum_los, maximum_los),
        )

        explorer_df = explorer_df[
            explorer_df["LOS_Days"].between(
                los_range[0],
                los_range[1],
            )
        ]

        if explorer_df.empty:
            st.info("No records fall within the selected stay range.")
        else:
            charge_maximum = float(
                explorer_df["Total_Charges"].max()
            )

            charge_maximum = max(charge_maximum, 1.0)

            charge_limit = st.number_input(
                "Maximum total charges ($)",
                min_value=0.0,
                max_value=charge_maximum,
                value=charge_maximum,
                step=max(charge_maximum / 20, 1.0),
            )

            explorer_df = explorer_df[
                explorer_df["Total_Charges"] <= charge_limit
            ]

            st.metric(
                "Records matching explorer filters",
                format_number(len(explorer_df)),
            )

            st.dataframe(
                explorer_df,
                use_container_width=True,
                hide_index=True,
                height=500,
            )

            st.download_button(
                "Download displayed records as CSV",
                data=explorer_df.to_csv(index=False).encode("utf-8"),
                file_name="mediscope_filtered_records.csv",
                mime="text/csv",
                use_container_width=True,
            )

            numeric_columns = [
                column for column in [
                    "LOS_Days",
                    "Total_Charges",
                    "Discharge_Year",
                ]
                if column in explorer_df.columns
                and pd.api.types.is_numeric_dtype(explorer_df[column])
            ]

            if numeric_columns:
                st.subheader("Numeric summary")

                st.dataframe(
                    explorer_df[numeric_columns].describe().transpose(),
                    use_container_width=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color:#00e5ff;">MediScope Health Analytics</strong><br>
        Hospital Inpatient Discharge Intelligence Dashboard<br>
        Built with Streamlit, Pandas, Plotly, scikit-learn, and Three.js<br>
        <span style="color:#526b85;">
            Analytics are descriptive and educational. They should not be
            interpreted as clinical advice or proof of hospital quality.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
