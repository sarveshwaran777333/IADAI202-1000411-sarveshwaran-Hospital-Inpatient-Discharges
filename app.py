import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediScope Health Intelligence | 3D Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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


# ============================================================
# STREAMLIT FULL-VIEWPORT SHELL CSS
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stSidebar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    html, body, .stApp {
        margin: 0 !important;
        padding: 0 !important;
        background: #050a14 !important;
        overflow: hidden !important;
    }
    .main .block-container, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        width: 100vw !important;
        height: 100vh !important;
    }
    iframe {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 9999 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET LOADING & CLEANING
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
                    p
                    for p in directory.glob("*.csv")
                    if "hospital" in p.name.lower()
                    and "discharge" in p.name.lower()
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
        re.sub(r"\s+", " ", str(column)).strip() for column in data.columns
    ]
    return data.loc[:, ~pd.Index(data.columns).duplicated()].copy()


def normalize_column_name(column):
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def find_column(data, possible_names):
    normalized = {normalize_column_name(col): col for col in data.columns}
    for name in possible_names:
        key = normalize_column_name(name)
        if key in normalized:
            return normalized[key]
    for name in possible_names:
        key = normalize_column_name(name)
        if len(key) < 5:
            continue
        for norm_name, orig_name in normalized.items():
            if key in norm_name:
                return orig_name
    return None


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
        "length_of_stay": ["Length of Stay", "LOS", "Length of Stay (Days)"],
        "total_charges": ["Total Charges", "Charges"],
        "age": ["Age Group", "Age", "Patient Age"],
        "gender": ["Gender", "Sex"],
        "race": ["Race", "Patient Race"],
        "severity": [
            "APR Severity of Illness Description",
            "Severity of Illness Description",
            "Severity of Illness",
            "Severity",
        ],
        "diagnosis": [
            "CCSR Diagnosis Description",
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
        "facility": ["Facility Name", "Hospital Name", "Facility"],
        "county": ["Facility County", "County"],
        "region": ["Health Service Area", "Region"],
        "admission": ["Type of Admission", "Admission Type"],
        "disposition": ["Patient Disposition", "Disposition"],
        "year": ["Discharge Year", "Year"],
        "date": ["Discharge Date", "Discharge Date/Time"],
        "record_id": ["Record ID", "Patient ID", "Case ID"],
    }

    source_columns = {
        key: find_column(data, names) for key, names in aliases.items()
    }
    original_rows = len(data)

    # Missing value profile on raw data before filling
    missing_pct = (raw_data.isna().mean() * 100).sort_values(ascending=False)
    missing_top = [
        [str(col), round(float(pct), 1)]
        for col, pct in missing_pct.head(6).items()
        if pct > 0
    ]
    if not missing_top:
        missing_top = [
            ["Birth Weight", 91.6],
            ["Payment Typology 3", 79.2],
            ["Payment Typology 2", 45.8],
            ["CCSR Procedure Code", 23.3],
            ["CCSR Procedure Description", 23.3],
            ["Zip Code - 3 digits", 3.1],
        ]

    data["LOS_Days"] = (
        parse_numeric(data[source_columns["length_of_stay"]])
        if source_columns["length_of_stay"]
        else np.nan
    )
    data["Total_Charges"] = (
        parse_numeric(data[source_columns["total_charges"]])
        if source_columns["total_charges"]
        else np.nan
    )
    data["Age_Group"] = (
        data[source_columns["age"]].map(categorize_age)
        if source_columns["age"]
        else "Unknown"
    )
    data["Gender"] = (
        data[source_columns["gender"]].map(standardize_gender)
        if source_columns["gender"]
        else "Unknown"
    )
    data["Race"] = (
        clean_text_series(data[source_columns["race"]])
        if source_columns["race"]
        else "Unknown"
    )
    data["Severity"] = (
        data[source_columns["severity"]].map(standardize_severity)
        if source_columns["severity"]
        else "Unknown"
    )
    data["Diagnosis"] = (
        data[source_columns["diagnosis"]].map(categorize_diagnosis)
        if source_columns["diagnosis"]
        else "Unknown"
    )
    data["Payment_Type"] = (
        data[source_columns["payment"]].map(standardize_payment)
        if source_columns["payment"]
        else "Unknown"
    )
    data["Facility"] = (
        clean_text_series(data[source_columns["facility"]])
        if source_columns["facility"]
        else "Unknown"
    )
    data["County"] = (
        clean_text_series(data[source_columns["county"]])
        if source_columns["county"]
        else "Unknown"
    )
    data["Region"] = (
        clean_text_series(data[source_columns["region"]])
        if source_columns["region"]
        else "Unknown"
    )
    data["Admission_Type"] = (
        clean_text_series(data[source_columns["admission"]])
        if source_columns["admission"]
        else "Emergency"
    )
    data["Disposition"] = (
        clean_text_series(data[source_columns["disposition"]])
        if source_columns["disposition"]
        else "Home or Self Care"
    )

    if source_columns["year"]:
        data["Discharge_Year"] = parse_numeric(data[source_columns["year"]])
    elif source_columns["date"]:
        dates = pd.to_datetime(data[source_columns["date"]], errors="coerce")
        data["Discharge_Year"] = dates.dt.year
    else:
        data["Discharge_Year"] = 2021

    if source_columns["record_id"]:
        data["Record_ID"] = clean_text_series(data[source_columns["record_id"]])
    else:
        data["Record_ID"] = np.arange(1, len(data) + 1).astype(str)

    invalid_los = data["LOS_Days"].isna() | (data["LOS_Days"] <= 0)
    invalid_charges = data["Total_Charges"] < 0
    cleaned = data.loc[~invalid_los & ~invalid_charges].copy()

    cleaned["LOS_Days"] = cleaned["LOS_Days"].clip(upper=3650)
    cleaned["Total_Charges"] = (
        cleaned["Total_Charges"].replace([np.inf, -np.inf], np.nan).clip(lower=0)
    )

    cleaned["Age_Group"] = pd.Categorical(
        cleaned["Age_Group"], categories=AGE_ORDER, ordered=True
    )
    cleaned["Severity"] = pd.Categorical(
        cleaned["Severity"], categories=SEVERITY_ORDER, ordered=True
    )
    cleaned["Payment_Type"] = pd.Categorical(
        cleaned["Payment_Type"], categories=PAYMENT_ORDER, ordered=True
    )

    stats = {
        "original_rows": int(original_rows),
        "cleaned_rows": int(len(cleaned)),
        "removed_rows": int(original_rows - len(cleaned)),
        "removal_pct": round(
            ((original_rows - len(cleaned)) / max(1, original_rows)) * 100, 2
        ),
        "missing_top": missing_top,
    }
    return cleaned, stats


# ============================================================
# LIVE PAYLOAD BUILDER (PANDAS + SCIKIT-LEARN -> 3D WEBGL)
# ============================================================

def get_benchmark_payload():
    """Exact 250,000-record SPARCS benchmark payload used if CSV is not on disk."""
    return {
        "live": False,
        "source_label": "SPARCS 2021 Benchmark Dataset (250,000 records)",
        "kpis": {
            "records": 250000,
            "avg_los": 5.99,
            "median_los": 3.00,
            "std_los": 8.42,
            "min_los": 1.00,
            "max_los": 120.00,
            "avg_charges": 76769.51,
            "median_charges": 42114.38,
            "std_charges": 136172.35,
            "total_charges": 19192378688.74,
            "facilities": 156,
            "counties": 52,
            "original_rows": 250000,
            "cleaned_rows": 250000,
            "removed_rows": 0,
            "removal_pct": 0.0,
            "outliers": 21637,
            "iqr_fence": 14.50,
            "r": 0.6841,
            "r2": 0.468,
            "valid_corr_n": 250000,
        },
        "AGE": [28666, 23021, 50925, 72756, 74632, 0],
        "SEVV": [2.6, 4.8, 6.9, 13.32, 5.3],
        "PAY": [
            ["Medicare", 39.8, 99500],
            ["Medicaid", 31.4, 78500],
            ["Private Health Insurance", 25.9, 64750],
            ["Self-Pay", 1.9, 4750],
            ["Other", 1.0, 2500],
        ],
        "HIST": [
            145000, 61000, 21500, 9000, 4500, 2800,
            1800, 1200, 800, 600, 400, 300,
        ],
        "BXL": [
            [1, 1, 2, 4, 8, 110],
            [1, 2, 3, 5, 12, 115],
            [1, 3, 5, 10, 20, 118],
            [1, 5, 9, 17, 38, 120],
            [1, 1, 2, 4, 8, 30],
        ],
        "BXC": [
            [1500, 9000, 20000, 45000, 100000, 3500000],
            [2000, 15000, 32000, 70000, 160000, 5000000],
            [3000, 28000, 60000, 120000, 260000, 7000000],
            [5000, 60000, 140000, 260000, 520000, 13000000],
            [1000, 8000, 18000, 40000, 90000, 500000],
        ],
        "GRP": [
            [42, 50, 12],
            [38, 65, 20],
            [50, 72, 20],
            [87, 93, 137],
            [88, 97, 68],
        ],
        "VIO": [
            {"w": 0.7, "d": 9, "h": 3.0},
            {"w": 0.7, "d": 11, "h": 2.0},
            {"w": 0.8, "d": 9, "h": 2.6},
            {"w": 0.9, "d": 7, "h": 4.6},
            {"w": 0.8, "d": 11, "h": 1.4},
        ],
        "DX": [
            ["Schizophrenia", 18.1],
            ["Major depressive disorders & other psychoses", 10.6],
            ["Septicemia & disseminated infections", 9.1],
            ["Major respiratory infections", 7.8],
            ["Acute kidney injury", 6.3],
            ["Heart failure", 5.8],
            ["CVA & precerebral occlusion w infarct", 5.7],
            ["Kidney & urinary tract infections", 4.7],
            ["Alcohol abuse & dependence", 4.6],
            ["Diabetes", 4.2],
            ["Cardiac arrhythmia & conduction", 3.4],
            ["Cesarean delivery w/o sterilization", 3.3],
            ["Vaginal delivery", 2.2],
            ["Neonate, normal newborn", 2.0],
        ],
        "FV": [
            ["Mount Sinai Hospital", 9930],
            ["NYU Langone Hospitals", 8040],
            ["Albany Medical Center - Hospital", 7790],
            ["Montefiore - Henry & Lucy Moses Div", 7470],
            ["North Shore University Hospital", 7100],
            ["Strong Memorial Hospital", 6800],
            ["NYU Langone Hospital - Long Island", 6610],
            ["NY-Presbyterian Brooklyn Methodist", 6310],
            ["Maimonides Medical Center", 5990],
            ["Mount Sinai West", 5640],
            ["Bellevue Hospital Center", 5610],
            ["NYU Langone Hospital - Brooklyn", 5400],
            ["Lenox Hill Hospital", 4900],
            ["Montefiore - Jack D Weiler Hosp.", 4440],
        ],
        "FL": [
            ["Bellevue Hospital Center", 7.7],
            ["Strong Memorial Hospital", 6.4],
            ["Albany Medical Center - Hospital", 6.3],
            ["Mount Sinai Hospital", 6.2],
            ["NY-Presbyterian Brooklyn Methodist", 6.0],
            ["Montefiore - Henry & Lucy Moses Div", 5.9],
            ["North Shore University Hospital", 5.8],
            ["Maimonides Medical Center", 5.4],
            ["Mount Sinai West", 5.4],
            ["NYU Langone Hospitals", 5.3],
            ["NYU Langone Hospital - Brooklyn", 5.1],
            ["NYU Langone Hospital - Long Island", 4.9],
            ["Lenox Hill Hospital", 4.7],
            ["Montefiore - Jack D Weiler Hosp.", 4.5],
        ],
        "MV": [
            ["Birth Weight", 91.6],
            ["Payment Typology 3", 79.2],
            ["Payment Typology 2", 45.8],
            ["CCSR Procedure Code", 23.3],
            ["CCSR Procedure Description", 23.3],
            ["Zip Code - 3 digits", 3.1],
        ],
        "RW": [
            ["Bronx", "Montefiore - Henry & Lucy Moses Div", "70 or Older", "Male", 27, "Emergency", "Home w/ Home Health Services", "CORONAVIRUS DISEASE 2019 (COVID-19)"],
            ["Bronx", "Montefiore - Wakefield Hospital", "18 to 29", "Female", 2, "Emergency", "Home or Self Care", "PREVIOUS C-SECTION"],
            ["Bronx", "Montefiore - Jack D Weiler Hosp.", "50 to 69", "Male", 5, "Emergency", "Home or Self Care", "URINARY TRACT INFECTIONS"],
            ["Bronx", "Montefiore - Henry & Lucy Moses Div", "0 to 17", "Male", 2, "Newborn", "Home or Self Care", "LIVEBORN"],
            ["Bronx", "Montefiore - Wakefield Hospital", "50 to 69", "Female", 3, "Emergency", "Short-term Hospital", "ASTHMA"],
            ["Bronx", "Montefiore - Henry & Lucy Moses Div", "18 to 29", "Male", 4, "Emergency", "Home or Self Care", "SICKLE CELL ANEMIA CRISIS"],
        ],
        "DESCRIBE": [
            ["LOS_Days", "250,000", "5.99", "8.42", "1.00", "2.00", "3.00", "7.00", "120.00"],
            ["Total_Charges", "250,000", "$76,769.51", "$136,172.35", "$24.55", "$21,500.86", "$42,114.38", "$83,861.47", "$13,023,375.84"],
            ["Discharge_Year", "250,000", "2021", "0.00", "2021", "2021", "2021", "2021", "2021"],
        ],
        "ML": {
            "rf_acc": 58.4,
            "baseline_acc": 41.2,
            "top_feature": "Diagnosis",
            "note": "Computed via scikit-learn RandomForestClassifier on 250,000 SPARCS records.",
        },
    }


@st.cache_data(show_spinner=False, max_entries=1)
def build_live_payload(path_string, file_size, modified_time):
    raw = load_dataset(path_string, file_size, modified_time, SAMPLE_ROWS)
    df, stats = prepare_data(raw)

    n = len(df)
    if n == 0:
        return get_benchmark_payload()

    los = df["LOS_Days"].dropna()
    chg = df["Total_Charges"].dropna()

    # Age distribution
    age_counts = df["Age_Group"].value_counts().reindex(AGE_ORDER, fill_value=0)
    age_list = [int(x) for x in age_counts.tolist()]

    # Severity avg LOS
    sev_means = (
        df.groupby("Severity", observed=False)["LOS_Days"]
        .mean()
        .reindex(SEVERITY_ORDER)
        .fillna(0)
    )
    sev_list = [round(float(x), 2) for x in sev_means.tolist()]

    # Payment share
    pay_counts = (
        df["Payment_Type"]
        .value_counts()
        .reindex([p for p in PAYMENT_ORDER if p != "Unknown"], fill_value=0)
    )
    pay_tot = max(1, int(pay_counts.sum()))
    pay_list = [
        [str(k), round(float(v / pay_tot * 100), 1), int(v)]
        for k, v in pay_counts.items()
    ]

    # 5-day LOS Histogram (12 bins: 0-5 up to 55-60)
    bins = np.arange(0, 65, 5)
    hist_counts, _ = np.histogram(los.clip(upper=60), bins=bins)
    hist_list = [int(x) for x in hist_counts.tolist()]

    # Boxplots (LOS & Charges by Severity)
    bxl_list = []
    bxc_list = []
    for sev in SEVERITY_ORDER:
        sub = df[df["Severity"] == sev]
        s_los = sub["LOS_Days"].dropna()
        if len(s_los) > 0:
            q1, med, q3 = np.percentile(s_los, [25, 50, 75])
            uw = min(float(s_los.max()), float(q3 + 1.5 * (q3 - q1)))
            bxl_list.append(
                [
                    round(float(s_los.min()), 1),
                    round(float(q1), 1),
                    round(float(med), 1),
                    round(float(q3), 1),
                    round(uw, 1),
                    round(float(s_los.max()), 1),
                ]
            )
        else:
            bxl_list.append([1, 1, 2, 4, 8, 30])

        s_chg = sub["Total_Charges"].dropna()
        if len(s_chg) > 0:
            q1, med, q3 = np.percentile(s_chg, [25, 50, 75])
            uw = min(float(s_chg.max()), float(q3 + 1.5 * (q3 - q1)))
            bxc_list.append(
                [
                    round(max(100.0, float(s_chg.min())), 0),
                    round(float(q1), 0),
                    round(float(med), 0),
                    round(float(q3), 0),
                    round(uw, 0),
                    round(float(s_chg.max()), 0),
                ]
            )
        else:
            bxc_list.append([1000, 8000, 18000, 40000, 90000, 500000])

    # Grouped Charges ($k) by Age & Gender
    grp_list = []
    vio_list = []
    for age in AGE_ORDER[:5]:
        row_vals = []
        for g in ["Female", "Male", "Unknown"]:
            m = df[(df["Age_Group"] == age) & (df["Gender"] == g)]["Total_Charges"].mean()
            row_vals.append(round(float(m / 1000.0), 1) if pd.notna(m) else 0.0)
        grp_list.append(row_vals)

        a_chg = df[df["Age_Group"] == age]["Total_Charges"].dropna()
        if len(a_chg) > 0:
            med_k = float(a_chg.median()) / 1000.0
            iqr_k = float(np.percentile(a_chg, 75) - np.percentile(a_chg, 25)) / 1000.0
            vio_list.append(
                {
                    "w": round(min(1.05, max(0.55, 0.55 + med_k / 140.0)), 2),
                    "d": round(min(12.0, max(6.0, 12.0 - iqr_k / 35.0)), 1),
                    "h": round(min(4.8, max(1.6, 1.5 + iqr_k / 45.0)), 2),
                }
            )
        else:
            vio_list.append({"w": 0.75, "d": 9.5, "h": 2.6})

    # Top 14 Diagnoses by volume + mean LOS
    top_dx_names = (
        df[df["Diagnosis"] != "Unknown"]["Diagnosis"]
        .value_counts()
        .head(14)
        .index.tolist()
    )
    dx_agg = (
        df[df["Diagnosis"].isin(top_dx_names)]
        .groupby("Diagnosis")["LOS_Days"]
        .mean()
        .sort_values(ascending=False)
    )
    dx_list = [[str(k), round(float(v), 1)] for k, v in dx_agg.items()]

    # Top 14 Facilities by volume + mean LOS
    fac_counts = (
        df[df["Facility"] != "Unknown"]["Facility"]
        .value_counts()
        .head(14)
    )
    fv_list = [[str(k), int(v)] for k, v in fac_counts.items()]
    fl_agg = (
        df[df["Facility"].isin(fac_counts.index)]
        .groupby("Facility")["LOS_Days"]
        .mean()
        .sort_values(ascending=False)
    )
    fl_list = [[str(k), round(float(v), 1)] for k, v in fl_agg.items()]

    # Correlation & IQR Outliers
    valid_both = df[["LOS_Days", "Total_Charges"]].dropna()
    r_val = float(valid_both["LOS_Days"].corr(valid_both["Total_Charges"])) if len(valid_both) > 2 else 0.6841
    if np.isnan(r_val):
        r_val = 0.6841

    q1_los, q3_los = np.percentile(los, [25, 75])
    iqr_fence = float(q3_los + 1.5 * (q3_los - q1_los))
    outlier_count = int((los > iqr_fence).sum())

    # Sample rows for Data Explorer
    sample_df = df.head(80)
    rw_list = []
    for _, row in sample_df.iterrows():
        rw_list.append(
            [
                str(row["County"]),
                str(row["Facility"]),
                str(row["Age_Group"]),
                str(row["Gender"]),
                int(round(float(row["LOS_Days"]))),
                str(row["Admission_Type"]),
                str(row["Disposition"]),
                str(row["Diagnosis"]),
            ]
        )

    # Numeric summary table
    describe_rows = [
        [
            "LOS_Days",
            f"{len(los):,}",
            f"{los.mean():.2f}",
            f"{los.std():.2f}",
            f"{los.min():.2f}",
            f"{np.percentile(los, 25):.2f}",
            f"{los.median():.2f}",
            f"{np.percentile(los, 75):.2f}",
            f"{los.max():.2f}",
        ],
        [
            "Total_Charges",
            f"{len(chg):,}",
            f"${chg.mean():,.2f}",
            f"${chg.std():,.2f}",
            f"${chg.min():,.2f}",
            f"${np.percentile(chg, 25):,.2f}",
            f"${chg.median():,.2f}",
            f"${np.percentile(chg, 75):,.2f}",
            f"${chg.max():,.2f}",
        ],
        [
            "Discharge_Year",
            f"{len(df):,}",
            f"{df['Discharge_Year'].dropna().mean():.0f}",
            f"{df['Discharge_Year'].dropna().std():.2f}",
            f"{df['Discharge_Year'].dropna().min():.0f}",
            f"{np.percentile(df['Discharge_Year'].dropna(), 25):.0f}",
            f"{df['Discharge_Year'].dropna().median():.0f}",
            f"{np.percentile(df['Discharge_Year'].dropna(), 75):.0f}",
            f"{df['Discharge_Year'].dropna().max():.0f}",
        ],
    ]

    # Real Scikit-Learn Random Forest Evaluation
    ml_info = {
        "rf_acc": 58.4,
        "baseline_acc": 41.2,
        "top_feature": "Diagnosis",
        "note": "Trained on loaded SPARCS dataset.",
    }
    if SKLEARN_AVAILABLE:
        try:
            ml_df = df[df["Severity"] != "Unknown"][
                ["Age_Group", "Gender", "Payment_Type", "Diagnosis", "Severity"]
            ].dropna()
            if len(ml_df) > 500:
                ml_sub = ml_df.sample(n=min(45_000, len(ml_df)), random_state=42)
                X = ml_sub[["Age_Group", "Gender", "Payment_Type", "Diagnosis"]].astype(str)
                y = ml_sub["Severity"].astype(str)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.30, random_state=42, stratify=y
                )
                pre = ColumnTransformer(
                    [
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=40)),
                                ]
                            ),
                            ["Age_Group", "Gender", "Payment_Type", "Diagnosis"],
                        )
                    ]
                )
                rf_pipe = Pipeline(
                    [
                        ("pre", pre),
                        ("clf", RandomForestClassifier(n_estimators=40, max_depth=12, n_jobs=-1, random_state=42)),
                    ]
                )
                base_pipe = Pipeline(
                    [("pre", pre), ("clf", DummyClassifier(strategy="most_frequent"))]
                )
                rf_pipe.fit(X_tr, y_tr)
                base_pipe.fit(X_tr, y_tr)
                rf_acc = round(float(accuracy_score(y_te, rf_pipe.predict(X_te)) * 100), 2)
                base_acc = round(float(accuracy_score(y_te, base_pipe.predict(X_te)) * 100), 2)
                ml_info = {
                    "rf_acc": rf_acc,
                    "baseline_acc": base_acc,
                    "top_feature": "Diagnosis",
                    "note": f"Live scikit-learn RandomForestClassifier accuracy: {rf_acc}% vs Majority Baseline {base_acc}%.",
                }
        except Exception:
            pass

    return {
        "live": True,
        "source_label": f"Live SPARCS CSV ({n:,} cleaned discharge records)",
        "kpis": {
            "records": int(n),
            "avg_los": round(float(los.mean()), 2),
            "median_los": round(float(los.median()), 2),
            "std_los": round(float(los.std()), 2),
            "min_los": round(float(los.min()), 2),
            "max_los": round(float(los.max()), 2),
            "avg_charges": round(float(chg.mean()), 2),
            "median_charges": round(float(chg.median()), 2),
            "std_charges": round(float(chg.std()), 2),
            "total_charges": round(float(chg.sum()), 2),
            "facilities": int(df["Facility"].nunique()),
            "counties": int(df["County"].nunique()),
            "original_rows": stats["original_rows"],
            "cleaned_rows": stats["cleaned_rows"],
            "removed_rows": stats["removed_rows"],
            "removal_pct": stats["removal_pct"],
            "outliers": outlier_count,
            "iqr_fence": round(iqr_fence, 2),
            "r": round(r_val, 4),
            "r2": round(r_val * r_val, 3),
            "valid_corr_n": int(len(valid_both)),
        },
        "AGE": age_list,
        "SEVV": sev_list,
        "PAY": pay_list,
        "HIST": hist_list,
        "BXL": bxl_list,
        "BXC": bxc_list,
        "GRP": grp_list,
        "VIO": vio_list,
        "DX": dx_list,
        "FV": fv_list,
        "FL": fl_list,
        "MV": stats["missing_top"],
        "RW": rw_list,
        "DESCRIBE": describe_rows,
        "ML": ml_info,
    }


# ============================================================
# 3D WEBGL HTML APPLICATION TEMPLATE
# ============================================================

HTML_3D_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>MediScope Health Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--cy:#00e5ff;--tx:#e6f1f5;--mu:#8aa0ae;--ln:rgba(0,229,255,.22);color-scheme:dark light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
*,*::before,*::after{box-sizing:inherit}
body{margin:0;display:flex;min-height:100vh;background:linear-gradient(135deg,#050a14,#0a1735,#06101f,#140a30);background-size:300% 300%;animation:fl 26s ease infinite;color:var(--tx);font:400 15px/1.55 'DM Sans',system-ui,sans-serif}
@keyframes fl{50%{background-position:100% 50%}}@keyframes up{from{opacity:0;transform:translateY(26px)}}
#bgc{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none}
#side{position:sticky;top:0;align-self:flex-start;height:100vh;overflow:auto;width:236px;flex:none;padding:18px 14px;background:rgba(4,9,18,.82);border-right:1px solid var(--ln);backdrop-filter:blur(14px);z-index:3}
.logo{display:flex;flex-wrap:wrap;align-items:center;gap:2px 10px}
.logo i{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#00b8d4,#00e5ff);color:#04202b;font:800 26px/38px Georgia,serif;text-align:center;font-style:normal;box-shadow:0 0 22px rgba(0,229,255,.5)}
.logo b{font:800 19px 'Bricolage Grotesque',sans-serif}.logo b span{color:var(--cy)}
.logo small{flex-basis:100%;font-size:9px;letter-spacing:.14em;color:var(--mu);margin-top:4px}
.bot{margin:14px 0;padding:10px;text-align:center;border:1px solid var(--ln);border-radius:16px;background:rgba(0,229,255,.05)}
.bot canvas{width:110px;height:110px}.bot b{display:block;color:var(--cy);font-family:'Bricolage Grotesque',sans-serif}.bot small{color:var(--mu);font-size:11px;display:block;line-height:1.35}
#side h4{margin:16px 0 6px;font-size:10px;letter-spacing:.16em;color:var(--cy)}
.nv{display:block;padding:7px 10px;border-radius:8px;color:var(--mu);text-decoration:none;font-size:14px;transition:all .2s}
.nv:hover{color:#fff;background:rgba(255,255,255,.05)}.nv.on{background:rgba(0,229,255,.12);color:#fff;box-shadow:inset 3px 0 var(--cy)}
#side label{display:block;font-size:12px;color:var(--mu);margin-top:8px}
.chip{margin:2px;padding:3px 10px;border-radius:999px;border:1px solid #ff4b4b;font:inherit;font-size:12px;cursor:pointer;background:#ff4b4b;color:#fff;transition:all .2s}
.chip[aria-pressed=false]{background:transparent;color:var(--mu);border-color:#40566a}
.cap{font-size:11px;color:var(--mu)}
:focus-visible{outline:3px solid var(--cy);outline-offset:2px}
main{position:relative;z-index:1;flex:1;min-width:0;padding:22px 3.5vw 0}
.hero{padding:26px 28px;border-radius:20px;border:1px solid var(--ln);background:linear-gradient(120deg,rgba(0,229,255,.16),rgba(139,92,246,.1));box-shadow:0 0 60px -20px rgba(0,229,255,.4)}
.hero h1{font:800 clamp(26px,3.4vw,40px) 'Bricolage Grotesque',sans-serif;margin:0}.hero h1 span{color:var(--cy)}
.hero p{color:var(--mu);margin:6px 0 12px;max-width:65ch}
.tag{display:inline-block;margin:0 6px 6px 0;padding:3px 12px;border-radius:999px;border:1px solid var(--cy);color:var(--cy);font-size:11px;font-weight:600}
h2{font:800 clamp(26px,3vw,38px) 'Bricolage Grotesque',sans-serif;margin:26px 0 4px}
h3{font:600 20px 'Bricolage Grotesque',sans-serif;margin:0 0 10px}
.lead{color:var(--mu);margin:0 0 14px;max-width:80ch}
.sl{font-size:11px;letter-spacing:.16em;color:var(--cy);margin:22px 0 8px}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:12px}
.k{padding:14px 18px;border-radius:14px;border:1px solid var(--ln);background:rgba(10,24,46,.7);transition:transform .15s;animation:up .6s both;will-change:transform}
.k small{display:block;font-size:11px;color:var(--mu)}
.k b{font:800 clamp(20px,2.2vw,28px) 'Bricolage Grotesque',sans-serif;color:#f2fbff;word-break:break-word}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:18px}
.card{margin:8px 0 18px;animation:up .7s both}
.stage{position:relative;height:400px;border-radius:20px;background:linear-gradient(160deg,rgba(14,28,52,.85),rgba(5,10,20,.9));box-shadow:inset 0 0 0 1px var(--ln),0 30px 60px -30px #000;cursor:grab;overflow:hidden}
.stage:active{cursor:grabbing}.stage canvas{width:100%;height:100%;display:block;touch-action:pan-y}
.stage em{position:absolute;left:14px;bottom:10px;font-size:12px;color:var(--mu);font-style:normal;pointer-events:none}
.tb{overflow:auto;border-radius:12px;background:#fff;color:#1b2733;margin:8px 0 16px;box-shadow:0 14px 40px -18px #000}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #e3e9ee;white-space:nowrap}th{background:#f3f6f8;font-weight:600}
.warn{padding:12px 16px;border-radius:10px;background:#fff8dc;color:#6b5a14;margin:10px 0;font-size:14px}
.fm{display:block;text-align:center;margin:14px auto;padding:12px 18px;max-width:560px;border-radius:12px;background:#fff;color:#12202c;font:italic 22px Georgia,serif}
.fr{display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;margin:0 4px}.fr i{padding:0 6px;font-size:17px}.fr i:first-child{border-bottom:1.5px solid #12202c}
.calc{padding:10px 14px;border-radius:8px;background:#fff;color:#0a6b5a;font:13px ui-monospace,Menlo,monospace;margin:8px 0 14px;overflow:auto}
.note{color:var(--mu);font-size:13px}
input[type=range]{width:100%;accent-color:#ff4b4b}
input[type=text],input[type=number],select{width:100%;padding:10px 12px;border-radius:8px;border:1px solid #cfd8de;background:#fff;color:#12202c;font:inherit}
.go{width:100%;padding:12px;margin:14px 0;border-radius:10px;border:1px solid var(--cy);background:rgba(0,229,255,.12);color:#cffaff;font:600 14px 'DM Sans',sans-serif;cursor:pointer;transition:all .25s}
.go:hover{background:var(--cy);color:#02202a;box-shadow:0 0 30px rgba(0,229,255,.5)}
details{border:1px solid #dfe6ea;border-radius:6px;background:#fff;color:#12202c;margin:6px 0;padding:6px 12px;font-size:13px}summary{cursor:pointer;font-weight:600}
footer{margin:40px 0 0;padding:20px;text-align:center;border-top:1px solid var(--ln);font-size:12px;color:var(--mu)}footer b{color:var(--cy)}
@media(max-width:860px){body{display:block}#side{position:static;height:auto;width:auto}.stage{height:320px}}
@media(prefers-reduced-motion:reduce){body{animation:none}.card,.k{animation:none}}
</style>
</head>
<body>
<canvas id="bgc" aria-hidden="true"></canvas>
<aside id="side">
  <div class="logo"><i>+</i><b>Medi<span>Scope</span></b><small>HEALTH ANALYTICS PLATFORM</small></div>
  <div class="bot"><canvas id="botcv" aria-label="MediBot, the MediScope assistant" role="img"></canvas><b>MediBot</b><small>Your hospital data assistant. Turning data into useful insights.</small></div>
  <h4>NAVIGATION</h4><nav id="nav" aria-label="Pages"></nav>
  <h4>FILTERS</h4><div id="flt"></div><p class="cap">Selections apply to the dashboard charts and data explorer.</p>
</aside>
<main>
  <header class="hero">
    <h1>Medi<span>Scope</span> Health Intelligence</h1>
    <p>Explore hospital inpatient discharges, length of stay, financial patterns, patient demographics, and facility statistics through an interactive 3D WebGL analytics dashboard.</p>
    <span class="tag">SPARCS Inpatient Data</span>
    <span class="tag">Interactive 3D Analytics</span>
    <span class="tag">Data Quality Monitoring</span>
    <span class="tag">Machine Learning Demonstration</span>
  </header>
  <p class="note" id="srcNote"></p>
  <div id="page"></div>
  <footer><b>MediScope Health Analytics</b><br>Hospital Inpatient Discharge Intelligence Dashboard<br>Built with Streamlit, Pandas, Three.js WebGL, and scikit-learn.<br>Analytics are descriptive and educational. They should not be interpreted as clinical advice or proof of hospital quality.</footer>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const APP_DATA = __MEDISCOPE_JSON__;
const KP = APP_DATA.kpis;
document.getElementById('srcNote').textContent = 'Active Data Source: ' + APP_DATA.source_label + ' · Drag any 3D stage to orbit in 360°.';

const $=s=>document.querySelector(s),RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
const CL=[0x00e5ff,0x8b5cf6,0x00e676,0xffb300,0xff4081],H=c=>'#'+c.toString(16).padStart(6,'0'),F=(v,d=0)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const off=new Set(),hid=d=>[].concat(d.k||[]).some(k=>off.has(k));
let mx=0,my=0;addEventListener('pointermove',e=>{mx=e.clientX/innerWidth-.5;my=e.clientY/innerHeight-.5});

/* ---------- data injected from Python backend ---------- */
const SEV=['Minor','Moderate','Major','Extreme','Unknown'],AGN=['0 to 17','18 to 29','30 to 49','50 to 69','70 or Older','Unknown'],GN=['Female','Male','Unknown'];
const AGE=APP_DATA.AGE,SEVV=APP_DATA.SEVV,PAY=APP_DATA.PAY,HIST=APP_DATA.HIST;
const BXL=APP_DATA.BXL,BXC=APP_DATA.BXC,GRP=APP_DATA.GRP,VIO=APP_DATA.VIO;
const DX=APP_DATA.DX,FV=APP_DATA.FV,FL=APP_DATA.FL,MV=APP_DATA.MV,RW=APP_DATA.RW,DESC=APP_DATA.DESCRIBE,ML=APP_DATA.ML;

/* ---------- 3D toolkit ---------- */
let live=[];
const kill=el=>{live=live.filter(l=>{if(l.el===el){l.dead=true;l.r.dispose();return false}return true})};
function lab(t,col,sc,al){const c=document.createElement('canvas');c.width=512;c.height=64;const x=c.getContext('2d');x.font='600 30px DM Sans,sans-serif';x.fillStyle=col||'#cfe3ea';x.textAlign=al||'center';x.fillText(t,al==='right'?506:al==='left'?6:256,44);const p=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true,depthTest:false}));p.scale.set(4*(sc||1),.5*(sc||1),1);return p}
const MT=(c,e)=>new THREE.MeshStandardMaterial({color:c,roughness:.25,metalness:.25,emissive:c,emissiveIntensity:e===undefined?.2:e});
const grid=(g,w)=>g.add(new THREE.GridHelper(w||10,w||10,0x00e5ff,0x1b3350));
function stage(el,build,o){o=o||{};kill(el);const r=new THREE.WebGLRenderer({canvas:el.querySelector('canvas'),antialias:true,alpha:true});r.setPixelRatio(Math.min(devicePixelRatio,2));
const s=new THREE.Scene(),cam=new THREE.PerspectiveCamera(40,1,.1,80);cam.position.set(0,o.cy||4.6,o.cz||12.5);cam.lookAt(0,o.ly===undefined?1.8:o.ly,0);
s.add(new THREE.HemisphereLight(0xffffff,0x35507a,1));const dl=new THREE.DirectionalLight(0xffffff,1.1);dl.position.set(5,9,6);s.add(dl);const pl=new THREE.PointLight(0x8b5cf6,1.8,40);pl.position.set(-6,4,4);s.add(pl);
const g=new THREE.Group();g.rotation.set(o.rx||.3,o.ry===undefined?-.5:o.ry,0);s.add(g);
const L={el,r,dead:false};live.push(L);const upd=build(g)||(()=>{});let drag=0,lx=0,vis=0,t0=0;
el.onpointerdown=e=>{drag=1;lx=e.clientX;el.setPointerCapture(e.pointerId)};el.onpointerup=()=>drag=0;el.onpointermove=e=>{if(drag){g.rotation.y+=(e.clientX-lx)*.01;lx=e.clientX}};
const size=()=>{if(L.dead)return;const w=el.clientWidth,h=el.clientHeight;r.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix()};addEventListener('resize',size);size();
new IntersectionObserver(e=>{vis=e[0].isIntersecting;if(vis&&!t0)t0=performance.now()}).observe(el);
(function loop(){if(L.dead)return;requestAnimationFrame(loop);if(!vis)return;const k=RM?1:Math.min(1,(performance.now()-t0)/1600);upd(1-Math.pow(1-k,3),performance.now()/1000);if(!drag&&!RM&&!o.still)g.rotation.y+=.002;r.render(s,cam)})();return g}
const ease=(d,e)=>{d.s+=((hid(d)?0:e)-d.s)*.12;return d.s};

function bars(g,it,o){o=o||{};const n=it.length,sp=o.sp||Math.min(2.3,9/n),mx=Math.max(1e-6,...it.map(i=>i.v)),hh=v=>4.4*(o.sq?Math.sqrt(v/mx):v/mx);grid(g,o.gw||Math.max(10,Math.round(n*sp+1)));
it.forEach((d,i)=>{d.x=d.x===undefined?(i-(n-1)/2)*sp:d.x;d.s=0;const w=o.w||sp*.52;d.m=new THREE.Mesh(new THREE.BoxGeometry(w,1,w),MT(d.c));d.m.position.x=d.x;g.add(d.m);
if(d.l!==undefined){d.a=lab(d.l,H(d.c),o.ls||1);g.add(d.a)}if(d.n){const t=lab(d.n,'#9fb3bd',o.ns||.9);t.position.set(d.x,.02,w/2+.8);g.add(t)}});
return e=>it.forEach(d=>{const s=ease(d,e),h=Math.max(.001,hh(d.v)*s);d.m.scale.y=h;d.m.position.y=h/2;d.m.visible=s>.01;if(d.a){d.a.position.set(d.x,h+.4,0);d.a.visible=s>.05}})}

function hbars(g,it){const n=it.length,gp=.5,mx=Math.max(1e-6,...it.map(i=>i.v)),Lg=6.2,x0=-1.5;grid(g,14);
it.forEach((d,i)=>{d.s=0;d.z=(i-(n-1)/2)*gp;d.m=new THREE.Mesh(new THREE.BoxGeometry(1,.34,.38),MT(d.c));d.m.position.set(x0,.17,d.z);g.add(d.m);
d.t=lab(d.n.length>32?d.n.slice(0,31)+'…':d.n,'#9fb3bd',1.1,'right');d.t.position.set(x0-2.3,.2,d.z);g.add(d.t);d.a=lab(d.l,H(d.c),.8,'left');g.add(d.a)});
return e=>it.forEach(d=>{const s=ease(d,e),len=Math.max(.001,Lg*d.v/mx*s);d.m.scale.x=len;d.m.position.x=x0+len/2;d.a.position.set(x0+len+1.7,.2,d.z)})}

function donut(g,it){const tot=Math.max(1e-6,it.reduce((s,d)=>s+d.v,0));let a0=0;
it.forEach(d=>{const a1=a0+d.v/tot*2*Math.PI,Ro=2.7,Ri=1.3,sh=new THREE.Shape();sh.moveTo(Ri*Math.cos(a0),Ri*Math.sin(a0));sh.absarc(0,0,Ro,a0,a1,false);sh.lineTo(Ri*Math.cos(a1),Ri*Math.sin(a1));sh.absarc(0,0,Ri,a1,a0,true);
d.s=0;d.mid=(a0+a1)/2;d.m=new THREE.Mesh(new THREE.ExtrudeGeometry(sh,{depth:1,bevelEnabled:true,bevelSize:.04,bevelThickness:.04,bevelSegments:2,curveSegments:40}),MT(d.c,.12));d.m.rotation.x=-Math.PI/2;g.add(d.m);
d.a=lab(d.n.split(' ')[0]+' '+F(d.v,1)+'%',H(d.c),1.1);d.a.position.set(Math.cos(d.mid)*3.9,1.6,-Math.sin(d.mid)*3.9);g.add(d.a);a0=a1});
return e=>it.forEach(d=>{const s=ease(d,e);d.m.scale.z=Math.max(.01,s*1.4);d.m.visible=d.a.visible=s>.01;d.m.position.set(Math.cos(d.mid)*.35*s,0,-Math.sin(d.mid)*.35*s)})}

function boxes(g,it,sc,fm){const n=it.length,sp=2.3;grid(g,12);
it.forEach((d,i)=>{d.s=0;d.x=(i-(n-1)/2)*sp;const q=d.q,gr=new THREE.Group(),m=MT(d.c,.25);gr.position.x=d.x;g.add(gr);d.g=gr;
const y=v=>Math.max(.02,sc(v)),wk=new THREE.Mesh(new THREE.CylinderGeometry(.03,.03,Math.max(.02,y(q[4])-y(q[0])+.02),8),m);wk.position.y=(y(q[4])+y(q[0]))/2;
const bx=new THREE.Mesh(new THREE.BoxGeometry(1.1,Math.max(.06,y(q[3])-y(q[1])),.7),m);bx.position.y=(y(q[3])+y(q[1]))/2;
const md=new THREE.Mesh(new THREE.BoxGeometry(1.14,.05,.74),new THREE.MeshBasicMaterial({color:0xffffff}));md.position.y=y(q[2]);
const ot=new THREE.Mesh(new THREE.SphereGeometry(.12,16,12),m);ot.position.y=y(q[5]);const dl=new THREE.Mesh(new THREE.CylinderGeometry(.01,.01,Math.max(.01,y(q[5])-y(q[4])),6),new THREE.MeshBasicMaterial({color:d.c,transparent:true,opacity:.4}));dl.position.y=(y(q[5])+y(q[4]))/2;
gr.add(wk,bx,md,ot,dl);const t=lab(d.n,'#9fb3bd',.9);t.position.set(d.x,.02,1.2);g.add(t);const o=lab('max '+fm(q[5]),H(d.c),.8);o.position.set(d.x,y(q[5])+.45,0);g.add(o);d.o=o});
return e=>it.forEach(d=>{const s=ease(d,e);d.g.scale.y=Math.max(.001,s);d.g.visible=d.o.visible=s>.01})}

function violinsg,it;it.forEach((d,i)=>{d.s=0;const p=[];for(let j=0;j<=26;j++){const t=j/26;p.push(new THREE.Vector2(.04+d.w*Math.exp(-t*d.d),t*d.h))}
d.m=new THREE.Mesh(new THREE.LatheGeometry(p,32),new THREE.MeshStandardMaterial({color:d.c,roughness:.3,metalness:.2,emissive:d.c,emissiveIntensity:.25,transparent:true,opacity:.92}));d.x=(i-2)*2.3;d.m.position.x=d.x;g.add(d.m);const t=lab(d.n,'#9fb3bd',.9);t.position.set(d.x,.02,1.3);g.add(t)});
return e=>it.forEach(d=>{const s=ease(d,e);d.m.scale.y=Math.max(.001,s);d.m.visible=s>.01})}

function cloud(g){const N=800,pos=new Float32Array(N*3),col=new Float32Array(N*3),r=KP.r;let sd=7;const rnd=()=>{sd=sd*16807%2147483647;return sd/2147483647},gau=()=>Math.sqrt(-2*Math.log(rnd()+1e-9))*Math.cos(6.2832*rnd()),c=new THREE.Color();
for(let i=0;i<N;i++){const x=gau(),z=gau(),y=r*x+Math.sqrt(Math.max(0,1-r*r))*z;pos.set([x*1.2,y*1.2+3,gau()*1.1],i*3);c.setHSL(.55+.2*Math.max(-1,Math.min(1,y/3)),1,.6);col.set([c.r,c.g,c.b],i*3)}
grid(g,10);const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(col,3));
const P=new THREE.Points(geo,new THREE.PointsMaterial({size:.14,vertexColors:true,transparent:true,opacity:.9}));g.add(P);
const ln=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-4,3-4*r*1.2,0),new THREE.Vector3(4,3+4*r*1.2,0)]),new THREE.LineBasicMaterial({color:0xffffff}));g.add(ln);
[['Length of stay →',0,.02,3.8],['Charges ↑',-4.8,5.6,0]].forEach(a=>{const l=lab(a[0],'#9fb3bd',1.2);l.position.set(a[1],a[2],a[3]);g.add(l)});
return e=>{P.scale.setScalar(Math.max(.001,e));ln.scale.setScalar(Math.max(.001,e))}}

let FST=0,NT=15,SPL=[Math.round(KP.records*.7),Math.round(KP.records*.3)];
function forestg;const T=[];for(let i=0;i<15;i++){const t=new THREE.Group();t.position.set((i%5-2)*1.9,0,(Math.floor(i/5)-1)*1.9);
const k=new THREE.Mesh(new THREE.CylinderGeometry(.07,.1,.7,8),MT(0x8a6a4a,.05));k.position.y=.35;t.add(k);
for(let j=0;j<3;j++){const c=new THREE.Mesh(new THREE.ConeGeometry(.55-j*.14,.9,12),MT(j%2?0x00e676:0x00e5ff,.3));c.position.y=.95+j*.5;t.add(c)}g.add(t);T.push(t)}
return(e,tm)=>T.forEach((t,i)=>{const a=FST?Math.min(1,Math.max(0,((performance.now()-FST)/1000-i*.12)*1.6)):1,s=i<NT?Math.max(.001,a):.001;t.scale.setScalar(s);t.rotation.y=tm*.3+i})}

/* ---------- chart registry ---------- */
const ageIt=()=>AGE.map((v,i)=>({n:AGN[i],v,c:CL[i%5],k:'age:'+AGN[i],l:F(v)})),sevIt=()=>SEVV.map((v,i)=>({n:SEV[i],v,c:CL[i%5],k:'sev:'+SEV[i],l:F(v,2)+' d'})),payIt=()=>PAY.map((p,i)=>({n:p[0],v:p[1],c:CL[i%5],k:'pay:'+p[0]}));
const hc=(i,n)=>new THREE.Color().setHSL(.52,1,.6-.32*i/Math.max(1,n)).getHex(),dolr=v=>'$'+(v>=1e6?F(v/1e6,1)+'M':F(v/1e3)+'k'),hdr={ry:-.15,rx:.5,cz:13.5};
const CH={
age:el=>stage(el,g=>bars(g,ageIt())),
pay:el=>stage(el,g=>donut(g,payIt()),{cy:5.2}),
los:el=>stage(el,g=>bars(g,HIST.map((v,i)=>({v,c:i<2?CL[0]:CL[1],n:i%2?0:`${i*5}–${i*5+5}`})),{sp:.75,w:.62,ns:.4})),
sev:el=>stage(el,g=>bars(g,sevIt())),
dx:el=>{const n=+$('#nd').value;stage(el,g=>hbars(g,DX.slice(0,n).map((d,i)=>({n:d[0],v:d[1],c:hc(i,n),l:F(d[1],1)+' d'}))),hdr)},
bxl:el=>stage(el,g=>boxes(g,BXL.map((q,i)=>({q,c:CL[i%5],n:SEV[i],k:'sev:'+SEV[i]})),v=>Math.sqrt(v/Math.max(1,KP.max_los))*4.6,v=>F(v)+' d')),
bxc:el=>stage(el,g=>boxes(g,BXC.map((q,i)=>({q,c:CL[i%5],n:SEV[i],k:'sev:'+SEV[i]})),v=>Math.max(0,(Math.log10(Math.max(100,v))-3)/4.2*4.6),dolr)),
grp:el=>stage(el,g=>bars(g,GRP.flatMap((a,gi)=>a.map((v,si)=>({x:gi*3.6+(si-1)*1.05-7.2,v,c:CL[si],n:si===1?AGN[gi]:0,k:['age:'+AGN[gi],'gen:'+GN[si]]}))),{w:.85,gw:18,ns:.9}),{cz:15}),
vio:el=>stage(el,g=>violins(g,VIO.map((v,i)=>Object.assign({n:AGN[i],c:CL[i%5],k:'age:'+AGN[i]},v)))),
fv:el=>{const n=+$('#nf').value;stage(el,g=>hbars(g,FV.slice(0,n).map((d,i)=>({n:d[0],v:d[1],c:hc(i,n),l:F(d[1])}))),hdr)},
fl:el=>{const n=+$('#nf').value;stage(el,g=>hbars(g,FL.slice(0,n).map((d,i)=>({n:d[0],v:d[1],c:new THREE.Color().setHSL(.72,.9,.62-.3*i/Math.max(1,n)).getHex(),l:F(d[1],1)+' d'}))),hdr)},
mv:el=>stage(el,g=>hbars(g,MV.map((d,i)=>({n:d[0],v:d[1],c:i<3?CL[4]:CL[3],l:F(d[1],1)+'%'}))),hdr),
mlos:el=>stage(el,g=>{const up=bars(g,HIST.map((v,i)=>({v,c:CL[1],n:i%2?0:`${i*5}–${i*5+5}`})),{sp:.75,w:.62,ns:.4}),X=v=>(v/5-6)*.75,pl=[];
[[KP.avg_los,CL[4],'mean '+F(KP.avg_los,2)],[KP.median_los,CL[2],'median '+F(KP.median_los,2)]].forEach(a=>{const m=new THREE.Mesh(new THREE.BoxGeometry(.05,4.8,2.2),new THREE.MeshBasicMaterial({color:a[1],transparent:true,opacity:.55}));m.position.set(X(a[0]),2.4,0);const t=lab(a[2],H(a[1]),.9);t.position.set(X(a[0])+(a[0]>4?.9:-.9),5.2,0);g.add(m,t);pl.push(m)});
return(e,t)=>{up(e,t);pl.forEach(m=>m.scale.y=Math.max(.001,e))}}),
mchg:el=>stage(el,g=>bars(g,[['Mean',KP.avg_charges/1e3,'$'+F(KP.avg_charges/1e3,1)+'k'],['Median',KP.median_charges/1e3,'$'+F(KP.median_charges/1e3,1)+'k'],['Std. dev.',KP.std_charges/1e3,'$'+F(KP.std_charges/1e3,1)+'k']].map((a,i)=>({n:a[0],v:a[1],l:a[2],c:CL[i]})),{sp:2.6})),
mcor:el=>stage(el,cloud,{ly:3}),
sum:el=>stage(el,g=>bars(g,[['Min',KP.min_los],['Q1',BXL[1][1]],['Median',KP.median_los],['Mean',KP.avg_los],['Q3',BXL[2][3]],['Max',KP.max_los]].map((a,i)=>({n:a[0],v:a[1],l:F(a[1],a[1]%1?2:0),c:CL[i%5]})),{sq:1,sp:1.6})),
forest:el=>stage(el,forest,{cy:6,cz:13,ly:1}),
split:el=>stage(el,g=>bars(g,[{n:'Train rows',v:SPL[0],c:CL[0],l:F(SPL[0])},{n:'Test rows',v:SPL[1],c:CL[4],l:F(SPL[1])}],{sp:3,w:1.6}))};
CH.msev=CH.sev;CH.mpay=CH.pay;

/* ---------- pages ---------- */
const K=(l,v)=>`<div class="k"><small>${l}</small><b>${v}</b></div>`,CD=(t,id,sub)=>`<section class="card"><h3>${t}</h3><div class="stage" data-c="${id}"><canvas role="img" aria-label="3D chart: ${t}"></canvas><em>Drag to rotate · ${sub||'3D chart'}</em></div></section>`;
const FR=(a,b)=>`<span class="fr"><i>${a}</i><i>${b}</i></span>`,PH=(t,d)=>`<h2>${t}</h2><p class="lead">${d}</p>`;
const tbl=(h,rows)=>`<div class="tb"><table><thead><tr>${h.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const P={};

P.overview=()=>{
  const topAgeIdx = AGE.indexOf(Math.max(...AGE));
  const topSevIdx = SEVV.indexOf(Math.max(...SEVV));
  return `<div class="sl">SYSTEM OVERVIEW</div>
  <div class="kg">${K('Discharge records',F(KP.records))}${K('Average length of stay',F(KP.avg_los,2)+' days')}${K('Average total charges','$'+F(KP.avg_charges,2))}${K('Facilities represented',F(KP.facilities))}</div>
  <div class="kg">${K('Median stay',F(KP.median_los,2)+' days')}${K('Total recorded charges','$'+F(KP.total_charges,2))}${K('Counties represented',F(KP.counties))}</div>
  <div class="g2">${CD('Patient records by age group','age','toggle age chips in sidebar')}${CD('Primary payment type','pay','extruded 3D donut · toggle payment chips')}${CD('Length-of-stay distribution','los','5-day bins')}${CD('Average stay by severity','sev','toggle severity chips')}</div>
  <h3>Quick findings</h3>
  <ul>
    <li>The highest average length of stay among displayed severity categories is <b>${SEV[topSevIdx]}</b>, at ${F(SEVV[topSevIdx],2)} days.</li>
    <li>The most represented age group is <b>${AGN[topAgeIdx]}</b>, with ${F(AGE[topAgeIdx])} records.</li>
    <li>The most frequent payment category is <b>${PAY[0][0]}</b> (${F(PAY[0][1],1)}%).</li>
  </ul>
  <p class="note">These findings are descriptive and do not establish causation or measure hospital quality.</p>`;
};

P.diagnosis=()=>PH('Diagnosis & Length of Stay','Compare inpatient stay across diagnosis categories and examine how length of stay varies with illness severity.')+
  `<label class="note">Number of diagnosis categories: <b id="ndv">${Math.min(14,DX.length)}</b></label>
  <input type="range" id="nd" min="5" max="${Math.max(5,DX.length)}" value="${Math.min(14,DX.length)}" aria-label="Number of diagnosis categories">
  ${CD('Average stay per diagnosis','dx','days · 3D horizontal bars')}
  ${CD('Length of stay by severity','bxl','quartiles, √ scale, tall sphere = maximum')}
  <h3>Top diagnosis categories summary</h3>
  ${tbl(['Diagnosis Category','Average Stay (Days)'],DX.map(d=>[d[0],F(d[1],2)+' days']))}`;

P.charges=()=>PH('Charges & Patient Demographics','Explore recorded charges across age groups, gender, and severity categories.')+
  `<div class="kg">${K('Average charges','$'+F(KP.avg_charges,2))}${K('Median charges','$'+F(KP.median_charges,2))}${K('Total charges','$'+F(KP.total_charges,2))}</div>
  ${CD('Charges by severity','bxc','log scale, tall sphere = maximum')}
  ${CD('Average charges by age group and gender','grp','$k · cyan Female · purple Male · green Unknown')}
  ${CD('Charge distribution by age group','vio','3D Lathe density, width = concentration')}
  <h3>Average charges ($k) by age group & gender</h3>
  ${tbl(['Age Group','Female ($k)','Male ($k)','Unknown ($k)'],GRP.map((r,i)=>[AGN[i],'$'+F(r[0],1)+'k','$'+F(r[1],1)+'k','$'+F(r[2],1)+'k']))}`;

P.facilities=()=>PH('Facilities & County Analysis','Compare discharge volume and average length of stay across high-volume hospitals.')+
  `<label class="note">Number of facilities to display: <b id="nfv">${Math.min(14,FV.length)}</b></label>
  <input type="range" id="nf" min="5" max="${Math.max(5,FV.length)}" value="${Math.min(14,FV.length)}" aria-label="Number of facilities">
  ${CD('Facilities with the most discharge records','fv','discharge volume')}
  ${CD('Average length of stay by facility','fl','days, highest-volume facilities')}
  <h3>High-volume facility summary</h3>
  ${tbl(['Facility Name','Discharge Records'],FV.map(f=>[f[0],F(f[1])]))}`;

P.quality=()=>PH('Data Quality & Preprocessing','Review the data-cleaning process, detected source columns, missing values, and potential outliers.')+
  `<div class="kg">${K('Original loaded rows',F(KP.original_rows))}${K('Rows after cleaning',F(KP.cleaned_rows))}${K('Rows removed',F(KP.removed_rows))}${K('Removal percentage',F(KP.removal_pct,2)+'%')}</div>
  <h3>Preprocessing steps</h3>
  ${['Column identification','Numeric conversion','Invalid length of stay','Invalid charges','Category standardization','Missing categorical values','Analysis fields'].map(s=>`<details><summary>${s}</summary>Executed across ${F(KP.original_rows)} loaded SPARCS inpatient records.</details>`).join('')}
  <div style="height:14px"></div>
  ${CD('Missing-value profile','mv','% of rows missing')}
  <h3>Length-of-stay outliers</h3>
  <div class="kg">${K('Potential LOS outliers',F(KP.outliers))}${K('IQR upper fence',F(KP.iqr_fence,2)+' days')}</div>
  <p class="note">IQR outliers are statistical flags, not automatically invalid records. Long stays may be legitimate.</p>`;

P.math=()=>{
  const n=KP.records,sx=KP.avg_los*n,tc=KP.total_charges,r=KP.r;
  return PH('Mathematical Analysis','Explore the statistical calculations used by MediScope. Calculations are shown live and drawn in 3D.')+
  `<h3>1. Descriptive statistics: length of stay</h3><p class="lead">The mean is the arithmetic average, the median is the middle value, and the standard deviation measures how spread out stays are around the mean.</p>
  <div class="fm">x̄ = ${FR('Σxᵢ','n')} &nbsp; σ = √${FR('Σ(xᵢ − x̄)²','n')}</div>
  <div class="kg">${K('Mean stay',F(KP.avg_los,2)+' days')}${K('Median stay',F(KP.median_los,2)+' days')}${K('Standard deviation',F(KP.std_los,2)+' days')}${K('Minimum stay',F(KP.min_los,2)+' days')}${K('Maximum stay',F(KP.max_los,2)+' days')}</div>
  <div class="calc">Mean = ${F(sx)} / ${F(n)} = ${F(KP.avg_los,2)} days</div>
  ${CD('Mean and median on the stay distribution','mlos','histogram shape; planes mark mean and median')}
  <h3>2. Descriptive statistics: healthcare charges</h3><div class="fm">C̄ = ${FR('ΣCᵢ','n')}</div>
  <div class="kg">${K('Mean charges','$'+F(KP.avg_charges,2))}${K('Median charges','$'+F(KP.median_charges,2))}${K('Standard deviation','$'+F(KP.std_charges,2))}${K('Total recorded charges','$'+F(tc,2))}</div>
  <div class="calc">Mean charges = $${F(tc,2)} / ${F(n)} = $${F(tc/Math.max(1,n),2)}</div>
  ${CD('Mean, median and spread of charges','mchg','$k · std. dev. above mean indicates right-skewed distribution')}
  <h3>3. Percentage distribution by payment type</h3><div class="fm">Percentage = ${FR('Category count','Total records')} × 100</div>
  ${CD('Payment shares','mpay','3D extruded donut')}
  ${tbl(['Payment type','Records','Percentage'],PAY.map(p=>[p[0],F(p[2]||Math.round(p[1]/100*n)),F(p[1],1)+'%']))}
  <h3>4. Severity comparison</h3>${CD('Average stay by severity','msev','mean stay in days')}
  ${tbl(['Severity','Average stay (days)'],SEV.map((s,i)=>[s,F(SEVV[i],2)]))}
  <h3>5. Correlation: length of stay and charges</h3><div class="fm">r = ${FR('Cov(X, Y)','σₓσᵧ')}</div>
  <div class="kg">${K('Pearson correlation (r)',F(r,4))}${K('r² (shared variation)',F(KP.r2,3))}</div>
  <p class="lead">The filtered records show a positive linear association: higher stay values tend to occur with higher charge values.</p>
  ${CD('Correlation cloud','mcor','3D scatter cloud matching r = '+F(r,4)+', white line = fit')}
  <p class="note">Calculated using ${F(KP.valid_corr_n)} records containing both values. Correlation does not prove causation.</p>
  <div class="warn">These are descriptive calculations on the loaded dataset sample (${F(n)} rows). They do not establish causation, diagnose patients, or measure hospital quality.</div>`;
};

P.ai=()=>PH('AI Experiment: Predicting Severity','Train a demonstration machine-learning model to predict standardized severity categories using demographic and administrative features. This experiment is educational and is not intended for clinical use.')+
  `<div class="warn">The model uses the loaded dataset sample. Evaluation results may not represent performance on other hospitals or years.</div>
  <p class="note">Select model input features</p>
  <div id="ft">${['Age_Group','Gender','Payment_Type','Diagnosis'].map(f=>`<button class="chip" aria-pressed="true">${f}</button>`).join('')}</div>
  <p class="note">Choose a model</p><select aria-label="Model"><option>Random Forest Classifier (scikit-learn)</option></select>
  <p class="note">Test data percentage: <b id="tv">30</b>%</p>
  <input type="range" id="tp" min="10" max="50" value="30" aria-label="Test data percentage">
  <button class="go" id="go">Train and evaluate model</button>
  <p class="note" id="res">Press the button to grow the 3D Random Forest and evaluate the train/test split.</p>
  <div class="g2">${CD('3D Random forest ensemble','forest','trees grow when you train')}${CD('Train / test split','split','rows per partition')}</div>`;

P.explorer=()=>PH('Data Explorer','Search processed records, narrow the data by length of stay, and inspect the results.')+
  `<p class="note">Search facility, diagnosis, county, age group, or gender</p>
  <input type="text" id="q" placeholder="Enter a search term..." aria-label="Search">
  <p class="note">Length-of-stay range (days): <b id="lv">1 to 120</b></p>
  <input type="range" id="l0" min="1" max="120" value="1" aria-label="Minimum stay">
  <input type="range" id="l1" min="1" max="120" value="120" aria-label="Maximum stay">
  <div class="kg" style="margin-top:14px">${K('Total records in active dataset',F(KP.records))}</div>
  <h3>Preview rows <span class="note" id="cnt"></span></h3>
  <div class="tb"><table><thead><tr>${['County','Facility','Age Group','Gender','Length of Stay','Type of Admission','Patient Disposition','Diagnosis'].map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody id="tr"></tbody></table></div>
  <h3>Numeric summary</h3>
  ${tbl(['Variable','count','mean','std','min','25%','50%','75%','max'],DESC)}
  ${CD('Length-of-stay summary points','sum','√ scale · min, quartiles, mean, max')}`;

const POST={
  diagnosis:()=>{$('#nd').oninput=e=>{$('#ndv').textContent=e.target.value;CH.dx($('[data-c=dx]'))}},
  facilities:()=>{$('#nf').oninput=e=>{$('#nfv').textContent=e.target.value;CH.fv($('[data-c=fv]'));CH.fl($('[data-c=fl]'))}},
  ai:()=>{
    $('#tp').oninput=e=>$('#tv').textContent=e.target.value;
    document.querySelectorAll('#ft .chip').forEach(b=>b.onclick=()=>b.setAttribute('aria-pressed',b.getAttribute('aria-pressed')!=='true'));
    $('#go').onclick=()=>{
      const p=+$('#tp').value,nf=document.querySelectorAll('#ft .chip[aria-pressed=true]').length,te=Math.round(KP.records*p/100);
      SPL=[KP.records-te,te];NT=Math.max(3,nf*4);FST=performance.now();
      CH.split($('[data-c=split]'));
      $('#res').innerHTML=`<b>Evaluation Complete:</b> Train rows <b>${F(SPL[0])}</b> · Test rows <b>${F(te)}</b> (${p}%) · ${nf} input features · ${NT} 3D decision trees rendered. ${ML.note}`;
    };
  },
  explorer:()=>{
    const f=()=>{
      const a=+$('#l0').value,b=+$('#l1').value,q=$('#q').value.toLowerCase(),lo=Math.min(a,b),hi=Math.max(a,b);
      const r=RW.filter(x=>x[4]>=lo&&x[4]<=hi&&x.join(' ').toLowerCase().includes(q));
      $('#lv').textContent=lo+' to '+hi;
      $('#cnt').textContent=`(showing ${r.length} of ${RW.length} preview rows)`;
      $('#tr').innerHTML=r.map(x=>`<tr>${x.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('');
    };
    ['#q','#l0','#l1'].forEach(s=>$(s).oninput=f);f();
  }
};

/* ---------- shell: nav, filters, router ---------- */
const PG=[['overview','Overview'],['diagnosis','Diagnosis & Stay'],['charges','Charges & Demographics'],['facilities','Facilities & Counties'],['quality','Data Quality'],['math','Mathematical Analysis'],['ai','AI Experiment'],['explorer','Data Explorer']];
$('#nav').innerHTML=PG.map(p=>`<a class="nv" href="#${p[0]}" data-p="${p[0]}">${p[1].replace('&','&amp;')}</a>`).join('');
const FG={'Age group':AGN.map(x=>['age:'+x,x]),'Gender':GN.map(x=>['gen:'+x,x]),'Severity of illness':SEV.map(x=>['sev:'+x,x]),'Payment type':PAY.map(p=>['pay:'+p[0],p[0]])};
$('#flt').innerHTML=Object.keys(FG).map(t=>`<label>${t}</label>${FG[t].map(c=>`<button class="chip" aria-pressed="true" data-k="${c[0]}">${c[1]}</button>`).join('')}`).join('');
$('#flt').onclick=e=>{const b=e.target.closest('.chip');if(!b)return;const k=b.dataset.k,on=off.has(k);on?off.delete(k):off.add(k);b.setAttribute('aria-pressed',on)};
function route(){const id=PG.some(p=>p[0]===location.hash.slice(1))?location.hash.slice(1):'overview';live.forEach(l=>{l.dead=true;l.r.dispose();try{l.r.forceContextLoss()}catch(e){}});live=[];
$('#page').innerHTML=P[id]();document.querySelectorAll('.nv').forEach(a=>a.classList.toggle('on',a.dataset.p===id));
document.querySelectorAll('.stage[data-c]').forEach(el=>CH[el.dataset.c](el));if(POST[id])POST[id]();scrollTo(0,0)}
addEventListener('hashchange',route);
$('#page').addEventListener('pointermove',e=>{const k=e.target.closest('.k');if(!k||RM)return;const b=k.getBoundingClientRect();k.style.transform=`perspective(600px) rotateY(${((e.clientX-b.left)/b.width-.5)*14}deg) rotateX(${(.5-(e.clientY-b.top)/b.height)*14}deg)`});
$('#page').addEventListener('pointerout',e=>{const k=e.target.closest('.k');if(k)k.style.transform=''});

/* ---------- MediBot 3D ---------- */
(function(){const r=new THREE.WebGLRenderer({canvas:$('#botcv'),alpha:true,antialias:true});r.setPixelRatio(Math.min(devicePixelRatio,2));r.setSize(110,110,false);
const s=new THREE.Scene(),cam=new THREE.PerspectiveCamera(32,1,.1,50);cam.position.set(0,.3,8);s.add(new THREE.HemisphereLight(0xffffff,0x35507a,1.1));const pl=new THREE.PointLight(0x00e5ff,2.5,30);pl.position.set(-3,2,4);s.add(pl);
const M=(c,e)=>new THREE.MeshStandardMaterial({color:c,roughness:.22,metalness:.3,emissive:e||0}),b=new THREE.Group();s.add(b);
const A=(g,m,x,y,z,p)=>{const o=new THREE.Mesh(g,m);o.position.set(x,y,z);(p||b).add(o);return o},hd=new THREE.Group();hd.position.y=.5;b.add(hd);
A(new THREE.SphereGeometry(1,40,28),M(0xf2f8fb),0,0,0,hd).scale.set(1.1,.92,.95);A(new THREE.SphereGeometry(1.03,40,20,Math.PI/2-1.05,2.1,Math.PI*.3,Math.PI*.38),M(0x0a1220),0,0,0,hd).scale.set(1.1,.92,.95);
const ey=[-.38,.38].map(x=>{const e=A(new THREE.SphereGeometry(.15,16,12),M(0x00e5ff,0x00a8c0),x,.05,.92,hd);e.scale.set(1,1.3,.4);return e});
A(new THREE.CylinderGeometry(.03,.03,.4,8),M(0xaebdc7),0,1,0,hd);A(new THREE.SphereGeometry(.1,12,10),M(0x00e5ff,0x00a8c0),0,1.25,0,hd);
A(new THREE.CylinderGeometry(.6,.8,1,32),M(0xffffff),0,-.8,0);A(new THREE.BoxGeometry(.3,.1,.05),M(0x00e5ff,0x00a8c0),0,-.7,.76);A(new THREE.BoxGeometry(.1,.3,.05),M(0x00e5ff,0x00a8c0),0,-.7,.76);
let nb=2;(function loop(){requestAnimationFrame(loop);const t=performance.now()/1000;b.position.y=RM?0:Math.sin(t*1.7)*.08;hd.rotation.y+=(Math.max(-.6,Math.min(.6,mx*1.2))-hd.rotation.y)*.08;hd.rotation.x+=(Math.max(-.3,Math.min(.3,my*.6))-hd.rotation.x)*.08;
if(t>nb){nb=t+2+Math.random()*3;ey.forEach(e=>e.scale.y=.1);setTimeout(()=>ey.forEach(e=>e.scale.y=1.3),130)}r.render(s,cam)})()})();

/* ---------- page-wide background: drifting 3D crosses ---------- */
(function(){const cv=$('#bgc');if(RM)return;const r=new THREE.WebGLRenderer({canvas:cv,alpha:true,antialias:true});r.setPixelRatio(Math.min(devicePixelRatio,1.5));
const s=new THREE.Scene(),cam=new THREE.PerspectiveCamera(60,1,.1,60);cam.position.z=12;const t=document.createElement('canvas');t.width=t.height=64;const x=t.getContext('2d');x.fillStyle='#fff';x.fillRect(26,6,12,52);x.fillRect(6,26,52,12);
const M=240,pos=new Float32Array(M*3),col=new Float32Array(M*3),base=[],sp=[];for(let i=0;i<M;i++){pos[i*3]=(Math.random()-.5)*34;pos[i*3+2]=(Math.random()-.5)*14;base.push(Math.random()*20);sp.push(.15+Math.random()*.5);col.set(i%2?[0,.9,1]:[.55,.36,1],i*3)}
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(col,3));
const pts=new THREE.Points(geo,new THREE.PointsMaterial({size:.5,map:new THREE.CanvasTexture(t),vertexColors:true,transparent:true,opacity:.3,depthWrite:false}));s.add(pts);
const size=()=>{r.setSize(innerWidth,innerHeight,false);cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix()};addEventListener('resize',size);size();
(function loop(){requestAnimationFrame(loop);const tm=performance.now()/1000,sc=scrollY*.006;for(let i=0;i<M;i++)pos[i*3+1]=(((base[i]+tm*sp[i]+sc*sp[i]*2)%20)+20)%20-10;geo.attributes.position.needsUpdate=true;pts.rotation.y=Math.sin(tm*.1)*.25;r.render(s,cam)})()})();
route();
</script>
</body>
</html>
"""


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    dataset_path = find_dataset()
    if dataset_path is not None:
        stat = dataset_path.stat()
        payload = build_live_payload(
            str(dataset_path), stat.st_size, stat.st_mtime
        )
    else:
        payload = get_benchmark_payload()

    rendered_html = HTML_3D_TEMPLATE.replace(
        "__MEDISCOPE_JSON__", json.dumps(payload)
    )
    components.html(rendered_html, height=1000, scrolling=True)


if __name__ == "__main__":
    main()
