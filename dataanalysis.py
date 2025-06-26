
# === OPTIMIZED DATA ANALYSIS DASHBOARD WITH CLEANING CONTROL AND PROGRESS BAR ===
from urllib.parse import quote
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import openai
import streamlit as st
import chardet

openai.api_key = os.getenv("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") or "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"


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_data
def cached_anomalies(df, method, threshold):
    return detect_anomalies(df, method, threshold)

@st.cache_data
def cached_trends(df):
    return detect_trends(df)

@st.cache_data
def cached_corr_matrix(df):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    path = f"{OUTPUT_DIR}/correlation_matrix.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig(path)
    plt.close()
    return path

@st.cache_data
def cached_histograms(df):
    paths = []
    for col in df.select_dtypes(include="number").columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        safe_col = re.sub(r'[\\/*?:"<>|]', "_", col)  # sanitize filename
        path = os.path.join(OUTPUT_DIR, f"{safe_col}_hist.png")
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def load_data(uploaded_file):
    filename = uploaded_file.name
    raw = uploaded_file.read()
    encoding_guess = chardet.detect(raw)
    uploaded_file.seek(0)
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding=encoding_guess['encoding'])
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format")
        bad_headers = df.columns.to_series().astype(str).str.match(r'^Unnamed|^\d+(\.\d+)?$').any()
        if bad_headers:
            st.warning("⚠️ Column headers may be malformed or missing.")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return pd.DataFrame()

def analyze_problems(df):
    issues = {}
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    if not missing_cols.empty:
        issues["Missing Values"] = missing_cols
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues["Constant Columns"] = constant_cols
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues["Duplicate Rows"] = duplicates
    return issues

def analyze_columns(df):
    return {
        "num_cols": df.select_dtypes(include="number").columns.tolist(),
        "cat_cols": [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() <= 20],
        "text_cols": [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() > 20],
        "date_cols": df.select_dtypes(include="datetime").columns.tolist(),
        "shape": df.shape,
    }

def detect_trends(df):
    trend_summary = {}
    numeric_cols = df.select_dtypes(include="number")
    for col in numeric_cols.columns:
        trend_summary[col] = {
            "mean": numeric_cols[col].mean(),
            "std_dev": numeric_cols[col].std(),
            "min": numeric_cols[col].min(),
            "max": numeric_cols[col].max(),
        }
    return trend_summary

def detect_anomalies(df, method="z-score", threshold=3):
    numeric_df = df.select_dtypes(include="number")
    anomalies = pd.DataFrame()
    explanations = []
    if method == "z-score":
        z = numeric_df.apply(zscore)
        mask = (z.abs() > threshold).any(axis=1)
        anomalies = df[mask].copy()
        for idx in anomalies.index:
            explanations.append(", ".join(f"{col} z={z.loc[idx, col]:.2f}" for col in numeric_df.columns if abs(z.loc[idx, col]) > threshold))
    elif method == "iqr":
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        mask = ((numeric_df < lower) | (numeric_df > upper)).any(axis=1)
        anomalies = df[mask].copy()
        for idx in anomalies.index:
            explanations.append(", ".join(f"{col} outlier (IQR)" for col in numeric_df.columns if numeric_df.loc[idx, col] < lower[col] or numeric_df.loc[idx, col] > upper[col]))
    elif method == "mad":
        median = numeric_df.median()
        mad = (numeric_df - median).abs().median()
        scores = ((numeric_df - median).abs() / mad).replace([np.inf, -np.inf], np.nan)
        mask = (scores > threshold).any(axis=1)
        anomalies = df[mask].copy()
        for idx in anomalies.index:
            explanations.append(", ".join(f"{col} MAD={scores.loc[idx, col]:.2f}" for col in numeric_df.columns if scores.loc[idx, col] > threshold))
    if not anomalies.empty:
        anomalies["Anomaly Reason"] = explanations
    return anomalies

def generate_conclusion(trends):
    return "\n".join(f"{col} shows high variability." if stats["std_dev"] > stats["mean"] * 0.5 else f"{col} is relatively stable." for col, stats in trends.items())

def gpt_summary(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analyst summarizing data trends and suggesting charts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT Summary Failed] {e}"

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📊 AI-Powered Data Analysis Dashboard")

with st.sidebar:
    st.markdown("### 🔍 Data Cleaning Settings")
    show_cleaning = st.checkbox("Analyze & Clean Data", value=True)
    missing_strategy = st.selectbox("Fill strategy for numeric columns", ["Median", "Mean", "Zero"])
    drop_constants = st.checkbox("Drop constant columns", value=True)
    drop_duplicates = st.checkbox("Drop duplicate rows", value=True)

    st.markdown("### 📈 Anomaly Detection")
    use_gpt = st.checkbox("Enable GPT Summary", value=True)
    method = st.selectbox("Anomaly Method", ["z-score", "iqr", "mad"])
    threshold = st.slider("Threshold", 1.0, 10.0, 3.0, 0.5)

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    progress = st.progress(0, text="Loading data...")
    df = load_data(uploaded_file)
    progress.progress(10, "Analyzing problems...")
    problems = analyze_problems(df)

    st.subheader("⚠️ Data Issues Summary (before cleaning)")
    if problems:
        for k, v in problems.items():
            st.write(f"**{k}:**")
            st.write(v)
    else:
        st.success("No significant issues detected.")

    df_clean = df.copy()
    if show_cleaning:
        progress.progress(20, "Cleaning missing values...")
        df_clean = df_clean.dropna(how="all").dropna(axis=1, how="all")
        for col in df_clean.select_dtypes(include="number").columns:
            if missing_strategy == "Median":
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif missing_strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif missing_strategy == "Zero":
                df_clean[col] = df_clean[col].fillna(0)
        for col in df_clean.select_dtypes(include="object").columns:
            df_clean[col] = df_clean[col].fillna("unknown")
        if drop_constants:
            df_clean = df_clean.loc[:, df_clean.apply(pd.Series.nunique) > 1]
        if drop_duplicates:
            df_clean = df_clean.drop_duplicates()

    progress.progress(40, "Analyzing columns...")
    column_info = analyze_columns(df_clean)
    progress.progress(60, "Calculating trends...")
    trends = cached_trends(df_clean)
    conclusion = generate_conclusion(trends)

    filename = uploaded_file.name
    shape_info = f"{df_clean.shape[0]} rows × {df_clean.shape[1]} columns"
    num_cols = column_info['num_cols']
    cat_cols = column_info['cat_cols']
    structure_prompt = f"You are a data analyst. This file is called '{filename}' with {shape_info}.\nNumeric columns: {', '.join(num_cols)}.\nCategorical columns: {', '.join(cat_cols)}.\nWhat trends or patterns should we look for?\nWhat else can you tell me about this dataset?"

    progress.progress(75, "GPT summary (optional)...")
    summary = gpt_summary(structure_prompt) if use_gpt else "(GPT disabled)"

    progress.progress(85, "Detecting anomalies and generating charts...")
    anomalies = cached_anomalies(df_clean, method, threshold)
    corr_path = cached_corr_matrix(df_clean)
    hist_paths = cached_histograms(df_clean)

    encoded_prompt = quote(structure_prompt)
    gpt_link = f"https://chatgpt.com/g/g-681a33e2f2ec8191ac28ed90f1ae1b16-insight-gpt?prompt={encoded_prompt}"
    progress.progress(100, "Done.")

    tabs = st.tabs(["📋 Overview", "📈 Trends", "📊 Visuals", "🚨 Anomalies", "📥 Downloads"])
    with tabs[0]:
        st.dataframe(df_clean.head())
        st.markdown(f"- Rows: {column_info['shape'][0]}")
        st.markdown(f"- Columns: {column_info['shape'][1]}")
        st.markdown(f"[🔗 Open in Insight GPT]({gpt_link})")
    with tabs[1]:
        st.json(trends)
        st.text(conclusion)
        if use_gpt:
            st.success(summary)
    with tabs[2]:
        st.image(corr_path)
        for p in hist_paths:
            st.image(p)
    with tabs[3]:
        st.dataframe(anomalies)
    with tabs[4]:
        st.download_button("📁 Download Cleaned Data", df_clean.to_csv(index=False), "cleaned.csv")
        st.download_button("🚨 Download Anomalies", anomalies.to_csv(index=False), "anomalies.csv")
        st.download_button("🧠 GPT Summary", summary.encode(), "summary.txt")
else:
    st.info("Upload a dataset to begin.")
