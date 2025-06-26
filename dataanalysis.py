# === INTELLIGENT DATA ANALYSIS DASHBOARD (MULTITHREADING + MEMORY AWARE + COLUMN COERCION FIX) ===
from urllib.parse import quote
import os
import re
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import openai
import streamlit as st
import chardet
import uuid
import psutil

openai.api_key = os.getenv("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") or "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"


session_uuid = str(uuid.uuid4())
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def report_memory():
    mem = psutil.virtual_memory()
    used = (mem.total - mem.available) / (1024 * 1024)
    avail = mem.available / (1024 * 1024)
    return used, avail

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
        if df.columns.to_series().astype(str).str.match(r'^Unnamed|^\d+(\.\d+)?$').any():
            st.warning("⚠️ Column headers may be malformed or missing.")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return pd.DataFrame()

def analyze_problems(df):
    issues = {}
    missing = df.isnull().sum()
    if (missing > 0).any():
        issues["Missing Values"] = missing[missing > 0]
    const_cols = [col for col in df.columns if df[col].nunique() == 1]
    if const_cols:
        issues["Constant Columns"] = const_cols
    dups = df.duplicated().sum()
    if dups > 0:
        issues["Duplicate Rows"] = dups
    return issues

def analyze_columns(df):
    return {
        "num_cols": df.select_dtypes(include="number").columns.tolist(),
        "cat_cols": [col for col in df.select_dtypes(include="object") if df[col].nunique() <= 20],
        "text_cols": [col for col in df.select_dtypes(include="object") if df[col].nunique() > 20],
        "date_cols": df.select_dtypes(include="datetime").columns.tolist(),
        "shape": df.shape
    }

def detect_trends(df):
    result = {}
    for col in df.select_dtypes(include="number").columns:
        result[col] = {
            "mean": df[col].mean(),
            "std_dev": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    return result

def detect_anomalies(df, method="z-score", threshold=3):
    numeric_df = df.select_dtypes(include="number")
    anomalies = pd.DataFrame()
    reasons = []

    if method == "z-score":
        z = numeric_df.apply(zscore)
        mask = (z.abs() > threshold).any(axis=1)
        anomalies = df[mask].copy()
        for i in anomalies.index:
            reasons.append(", ".join(f"{col} z={z.loc[i, col]:.2f}" for col in numeric_df if abs(z.loc[i, col]) > threshold))

    if not anomalies.empty:
        anomalies["Anomaly Reason"] = reasons
    return anomalies

def gpt_summary(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst summarizing data trends."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT Summary Error] {e}"

def threaded_plot(df, col, kind):
    mem_used, mem_avail = report_memory()
    if mem_avail < 200:
        return f"Skipped {col} ({kind}) – Not enough memory"

    plt.figure()
    if kind == "Histogram":
        sns.histplot(df[col], kde=True)
    elif kind == "Boxplot":
        sns.boxplot(y=df[col])
    elif kind == "Bar Chart":
        df[col].value_counts().plot(kind='bar')
    elif kind == "Pie Chart":
        df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
    else:
        return f"Unsupported chart type: {kind}"

    safe_name = re.sub(r'[\\/*?:"<>|]', "_", col)
    path = os.path.join(OUTPUT_DIR, f"{safe_name}_{kind}.png")
    plt.title(f"{kind} of {col}")
    plt.savefig(path)
    plt.close()
    return path

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📊 AI Data Analysis Dashboard")

used, avail = report_memory()
st.sidebar.markdown(f"💾 Memory Used: `{used:.1f} MB` | Free: `{avail:.1f} MB`")

with st.sidebar:
    st.markdown("### Data Cleaning")
    show_cleaning = st.checkbox("Clean data", value=True)
    fill_strategy = st.selectbox("Fill strategy (numeric)", ["Median", "Mean", "Zero"])
    drop_const = st.checkbox("Drop constant columns", value=True)
    drop_dups = st.checkbox("Drop duplicates", value=True)

    st.markdown("### GPT + Anomalies")
    enable_gpt = st.checkbox("Enable GPT Summary", value=True)
    method = st.selectbox("Anomaly Method", ["z-score"])
    threshold = st.slider("Threshold", 1.0, 10.0, 3.0, 0.5)

uploaded = st.file_uploader("📥 Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded:
    df = load_data(uploaded)

    st.subheader("⚠️ Issues Before Cleaning")
    issues = analyze_problems(df)
    if issues:
        for k, v in issues.items():
            st.write(f"**{k}**", v)
    else:
        st.success("No major issues.")

    df_clean = df.copy()
    if show_cleaning:
        df_clean.dropna(how="all", inplace=True)
        df_clean.dropna(axis=1, how="all", inplace=True)
        for col in df_clean.select_dtypes(include="number"):
            if fill_strategy == "Median":
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif fill_strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif fill_strategy == "Zero":
                df_clean[col] = df_clean[col].fillna(0)
        for col in df_clean.select_dtypes(include="object"):
            df_clean[col] = df_clean[col].fillna("unknown")

        # Coerce string-like numbers to actual numerics
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col])
                except:
                    pass

        if drop_const:
            df_clean = df_clean.loc[:, df_clean.apply(pd.Series.nunique) > 1]
        if drop_dups:
            df_clean = df_clean.drop_duplicates()

    info = analyze_columns(df_clean)
    trends = detect_trends(df_clean)
    filename = uploaded.name
    structure_prompt = (
    f"Session {session_uuid}: File '{filename}' with {df_clean.shape[0]} rows and {df_clean.shape[1]} columns.\n"
    f"- Numeric Columns: {', '.join(info['num_cols']) or 'None'}\n"
    f"- Categorical Columns: {', '.join(info['cat_cols']) or 'None'}\n"
    f"- Text Columns: {', '.join(info['text_cols']) or 'None'}\n"
    f"- Date Columns: {', '.join(info['date_cols']) or 'None'}\n"
    "Do you understand file structure? What trends should I look for? Anything outstanding in the data? Any recommendations?"
)


    gpt_summary_text = gpt_summary(structure_prompt) if enable_gpt else "(GPT disabled)"
    anomalies = detect_anomalies(df_clean, method, threshold)
    encoded_prompt = quote(structure_prompt)
    gpt_link = f"https://chatgpt.com/g/g-681a33e2f2ec8191ac28ed90f1ae1b16-insight-gpt?prompt={encoded_prompt}"

    tabs = st.tabs(["📋 Overview", "📈 Trends", "🧠 GPT", "🚨 Anomalies", "📊 Visualize", "📥 Downloads"])

    with tabs[0]:
        st.dataframe(df_clean.head())
        st.markdown(f"[🔗 Insight GPT Link]({gpt_link})")

    with tabs[1]:
        st.json(trends)

    with tabs[2]:
        st.text(gpt_summary_text)

    with tabs[3]:
        st.dataframe(anomalies)

    with tabs[4]:
        st.markdown("### Select Columns and Chart Types")
        selected_cols = st.multiselect("Columns", info['num_cols'] + info['cat_cols'])
        chart_types = st.multiselect("Chart Types", ["Histogram", "Boxplot", "Bar Chart", "Pie Chart"])
        if st.button("Generate Charts"):
            st.info("Running visualizations in threads...")
            for col in selected_cols:
                for chart in chart_types:
                    result = threaded_plot(df_clean, col, chart)
                    if isinstance(result, str) and result.endswith(".png"):
                        st.image(result, caption=f"{chart} of {col}")
                    else:
                        st.warning(result)

    with tabs[5]:
        st.download_button("📁 Download Cleaned Data", df_clean.to_csv(index=False), "cleaned.csv")
        st.download_button("🚨 Download Anomalies", anomalies.to_csv(index=False), "anomalies.csv")
        st.download_button("🧠 GPT Summary", gpt_summary_text.encode(), "summary.txt")

else:
    st.info("Upload a file to get started.")
