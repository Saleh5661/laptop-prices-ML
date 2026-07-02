"""
Laptop Price Prediction — Interactive Streamlit App
Follows the exact cleaning / feature-engineering / modeling pipeline
from the source notebook, but presents EDA with different chart types
(Plotly interactive charts instead of the notebook's matplotlib/seaborn
boxplots & heatmap) so the app doesn't just repeat the notebook visuals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
@st.cache_data
def load_raw_data(file):
    return pd.read_csv(file)


st.sidebar.title("💻 Laptop Price Predictor")

try:
    raw_df = load_raw_data("laptopData.csv")
except FileNotFoundError:
    st.error("Couldn't find `laptopData.csv`. Please place it next to app.py.")
    st.stop()

section = st.sidebar.radio(
    "Go to section",
    [
        "1️⃣ Data Overview",
        "2️⃣ EDA",
        "3️⃣ Cleaning & Preprocessing",
        "4️⃣ Model Results",
        "5️⃣ Predict a Price",
    ],
)

st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "🤖 Model",
    ["Random Forest", "Linear Regression"],
    help="Switch the model used for evaluation and price prediction.",
)

# ----------------------------------------------------------------------
# 2. CLEANING / FEATURE ENGINEERING (mirrors the notebook exactly)
# ----------------------------------------------------------------------
@st.cache_data
def clean_and_engineer(df):
    stats = {}
    df = df.copy()

    stats["rows_before"] = len(df)
    stats["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    stats["missing_before"] = df.isna().sum().sum()
    df.dropna(how="all", inplace=True)

    # Ram / Weight / Inches type fixes
    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", case=False).astype(int)
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", case=False)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

    df.drop(columns="Unnamed: 0", inplace=True, errors="ignore")

    stats["missing_after"] = df.isna().sum().sum()
    df = df.dropna()

    # Outlier removal on Inches
    stats["rows_before_outlier"] = len(df)
    df = df[df["Inches"] <= 20]
    stats["rows_after_outlier"] = len(df)

    # Memory -> SSD / HDD
    df["Memory"] = (
        df["Memory"].astype(str)
        .str.replace("1.0TB", "1000")
        .astype(str)
        .str.replace("1TB", "1000")
        .str.replace("2TB", "2000")
        .str.replace("GB", "")
    )

    def extract_ssd(text):
        words = str(text).split()
        if "SSD" in words:
            idx = words.index("SSD")
            return int(words[idx - 1])
        return 0

    def extract_hdd(text):
        words = str(text).split()
        if "HDD" in words:
            idx = words.index("HDD")
            return int(words[idx - 1])
        return 0

    df["SSD"] = df["Memory"].apply(extract_ssd)
    df["HDD"] = df["Memory"].apply(extract_hdd)

    # Cpu / Gpu / Touchscreen / OS
    def get_cpu_brand(text):
        return " ".join(str(text).split()[0:3])

    def get_gpu_brand(text):
        return str(text).split()[0]

    def check_touchscreen(text):
        return 1 if "touchscreen" in str(text).lower() else 0

    def categorize_os(os_name):
        text = str(os_name).lower()
        if "windows" in text:
            return "Windows"
        elif "mac" in text:
            return "Mac"
        elif "linux" in text:
            return "Linux"
        return "No OS"

    df["Cpu_Brand"] = df["Cpu"].apply(get_cpu_brand)
    df["Gpu_Brand"] = df["Gpu"].apply(get_gpu_brand)
    df["Touchscreen"] = df["ScreenResolution"].apply(check_touchscreen)
    df["OpSys"] = df["OpSys"].apply(categorize_os)

    df.drop(columns=["Gpu", "Cpu", "ScreenResolution"], inplace=True, errors="ignore")

    return df, stats


clean_df, stats = clean_and_engineer(raw_df)

text_columns = ["Company", "TypeName", "Cpu_Brand", "Gpu_Brand", "OpSys"]
df_encoded = pd.get_dummies(clean_df, columns=text_columns, drop_first=True, dtype=int)

X = df_encoded.drop(columns=["Price", "Memory"])
X = X.fillna(X.median(numeric_only=True))
y = df_encoded["Price"]


# ----------------------------------------------------------------------
# 3. TRAIN MODEL (switchable: Random Forest or Linear Regression, no PCA)
# ----------------------------------------------------------------------
@st.cache_resource
def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    metrics = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
    }
    return model, scaler, metrics, y_test, preds


model, scaler, metrics, y_test, preds = train_model(X, y, model_choice)

# ========================================================================
# SECTION 1 — DATA OVERVIEW
# ========================================================================
if section == "1️⃣ Data Overview":
    st.title("Laptop Price Prediction — Data Overview")
    st.markdown(
        """
        This app predicts a laptop's **price** from its specs (brand, RAM, storage,
        CPU/GPU, screen size, weight, OS, etc.) using a **{model}**
        trained on the *laptopData.csv* dataset. You can switch models from the sidebar.
        """.format(model=model_choice)
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (raw)", raw_df.shape[0])
    c2.metric("Columns (raw)", raw_df.shape[1])
    c3.metric("Rows after cleaning", clean_df.shape[0])

    st.subheader("Sample of the raw dataset")
    st.dataframe(raw_df.head(10))

    st.subheader("Column descriptions")
    st.table(
        pd.DataFrame(
            {
                "Column": ["Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram",
                           "Memory", "Gpu", "OpSys", "Weight", "Price"],
                "Meaning": [
                    "Laptop manufacturer",
                    "Category (Notebook, Gaming, Ultrabook, ...)",
                    "Screen size in inches",
                    "Resolution + touchscreen info",
                    "Processor description",
                    "RAM size (GB)",
                    "Storage description (SSD/HDD, size)",
                    "Graphics card description",
                    "Operating system",
                    "Weight in kg",
                    "Target: price of the laptop",
                ],
            }
        )
    )

# ========================================================================
# SECTION 2 — EDA  (different chart types than the notebook)
# ========================================================================
elif section == "2️⃣ EDA":
    st.title("Exploratory Data Analysis")
    st.caption("Interactive charts — different views than the ones in the notebook.")

    st.subheader("Price distribution")
    fig = px.histogram(clean_df, x="Price", nbins=40, marginal="rug",
                        color_discrete_sequence=["#636EFA"])
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Prices are right-skewed — most laptops are mid-range, with a long tail of premium models.")

    st.subheader("Average price by brand")
    avg_price = clean_df.groupby("Company")["Price"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg_price, x="Company", y="Price", color="Price", color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RAM vs Price, colored by laptop type")
    fig = px.scatter(clean_df, x="Ram", y="Price", color="TypeName", size="Weight",
                      hover_data=["Company"], opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Touchscreen share")
        ts = clean_df["Touchscreen"].map({0: "No", 1: "Yes"}).value_counts().reset_index()
        ts.columns = ["Touchscreen", "Count"]
        fig = px.pie(ts, names="Touchscreen", values="Count", hole=0.45,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Operating system mix")
        os_counts = clean_df["OpSys"].value_counts().reset_index()
        os_counts.columns = ["OpSys", "Count"]
        fig = px.pie(os_counts, names="OpSys", values="Count", hole=0.45,
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price by storage type (SSD vs HDD present)")
    tmp = clean_df.copy()
    tmp["Storage Type"] = np.where((tmp["SSD"] > 0) & (tmp["HDD"] > 0), "SSD + HDD",
                             np.where(tmp["SSD"] > 0, "SSD only",
                             np.where(tmp["HDD"] > 0, "HDD only", "Other")))
    fig = px.violin(tmp, x="Storage Type", y="Price", color="Storage Type", box=True, points="outliers")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full correlation heatmap")
    num_cols = clean_df.select_dtypes(include=[np.number])
    corr = num_cols.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("RAM, SSD capacity and CPU/GPU tier show the strongest positive correlation with price.")

# ========================================================================
# SECTION 3 — CLEANING & PREPROCESSING SUMMARY
# ========================================================================
elif section == "3️⃣ Cleaning & Preprocessing":
    st.title("Cleaning & Preprocessing Summary")

    st.markdown(
        f"""
        - **Duplicate rows removed:** {stats['duplicates']}
        - **Fully-empty row removed:** yes (`dropna(how="all")`)
        - **Missing values before final drop:** {int(stats['missing_before'])}
        - **Missing values after final drop:** {int(stats['missing_after'])}
        - **Outlier filtering:** removed laptops with `Inches > 20`
          ({stats['rows_before_outlier']} → {stats['rows_after_outlier']} rows)
        - **Type fixes:** `Ram` ("8GB" → `8`), `Weight` ("1.5kg" → `1.5`), `Inches` → numeric
        - **Feature engineering:**
            - `Memory` split into numeric **SSD** and **HDD** capacity (GB)
            - `Cpu` → **Cpu_Brand** (first 3 words)
            - `Gpu` → **Gpu_Brand** (first word)
            - `ScreenResolution` → binary **Touchscreen** flag
            - `OpSys` grouped into **Windows / Mac / Linux / No OS**
        - **Encoding:** one-hot encoding (`pd.get_dummies`, `drop_first=True`) on
          Company, TypeName, Cpu_Brand, Gpu_Brand, OpSys
        - **Scaling:** `StandardScaler` fit on the training split
        """
    )

    st.subheader("Before vs after: raw vs cleaned sample")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before cleaning**")
        st.dataframe(raw_df.head(5))
    with c2:
        st.markdown("**After cleaning & feature engineering**")
        st.dataframe(clean_df.head(5))

    st.subheader("Numeric feature ranges after cleaning")
    fig = px.box(clean_df.melt(value_vars=["Inches", "Ram", "Weight"]),
                 x="variable", y="value", color="variable", points=False)
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# SECTION 4 — MODEL RESULTS
# ========================================================================
elif section == "4️⃣ Model Results":
    st.title(f"Model Performance — {model_choice}")
    st.caption("Trained on scaled, one-hot encoded features (no PCA), 80/20 train-test split. "
                "Switch the model from the sidebar to compare results.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{metrics['R2']:.3f}")
    c2.metric("MAE", f"{metrics['MAE']:.1f}")
    c3.metric("MSE", f"{metrics['MSE']:.1f}")
    c4.metric("RMSE", f"{metrics['RMSE']:.1f}")

    st.subheader("Actual vs Predicted price")
    fig = px.scatter(x=y_test, y=preds, labels={"x": "Actual Price", "y": "Predicted Price"},
                      opacity=0.6, color_discrete_sequence=["#00CC96"])
    min_v, max_v = float(min(y_test.min(), preds.min())), float(max(y_test.max(), preds.max()))
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                              line=dict(color="red", dash="dash"), name="Perfect prediction"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residuals distribution")
    residuals = y_test.values - preds
    fig = px.histogram(residuals, nbins=40, labels={"value": "Residual (Actual − Predicted)"},
                        color_discrete_sequence=["#AB63FA"])
    st.plotly_chart(fig, use_container_width=True)

    if model_choice == "Linear Regression":
        st.subheader("Top 15 features by coefficient magnitude")
        importances = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False).head(15)
        value_label = "Absolute coefficient"
    else:
        st.subheader("Top 15 most important features")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        value_label = "Importance"

    fig = px.bar(importances[::-1], orientation="h", labels={"value": value_label, "index": "Feature"})
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# SECTION 5 — INTERACTIVE PREDICTION
# ========================================================================
elif section == "5️⃣ Predict a Price":
    st.title("Predict a Laptop's Price")
    st.markdown("Set the specs below and get a live price prediction from the trained model.")

    companies = sorted(clean_df["Company"].unique())
    types = sorted(clean_df["TypeName"].unique())
    cpus = sorted(clean_df["Cpu_Brand"].unique())
    gpus = sorted(clean_df["Gpu_Brand"].unique())
    oses = sorted(clean_df["OpSys"].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        company = st.selectbox("Company", companies)
        type_name = st.selectbox("Type", types)
        opsys = st.selectbox("Operating System", oses)
    with c2:
        cpu_brand = st.selectbox("CPU brand", cpus)
        gpu_brand = st.selectbox("GPU brand", gpus)
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    with c3:
        ram = st.select_slider("RAM (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        inches = st.slider("Screen size (inches)", 10.0, 18.0, 15.6, 0.1)
        weight = st.slider("Weight (kg)", 0.5, 4.5, 2.0, 0.1)

    c4, c5 = st.columns(2)
    with c4:
        ssd = st.select_slider("SSD (GB)", options=[0, 128, 256, 512, 1000, 2000], value=256)
    with c5:
        hdd = st.select_slider("HDD (GB)", options=[0, 500, 1000, 2000], value=0)

    if st.button("🔮 Predict Price", type="primary"):
        input_row = pd.DataFrame([{
            "Inches": inches, "Ram": ram, "Weight": weight,
            "SSD": ssd, "HDD": hdd, "Touchscreen": 1 if touchscreen == "Yes" else 0,
            "Company": company, "TypeName": type_name,
            "Cpu_Brand": cpu_brand, "Gpu_Brand": gpu_brand, "OpSys": opsys,
        }])

        input_encoded = pd.get_dummies(input_row, columns=text_columns, drop_first=True, dtype=int)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]

        st.success(f"### 💰 Estimated Price: **{prediction:,.0f}**")

        st.caption("This estimate comes from a {} model trained on the cleaned dataset "
                   "(R² ≈ {:.2f} on the held-out test set).".format(model_choice, metrics["R2"]))