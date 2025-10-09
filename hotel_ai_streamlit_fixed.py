# hotel_ai_streamlit_fixed.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Config / Feature lists
# ---------------------------
TARGET = "is_canceled"

NUMERIC_FIELDS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "previous_cancellations",
    "booking_changes",
]

CATEGORICAL_FIELDS = ["deposit_type", "customer_type", "market_segment"]

LOYALTY_FEATURES = ["total_of_special_requests", "adr", "previous_bookings_not_canceled"]

REQUIRED = set([TARGET] + NUMERIC_FIELDS + CATEGORICAL_FIELDS + LOYALTY_FEATURES)

default_path = "C:/Users/2309m/Downloads/hotel_bookings.csv/hotel_bookings.csv"

# ---------------------------
# Helpers
# ---------------------------
def prepare_df_from_df(df_raw):
    """Given a raw DataFrame, validate/encode/fill and return df, encoders, feature_columns, X, y"""
    df = df_raw.copy()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    encoders = {}
    for col in CATEGORICAL_FIELDS:
        df[col] = df[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le
        df[col + "_enc"] = le.transform(df[col])

    df[NUMERIC_FIELDS + LOYALTY_FEATURES] = df[NUMERIC_FIELDS + LOYALTY_FEATURES].fillna(0)
    feature_columns = NUMERIC_FIELDS + [c + "_enc" for c in CATEGORICAL_FIELDS]
    X = df[feature_columns]
    y = df[TARGET].astype(int)
    return df, encoders, feature_columns, X, y

@st.cache_data
def load_and_prepare_from_path(csv_path: str):
    df_raw = pd.read_csv(csv_path)
    return prepare_df_from_df(df_raw)

@st.cache_resource
def train_models(df, feature_columns, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X_train, y_train)
    df["cancellation_probability"] = clf.predict_proba(X)[:, 1]

    X_loyal = df[LOYALTY_FEATURES]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_loyal)
    df["loyalty_cluster"] = kmeans.predict(X_loyal)

    centers = kmeans.cluster_centers_
    center_sums = centers.sum(axis=1)
    order = center_sums.argsort()[::-1]
    label_names = ["High Loyalty", "Medium Loyalty", "Low Loyalty"]
    cluster_to_label = {cluster_id: label_names[rank] for rank, cluster_id in enumerate(order)}
    df["loyalty_segment"] = df["loyalty_cluster"].map(cluster_to_label)

    return clf, kmeans, cluster_to_label, df

# ---------------------------
# Streamlit UI & Flow
# ---------------------------
st.set_page_config(page_title="Hotel Booking AI Assistant", layout="centered")
st.title("🏨 Hotel Booking AI — Cancellation Prediction & Recommendations")

st.markdown("Upload `hotel_bookings.csv` or use the default path.")

uploaded_file = st.file_uploader("Upload hotel_bookings.csv (optional)", type=["csv"])
use_default_path = st.checkbox("Use default CSV path instead of uploaded file", value=False)

# Attempt to load/prep data; if fails, show error and stop before building inputs
df = encoders = feature_columns = X = y = None
clf = kmeans = cluster_to_label = None

try:
    if uploaded_file is not None and not use_default_path:
        df_raw = pd.read_csv(uploaded_file)
        df, encoders, feature_columns, X, y = prepare_df_from_df(df_raw)
        st.success("Uploaded CSV loaded and prepared.")
        clf, kmeans, cluster_to_label, df = train_models(df, feature_columns, X, y)
    elif use_default_path:
        df, encoders, feature_columns, X, y = load_and_prepare_from_path(default_path)
        clf, kmeans, cluster_to_label, df = train_models(df, feature_columns, X, y)
        st.success(f"Loaded CSV from default path: {default_path}")
    else:
        st.info("Please upload a CSV or check 'Use default CSV path' to load data from disk.")
        st.stop()
except Exception as e:
    st.error(f"Error loading/preparing data: {e}")
    st.stop()

# By this point df, encoders, clf, kmeans should be defined
st.write("---")
st.header("Input booking details")

# Numeric inputs
col1, col2 = st.columns(2)
with col1:
    lead_time = st.number_input("Lead time (days)", min_value=0.0, max_value=3650.0, value=30.0, step=1.0)
    stays_weekend = st.number_input("Stays in weekend nights", min_value=0.0, max_value=365.0, value=0.0, step=1.0)
    stays_week = st.number_input("Stays in week nights", min_value=0.0, max_value=365.0, value=2.0, step=1.0)
    adults = st.number_input("Adults", min_value=0.0, max_value=10.0, value=2.0, step=1.0)

with col2:
    children = st.number_input("Children", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    prev_canc = st.number_input("Previous cancellations", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    booking_changes = st.number_input("Booking changes", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

# Categorical selects (safe because df is defined)
st.subheader("Booking Type")
cat_cols = st.columns(len(CATEGORICAL_FIELDS))
cat_values = {}
for i, col in enumerate(CATEGORICAL_FIELDS):
    values = sorted(df[col].dropna().astype(str).unique().tolist())
    cat_values[col] = cat_cols[i].selectbox(col.replace("_", " ").title(), options=values)

# Loyalty inputs
st.subheader("Loyalty Indicators")
lor_col1, lor_col2, lor_col3 = st.columns(3)
with lor_col1:
    special_requests = st.number_input("Total special requests", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
with lor_col2:
    adr = st.number_input("ADR (avg daily rate)", min_value=0.0, max_value=10000.0, value=100.0, step=1.0)
with lor_col3:
    prev_not_canc = st.number_input("Previous bookings not canceled", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)

# Build input dictionary
input_data = {}
for col in NUMERIC_FIELDS + LOYALTY_FEATURES:
    if col == "lead_time":
        input_data[col] = lead_time
    elif col == "stays_in_weekend_nights":
        input_data[col] = stays_weekend
    elif col == "stays_in_week_nights":
        input_data[col] = stays_week
    elif col == "adults":
        input_data[col] = adults
    elif col == "children":
        input_data[col] = children
    elif col == "previous_cancellations":
        input_data[col] = prev_canc
    elif col == "booking_changes":
        input_data[col] = booking_changes
    elif col == "total_of_special_requests":
        input_data[col] = special_requests
    elif col == "adr":
        input_data[col] = adr
    elif col == "previous_bookings_not_canceled":
        input_data[col] = prev_not_canc
    else:
        input_data[col] = 0.0

for col in CATEGORICAL_FIELDS:
    val = cat_values[col]
    le = encoders[col]
    if val not in le.classes_:
        st.error(f"Invalid option for {col}: {val}")
        st.stop()
    input_data[col + "_enc"] = int(le.transform([val])[0])

# Predict button
if st.button("🔮 Predict & Recommend"):
    X_input = pd.DataFrame([input_data])[feature_columns]
    prob = clf.predict_proba(X_input)[0, 1]
    loyalty_input = pd.DataFrame([input_data])[LOYALTY_FEATURES]
    cluster_id = kmeans.predict(loyalty_input)[0]
    loyalty_label = cluster_to_label[cluster_id]

    recommendation_lines = []
    if prob > 0.7:
        recommendation_lines.append("⚠ High risk of cancellation! Ask for confirmation or prepayment.")
    elif 0.4 < prob <= 0.7:
        recommendation_lines.append("⚠ Moderate risk. Send reminder and small incentive.")
    else:
        recommendation_lines.append("✅ Low risk. Booking is likely secure.")

    if loyalty_label == "High Loyalty":
        recommendation_lines.append("💎 Recommend premium upsells (spa, room upgrade, loyalty points).")
    elif loyalty_label == "Medium Loyalty":
        recommendation_lines.append("🌟 Recommend standard upsells (breakfast, late checkout, small discounts).")
    else:
        recommendation_lines.append("🎁 Recommend loyalty incentive (discount for next booking, special offer).")

    st.metric("Cancellation Probability", f"{prob*100:.2f}%")
    st.write("**Loyalty Segment:**", loyalty_label)
    st.write("**Recommendations:**")
    for line in recommendation_lines:
        st.write("-", line)

st.write("---")
st.caption("Powered by AI & ML • Hotel Booking Insights — converted from Tkinter to Streamlit")
