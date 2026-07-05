import streamlit as st
import pandas as pd
import pickle
import time

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_artifacts():
    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, label_encoders = load_artifacts()

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

.main {
    background-color: #F7F9FC;
}

.title {
    font-size:45px;
    font-weight:bold;
    text-align:center;
    color:#2563EB;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:30px;
}

div[data-testid="metric-container"]{
    border:1px solid #E5E7EB;
    padding:20px;
    border-radius:15px;
    background-color:white;
    box-shadow:0px 2px 8px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------

st.markdown(
    "<div class='title'>📦 AI Demand Forecasting System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict Product Demand using Machine Learning (XGBoost)</div>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- SIDEBAR --------------------

st.sidebar.header("⚙ Product Information")

price = st.sidebar.number_input(
    "Price",
    min_value=0.0,
    value=50.0
)

discount = st.sidebar.slider(
    "Discount (%)",
    0,
    100,
    10
)

inventory_level = st.sidebar.number_input(
    "Inventory Level",
    min_value=0,
    value=100
)

promotion = st.sidebar.selectbox(
    "Promotion",
    ["No", "Yes"]
)

promotion = 1 if promotion == "Yes" else 0

competitor_pricing = st.sidebar.number_input(
    "Competitor Pricing",
    min_value=0.0,
    value=50.0
)

category = st.sidebar.selectbox(
    "Category",
    label_encoders["Category"].classes_.tolist()
)

# -------------------- INPUT DATA --------------------

input_data = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory Level": [inventory_level],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category": [category]
})

# Encode categorical columns

for col, encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

# -------------------- SUMMARY --------------------

st.subheader("📋 Current Input Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💲 Price", f"₹ {price}")
    st.metric("🏷 Discount", f"{discount}%")

with col2:
    st.metric("📦 Inventory", inventory_level)
    st.metric("🎁 Promotion", "Yes" if promotion else "No")

with col3:
    st.metric("💹 Competitor Price", f"₹ {competitor_pricing}")
    st.metric("🗂 Category", category)

st.divider()

# -------------------- BUTTON --------------------

if st.button("🚀 Predict Demand", use_container_width=True):

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    progress.empty()

    with st.spinner("Running prediction..."):

        prediction = model.predict(input_data)[0]

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="📈 Predicted Demand",
        value=f"{int(prediction)} Units"
    )

    # Demand Status

    if prediction < 100:
        st.error("🔴 Low Demand")

    elif prediction < 300:
        st.warning("🟡 Moderate Demand")

    else:
        st.success("🟢 High Demand")

    st.divider()

    with st.expander("🔍 View Processed Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

# -------------------- FOOTER --------------------
st.markdown("""
<style>

.main .block-container{
    padding-bottom:80px;
}

.footer{
    position:fixed;
    left:0;
    bottom:0;
    width:100%;
    background:linear-gradient(90deg,#0F172A,#1E3A8A);
    color:white;
    padding:14px;
    text-align:center;
    font-size:14px;
    font-weight:500;
    box-shadow:0 -3px 10px rgba(0,0,0,0.25);
    z-index:99999;
}

.footer span{
    color:#60A5FA;
    font-weight:bold;
}

</style>

<div class="footer">
    © 2026 <span>Khushi</span> | AI Demand Forecasting System | Powered by Python • Streamlit • XGBoost | All Rights Reserved
</div>
""", unsafe_allow_html=True)