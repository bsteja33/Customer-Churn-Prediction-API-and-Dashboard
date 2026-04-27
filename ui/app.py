import streamlit as st
import requests
import plotly.graph_objects as go
import os

# Configure the API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered",
    initial_sidebar_state="collapsed",
)


if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None


st.markdown("""
<style>
    .hero {
        text-align: center;
        padding: 3rem 0;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        height: 3rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def go_home():
    st.session_state.page = 'home'
    st.session_state.prediction_result = None

def render_home():
    st.markdown("<div class='hero'><h1>Customer Churn Prediction</h1><p>Enter telecom customer data below to instantly assess retention risk.</p></div>", unsafe_allow_html=True)

    with st.form("customer_form"):
        st.subheader("Demographics & Account Details")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0)
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            
        with col2:
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Bank Withdrawal", "Credit Card", "Mailed Check"
            ])
            monthly_charges = st.slider("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=65.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)

        st.subheader("Service Subscriptions")
        col3, col4 = st.columns(2)
        
        with col3:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "Unknown"])
            internet = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "Unknown"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "Unknown"])
            
        with col4:
            online_security = st.selectbox("Online Security", ["Yes", "No", "Unknown"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "Unknown"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "Unknown"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "Unknown"])

        submit_button = st.form_submit_button(label="Predict Churn Risk", type="primary")

    if submit_button:
        # Determine Internet Service (binary Yes/No) and Internet Type from the user's selection
        internet_type = internet if internet != "None" else "Unknown"
        internet_service = "Yes" if internet != "None" else "No"

        payload = {
            # Fields captured from UI controls
            "Gender": gender,
            "Married": partner,
            "Tenure in Months": tenure,
            "Phone Service": phone,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Internet Type": internet_type,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection Plan": device_protection,
            "Premium Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charge": monthly_charges,
            "Total Charges": total_charges,
            # Default baseline values for fields not exposed in the UI
            "Number of Dependents": dependents,
            "Number of Referrals": 0,
            "Age": 45,
            "Offer": "Unknown",
            "Avg Monthly Long Distance Charges": 0.0,
            "Avg Monthly GB Download": 20.0,
            "Streaming Music": "No",
            "Unlimited Data": "Yes",
            "Total Refunds": 0.0,
            "Total Extra Data Charges": 0.0,
            "Total Long Distance Charges": 0.0,
            "Total Revenue": float(total_charges),
        }

        with st.spinner("Analyzing customer profile..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                if response.status_code == 200:
                    st.session_state.prediction_result = response.json()
                    st.session_state.page = 'results'
                    st.rerun()
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.Timeout:
                st.error(
                    "The prediction engine is taking longer than expected to warm up. "
                    "Please ensure the backend server is fully running and try again."
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend API at {API_URL}.")
                st.write(str(e))


def render_results():
    st.markdown("<div class='hero'><h1>Analysis Complete</h1><p>Review the comprehensive retention risk profile below.</p></div>", unsafe_allow_html=True)
    
    result = st.session_state.prediction_result
    prob = result["churn_probability"]
    risk_tier = result["retention_risk"]

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': "#2ecc71"},
                {'range': [40, 70], 'color': "#f1c40f"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    st.plotly_chart(fig, width="stretch")

    # Risk Assessment Text
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Risk Assessment</h3>", unsafe_allow_html=True)
    
    if risk_tier == "High":
        st.error(f"**Risk Tier: {risk_tier} (>{70}%)**")
        st.markdown("**Immediate intervention recommended.** This customer exhibits a high likelihood of canceling their subscription. Consider targeted promotional offers, immediate account manager outreach, or contract renegotiation.")
    elif risk_tier == "Medium":
        st.warning(f"**Risk Tier: {risk_tier} (40-70%)**")
        st.markdown("**Monitor closely.** This customer is showing signs of potential churn. A proactive check-in or a slight discount on their next billing cycle could stabilize the account.")
    else:
        st.success(f"**Risk Tier: {risk_tier} (<40%)**")
        st.markdown("**Customer is stable.** The model indicates a very low likelihood of churn. Maintain standard service excellence.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Make Another Prediction", on_click=go_home):
        pass


# Routing Logic
if st.session_state.page == 'home':
    render_home()
elif st.session_state.page == 'results':
    render_results()
