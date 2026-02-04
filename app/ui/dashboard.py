import streamlit as st
import requests
import plotly.graph_objects as go
import time

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Minimal & Robust) ---
# We use standard Streamlit classes where possible to avoid theming conflicts, 
# and only add custom styling for specific components.
st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Risk Badges */
    .risk-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
        border: 1px solid #f87171;
    }
    .risk-badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
        border: 1px solid #4ade80;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Title Styling */
    h1 {
        color: #1f2937;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        height: 3em;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
def call_api(endpoint, data):
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def main():
    # --- HEADER ---
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.title("Customer Churn Intelligence")
        st.markdown("Predictive analytics for customer retention.")
    
    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("📋 Customer Profile")
        with st.form("input_form"):
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            st.subheader("Services")
            tenure = st.slider("Tenure (Months)", 0, 72, 12, help="How long the customer has been with us.")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            
            with st.expander("More Options"):
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiline = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                online_sec = st.checkbox("Online Security")
                tech_sup = st.checkbox("Tech Support")
                paperless = st.checkbox("Paperless Billing", value=True)
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

            st.subheader("Billing")
            monthly = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0)
            total = st.number_input("Total Charges ($)", 0.0, 20000.0, 70.0 * tenure)
            
            st.markdown("---")
            submit = st.form_submit_button("Run Analysis", help="Click to predict churn probability")

    # --- MAIN DASHBOARD ---
    if submit:
        # 1. Prepare Data
        payload = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone if 'phone' in locals() else "Yes",
            "MultipleLines": multiline if 'multiline' in locals() else "No",
            "InternetService": internet, 
            "OnlineSecurity": "Yes" if online_sec else "No",
            "TechSupport": "Yes" if tech_sup else "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": contract, "PaperlessBilling": "Yes" if paperless else "No",
            "PaymentMethod": payment if 'payment' in locals() else "Electronic check",
            "MonthlyCharges": monthly, "TotalCharges": total
        }

        # 2. Call API with Spinner
        with st.spinner("Analyzing data models..."):
            time.sleep(0.5) # Slight delay for UX
            result = call_api("predict", payload)
        
        if result:
            prob = result['churn_probability']
            churn_risk = prob > 0.5
            
            # --- TOP METRICS ROW ---
            st.markdown("### Analysis Results")
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric("Churn Probability", f"{prob:.1%}", delta=f"{'High' if churn_risk else 'Low'}", delta_color="inverse")
            with m2:
                # Custom HTML Badge
                badge_class = "risk-badge-high" if churn_risk else "risk-badge-low"
                badge_text = "HIGH RISK" if churn_risk else "LOW RISK"
                st.markdown(f"""
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">Risk Category</div>
                    <span class="{badge_class}">{badge_text}</span>
                """, unsafe_allow_html=True)
            with m3:
                st.metric("Estimated LTV Loss", f"${monthly * 12:.2f}", help="Potential annual revenue loss")

            # --- CHARTS ROW ---
            st.markdown("---")
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.subheader("Risk Meter")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#dc2626" if churn_risk else "#16a34a"},
                        'steps': [
                            {'range': [0, 50], 'color': "#f3f4f6"},
                            {'range': [50, 100], 'color': "#fee2e2"} if churn_risk else {'range': [50, 100], 'color': "#f3f4f6"}
                        ],
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("Key Drivers")
                # Explanation API
                explain = call_api("explain", payload)
                if explain:
                    imp = explain['feature_importance']
                    
                    # Sort and plot
                    sorted_imp = dict(sorted(imp.items(), key=lambda item: abs(item[1]), reverse=False))
                    
                    fig_bar = go.Figure(go.Bar(
                        x=list(sorted_imp.values()),
                        y=list(sorted_imp.keys()),
                        orientation='h',
                        marker_color='#3b82f6'
                    ))
                    fig_bar.update_layout(
                        title="Top Factors Influencing Prediction",
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Impact Magnitude"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # --- STRATEGY ROW ---
            st.markdown("### 🤖 Recommended Strategy")
            try:
                retention = requests.post(f"{API_URL}/retention", params={"churn_prob": prob}, json=list(imp.keys())).json()
                
                with st.expander(f"Strategy: {retention['strategy']}", expanded=True):
                    for item in retention['action_items']:
                        st.markdown(f"✅ {item}")
                    
                    st.info(f"**Draft Email Subject:** {retention['strategy']}")
                    st.code(retention['email_draft'], language="text")
            except:
                st.warning("Could not generate AI strategy.")

        else:
            st.error("Error: Could not connect to the inference engine. Please make sure the backend is running.")

    else:
        # Empty State
        st.info("👈 Please enter customer details in the sidebar to begin analysis.")
        st.markdown("""
        ### How it works
        1. **Enter Details**: Fill in the customer demographics and service info.
        2. **Run Analysis**: Our XGBoost model predicts the probability of churn.
        3. **Get Insights**: See *why* they might churn and *how* to keep them.
        """)

if __name__ == "__main__":
    main()
