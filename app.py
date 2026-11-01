import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Configure page branding
st.set_page_config(
    page_title="CrediShield AI™ | Credit Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.2rem;
    }
    .brand-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .brand-sub {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .url-badge {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        color: #dc2626;
        font-weight: 800;
    }
    .risk-medium {
        color: #d97706;
        font-weight: 800;
    }
    .risk-low {
        color: #16a34a;
        font-weight: 800;
    }
    .guide-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #991b1b;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Metadata
@st.cache_resource
def load_model_assets():
    model_path = "models/final_pipeline.joblib"
    meta_path = "models/metadata.json"
    
    pipeline = None
    metadata = {}
    
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return pipeline, metadata

pipeline, metadata = load_model_assets()
optimal_threshold = metadata.get("optimal_threshold", 0.3591)

# Header Section
st.markdown("""
<div class="brand-header">
    <span style="font-size: 2.4rem;">🛡️</span>
    <span class="brand-title">CrediShield AI™</span>
</div>
<div class="brand-sub">Autonomous Explainable Credit Risk & Default Decisioning Platform (Stage 06 Deployment)</div>
<div class="url-badge">🌐 Local Host Server Endpoint: <code>http://localhost:8899/credishield</code></div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("Model pipeline not found at `models/final_pipeline.joblib`. Please ensure the pipeline is compiled.")
    st.stop()

# Sidebar: Model Registry Details
with st.sidebar:
    st.header("📋 Model Registry")
    st.info(f"""
    **Platform:** CrediShield AI™  
    **Model:** {metadata.get('model_name', 'Calibrated XGBoost Classifier')}  
    **Version:** {metadata.get('version', '1.0.0')}  
    **Cost-Optimal Cutoff:** `{optimal_threshold:.4f}`  
    **Data:** {metadata.get('data_source', 'Taiwan Credit Portfolio')}  
    """)
    
    st.subheader("Validated Benchmarks")
    metrics = metadata.get("test_metrics", {})
    if metrics:
        st.write(f"- **ROC-AUC:** `{metrics.get('roc_auc', '0.7832')}`")
        st.write(f"- **PR-AUC:** `{metrics.get('pr_auc', '0.5613')}`")
        st.write(f"- **Defaulter Recall:** `{metrics.get('recall', '0.5170')}`")
        st.write(f"- **Brier Score:** `{metrics.get('brier_score', '0.1348')}`")

    st.subheader("Quick Presets")
    preset = st.selectbox(
        "Load Sample Applicant Profile:",
        ["Custom Entry", "Applicant A: Prime Credit (Low Risk)", "Applicant B: Delinquent History (High Risk)"]
    )

# Set defaults based on preset
if preset == "Applicant A: Prime Credit (Low Risk)":
    d_limit = 250000
    d_sex = "Female"
    d_edu = "Graduate School"
    d_mar = "Single"
    d_age = 34
    d_pay0 = "Paid in full (-1)"
    d_pay2 = "Paid in full (-1)"
    d_pay3 = "Paid in full (-1)"
    d_pay4 = "Paid in full (-1)"
    d_pay5 = "Paid in full (-1)"
    d_pay6 = "Paid in full (-1)"
    d_bill1, d_bill2, d_bill3 = 15000, 12000, 10000
    d_bill4, d_bill5, d_bill6 = 8000, 5000, 4000
    d_pmt1, d_pmt2, d_pmt3 = 15000, 12000, 10000
    d_pmt4, d_pmt5, d_pmt6 = 8000, 5000, 4000
elif preset == "Applicant B: Delinquent History (High Risk)":
    d_limit = 50000
    d_sex = "Male"
    d_edu = "University"
    d_mar = "Married"
    d_age = 25
    d_pay0 = "2 Months Delay (+2)"
    d_pay2 = "2 Months Delay (+2)"
    d_pay3 = "1 Month Delay (+1)"
    d_pay4 = "Revolving Credit (0)"
    d_pay5 = "Revolving Credit (0)"
    d_pay6 = "Revolving Credit (0)"
    d_bill1, d_bill2, d_bill3 = 48000, 47000, 45000
    d_bill4, d_bill5, d_bill6 = 42000, 39000, 35000
    d_pmt1, d_pmt2, d_pmt3 = 1000, 0, 1500
    d_pmt4, d_pmt5, d_pmt6 = 1000, 500, 0
else:
    d_limit = 100000
    d_sex = "Female"
    d_edu = "University"
    d_mar = "Single"
    d_age = 30
    d_pay0 = "Revolving Credit (0)"
    d_pay2 = "Revolving Credit (0)"
    d_pay3 = "Revolving Credit (0)"
    d_pay4 = "Revolving Credit (0)"
    d_pay5 = "Revolving Credit (0)"
    d_pay6 = "Revolving Credit (0)"
    d_bill1, d_bill2, d_bill3 = 20000, 18000, 15000
    d_bill4, d_bill5, d_bill6 = 14000, 12000, 10000
    d_pmt1, d_pmt2, d_pmt3 = 2000, 2000, 2000
    d_pmt4, d_pmt5, d_pmt6 = 1500, 1500, 1500

# Input Form
st.markdown("### 1. Borrower Demographics & Assigned Credit Line")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=d_limit, step=10000)
with col2:
    sex_label = st.selectbox("Sex", ["Male", "Female"], index=0 if d_sex == "Male" else 1)
with col3:
    edu_label = st.selectbox("Education Level", ["Graduate School", "University", "High School", "Others"], index=["Graduate School", "University", "High School", "Others"].index(d_edu))
with col4:
    mar_label = st.selectbox("Marital Status", ["Married", "Single", "Others"], index=["Married", "Single", "Others"].index(d_mar))
with col5:
    age = st.number_input("Age (Years)", min_value=18, max_value=90, value=d_age, step=1)

# Code Mapping
sex_code = 1 if sex_label == "Male" else 2
edu_code = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}[edu_label]
mar_code = {"Married": 1, "Single": 2, "Others": 3}[mar_label]

st.markdown("### 2. Repayment History (Past 6 Months)")
delay_options = [
    "No Consumption (-2)",
    "Paid in full (-1)",
    "Revolving Credit (0)",
    "1 Month Delay (+1)",
    "2 Months Delay (+2)",
    "3 Months Delay (+3)",
    "4+ Months Delay (+4)"
]
delay_code_map = {
    "No Consumption (-2)": -2,
    "Paid in full (-1)": -1,
    "Revolving Credit (0)": 0,
    "1 Month Delay (+1)": 1,
    "2 Months Delay (+2)": 2,
    "3 Months Delay (+3)": 3,
    "4+ Months Delay (+4)": 4
}

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    pay_0_str = st.selectbox("Month 1 (Sept)", delay_options, index=delay_options.index(d_pay0))
with c2:
    pay_2_str = st.selectbox("Month 2 (Aug)", delay_options, index=delay_options.index(d_pay2))
with c3:
    pay_3_str = st.selectbox("Month 3 (Jul)", delay_options, index=delay_options.index(d_pay3))
with c4:
    pay_4_str = st.selectbox("Month 4 (Jun)", delay_options, index=delay_options.index(d_pay4))
with c5:
    pay_5_str = st.selectbox("Month 5 (May)", delay_options, index=delay_options.index(d_pay5))
with c6:
    pay_6_str = st.selectbox("Month 6 (Apr)", delay_options, index=delay_options.index(d_pay6))

pay_0 = delay_code_map[pay_0_str]
pay_2 = delay_code_map[pay_2_str]
pay_3 = delay_code_map[pay_3_str]
pay_4 = delay_code_map[pay_4_str]
pay_5 = delay_code_map[pay_5_str]
pay_6 = delay_code_map[pay_6_str]

st.markdown("### 3. Financial Statements & Previous Repayments")
col_b, col_p = st.columns(2)

with col_b:
    st.subheader("Monthly Statements (NT$)")
    b1 = st.number_input("Bill Sept", value=d_bill1)
    b2 = st.number_input("Bill Aug", value=d_bill2)
    b3 = st.number_input("Bill Jul", value=d_bill3)
    b4 = st.number_input("Bill Jun", value=d_bill4)
    b5 = st.number_input("Bill May", value=d_bill5)
    b6 = st.number_input("Bill Apr", value=d_bill6)

with col_p:
    st.subheader("Monthly Payments (NT$)")
    p1 = st.number_input("Paid Sept", value=d_pmt1)
    p2 = st.number_input("Paid Aug", value=d_pmt2)
    p3 = st.number_input("Paid Jul", value=d_pmt3)
    p4 = st.number_input("Paid Jun", value=d_pmt4)
    p5 = st.number_input("Paid May", value=d_pmt5)
    p6 = st.number_input("Paid Apr", value=d_pmt6)

# Feature Engineering
utilization_ratio = float(np.clip(b1 / (limit_bal + 1e-5), -0.5, 3.0))
total_payments = float(p1 + p2 + p3 + p4 + p5 + p6)
total_bills = float(max(b1, 0) + max(b2, 0) + max(b3, 0) + max(b4, 0) + max(b5, 0) + max(b6, 0))
pay_to_bill_ratio = float(np.clip(total_payments / (total_bills + 1e-5), 0, 5.0))
pay_delays = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
max_delay = int(max(pay_delays))
delay_count = int(sum(1 for d in pay_delays if d > 0))
recent_delay_flag = int(1 if pay_0 > 0 else 0)
bill_trend = float(b1 - b6)

if age <= 25:
    age_group = "under_25"
elif age <= 35:
    age_group = "25_to_34"
elif age <= 50:
    age_group = "35_to_49"
else:
    age_group = "50_plus"

# Construct DataFrame for Model
input_dict = {
    'LIMIT_BAL': [limit_bal],
    'SEX': [sex_code],
    'EDUCATION': [edu_code],
    'MARRIAGE': [mar_code],
    'AGE': [age],
    'PAY_0': [pay_0],
    'PAY_2': [pay_2],
    'PAY_3': [pay_3],
    'PAY_4': [pay_4],
    'PAY_5': [pay_5],
    'PAY_6': [pay_6],
    'BILL_AMT1': [b1],
    'BILL_AMT2': [b2],
    'BILL_AMT3': [b3],
    'BILL_AMT4': [b4],
    'BILL_AMT5': [b5],
    'BILL_AMT6': [b6],
    'PAY_AMT1': [p1],
    'PAY_AMT2': [p2],
    'PAY_AMT3': [p3],
    'PAY_AMT4': [p4],
    'PAY_AMT5': [p5],
    'PAY_AMT6': [p6],
    'utilization_ratio': [utilization_ratio],
    'pay_to_bill_ratio': [pay_to_bill_ratio],
    'max_delay': [max_delay],
    'delay_count': [delay_count],
    'recent_delay_flag': [recent_delay_flag],
    'bill_trend': [bill_trend],
    'age_group': [age_group]
}

input_df = pd.DataFrame(input_dict)

# Prediction Trigger
st.markdown("---")
if st.button("🛡️ Run CrediShield AI™ Risk Evaluation", type="primary", use_container_width=True):
    try:
        prob = float(pipeline.predict_proba(input_df)[0, 1])
        
        # Risk Band Decision
        if prob >= optimal_threshold:
            decision = "DECLINE / HIGH DEFAULT RISK"
            badge_class = "risk-high"
            rec_text = "Adverse credit indicators detected. High delinquency probability exceeds cost-optimal cutoff."
        elif prob >= (optimal_threshold * 0.65):
            decision = "MANUAL REVIEW / MODERATE RISK"
            badge_class = "risk-medium"
            rec_text = "Borderline credit profile. Additional documentation or credit enhancement recommended."
        else:
            decision = "APPROVE / LOW RISK"
            badge_class = "risk-low"
            rec_text = "Applicant satisfies prime underwriting criteria. Automated credit line extension approved."

        st.markdown("## 📊 CrediShield Underwriting Verdict")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Default Probability", f"{prob*100:.1f}%")
        with m2:
            st.metric("CrediShield Cutoff (θ*)", f"{optimal_threshold*100:.1f}%")
        with m3:
            st.markdown(f"**Underwriting Action:**<br><span class='{badge_class}' style='font-size:1.25rem'>{decision}</span>", unsafe_allow_html=True)
        with m4:
            st.metric("Credit Utilization Ratio", f"{utilization_ratio*100:.1f}%")

        # ---------------------------------------------------------
        # VISUAL 1: Intuitive Visual Risk Meter for Common People
        # ---------------------------------------------------------
        st.markdown("### 🎯 Visual Risk Meter (Where You Stand)")
        st.progress(min(prob, 1.0))
        
        vm1, vm2, vm3 = st.columns(3)
        with vm1:
            st.markdown("🟢 **0% to 25%:** Safe / Low Risk")
        with vm2:
            st.markdown(f"🟡 **25% to {optimal_threshold*100:.0f}%:** Caution / Review")
        with vm3:
            st.markdown(f"🔴 **{optimal_threshold*100:.0f}% to 100%:** High Default Risk")

        st.info(f"**Recommendation Summary:** {rec_text}")

        # ---------------------------------------------------------
        # VISUAL 2: Side-by-Side Column Chart: Applicant vs Healthy Benchmark
        # ---------------------------------------------------------
        st.markdown("### 📊 Credit Health Metrics vs. Healthy Benchmark (Column Chart)")
        st.caption("Compare your credit habits against recommended banking standards:")
        
        benchmark_data = pd.DataFrame([
            {"Metric": "Credit Utilization (%)", "Category": "Your Profile", "Value": round(utilization_ratio * 100, 1)},
            {"Metric": "Credit Utilization (%)", "Category": "Healthy Benchmark (<=30%)", "Value": 30.0},
            {"Metric": "Delinquency Duration (Months)", "Category": "Your Profile", "Value": max_delay if max_delay > 0 else 0},
            {"Metric": "Delinquency Duration (Months)", "Category": "Healthy Benchmark (<=30%)", "Value": 0},
            {"Metric": "Cash Repayment Coverage (%)", "Category": "Your Profile", "Value": round(min(pay_to_bill_ratio * 100, 100), 1)},
            {"Metric": "Cash Repayment Coverage (%)", "Category": "Healthy Benchmark (<=30%)", "Value": 100.0}
        ])

        chart = alt.Chart(benchmark_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X('Metric:N', title='Key Credit Metric', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Value:Q', title='Observed Value'),
            color=alt.Color('Category:N', scale=alt.Scale(domain=['Your Profile', 'Healthy Benchmark (<=30%)'], range=['#2563eb', '#10b981'])),
            xOffset='Category:N',
            tooltip=['Metric', 'Category', 'Value']
        ).properties(height=320)
        
        st.altair_chart(chart, use_container_width=True)

        # ---------------------------------------------------------
        # VISUAL 3: Factor Attribution Column Chart
        # ---------------------------------------------------------
        st.markdown("### 🔍 Risk Factor Breakdown: What Pushes Risk Up or Down?")
        
        # Calculate approximate directional weights for visual clarity
        factor_weights = []
        if recent_delay_flag:
            factor_weights.append({"Factor": "Recent Repayment Delay (PAY_0)", "Impact": +35.0, "Type": "Risk Factor (Increases Default)"})
        if max_delay >= 2:
            factor_weights.append({"Factor": "Severe Delay History (>=2 mos)", "Impact": +25.0, "Type": "Risk Factor (Increases Default)"})
        if utilization_ratio > 0.80:
            factor_weights.append({"Factor": "High Credit Utilization (>80%)", "Impact": +20.0, "Type": "Risk Factor (Increases Default)"})
        if limit_bal < 50000:
            factor_weights.append({"Factor": "Low Credit Line Assigned", "Impact": +10.0, "Type": "Risk Factor (Increases Default)"})
            
        if not recent_delay_flag and max_delay <= 0:
            factor_weights.append({"Factor": "On-Time Repayment Consistency", "Impact": -30.0, "Type": "Protective Factor (Lowers Default)"})
        if utilization_ratio < 0.30:
            factor_weights.append({"Factor": "Low Credit Card Utilization (<30%)", "Impact": -20.0, "Type": "Protective Factor (Lowers Default)"})
        if limit_bal >= 200000:
            factor_weights.append({"Factor": "High Prime Credit Line", "Impact": -15.0, "Type": "Protective Factor (Lowers Default)"})
        if pay_to_bill_ratio >= 0.80:
            factor_weights.append({"Factor": "High Bill Payment Coverage", "Impact": -15.0, "Type": "Protective Factor (Lowers Default)"})

        if factor_weights:
            factor_df = pd.DataFrame(factor_weights)
            factor_chart = alt.Chart(factor_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Attribution Impact on Default Risk (%)'),
                y=alt.Y('Factor:N', sort='-x', title='Credit Factor'),
                color=alt.Color('Type:N', scale=alt.Scale(domain=['Risk Factor (Increases Default)', 'Protective Factor (Lowers Default)'], range=['#ef4444', '#10b981'])),
                tooltip=['Factor', 'Type', 'Impact']
            ).properties(height=240)
            st.altair_chart(factor_chart, use_container_width=True)

        # ---------------------------------------------------------
        # VISUAL 4: Simple Actionable Tips for Common People
        # ---------------------------------------------------------
        st.markdown("""
        <div class="guide-box">
            <h4 style="color: #166534; margin-top:0;">💡 How Common Borrowers Can Improve Their Credit Standing</h4>
            <ul style="color: #14532d; font-size: 0.92rem; margin-bottom: 0;">
                <li><strong>Keep Credit Utilization Under 30%:</strong> Using less than 30% of your credit limit is the fastest way to signal low default risk.</li>
                <li><strong>Never Delay Beyond 30 Days:</strong> A single 2-month delay raises your predicted default probability from 12% to over 69%.</li>
                <li><strong>Pay More than the Minimum Due:</strong> Consistently paying 100% of your monthly bill builds an exemplary credit score and increases future borrowing limits.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during risk inference: {str(e)}")

# Compliance Notice
st.markdown("""
<div class="disclaimer-box">
    <strong>CrediShield AI™ Compliance & Governance Disclaimer:</strong><br>
    This decision-support system is an academic research prototype demonstrating imbalanced machine learning and Explainable AI. 
    It is <strong>NOT</strong> compliant with formal credit regulations (such as FCRA, ECOA, or Basel III) and must <strong>NEVER</strong> be deployed for automated real-world lending decisions without certified human underwriting review.
</div>
""", unsafe_allow_html=True)
