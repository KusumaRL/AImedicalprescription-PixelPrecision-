import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="🏥 AI Medical Prescription Verification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    .alert-high {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .alert-moderate {
        background-color: #fff4e6;
        border: 1px solid #ffd700;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .alert-low {
        background-color: #f0f8ff;
        border: 1px solid #87ceeb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .success-box {
        background-color: #f0fff0;
        border: 1px solid #90ee90;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
    }

    .drug-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }

    .interaction-high {
        border-left-color: #dc3545 !important;
    }

    .interaction-moderate {
        border-left-color: #ffc107 !important;
    }

    .interaction-low {
        border-left-color: #17a2b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'patient_drugs' not in st.session_state:
    st.session_state.patient_drugs = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


# ==================== UTILITY FUNCTIONS ====================

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "POST":
            response = requests.post(url, json=data, timeout=60)
        else:
            response = requests.get(url, timeout=30)

        response.raise_for_status()
        return {"success": True, "data": response.json()}

    except requests.exceptions.ConnectionError:
        return {"success": False,
                "error": "❌ Cannot connect to AI backend server. Please ensure the FastAPI server is running on localhost:8000"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "⏱️ Request timed out. The AI analysis is taking longer than expected."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"🚫 Server error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": f"🔥 Unexpected error: {str(e)}"}


def get_severity_color(severity: str) -> str:
    """Get color for severity level"""
    colors = {
        "High": "#dc3545",
        "Moderate": "#ffc107",
        "Low": "#17a2b8"
    }
    return colors.get(severity, "#6c757d")


def create_safety_gauge(score: float) -> go.Figure:
    """Create safety score gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Safety Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_interaction_chart(interactions: List[Dict]) -> go.Figure:
    """Create drug interactions visualization"""
    if not interactions:
        return None

    severity_counts = {}
    for interaction in interactions:
        severity = interaction.get('severity', 'Unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=['#dc3545' if x == 'High' else '#ffc107' if x == 'Moderate' else '#17a2b8' for x in
                          severity_counts.keys()]
        )
    ])

    fig.update_layout(
        title="Drug Interactions by Severity",
        xaxis_title="Severity Level",
        yaxis_title="Count",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Medical Prescription Verification System</h1>
        <p>Advanced AI-Powered Drug Interaction & Safety Analysis with HuggingFace & IBM Watson</p>
        <p>🤖 Intelligent Clinical Decision Support • 🔍 Advanced NLP Analysis • ⚡ Real-time Verification</p>
    </div>
    """, unsafe_allow_html=True)

    # Check API connection
    api_status = make_api_request("/")
    if not api_status["success"]:
        st.error(api_status["error"])
        st.info(
            "💡 **Quick Fix**: Start the FastAPI backend server with: `python main.py` or `uvicorn main:app --reload`")
        return

    # Display API status
    api_data = api_status["data"]
    with st.expander("🔌 AI System Status", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"**Backend Status**: {api_data.get('status', 'Unknown')}")
            st.info(f"**Version**: {api_data.get('version', 'Unknown')}")

        with col2:
            ai_services = api_data.get('ai_services', {})
            for service, info in ai_services.items():
                st.write(f"**{service.title()}**: {info.get('status', 'Unknown')}")

    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")

    # Navigation
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🏠 Home", "📊 Analysis Dashboard", "🤖 AI Insights", "📋 History", "⚙️ Settings"]
    )

    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Analysis Dashboard":
        show_analysis_dashboard()
    elif page == "🤖 AI Insights":
        show_ai_insights()
    elif page == "📋 History":
        show_history()
    elif page == "⚙️ Settings":
        show_settings()


def show_home_page():
    """Main prescription analysis page"""

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Patient Information")

    # Patient Information
    patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45, help="Patient's age in years")
    patient_weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0, step=0.1,
                                             help="Patient's weight in kilograms")

    # Medical conditions
    conditions = st.sidebar.text_area(
        "Medical Conditions",
        placeholder="Enter conditions separated by commas (e.g., diabetes, hypertension, COPD)",
        help="List patient's current medical conditions"
    )
    conditions_list = [c.strip() for c in conditions.split(",") if c.strip()] if conditions else []

    # Allergies
    allergies = st.sidebar.text_area(
        "Known Allergies",
        placeholder="Enter allergies separated by commas (e.g., penicillin, aspirin, sulfa)",
        help="List patient's known drug allergies"
    )
    allergies_list = [a.strip() for a in allergies.split(",") if a.strip()] if allergies else []

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Analysis Options")

    use_ai_analysis = st.sidebar.checkbox("🤖 Enable Advanced AI Analysis", value=True,
                                          help="Use HuggingFace & IBM Watson AI for enhanced analysis")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 💊 Prescription Entry")

        # Method selection
        entry_method = st.radio(
            "Choose prescription entry method:",
            ["📝 Manual Entry", "📄 Text Analysis (NLP)", "💊 Quick Drug Search"]
        )

        if entry_method == "📝 Manual Entry":
            show_manual_entry()
        elif entry_method == "📄 Text Analysis (NLP)":
            show_text_analysis()
        elif entry_method == "💊 Quick Drug Search":
            show_quick_search()

    with col2:
        st.markdown("## 📋 Current Prescription")

        if st.session_state.patient_drugs:
            for i, drug in enumerate(st.session_state.patient_drugs):
                with st.container():
                    st.markdown(f"""
                    <div class="drug-card">
                        <strong>{drug['name']}</strong><br>
                        📊 Dosage: {drug.get('dosage', 'Not specified')}<br>
                        ⏰ Frequency: {drug.get('frequency', 'Not specified')}
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"❌ Remove {drug['name']}", key=f"remove_{i}"):
                        st.session_state.patient_drugs.pop(i)
                        st.rerun()

            st.markdown("---")

            # Analysis button
            if st.button("🔍 **ANALYZE PRESCRIPTION**", type="primary", use_container_width=True):
                perform_analysis(patient_age, patient_weight, conditions_list, allergies_list, use_ai_analysis)
        else:
            st.info("📝 No drugs added yet. Add medications using the form on the left.")

    # Results display
    if st.session_state.analysis_results:
        st.markdown("---")
        display_analysis_results()


def show_manual_entry():
    """Manual drug entry form"""
    with st.form("drug_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            drug_name = st.text_input("💊 Drug Name", placeholder="e.g., Aspirin")
            dosage = st.text_input("📊 Dosage", placeholder="e.g., 100mg")

        with col2:
            frequency = st.text_input("⏰ Frequency", placeholder="e.g., Once daily")

        submitted = st.form_submit_button("➕ Add Drug", use_container_width=True)

        if submitted and drug_name:
            drug_info = {
                "name": drug_name.strip(),
                "dosage": dosage.strip() if dosage else None,
                "frequency": frequency.strip() if frequency else None
            }
            st.session_state.patient_drugs.append(drug_info)
            st.success(f"✅ Added {drug_name} to prescription")
            st.rerun()


def show_text_analysis():
    """Text-based prescription analysis"""
    st.markdown("### 📄 Prescription Text Analysis")
    st.info(
        "🤖 **AI-Powered**: Our advanced NLP system will extract drug information from medical text using HuggingFace models.")

    medical_text = st.text_area(
        "Enter prescription or medical text:",
        placeholder="""Example:
Patient prescribed:
- Aspirin 100mg once daily for cardioprotection
- Metformin 500mg twice daily with meals for diabetes
- Lisinopril 10mg once daily for hypertension
- Warfarin 5mg daily (target INR 2.0-3.0)""",
        height=200
    )

    if st.button("🔍 Extract Drugs with AI", type="primary"):
        if medical_text.strip():
            extract_drugs_from_text(medical_text.strip())
        else:
            st.warning("Please enter some medical text first.")


def show_quick_search():
    """Quick drug search and add"""
    st.markdown("### 🔍 Quick Drug Search")

    # Common drugs list
    common_drugs = [
        "Aspirin", "Paracetamol", "Metformin", "Warfarin", "Digoxin",
        "Furosemide", "Lisinopril", "Spironolactone", "Simvastatin",
        "Clopidogrel", "Rivaroxaban", "Apixaban", "Atorvastatin",
        "Amlodipine", "Losartan", "Omeprazole", "Ibuprofen"
    ]

    selected_drug = st.selectbox("Select from common drugs:", [""] + common_drugs)

    col1, col2 = st.columns(2)
    with col1:
        custom_drug = st.text_input("Or enter custom drug name:")

    with col2:
        if selected_drug:
            drug_name = selected_drug
        elif custom_drug:
            drug_name = custom_drug
        else:
            drug_name = ""

    if drug_name:
        col1, col2 = st.columns(2)
        with col1:
            dosage = st.text_input("Dosage:", key="quick_dosage")
        with col2:
            frequency = st.text_input("Frequency:", key="quick_frequency")

        if st.button("➕ Add to Prescription", type="primary"):
            drug_info = {
                "name": drug_name,
                "dosage": dosage if dosage else None,
                "frequency": frequency if frequency else None
            }
            st.session_state.patient_drugs.append(drug_info)
            st.success(f"✅ Added {drug_name}")
            st.rerun()


def extract_drugs_from_text(medical_text: str):
    """Extract drugs from medical text using AI"""
    with st.spinner("🤖 AI is analyzing the medical text..."):
        # Prepare request data
        request_data = {
            "drugs": [],
            "patient": {
                "age": 45,  # Default values for extraction
                "conditions": [],
                "allergies": []
            },
            "medical_text": medical_text,
            "use_ai_analysis": True
        }

        # Make API request
        result = make_api_request("/analyze", method="POST", data=request_data)

        if result["success"]:
            extracted_drugs = result["data"].get("extracted_drugs", [])

            if extracted_drugs:
                st.success(f"🎉 AI extracted {len(extracted_drugs)} drugs from the text!")

                # Display extracted drugs for review
                st.markdown("### 📋 Review Extracted Drugs:")
                for i, drug in enumerate(extracted_drugs):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.write(f"**{drug['name']}**")
                            if drug.get('ai_confidence'):
                                confidence = drug['ai_confidence'] * 100
                                st.progress(confidence / 100)
                                st.caption(f"AI Confidence: {confidence:.1f}%")

                        with col2:
                            st.write(f"📊 {drug.get('dosage', 'N/A')}")

                        with col3:
                            if st.button(f"➕ Add", key=f"add_extracted_{i}"):
                                st.session_state.patient_drugs.append(drug)
                                st.success(f"Added {drug['name']}")
                                st.rerun()
            else:
                st.warning("🤔 No drugs were extracted from the text. Try adding them manually.")
        else:
            st.error(result["error"])


def perform_analysis(age: int, weight: float, conditions: List[str], allergies: List[str], use_ai: bool):
    """Perform comprehensive prescription analysis"""

    if not st.session_state.patient_drugs:
        st.warning("Please add at least one drug before analyzing.")
        return

    # Prepare request data
    request_data = {
        "drugs": st.session_state.patient_drugs,
        "patient": {
            "age": age,
            "weight": weight,
            "conditions": conditions,
            "allergies": allergies
        },
        "use_ai_analysis": use_ai
    }

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("🔬 Performing comprehensive AI analysis..."):
        status_text.text("🤖 Initializing AI systems...")
        progress_bar.progress(20)
        time.sleep(0.5)

        status_text.text("🧠 Running HuggingFace NLP analysis...")
        progress_bar.progress(40)
        time.sleep(0.5)

        status_text.text("💻 Executing IBM Watson AI analysis...")
        progress_bar.progress(60)
        time.sleep(0.5)

        status_text.text("🔍 Analyzing drug interactions...")
        progress_bar.progress(80)

        # Make API request
        result = make_api_request("/analyze", method="POST", data=request_data)

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()

    if result["success"]:
        st.session_state.analysis_results = result["data"]

        # Add to history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_age": age,
            "drugs_count": len(st.session_state.patient_drugs),
            "safety_score": result["data"].get("safety_score", 0),
            "ai_enabled": use_ai
        }
        st.session_state.analysis_history.append(history_entry)

        st.success("🎉 Analysis completed successfully!")
    else:
        st.error(result["error"])


def display_analysis_results():
    """Display comprehensive analysis results"""
    results = st.session_state.analysis_results
    if not results:
        return

    st.markdown("# 📊 Analysis Results")

    # Safety Score
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        safety_score = results.get("safety_score", 0)
        fig = create_safety_gauge(safety_score)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Safety Score", f"{safety_score:.1f}/100", delta=None)

        if safety_score >= 80:
            st.success("🟢 Low Risk")
        elif safety_score >= 60:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

    with col3:
        interactions = results.get("interactions", [])
        st.metric("Interactions Found", len(interactions))

        allergy_warnings = results.get("allergy_warnings", [])
        st.metric("Allergy Warnings", len(allergy_warnings))

    # Tabs for different result sections
    tabs = st.tabs(["🚨 Interactions", "💊 Dosage", "⚠️ Allergies", "🔄 Alternatives", "🤖 AI Insights"])

    with tabs[0]:
        display_interactions(interactions)

    with tabs[1]:
        display_dosage_recommendations(results.get("dosage_recommendations", []))

    with tabs[2]:
        display_allergy_warnings(allergy_warnings)

    with tabs[3]:
        display_alternatives(results.get("alternatives", []))

    with tabs[4]:
        display_ai_insights(results)


def display_interactions(interactions: List[Dict]):
    """Display drug interactions"""
    if not interactions:
        st.success("✅ No drug interactions detected!")
        return

    st.warning(f"⚠️ Found {len(interactions)} drug interactions")

    # Chart
    fig = create_interaction_chart(interactions)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Detailed interactions
    for interaction in interactions:
        severity = interaction.get("severity", "Unknown")
        css_class = f"interaction-{severity.lower()}"

        st.markdown(f"""
        <div class="drug-card {css_class}">
            <h4>⚠️ {interaction.get('drug1')} ↔️ {interaction.get('drug2')}</h4>
            <p><strong>Severity:</strong> <span style="color: {get_severity_color(severity)}">{severity}</span></p>
            <p><strong>Description:</strong> {interaction.get('description', 'N/A')}</p>
            <p><strong>Recommendation:</strong> {interaction.get('recommendation', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_dosage_recommendations(recommendations: List[Dict]):
    """Display dosage recommendations"""
    if not recommendations:
        st.success("✅ No dosage adjustments needed based on current information")
        return

    st.info(f"📊 {len(recommendations)} dosage recommendations found")

    for rec in recommendations:
        st.markdown(f"""
        <div class="drug-card">
            <h4>💊 {rec.get('drug_name')}</h4>
            <p><strong>Current Dosage:</strong> {rec.get('current_dosage')}</p>
            <p><strong>Recommended Dosage:</strong> {rec.get('recommended_dosage')}</p>
            <p><strong>Notes:</strong> {rec.get('notes', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_allergy_warnings(warnings: List[Dict]):
    """Display allergy warnings"""
    if not warnings:
        st.success("✅ No allergy contraindications detected")
        return

    st.error(f"🚨 CRITICAL: {len(warnings)} allergy warnings!")

    for warning in warnings:
        st.markdown(f"""
        <div class="alert-high">
            <h4>🚨 {warning.get('drug_name')} - CONTRAINDICATED</h4>
            <p><strong>Allergen:</strong> {warning.get('allergen')}</p>
            <p><strong>Severity:</strong> {warning.get('severity')}</p>
            <p><strong>Recommendation:</strong> {warning.get('recommendation')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_alternatives(alternatives: List[Dict]):
    """Display alternative medications"""
    if not alternatives:
        st.success("✅ No alternative medications needed")
        return

    st.info(f"🔄 {len(alternatives)} alternative medications suggested")

    for alt in alternatives:
        st.markdown(f"""
        <div class="success-box">
            <h4>🔄 Alternative for {alt.get('original_drug')}</h4>
            <p><strong>Suggested Alternative:</strong> {alt.get('alternative')}</p>
            <p><strong>Dosage:</strong> {alt.get('dosage')}</p>
            <p><strong>Reason:</strong> {alt.get('reason')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_ai_insights(results: Dict):
    """Display AI insights and analysis"""

    # Traditional AI Insights
    ai_insights = results.get("ai_insights", [])
    if ai_insights:
        st.markdown("### 🧠 Traditional AI Insights")
        for insight in ai_insights:
            st.markdown(f"""
            <div class="ai-insight">
                {insight}
            </div>
            """, unsafe_allow_html=True)

    # Enhanced AI Insights
    enhanced_insights = results.get("enhanced_ai_insights", [])
    if enhanced_insights:
        st.markdown("### 🚀 Enhanced AI Insights")
        for insight in enhanced_insights:
            confidence_pct = insight.get('confidence', 0) * 100
            st.markdown(f"""
            <div class="ai-insight">
                <h4>🤖 {insight.get('source', 'AI System')}</h4>
                <p>{insight.get('insight', 'No insight available')}</p>
                <p><strong>Type:</strong> {insight.get('type', 'general').replace('_', ' ').title()}</p>
                <p><strong>Confidence:</strong> {confidence_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # HuggingFace Analysis
    hf_analysis = results.get("huggingface_analysis", {})
    if hf_analysis:
        st.markdown("### 🤗 HuggingFace AI Analysis")

        with st.expander("View HuggingFace Results", expanded=False):
            st.json(hf_analysis)

    # IBM Watson Analysis
    ibm_analysis = results.get("ibm_analysis", {})
    if ibm_analysis:
        st.markdown("### 🔵 IBM Watson AI Analysis")

        with st.expander("View IBM Watson Results", expanded=False):
            st.json(ibm_analysis)

    # Warnings
    warnings = results.get("warnings", [])
    if warnings:
        st.markdown("### ⚠️ System Warnings")
        for warning in warnings:
            st.warning(warning)


def show_analysis_dashboard():
    """Analysis dashboard with visualizations"""
    st.markdown("# 📊 Analysis Dashboard")

    if not st.session_state.analysis_results:
        st.info("🔍 No analysis results available. Please run an analysis first.")
        return

    results = st.session_state.analysis_results

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        safety_score = results.get("safety_score", 0)
        st.metric("Safety Score", f"{safety_score:.1f}", delta=None)

    with col2:
        interactions = len(results.get("interactions", []))
        st.metric("Interactions", interactions)

    with col3:
        allergy_warnings = len(results.get("allergy_warnings", []))
        st.metric("Allergy Warnings", allergy_warnings)

    with col4:
        alternatives = len(results.get("alternatives", []))
        st.metric("Alternatives", alternatives)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Safety gauge
        fig = create_safety_gauge(safety_score)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Interactions chart
        interactions_data = results.get("interactions", [])
        fig = create_interaction_chart(interactions_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No interactions to display")

    # Drug analysis table
    st.markdown("### 💊 Drug Analysis Summary")

    if st.session_state.patient_drugs:
        drug_data = []
        for drug in st.session_state.patient_drugs:
            # Check if this drug has interactions
            drug_interactions = 0
            for interaction in interactions_data:
                if drug['name'].lower() in [interaction.get('drug1', '').lower(), interaction.get('drug2', '').lower()]:
                    drug_interactions += 1

            # Check allergies
            has_allergy = any(warning.get('drug_name', '').lower() == drug['name'].lower()
                              for warning in results.get("allergy_warnings", []))

            drug_data.append({
                "Drug Name": drug['name'],
                "Dosage": drug.get('dosage', 'Not specified'),
                "Frequency": drug.get('frequency', 'Not specified'),
                "Interactions": drug_interactions,
                "Allergy Risk": "⚠️ YES" if has_allergy else "✅ No",
                "Status": "🔴 High Risk" if has_allergy or drug_interactions > 0 else "🟢 Safe"
            })

        df = pd.DataFrame(drug_data)
        st.dataframe(df, use_container_width=True)


def show_ai_insights():
    """Dedicated AI insights page"""
    st.markdown("# 🤖 AI Insights Center")

    if not st.session_state.analysis_results:
        st.info("🔍 No AI analysis results available. Please run an analysis first.")
        return

    results = st.session_state.analysis_results

    # AI Service Status
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤗 HuggingFace AI")
        hf_analysis = results.get("huggingface_analysis", {})
        if hf_analysis:
            st.success("✅ Analysis Complete")

            # Drug interactions from HF
            drug_interactions = hf_analysis.get('drug_interactions', [])
            if drug_interactions:
                st.write(f"🔍 Found {len(drug_interactions)} interaction analyses")

            # Medical insights
            medical_insights = hf_analysis.get('medical_insights', {})
            if medical_insights and medical_insights.get('status') == 'success':
                st.success("🧠 Medical insights generated")

        else:
            st.warning("⚠️ No HuggingFace analysis available")

    with col2:
        st.markdown("### 🔵 IBM Watson AI")
        ibm_analysis = results.get("ibm_analysis", {})
        if ibm_analysis:
            st.success("✅ Analysis Complete")

            # Safety analysis
            safety_analysis = ibm_analysis.get('safety_analysis', {})
            if safety_analysis and safety_analysis.get('status') == 'success':
                st.success("🛡️ Safety analysis complete")

            # Drug recommendations
            drug_recs = ibm_analysis.get('drug_recommendations', {})
            if drug_recs and drug_recs.get('status') == 'success':
                st.success("💊 Drug recommendations generated")
        else:
            st.warning("⚠️ No IBM Watson analysis available")

    # Enhanced AI Insights
    enhanced_insights = results.get("enhanced_ai_insights", [])
    if enhanced_insights:
        st.markdown("### 🚀 Enhanced AI Insights")

        for insight in enhanced_insights:
            confidence_pct = insight.get('confidence', 0) * 100

            # Color code by confidence
            if confidence_pct >= 80:
                color = "#28a745"
            elif confidence_pct >= 60:
                color = "#ffc107"
            else:
                color = "#dc3545"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                        border-left: 4px solid {color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: {color};">🤖 {insight.get('source', 'AI System')}</h4>
                <p><strong>Insight:</strong> {insight.get('insight', 'No insight available')}</p>
                <p><strong>Category:</strong> {insight.get('type', 'general').replace('_', ' ').title()}</p>
                <div style="background-color: #f8f9fa; border-radius: 4px; padding: 0.5rem;">
                    <strong>Confidence Score:</strong> {confidence_pct:.1f}%
                    <div style="background-color: {color}; height: 10px; width: {confidence_pct}%; 
                               border-radius: 5px; margin-top: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Raw AI Data Sections
    st.markdown("### 📊 Detailed AI Analysis")

    tab1, tab2 = st.tabs(["🤗 HuggingFace Data", "🔵 IBM Watson Data"])

    with tab1:
        hf_data = results.get("huggingface_analysis", {})
        if hf_data:
            # Medical insights
            medical_insights = hf_data.get('medical_insights', {})
            if medical_insights:
                st.markdown("#### 🧠 Medical Text Analysis")
                if medical_insights.get('status') == 'success':
                    st.success("Analysis completed successfully")
                    insights_data = medical_insights.get('insights')
                    if insights_data:
                        st.json(insights_data)
                else:
                    st.error("Medical insights analysis failed")

            # Drug interactions
            drug_interactions = hf_data.get('drug_interactions', [])
            if drug_interactions:
                st.markdown("#### 🔄 Drug Interaction Analysis")
                for i, interaction in enumerate(drug_interactions):
                    with st.expander(f"Interaction {i + 1}: {interaction.get('drug1')} ↔️ {interaction.get('drug2')}"):
                        st.json(interaction)
        else:
            st.info("No HuggingFace analysis data available")

    with tab2:
        ibm_data = results.get("ibm_analysis", {})
        if ibm_data:
            # Safety analysis
            safety_data = ibm_data.get('safety_analysis', {})
            if safety_data:
                st.markdown("#### 🛡️ Prescription Safety Analysis")
                if safety_data.get('status') == 'success':
                    st.success("Safety analysis completed")
                    analysis_text = safety_data.get('analysis')
                    if analysis_text:
                        st.markdown("**AI-Generated Safety Analysis:**")
                        st.write(analysis_text)
                else:
                    st.error("Safety analysis failed")

            # Drug recommendations
            rec_data = ibm_data.get('drug_recommendations', {})
            if rec_data:
                st.markdown("#### 💊 AI Drug Recommendations")
                if rec_data.get('status') == 'success':
                    st.success("Recommendations generated")
                    recommendations_text = rec_data.get('recommendations')
                    if recommendations_text:
                        st.markdown("**AI-Generated Recommendations:**")
                        st.write(recommendations_text)
                else:
                    st.error("Drug recommendations failed")
        else:
            st.info("No IBM Watson analysis data available")

    # AI Insights Request Section
    st.markdown("---")
    st.markdown("### 🔍 Request Additional AI Insights")

    col1, col2 = st.columns(2)

    with col1:
        additional_text = st.text_area(
            "Additional medical text for analysis:",
            placeholder="Enter additional clinical notes, symptoms, or medical history...",
            height=100
        )

    with col2:
        st.markdown("**Patient Summary:**")
        if st.session_state.patient_drugs:
            st.write(f"📊 Drugs: {len(st.session_state.patient_drugs)}")
            for drug in st.session_state.patient_drugs[:3]:  # Show first 3
                st.write(f"• {drug['name']}")
            if len(st.session_state.patient_drugs) > 3:
                st.write(f"... and {len(st.session_state.patient_drugs) - 3} more")

    if st.button("🤖 Get Additional AI Insights", type="primary"):
        if additional_text.strip():
            get_additional_insights(additional_text.strip())
        else:
            st.warning("Please enter some additional medical text.")


def get_additional_insights(medical_text: str):
    """Get additional AI insights"""

    # Prepare patient data
    patient_data = {
        "age": 45,  # Default or get from session
        "medications": [drug['name'] for drug in st.session_state.patient_drugs],
        "conditions": [],  # Could get from session state
        "allergies": []  # Could get from session state
    }

    request_data = {
        "medical_text": medical_text,
        "patient_data": patient_data
    }

    with st.spinner("🤖 Requesting additional AI insights..."):
        result = make_api_request("/ai-insights", method="POST", data=request_data)

        if result["success"]:
            insights_data = result["data"]["insights"]

            st.success("✅ Additional AI insights generated!")

            # Display HuggingFace insights
            if "huggingface" in insights_data:
                hf_data = insights_data["huggingface"]

                st.markdown("#### 🤗 HuggingFace Analysis")

                # Entities
                entities = hf_data.get("entities", [])
                if entities:
                    st.write(f"🔍 Extracted {len(entities)} medical entities:")
                    entity_df = pd.DataFrame(entities)
                    st.dataframe(entity_df)

                # Medical insights
                medical_insights = hf_data.get("medical_insights", {})
                if medical_insights.get("status") == "success":
                    st.success("🧠 Medical insights generated successfully")

            # Display IBM Watson insights
            if "ibm_watson" in insights_data:
                ibm_data = insights_data["ibm_watson"]

                st.markdown("#### 🔵 IBM Watson Analysis")

                # Safety analysis
                safety_analysis = ibm_data.get("safety_analysis", {})
                if safety_analysis.get("status") == "success":
                    analysis_text = safety_analysis.get("analysis")
                    if analysis_text:
                        st.markdown("**Safety Analysis:**")
                        st.write(analysis_text)

                # Recommendations
                recommendations = ibm_data.get("recommendations", {})
                if recommendations.get("status") == "success":
                    rec_text = recommendations.get("recommendations")
                    if rec_text:
                        st.markdown("**Recommendations:**")
                        st.write(rec_text)
        else:
            st.error(result["error"])


def show_history():
    """Analysis history page"""
    st.markdown("# 📋 Analysis History")

    if not st.session_state.analysis_history:
        st.info("📊 No analysis history available yet. Complete some analyses to see them here.")
        return

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyses", len(st.session_state.analysis_history))

    with col2:
        avg_safety = sum(h.get("safety_score", 0) for h in st.session_state.analysis_history) / len(
            st.session_state.analysis_history)
        st.metric("Avg Safety Score", f"{avg_safety:.1f}")

    with col3:
        ai_analyses = sum(1 for h in st.session_state.analysis_history if h.get("ai_enabled", False))
        st.metric("AI Analyses", ai_analyses)

    with col4:
        total_drugs = sum(h.get("drugs_count", 0) for h in st.session_state.analysis_history)
        st.metric("Total Drugs Analyzed", total_drugs)

    # History table
    st.markdown("### 📊 Analysis History")

    history_data = []
    for i, entry in enumerate(reversed(st.session_state.analysis_history)):  # Most recent first
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        history_data.append({
            "Analysis #": len(st.session_state.analysis_history) - i,
            "Timestamp": timestamp,
            "Patient Age": entry.get("patient_age", "N/A"),
            "Drugs Count": entry.get("drugs_count", 0),
            "Safety Score": f"{entry.get('safety_score', 0):.1f}",
            "AI Enabled": "🤖 Yes" if entry.get("ai_enabled", False) else "❌ No",
            "Risk Level": "🟢 Low" if entry.get("safety_score", 0) >= 80 else
            "🟡 Moderate" if entry.get("safety_score", 0) >= 60 else "🔴 High"
        })

    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)

    # History chart
    if len(st.session_state.analysis_history) > 1:
        st.markdown("### 📈 Safety Score Trend")

        # Prepare data for chart
        timestamps = [datetime.fromisoformat(h["timestamp"]) for h in st.session_state.analysis_history]
        safety_scores = [h.get("safety_score", 0) for h in st.session_state.analysis_history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=safety_scores,
            mode='lines+markers',
            name='Safety Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))

        # Add threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Safe Threshold")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")

        fig.update_layout(
            title="Safety Score Over Time",
            xaxis_title="Analysis Date",
            yaxis_title="Safety Score",
            height=400,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig, use_container_width=True)

    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        if st.checkbox("I confirm I want to clear all history"):
            st.session_state.analysis_history = []
            st.success("✅ History cleared successfully!")
            st.rerun()


def show_settings():
    """Settings and configuration page"""
    st.markdown("# ⚙️ Settings & Configuration")

    # API Configuration
    st.markdown("### 🔌 API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        current_url = st.text_input("Backend API URL", value=API_BASE_URL, help="URL of the FastAPI backend")

        if st.button("🧪 Test Connection"):
            test_result = make_api_request("/", method="GET")
            if test_result["success"]:
                st.success("✅ Connection successful!")
                api_info = test_result["data"]
                st.json(api_info)
            else:
                st.error(test_result["error"])

    with col2:
        st.markdown("**Current System Status:**")

        # Test API connection
        status = make_api_request("/health")
        if status["success"]:
            health_data = status["data"]
            st.success(f"✅ System: {health_data.get('status', 'Unknown')}")
            st.info(f"📊 Version: {health_data.get('version', 'Unknown')}")

            # AI Services status
            ai_services = health_data.get("ai_services", {})
            for service, info in ai_services.items():
                service_status = info.get("status", "unknown")
                if service_status == "connected":
                    st.success(f"✅ {service.title()}: Connected")
                else:
                    st.warning(f"⚠️ {service.title()}: {service_status}")
        else:
            st.error("❌ Cannot connect to backend")

    st.markdown("---")

    # Display Settings
    st.markdown("### 🎨 Display Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Show detailed AI insights", value=True, help="Display detailed AI analysis results")
        st.checkbox("Enable animation effects", value=True, help="Enable animated progress bars and transitions")
        st.selectbox("Theme preference", ["Auto", "Light", "Dark"], help="Choose display theme")

    with col2:
        st.slider("Results per page", 5, 50, 20, help="Number of results to show per page")
        st.checkbox("Auto-refresh results", value=False, help="Automatically refresh analysis results")

    st.markdown("---")

    # Advanced Settings
    st.markdown("### 🔬 Advanced Settings")

    with st.expander("AI Model Configuration", expanded=False):
        st.markdown("**HuggingFace Models:**")
        st.text("• Biomedical NER: d4data/biomedical-ner-all")
        st.text("• Clinical BERT: emilyalsentzer/Bio_ClinicalBERT")
        st.text("• Drug Interaction: cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

        st.markdown("**IBM Watson Models:**")
        st.text("• LLaMA2: meta-llama/llama-2-70b-chat")
        st.text("• Granite: ibm/granite-13b-chat-v2")
        st.text("• Flan-T5: google/flan-t5-xxl")

        st.info("💡 Model configuration is managed by the backend API")

    with st.expander("Data Management", expanded=False):
        st.markdown("**Session Data:**")
        st.write(f"• Current drugs: {len(st.session_state.patient_drugs)}")
        st.write(f"• Analysis history: {len(st.session_state.analysis_history)}")
        st.write(f"• Has current results: {'Yes' if st.session_state.analysis_results else 'No'}")

        if st.button("🗑️ Clear All Session Data"):
            if st.checkbox("Confirm data clearing"):
                st.session_state.patient_drugs = []
                st.session_state.analysis_results = None
                st.session_state.analysis_history = []
                st.success("✅ All session data cleared!")
                st.rerun()

    st.markdown("---")

    # System Information
    st.markdown("### ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Frontend Information:**")
        st.write("• Framework: Streamlit")
        st.write("• Version: 3.0.0")
        st.write("• Features: AI Integration, NLP, Real-time Analysis")

    with col2:
        st.markdown("**Backend Information:**")
        backend_status = make_api_request("/")
        if backend_status["success"]:
            backend_info = backend_status["data"]
            st.write(f"• Version: {backend_info.get('version', 'Unknown')}")
            st.write(f"• Status: {backend_info.get('status', 'Unknown')}")
            st.write(f"• AI Services: {len(backend_info.get('ai_services', {}))}")

    # Documentation
    st.markdown("---")
    st.markdown("### 📚 Documentation & Help")

    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. 👤 Enter patient information in the sidebar
        2. 💊 Add medications using manual entry or text analysis
        3. 🔍 Click "Analyze Prescription" to run AI analysis
        4. 📊 Review results in the detailed analysis sections
        5. 🤖 Check AI insights for additional recommendations

        **AI Features:**
        • 🤗 HuggingFace NLP for drug extraction
        • 🔵 IBM Watson for safety analysis
        • 🧠 Advanced drug interaction detection
        • 📊 Comprehensive risk scoring
        """)

    with st.expander("❓ Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**

        **❌ Cannot connect to backend:**
        • Ensure FastAPI server is running on localhost:8000
        • Check if the backend URL in settings is correct
        • Verify no firewall is blocking the connection

        **⏱️ Analysis taking too long:**
        • AI analysis can take 30-60 seconds
        • Check your internet connection for API calls
        • Try disabling AI analysis for faster results

        **🤖 AI features not working:**
        • Verify API keys are configured in the backend
        • Check AI service status in settings
        • Review backend logs for error messages
        """)


# ==================== UTILITY FUNCTIONS (CONTINUED) ====================

def create_risk_analysis_chart(results: Dict) -> go.Figure:
    """Create comprehensive risk analysis chart"""

    # Risk categories
    categories = ['Interactions', 'Allergies', 'Age Factors', 'Polypharmacy', 'Overall Safety']

    # Calculate risk scores
    interaction_risk = len(results.get('interactions', [])) * 10
    allergy_risk = len(results.get('allergy_warnings', [])) * 25
    age_risk = 20 if results.get('patient_age', 45) > 75 else 5
    poly_risk = max(0, (len(results.get('drugs', [])) - 3) * 5)
    overall_risk = 100 - results.get('safety_score', 100)

    risk_scores = [interaction_risk, allergy_risk, age_risk, poly_risk, overall_risk]

    fig = go.Figure(data=go.Scatterpolar(
        r=risk_scores,
        theta=categories,
        fill='toself',
        name='Risk Assessment',
        line_color='rgb(255, 99, 71)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="Comprehensive Risk Analysis",
        height=500
    )

    return fig


# ==================== MAIN APP EXECUTION ====================

if __name__ == "__main__":
    main()