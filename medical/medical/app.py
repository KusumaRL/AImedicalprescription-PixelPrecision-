# ==================== ENHANCED STREAMLIT FRONTEND (app.py) ====================

import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="AI Medical Prescription Verification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #667eea;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.warning-high { 
    border-left-color: #ff4757 !important; 
    background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
}

.warning-moderate { 
    border-left-color: #ffa502 !important; 
    background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
}

.warning-low { 
    border-left-color: #2ed573 !important; 
    background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
}

.safety-score-high { color: #2ed573; font-weight: bold; }
.safety-score-medium { color: #ffa502; font-weight: bold; }
.safety-score-low { color: #ff4757; font-weight: bold; }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    transform: translateY(-1px);
}

.error-message {
    background: #fff5f5;
    border-left: 4px solid #ff4757;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}

.success-message {
    background: #f0fff4;
    border-left: 4px solid #2ed573;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}

.info-message {
    background: #f8f9fa;
    border-left: 4px solid #3742fa;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000"
MAX_RETRIES = 3
RETRY_DELAY = 1


def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def extract_drugs_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract drug information from medical text using regex patterns
    Returns a list of drug dictionaries with name, dosage, and frequency
    """
    if not text or not text.strip():
        return []

    drugs = []

    # Common drug name patterns
    drug_patterns = [
        r'\b(aspirin|warfarin|metformin|lisinopril|furosemide|digoxin|simvastatin|atorvastatin|amlodipine|omeprazole)\b',
        r'\b([A-Z][a-z]+(?:mab|pril|sartan|olol|pine|statin|mycin|cillin))\b',  # Common drug suffixes
        r'\b([A-Z][a-z]{3,})\s+\d+(?:\.\d+)?\s*mg\b'  # Pattern: DrugName XXmg
    ]

    # Dosage patterns
    dosage_pattern = r'(\d+(?:\.\d+)?)\s*mg'

    # Frequency patterns
    frequency_patterns = [
        r'once\s+daily|daily|od',
        r'twice\s+daily|bid|bd',
        r'three\s+times\s+daily|tid|tds',
        r'four\s+times\s+daily|qid',
        r'every\s+\d+\s+hours?',
        r'as\s+needed|prn'
    ]

    # Find all potential drug mentions
    for pattern in drug_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            drug_name = match.group(1) if match.group(1) else match.group(0)

            # Look for dosage near the drug name
            drug_context = text[max(0, match.start() - 50):match.end() + 50]
            dosage_match = re.search(dosage_pattern, drug_context)
            dosage = dosage_match.group(1) + "mg" if dosage_match else "Not specified"

            # Look for frequency
            frequency = "Not specified"
            for freq_pattern in frequency_patterns:
                if re.search(freq_pattern, drug_context, re.IGNORECASE):
                    frequency_match = re.search(freq_pattern, drug_context, re.IGNORECASE)
                    frequency = frequency_match.group(0)
                    break

            # Avoid duplicates
            if not any(d['name'].lower() == drug_name.lower() for d in drugs):
                drugs.append({
                    'name': drug_name.title(),
                    'dosage': dosage,
                    'frequency': frequency,
                    'route': 'oral'
                })

    return drugs


def validate_drug_input(drugs: List[Dict[str, Any]]) -> bool:
    """Validate that we have at least one drug with required fields"""
    if not drugs or len(drugs) == 0:
        return False

    for drug in drugs:
        if not drug.get('name') or not drug.get('name').strip():
            return False

    return True


def handle_demo_mode(demo_mode, age, weight, conditions, allergies):
    """Handle demo mode execution with error handling"""
    try:
        if demo_mode == "high_risk":
            demo_drugs = [
                {"name": "warfarin", "dosage": "5mg", "frequency": "once daily"},
                {"name": "aspirin", "dosage": "325mg", "frequency": "once daily"}
            ]
            demo_age = 75
            demo_allergies = ["penicillin"]
            demo_text = "Patient on warfarin 5mg daily and aspirin 325mg daily for atrial fibrillation"
            analyze_prescription_safe(demo_drugs, demo_age, 70, ["atrial fibrillation"], demo_allergies, demo_text)

        elif demo_mode == "safe":
            demo_drugs = [
                {"name": "metformin", "dosage": "500mg", "frequency": "twice daily"},
                {"name": "lisinopril", "dosage": "10mg", "frequency": "once daily"}
            ]
            demo_age = 45
            analyze_prescription_safe(demo_drugs, demo_age, 80, ["diabetes", "hypertension"], [], "")

        elif demo_mode == "elderly":
            demo_drugs = [
                {"name": "digoxin", "dosage": "250mcg", "frequency": "once daily"},
                {"name": "furosemide", "dosage": "40mg", "frequency": "once daily"},
                {"name": "simvastatin", "dosage": "80mg", "frequency": "evening"}
            ]
            demo_age = 85
            demo_text = "Elderly patient on digoxin 250mcg daily, furosemide 40mg daily, and simvastatin 80mg evening"
            analyze_prescription_safe(demo_drugs, demo_age, 65, ["heart failure", "hyperlipidemia"], [], demo_text)

    except Exception as e:
        st.error(f"Demo failed: {str(e)}")


def handle_manual_input():
    """Handle manual drug input with improved UX"""
    st.subheader("📝 Manual Drug Entry")

    manual_drugs = []

    # Initialize session state for dynamic drugs
    if 'num_drugs' not in st.session_state:
        st.session_state.num_drugs = 2

    with st.form("drugs_form"):
        st.markdown("*Enter each medication with dosage and frequency:*")

        for i in range(st.session_state.num_drugs):
            with st.expander(f"💊 Drug {i + 1}", expanded=True):
                col_a, col_b, col_c = st.columns([3, 2, 2])

                with col_a:
                    drug_name = st.text_input(
                        f"Drug Name",
                        key=f"drug_name_{i}",
                        placeholder="e.g., aspirin, metformin",
                        help="Enter the generic or brand name"
                    )
                with col_b:
                    dosage = st.text_input(
                        f"Dosage",
                        key=f"dosage_{i}",
                        placeholder="e.g., 75mg, 500mg",
                        help="Include units (mg, g, ml)"
                    )
                with col_c:
                    frequency = st.selectbox(
                        f"Frequency",
                        ["", "once daily", "twice daily", "three times daily", "four times daily",
                         "every 8 hours", "every 12 hours", "as needed", "other"],
                        key=f"freq_{i}",
                        help="How often to take"
                    )

                if drug_name and drug_name.strip():
                    manual_drugs.append({
                        "name": drug_name.strip(),
                        "dosage": dosage.strip() if dosage else None,
                        "frequency": frequency if frequency else None
                    })

        col_x, col_y = st.columns(2)
        with col_x:
            if st.form_submit_button("➕ Add Drug") and st.session_state.num_drugs < 10:
                st.session_state.num_drugs += 1
                st.rerun()

        with col_y:
            if st.form_submit_button("➖ Remove") and st.session_state.num_drugs > 1:
                st.session_state.num_drugs -= 1
                st.rerun()

    return manual_drugs


def handle_text_input():
    """Handle text analysis input with examples and drug extraction"""
    st.subheader("🧠 AI-Powered Text Analysis")

    # Example texts
    example_option = st.selectbox(
        "📋 Load example text:",
        ["Custom Text", "Cardiology Example", "Diabetes Example", "Elderly Care Example"]
    )

    example_texts = {
        "Cardiology Example": "Patient prescribed aspirin 75mg once daily for cardiovascular protection. Also starting warfarin 5mg daily for atrial fibrillation prevention. Consider monitoring INR levels.",
        "Diabetes Example": "Start metformin 500mg twice daily with meals for type 2 diabetes. Add lisinopril 10mg once daily for blood pressure control. Monitor kidney function.",
        "Elderly Care Example": "Elderly patient with heart failure on furosemide 40mg daily and digoxin 125mcg daily. Reduce simvastatin from 80mg to 40mg evening due to age-related muscle toxicity risk."
    }

    default_text = example_texts.get(example_option, "")

    medical_text = st.text_area(
        "Enter prescription or clinical text:",
        value=default_text,
        height=180,
        placeholder="""Example:
Patient prescribed aspirin 75mg once daily for cardiovascular protection.
Also starting metformin 500mg twice daily for diabetes management.
Consider adding lisinopril 10mg daily for blood pressure control.""",
        help="AI will extract drug names, dosages, and frequencies automatically"
    )

    # Extract drugs from the text
    extracted_drugs = []
    if medical_text and medical_text.strip():
        extracted_drugs = extract_drugs_from_text(medical_text)

        if extracted_drugs:
            st.success(f"🎯 Extracted {len(extracted_drugs)} drug(s) from text")

            # Display extracted drugs
            with st.expander("📊 Extracted Drug Information", expanded=True):
                for i, drug in enumerate(extracted_drugs, 1):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{i}. {drug['name']}**")
                    with col2:
                        st.write(f"Dose: {drug['dosage']}")
                    with col3:
                        st.write(f"Freq: {drug['frequency']}")
        else:
            if medical_text != default_text:
                st.warning(
                    "⚠️ No drugs detected in the text. Please ensure drug names and dosages are clearly mentioned.")

    return medical_text


def analyze_prescription_safe(drugs, age, weight, conditions, allergies, medical_text):
    """Safely analyze prescription with comprehensive error handling"""
    if not st.session_state.api_available:
        st.error("❌ Cannot analyze - Backend API is not available")
        return

    with st.spinner("🔄 AI is performing comprehensive analysis..."):
        try:
            # Prepare request payload
            request_data = {
                "drugs": drugs,
                "patient": {
                    "age": int(age),
                    "weight": float(weight),
                    "conditions": conditions,
                    "allergies": allergies
                },
                "medical_text": medical_text if medical_text and medical_text.strip() else None
            }

            # Add retry logic
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.post(
                        f"{API_URL}/analyze",
                        json=request_data,
                        timeout=30,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.analysis_result = result
                        st.success("✅ AI analysis completed successfully!")
                        st.rerun()
                        return
                    elif response.status_code == 422:
                        error_detail = response.json().get("detail", "Validation error")
                        st.error(f"❌ Input validation failed: {error_detail}")
                        return
                    elif response.status_code == 400:
                        error_detail = response.json().get("detail", "Bad request")
                        st.error(f"❌ Request error: {error_detail}")
                        return
                    else:
                        st.error(f"❌ Analysis failed with status {response.status_code}")
                        if response.text:
                            st.error(f"Error details: {response.text}")
                        return

                except requests.exceptions.Timeout:
                    if attempt < MAX_RETRIES - 1:
                        st.warning(f"⏰ Request timeout, retrying... (attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        st.error("⏰ Analysis timeout after multiple attempts")
                        return

                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to AI backend server")
                    st.session_state.api_available = False
                    return

        except Exception as e:
            st.error(f"❌ Unexpected error during analysis: {str(e)}")


def display_enhanced_results(result):
    """Display comprehensive analysis results with enhanced visualization"""
    # Safety score with enhanced styling
    safety_score = result.get("safety_score", 0)
    if safety_score >= 80:
        score_class, score_emoji, score_desc = "safety-score-high", "🟢", "Excellent"
        score_color = "#2ed573"
    elif safety_score >= 60:
        score_class, score_emoji, score_desc = "safety-score-medium", "🟡", "Moderate"
        score_color = "#ffa502"
    else:
        score_class, score_emoji, score_desc = "safety-score-low", "🔴", "High Risk"
        score_color = "#ff4757"

    # Top-level metrics dashboard with plotly gauge
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Create gauge chart for safety score
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=safety_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Safety Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': score_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))

        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        interaction_count = len(result.get("interactions", []))
        high_interactions = sum(1 for i in result.get("interactions", []) if i.get("severity") == "High")
        color = "🔴" if high_interactions > 0 else "🟡" if interaction_count > 0 else "🟢"
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h2>{color} {interaction_count}</h2>
            <p><strong>Interactions</strong><br><small>{high_interactions} High Risk</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        allergy_count = len(result.get("allergy_warnings", []))
        warning_count = len(result.get("warnings", []))
        color = "🚨" if allergy_count > 0 else "⚠️" if warning_count > 0 else "✅"
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h2>{color} {warning_count}</h2>
            <p><strong>Warnings</strong><br><small>{allergy_count} Allergies</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        extracted_count = len(result.get("extracted_drugs", []))
        alternative_count = len(result.get("alternatives", []))
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h2>🧠 {extracted_count}</h2>
            <p><strong>AI Extracted</strong><br><small>{alternative_count} Alternatives</small></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Critical alerts section
    allergy_warnings = result.get("allergy_warnings", [])
    if allergy_warnings:
        st.markdown("### 🚨 CRITICAL ALLERGY ALERTS")
        for warning in allergy_warnings:
            st.error(
                f"**{warning.get('drug_name', 'Unknown')}** - {warning.get('recommendation', 'No recommendation')}")

    # General warnings with improved formatting
    warnings = result.get("warnings", [])
    if warnings:
        st.markdown("### ⚠️ Safety Warnings")
        for i, warning in enumerate(warnings, 1):
            st.warning(f"**{i}.** {warning}")

    # AI insights with better presentation
    ai_insights = result.get("ai_insights", [])
    if ai_insights:
        st.markdown("### 🤖 AI-Powered Clinical Insights")
        for insight in ai_insights:
            st.info(insight)

    # Detailed analysis in enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Drug Interactions",
        "💊 Dosage Analysis",
        "🔄 Alternatives",
        "🧠 AI Extraction",
        "📊 Summary Report"
    ])

    with tab1:
        display_interactions_tab(result.get("interactions", []))

    with tab2:
        display_dosage_tab(result.get("dosage_recommendations", []))

    with tab3:
        display_alternatives_tab(result.get("alternatives", []))

    with tab4:
        display_extraction_tab(result.get("extracted_drugs", []))

    with tab5:
        display_summary_tab(result, safety_score, score_desc)


def display_interactions_tab(interactions):
    """Display drug interactions with enhanced formatting"""
    if interactions:
        st.subheader(f"🔍 {len(interactions)} Drug Interactions Detected")

        # Create visualization of interactions
        if len(interactions) > 1:
            severity_counts = {}
            for interaction in interactions:
                severity = interaction.get("severity", "Unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            fig = px.bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                title="Interactions by Severity",
                color=list(severity_counts.keys()),
                color_discrete_map={"High": "#ff4757", "Moderate": "#ffa502", "Low": "#2ed573"}
            )
            st.plotly_chart(fig, use_container_width=True)

        for i, interaction in enumerate(interactions, 1):
            severity = interaction.get("severity", "Unknown")
            css_class = f"warning-{severity.lower()}"
            severity_emoji = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(severity, "⚪")

            st.markdown(f"""
            <div class="metric-card {css_class}">
                <h4>{severity_emoji} Interaction {i}: {interaction.get('drug1', '')} ↔ {interaction.get('drug2', '')}</h4>
                <p><strong>Severity Level:</strong> {severity}</p>
                <p><strong>Clinical Issue:</strong> {interaction.get('description', 'No description available')}</p>
                <p><strong>Recommended Action:</strong> {interaction.get('recommendation', 'No recommendation available')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No significant drug interactions detected in current prescription")
        st.info("🔍 The AI has analyzed all possible drug combinations and found no concerning interactions")


def display_dosage_tab(dosage_recs):
    """Display dosage recommendations"""
    if dosage_recs:
        st.subheader(f"💊 Age-Appropriate Dosage Analysis ({len(dosage_recs)} drugs)")

        for rec in dosage_recs:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💊 {rec.get('drug_name', 'Unknown Drug')} - Dosage Review</h4>
                <p><strong>Current Prescription:</strong> {rec.get('current_dosage', 'Not specified')}</p>
                <p><strong>AI Recommendation:</strong> {rec.get('recommended_dosage', 'No recommendation')}</p>
                <p><strong>Clinical Notes:</strong> {rec.get('notes', 'No additional notes')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No specific dosage recommendations for current medications")


def display_alternatives_tab(alternatives):
    """Display alternative medications"""
    if alternatives:
        st.subheader(f"🔄 Alternative Medication Suggestions ({len(alternatives)} options)")

        for alt in alternatives:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔄 Alternative for {alt.get('original_drug', 'Unknown')}</h4>
                <p><strong>Suggested Alternative:</strong> {alt.get('alternative', 'Not specified')}</p>
                <p><strong>Clinical Rationale:</strong> {alt.get('reason', 'No reason provided')}</p>
                <p><strong>Recommended Dosage:</strong> {alt.get('dosage', 'Dosage not specified')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No alternative medications needed")
        st.info("Current prescription appears to be well-tolerated with acceptable risk profile")


def display_extraction_tab(extracted):
    """Display AI-extracted drug information"""
    if extracted:
        st.subheader(f"🧠 AI-Extracted Drug Information ({len(extracted)} drugs found)")

        # Create a nice table display
        df_data = []
        for drug in extracted:
            df_data.append({
                "Drug Name": drug.get('name', 'Unknown').title(),
                "Dosage": drug.get('dosage') or '❓ Not specified',
                "Frequency": drug.get('frequency') or '❓ Not specified',
                "Status": "✅ Extracted"
            })

        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

        st.success(f"🎯 AI successfully identified {len(extracted)} medications using advanced NLP processing")
    else:
        st.info("ℹ️ No drugs were extracted from medical text")


def display_summary_tab(result, safety_score, score_desc):
    """Display comprehensive analysis summary"""
    st.subheader("📊 Comprehensive Analysis Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Risk Assessment")
        st.write(f"**Overall Safety Score:** {safety_score:.1f}/100 ({score_desc})")
        st.write(f"**Total Drug Interactions:** {len(result.get('interactions', []))}")
        st.write(
            f"**High-Risk Interactions:** {sum(1 for i in result.get('interactions', []) if i.get('severity') == 'High')}")
        st.write(f"**Allergy Warnings:** {len(result.get('allergy_warnings', []))}")
        st.write(f"**Clinical Warnings:** {len(result.get('warnings', []))}")

    with col_b:
        st.markdown("#### 🔬 AI Analysis Metrics")
        total_drugs = len(result.get("extracted_drugs", []))
        st.write(f"**Drugs Analyzed:** {total_drugs}")
        st.write(f"**AI-Extracted Drugs:** {len(result.get('extracted_drugs', []))}")
        st.write(f"**Dosage Recommendations:** {len(result.get('dosage_recommendations', []))}")
        st.write(f"**Alternative Suggestions:** {len(result.get('alternatives', []))}")
        st.write(f"**AI Insights Generated:** {len(result.get('ai_insights', []))}")

    # Clinical recommendation
    st.markdown("#### 🎯 Clinical Recommendation")
    if safety_score >= 90:
        st.success("✅ **LOW RISK**: Prescription appears safe with minimal concerns")
    elif safety_score >= 70:
        st.warning("⚠️ **MODERATE RISK**: Some concerns identified - enhanced monitoring recommended")
    elif safety_score >= 50:
        st.error("🔴 **HIGH RISK**: Significant issues detected - clinical review strongly recommended")
    else:
        st.error("🚨 **CRITICAL RISK**: Dangerous combinations detected - immediate clinical intervention required")


def display_welcome_message():
    """Display welcome message and instructions"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;'>
        <h3>🎯 Welcome to AI Prescription Analysis</h3>
        <p>👈 Enter prescription information on the left and click <strong>'Analyze'</strong></p>
        <br>
        <h4>🌟 System Features:</h4>
        <ul style='text-align: left; display: inline-block;'>
            <li>🔍 <strong>Drug Interaction Detection</strong> - Identifies harmful combinations</li>
            <li>💊 <strong>Age-Specific Dosing</strong> - Recommends appropriate doses</li>
            <li>🚨 <strong>Allergy Alerts</strong> - Prevents dangerous reactions</li>
            <li>🤖 <strong>AI Text Processing</strong> - Extracts drugs from clinical notes</li>
            <li>🔄 <strong>Alternative Suggestions</strong> - Safer medication options</li>
            <li>📊 <strong>Safety Scoring</strong> - Overall prescription risk assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Medical Prescription Verification System</h1>
        <p><strong>Advanced AI-Powered Drug Analysis & Safety Assessment</strong></p>
        <p>🔬 IBM Watson AI • 🧠 Hugging Face NLP • ⚡ FastAPI • 📊 Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'api_available' not in st.session_state:
        st.session_state.api_available = None

    # Check API availability
    if st.session_state.api_available is None:
        with st.spinner("🔍 Checking API connection..."):
            st.session_state.api_available = check_api_health()

    # API status indicator
    if st.session_state.api_available:
        st.success("🟢 Backend API: Connected")
    else:
        st.error("🔴 Backend API: Disconnected")
        st.markdown("""
        <div class="error-message">
            <strong>⚠️ Backend Server Not Running</strong><br>
            To start the backend server:
            <ol>
                <li>Save the backend code as <code>main.py</code></li>
                <li>Install dependencies: <code>pip install fastapi uvicorn pydantic</code></li>
                <li>Run: <code>uvicorn main:app --reload --port 8000</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar - Patient Information
    with st.sidebar:
        st.header("👤 Patient Information")

        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=45,
                                      help="Patient age for dosage calculations")
            with col2:
                weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0,
                                         help="Patient weight")

            st.subheader("📋 Medical Conditions")
            conditions_text = st.text_area(
                "Enter conditions (one per line)",
                placeholder="diabetes\nhypertension\nheart disease\nkidney disease",
                help="List any relevant medical conditions"
            )
            conditions = [c.strip() for c in conditions_text.split('\n') if c.strip()]

            st.subheader("🚨 Known Allergies")
            allergies_text = st.text_area(
                "Enter allergies (one per line)",
                placeholder="penicillin\nsulfa drugs\naspirin\nstatins",
                help="List all known drug allergies"
            )
            allergies = [a.strip() for a in allergies_text.split('\n') if a.strip()]

            patient_submitted = st.form_submit_button("💾 Save Patient Info", type="primary")
            if patient_submitted:
                st.success("✅ Patient information saved!")

        # Demo buttons
        st.markdown("---")
        st.subheader("🎯 Quick Demos")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚠ High Risk", help="Warfarin + Aspirin interaction"):
                st.session_state.demo_mode = "high_risk"
                st.rerun()

        with col_b:
            if st.button("✅ Safe Profile", help="Metformin + Lisinopril"):
                st.session_state.demo_mode = "safe"
                st.rerun()

        if st.button("🧓 Elderly Patient", help="Age-specific dosing concerns", use_container_width=True):
            st.session_state.demo_mode = "elderly"
            st.rerun()

        # API status and controls
        st.markdown("---")
        st.subheader("🔧 System Controls")
        if st.button("🔄 Refresh API Status", use_container_width=True):
            st.session_state.api_available = check_api_health()
            st.rerun()

        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.analysis_result = None
            st.rerun()

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("💊 Prescription Input")

        # Handle demo modes
        if st.session_state.demo_mode:
            handle_demo_mode(st.session_state.demo_mode, age, weight, conditions, allergies)
            st.session_state.demo_mode = False

        # Input method selection
        input_method = st.selectbox(
            "🔍 Choose input method:",
            ["Manual Drug Entry", "Smart Text Analysis", "Combined Input"],
            help="Select how you want to enter prescription information"
        )

        # Manual drug entry
        manual_drugs = []
        if input_method in ["Manual Drug Entry", "Combined Input"]:
            manual_drugs = handle_manual_input()

        # Text analysis section
        medical_text = ""
        if input_method in ["Smart Text Analysis", "Combined Input"]:
            medical_text = handle_text_input()

        # Main analysis button
        st.markdown("---")
        analyze_button = st.button("🚀 ANALYZE PRESCRIPTION", type="primary", use_container_width=True)

        if analyze_button:
            if not st.session_state.api_available:
                st.error("❌ Cannot analyze - API server is not available")
            elif manual_drugs or medical_text:
                analyze_prescription_safe(manual_drugs, age, weight, conditions, allergies, medical_text)
            else:
                st.error("⚠️ Please enter either drug information or prescription text")

    with col2:
        st.header("📊 Analysis Results")

        if st.session_state.analysis_result:
            display_enhanced_results(st.session_state.analysis_result)
        else:
            display_welcome_message()

            # Quick demo section in welcome area
            st.markdown("### 🎮 Try Quick Demos:")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("⚠️ High Risk", help="See a high-risk drug combination", key="demo_high_risk"):
                    st.session_state.demo_mode = "high_risk"
                    st.rerun()

            with col_b:
                if st.button("✅ Safe Profile", help="See a safe prescription", key="demo_safe"):
                    st.session_state.demo_mode = "safe"
                    st.rerun()

            with col_c:
                if st.button("👴 Elderly Care", help="See elderly patient considerations", key="demo_elderly"):
                    st.session_state.demo_mode = "elderly"
                    st.rerun()

    # Footer with system information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔬 <strong>Powered by Advanced AI Technologies:</strong></p>
        <p>IBM Watson Natural Language Processing • Hugging Face Transformers • FastAPI • Streamlit</p>
        <p>⚕️ <strong>For Healthcare Professionals:</strong> This system provides decision support - always verify with clinical judgment</p>
        <p>⚠️ <em>Educational and demonstration purposes - Consult healthcare professionals for medical decisions</em></p>
        <p><small>Version 2.0.0 | AI Medical Prescription Verification System</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()