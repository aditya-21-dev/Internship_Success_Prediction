from __future__ import annotations

from datetime import datetime
from io import BytesIO, StringIO

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import shap
except Exception:
    shap = None


# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Internship Success Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


FEATURE_COLUMNS = [
    "CGPA",
    "Number_of_Relevant_Projects",
    "Average_Project_Complexity",
    "Previous_Internship_Experience",
    "Relevant_Certifications",
    "Number_of_Programming_Languages",
    "Number_of_Technical_Skills",
    "Internship_Domain_Applied",
]

DEFAULT_INPUTS = {
    "cgpa": 8.0,
    "projects": 3,
    "complexity": 3,
    "internship": "No",
    "certifications": 2,
    "languages": 3,
    "skills": 5,
    "domain_name": "Machine Learning",
}

DOMAIN_FALLBACK_RECOMMENDATIONS = {
    "Cloud Computing": [
        "Earn an AWS, Azure, or Google Cloud certification.",
        "Deploy a real application using cloud storage, compute, and monitoring.",
        "Practice DevOps workflows with Docker, CI/CD, and infrastructure basics.",
    ],
    "Data Analytics": [
        "Build dashboard projects using SQL, Excel, Power BI, or Tableau.",
        "Improve SQL proficiency with joins, aggregations, and window functions.",
        "Practice business storytelling with clean data visualizations.",
    ],
    "Machine Learning": [
        "Build advanced ML projects with deployment-ready notebooks or APIs.",
        "Participate in Kaggle-style competitions to strengthen experimentation.",
        "Learn model evaluation, feature engineering, and deep learning basics.",
    ],
    "Software Development": [
        "Build scalable software projects with clean architecture and tests.",
        "Practice data structures, algorithms, and system design fundamentals.",
        "Contribute to open-source repositories to show production collaboration.",
    ],
    "Web Development": [
        "Build full-stack web applications with authentication and databases.",
        "Contribute to open-source web projects or publish reusable components.",
        "Learn modern frontend frameworks and API integration patterns.",
    ],
}

FEATURE_LABELS = {
    "CGPA": "CGPA",
    "Number_of_Relevant_Projects": "Projects",
    "Average_Project_Complexity": "Project Complexity",
    "Previous_Internship_Experience": "Internship Experience",
    "Relevant_Certifications": "Certifications",
    "Number_of_Programming_Languages": "Programming Languages",
    "Number_of_Technical_Skills": "Technical Skills",
    "Internship_Domain_Applied": "Internship Domain",
}

IDEAL_PROFILES = {
    "Cloud Computing": {"cgpa": 8.0, "projects": 3, "complexity": 4, "certifications": 3, "languages": 3, "skills": 7, "internship": "Yes"},
    "Data Analytics": {"cgpa": 8.0, "projects": 3, "complexity": 4, "certifications": 3, "languages": 3, "skills": 7, "internship": "Yes"},
    "Machine Learning": {"cgpa": 8.2, "projects": 4, "complexity": 4, "certifications": 3, "languages": 3, "skills": 7, "internship": "Yes"},
    "Software Development": {"cgpa": 8.0, "projects": 4, "complexity": 4, "certifications": 2, "languages": 4, "skills": 7, "internship": "Yes"},
    "Web Development": {"cgpa": 7.8, "projects": 4, "complexity": 4, "certifications": 2, "languages": 3, "skills": 7, "internship": "Yes"},
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Apply a focused visual layer for a polished Streamlit dashboard."""
    st.markdown(
        """
        <style>
            :root {
                --ink: #f6fbff;
                --muted: #9caeb9;
                --panel: #121b22;
                --panel-soft: #18242d;
                --line: #283842;
                --line-soft: rgba(148, 163, 184, 0.18);
                --success: #35d39d;
                --warning: #ffc857;
                --danger: #ff6b6b;
                --accent: #37c7e6;
                --accent-soft: rgba(55, 199, 230, 0.13);
                --shadow: rgba(0, 0, 0, 0.38);
                --radius: 16px;
                --card-pad: 1rem;
                --section-gap: 1rem;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(55, 199, 230, 0.10), transparent 30%),
                    linear-gradient(135deg, #071016 0%, #0d171d 44%, #101f24 100%);
                color: var(--ink);
            }

            .main .block-container {
                padding: 1.35rem 1.6rem 3rem;
                max-width: 1280px;
                margin-left: 0;
                margin-right: auto;
            }

            .hero {
                background:
                    linear-gradient(135deg, rgba(55, 199, 230, 0.18), rgba(53, 211, 157, 0.11)),
                    radial-gradient(circle at top right, rgba(255, 200, 87, 0.2), transparent 28%);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                padding: 1.45rem 1.55rem;
                margin-bottom: var(--section-gap);
                box-shadow: 0 24px 70px var(--shadow);
                animation: fadeUp 520ms ease both;
            }

            .hero h1 {
                color: var(--ink);
                font-size: 2.05rem;
                line-height: 1.15;
                margin: 0 0 0.35rem;
                letter-spacing: 0;
                font-weight: 800;
            }

            .hero p {
                color: var(--muted);
                font-size: 0.98rem;
                margin: 0;
            }

            .section-card {
                background: linear-gradient(180deg, rgba(18, 27, 34, 0.96), rgba(15, 23, 30, 0.96));
                border: 1px solid var(--line);
                border-radius: var(--radius);
                padding: var(--card-pad);
                margin-bottom: var(--section-gap);
                box-shadow: 0 16px 40px var(--shadow);
                animation: fadeUp 560ms ease both;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 0.75rem;
            }

            .summary-item {
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 0.8rem;
                background: var(--panel-soft);
                min-width: 0;
                transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
            }

            .summary-item:hover {
                transform: translateY(-1px);
                border-color: rgba(55, 199, 230, 0.42);
                background: #1a2933;
            }

            .summary-label {
                color: var(--muted);
                font-size: 0.78rem;
                margin-bottom: 0.2rem;
                text-transform: uppercase;
            }

            .summary-value {
                color: var(--ink);
                font-size: 1.05rem;
                font-weight: 700;
                overflow-wrap: anywhere;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.35rem 0.65rem;
                font-weight: 700;
                font-size: 0.88rem;
            }

            .status-excellent { background: rgba(53, 211, 157, 0.14); color: var(--success); border: 1px solid rgba(53, 211, 157, 0.32); }
            .status-good { background: rgba(55, 199, 230, 0.14); color: var(--accent); border: 1px solid rgba(55, 199, 230, 0.32); }
            .status-moderate { background: rgba(255, 200, 87, 0.14); color: var(--warning); border: 1px solid rgba(255, 200, 87, 0.32); }
            .status-poor { background: rgba(255, 107, 107, 0.14); color: var(--danger); border: 1px solid rgba(255, 107, 107, 0.32); }

            .gauge-wrap {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 220px;
            }

            .gauge {
                --value: 0;
                width: 210px;
                height: 210px;
                border-radius: 50%;
                display: grid;
                place-items: center;
                background: conic-gradient(var(--accent) calc(var(--value) * 1%), #263741 0);
                position: relative;
            }

            .gauge::before {
                content: "";
                position: absolute;
                width: 150px;
                height: 150px;
                background: #0f171d;
                border-radius: 50%;
                box-shadow: inset 0 0 0 1px var(--line);
            }

            .gauge-value {
                position: relative;
                color: var(--ink);
                font-size: 2rem;
                font-weight: 800;
            }

            .history-table {
                font-size: 0.9rem;
            }

            h1, h2, h3, .stMarkdown, .stMetric, label {
                color: var(--ink) !important;
            }

            h2, [data-testid="stMarkdownContainer"] h2 {
                font-size: 1.45rem;
                line-height: 1.25;
                margin: 1.15rem 0 0.55rem;
                font-weight: 760;
            }

            h3, [data-testid="stMarkdownContainer"] h3 {
                font-size: 1.08rem;
                line-height: 1.3;
                margin: 0.75rem 0 0.45rem;
                font-weight: 720;
            }

            p, .stCaptionContainer, [data-testid="stMarkdownContainer"] p {
                color: var(--muted);
                font-size: 0.92rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0c151b, #111d24);
                border-right: 1px solid var(--line);
                width: 300px !important;
                min-width: 280px !important;
                max-width: 320px !important;
            }

            [data-testid="stSidebar"] > div {
                width: 300px !important;
                padding: 1.2rem 0.9rem;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.55rem;
            }

            [data-testid="stSidebar"] label {
                font-size: 0.88rem;
                font-weight: 650;
            }

            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                padding: 0.95rem 1rem;
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
                min-height: 92px;
            }

            [data-testid="stMetric"] label,
            [data-testid="stMetricLabel"] {
                color: var(--muted) !important;
                font-size: 0.82rem !important;
                font-weight: 650;
            }

            [data-testid="stMetricValue"] {
                color: var(--ink) !important;
                font-size: 1.45rem !important;
                line-height: 1.18;
            }

            [data-testid="stAlert"], [data-testid="stExpander"] {
                background: linear-gradient(180deg, rgba(18, 27, 34, 0.96), rgba(14, 22, 29, 0.96));
                border: 1px solid var(--line-soft);
                border-radius: var(--radius);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
            }

            [data-testid="stAlert"] div {
                color: var(--ink);
            }

            [data-testid="stExpander"] summary {
                color: var(--ink);
                font-weight: 680;
            }

            .stButton > button, .stDownloadButton > button {
                min-height: 48px;
                border-radius: 13px;
                padding: 0.72rem 1rem;
                font-weight: 760;
                letter-spacing: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.45rem;
                white-space: normal;
                line-height: 1.2;
                transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
            }

            .stButton > button:hover, .stDownloadButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.26);
            }

            div[data-testid="stButton"] button[kind="primary"] {
                width: 100%;
                min-height: 56px;
                font-size: 1rem;
                background: linear-gradient(135deg, #37c7e6, #35d39d);
                color: #071016;
                border: 0;
            }

            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, var(--accent), var(--success));
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                overflow: hidden;
                width: 100%;
                background: var(--panel);
            }

            div[data-testid="stPlotlyChart"] {
                width: 100%;
                border: 1px solid var(--line-soft);
                border-radius: var(--radius);
                padding: 0.35rem;
                background: rgba(18, 27, 34, 0.48);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
            }

            .block-container [data-testid="stVerticalBlock"] {
                gap: 0.65rem;
            }

            [data-testid="column"] {
                min-width: 0;
            }

            @keyframes fadeUp {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 900px) {
                .main .block-container { padding: 1rem 1rem 2rem; }
                .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
                    width: 285px !important;
                    min-width: 280px !important;
                }
            }

            @media (max-width: 560px) {
                .summary-grid { grid-template-columns: 1fr; }
                .hero h1 { font-size: 1.65rem; }
                .hero { padding: 1.15rem; }
                [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load the trained Random Forest model and domain label encoder once."""
    trained_model = joblib.load("internship_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return trained_model, label_encoder


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def validate_inputs(profile: dict) -> tuple[dict, list[str]]:
    """Normalize profile values and return non-blocking validation messages."""
    validated = profile.copy()
    messages: list[str] = []

    validated["cgpa"] = min(max(float(validated["cgpa"]), 0.0), 10.0)
    validated["projects"] = min(max(int(validated["projects"]), 0), 10)
    validated["certifications"] = min(max(int(validated["certifications"]), 0), 10)
    validated["languages"] = min(max(int(validated["languages"]), 0), 10)
    validated["skills"] = min(max(int(validated["skills"]), 0), 20)

    if validated["projects"] == 0:
        validated["complexity"] = 0
        messages.append("Project complexity was set to 0 because no projects were entered.")
    else:
        validated["complexity"] = min(max(int(validated["complexity"]), 1), 5)

    return validated, messages


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def preprocess_input(profile: dict, label_encoder) -> pd.DataFrame:
    """Convert the validated UI profile into the model's expected DataFrame."""
    internship_value = 1 if profile["internship"] == "Yes" else 0
    domain_encoded = int(label_encoder.transform([profile["domain_name"]])[0])

    input_df = pd.DataFrame(
        [
            {
                "CGPA": profile["cgpa"],
                "Number_of_Relevant_Projects": profile["projects"],
                "Average_Project_Complexity": profile["complexity"],
                "Previous_Internship_Experience": internship_value,
                "Relevant_Certifications": profile["certifications"],
                "Number_of_Programming_Languages": profile["languages"],
                "Number_of_Technical_Skills": profile["skills"],
                "Internship_Domain_Applied": domain_encoded,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    return input_df


def predict_success(trained_model, input_df: pd.DataFrame) -> float:
    """Return the Random Forest success probability as a percentage."""
    return float(trained_model.predict_proba(input_df)[0][1] * 100)


def classify_suitability(probability: float) -> tuple[str, str, str]:
    """Map probability into a presentation-friendly suitability label."""
    if probability >= 80:
        return "Excellent Match", "status-excellent", "●"
    if probability >= 65:
        return "Good Match", "status-good", "●"
    if probability >= 35:
        return "Moderate Match", "status-moderate", "●"
    return "Poor Match", "status-poor", "●"


# ---------------------------------------------------------------------------
# Domain Analysis
# ---------------------------------------------------------------------------

def analyze_domain(profile: dict) -> str:
    """Describe why the selected domain fits the profile."""
    domain_name = profile["domain_name"]
    project_signal = "project-backed" if profile["projects"] >= 3 else "early-stage"
    skill_signal = "skill-rich" if profile["skills"] >= 6 else "skill-building"
    return f"{domain_name} profile is currently {project_signal} and {skill_signal}."


# ---------------------------------------------------------------------------
# Profile Strength Analysis
# ---------------------------------------------------------------------------

def analyze_strengths(profile: dict) -> list[str]:
    """Identify visible strengths in the student profile."""
    strengths: list[str] = []

    if profile["cgpa"] >= 8:
        strengths.append("Strong academic performance")
    if profile["projects"] >= 4:
        strengths.append("Strong project portfolio")
    if profile["projects"] > 0 and profile["complexity"] >= 4:
        strengths.append("Experience with complex projects")
    if profile["certifications"] >= 3:
        strengths.append("Good certification profile")
    if profile["internship"] == "Yes":
        strengths.append("Prior internship experience")
    if profile["languages"] >= 3:
        strengths.append("Comfortable with multiple programming languages")
    if profile["skills"] >= 6:
        strengths.append("Broad technical skill coverage")

    return strengths


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------

def analyze_gaps(profile: dict) -> list[str]:
    """Find domain-specific gaps using the same decision logic as the original app."""
    gaps: list[str] = []
    domain_name = profile["domain_name"]

    if profile["cgpa"] < 7:
        gaps.append("Improve academic consistency and fundamentals")

    if domain_name == "Data Analytics":
        if profile["certifications"] < 2:
            gaps.append("Need more analytics certifications")
        if profile["skills"] < 5:
            gaps.append("Need more analytics-related technical skills")
        if profile["projects"] < 3:
            gaps.append("Need more data analytics projects")

    elif domain_name == "Machine Learning":
        if profile["projects"] == 0:
            gaps.append("Need ML-related projects")
        elif profile["complexity"] < 4:
            gaps.append("Need more advanced ML projects")
        if profile["projects"] < 3:
            gaps.append("Need more ML-related projects")
        if profile["certifications"] < 2:
            gaps.append("Need more ML certifications")

    elif domain_name == "Web Development":
        if profile["projects"] < 4:
            gaps.append("Need more web development projects")
        if profile["languages"] < 3:
            gaps.append("Learn additional web-related programming languages")
        if profile["projects"] > 0 and profile["complexity"] < 3:
            gaps.append("Build more advanced web applications")

    elif domain_name == "Software Development":
        if profile["languages"] < 3:
            gaps.append("Learn additional programming languages")
        if profile["projects"] < 4:
            gaps.append("Build more software development projects")
        if profile["projects"] > 0 and profile["complexity"] < 4:
            gaps.append("Work on more complex software projects")

    elif domain_name == "Cloud Computing":
        if profile["certifications"] < 2:
            gaps.append("Earn cloud-related certifications")
        if profile["skills"] < 5:
            gaps.append("Develop more cloud-related technical skills")
        if profile["projects"] < 3:
            gaps.append("Build cloud computing projects")

    return gaps


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------

def recommend_improvements(profile: dict, gaps: list[str]) -> list[str]:
    """Turn detected gaps into practical next actions without duplicating advice."""
    recommendations: list[str] = []

    for gap in gaps:
        normalized_gap = gap.lower()
        if "certification" in normalized_gap:
            recommendations.append("Earn relevant certifications for the target domain.")
        elif "project" in normalized_gap or "application" in normalized_gap:
            recommendations.append("Build additional domain-focused projects.")
        elif "technical skill" in normalized_gap or "fundamentals" in normalized_gap:
            recommendations.append("Develop more practical technical skills.")
        elif "programming language" in normalized_gap:
            recommendations.append("Learn additional programming languages used in the domain.")

    if not recommendations:
        recommendations = DOMAIN_FALLBACK_RECOMMENDATIONS[profile["domain_name"]]

    return list(dict.fromkeys(recommendations))


# ---------------------------------------------------------------------------
# Expected Success Probability After Improvement
# ---------------------------------------------------------------------------

def simulate_profile_improvement(profile: dict, label_encoder, trained_model) -> tuple[dict, pd.DataFrame, float]:
    """Improve only weak profile areas and run the model again."""
    improved_profile = profile.copy()

    if improved_profile["cgpa"] < 7:
        improved_profile["cgpa"] = 7.0
    if improved_profile["projects"] < 4:
        improved_profile["projects"] = 4
    if improved_profile["certifications"] < 3:
        improved_profile["certifications"] = 3
    if improved_profile["languages"] < 3:
        improved_profile["languages"] = 3
    if improved_profile["skills"] < 6:
        improved_profile["skills"] = 6

    if improved_profile["projects"] == 0:
        improved_profile["complexity"] = 0
    elif improved_profile["complexity"] < 4:
        improved_profile["complexity"] = 4

    improved_df = preprocess_input(improved_profile, label_encoder)
    improved_probability = predict_success(trained_model, improved_df)
    return improved_profile, improved_df, improved_probability


def calculate_profile_scores(profile: dict) -> dict:
    """Score the candidate profile as an analysis layer only."""
    academic_score = min(profile["cgpa"] / 10 * 100, 100)
    projects_score = min((profile["projects"] / 4 * 70) + (profile["complexity"] / 5 * 30), 100)
    skills_score = min(((profile["languages"] / 4) * 45) + ((profile["skills"] / 8) * 55), 100)
    certifications_score = min(profile["certifications"] / 3 * 100, 100)
    professional_exposure_score = 100 if profile["internship"] == "Yes" else 35

    overall_score = (
        academic_score * 0.22
        + projects_score * 0.26
        + skills_score * 0.24
        + certifications_score * 0.14
        + professional_exposure_score * 0.14
    )

    return {
        "Overall Score": round(overall_score, 1),
        "Academic Score": round(academic_score, 1),
        "Projects Score": round(projects_score, 1),
        "Skills Score": round(skills_score, 1),
        "Certifications Score": round(certifications_score, 1),
        "Professional Exposure score": round(professional_exposure_score, 1),
    }


def build_candidate_comparison(profile: dict) -> pd.DataFrame:
    """Compare the candidate against a domain-specific ideal profile."""
    ideal = IDEAL_PROFILES[profile["domain_name"]]
    rows = [
        ("CGPA", f"{profile['cgpa']:.1f}", f"{ideal['cgpa']:.1f}", profile["cgpa"] >= ideal["cgpa"]),
        ("Relevant Projects", profile["projects"], ideal["projects"], profile["projects"] >= ideal["projects"]),
        ("Project Complexity", profile["complexity"], ideal["complexity"], profile["complexity"] >= ideal["complexity"]),
        ("Certifications", profile["certifications"], ideal["certifications"], profile["certifications"] >= ideal["certifications"]),
        ("Programming Languages", profile["languages"], ideal["languages"], profile["languages"] >= ideal["languages"]),
        ("Technical Skills", profile["skills"], ideal["skills"], profile["skills"] >= ideal["skills"]),
        ("Internship Experience", profile["internship"], ideal["internship"], profile["internship"] == ideal["internship"]),
    ]

    return pd.DataFrame(
        [
            {
                "Feature": feature,
                "Student": student,
                "Ideal Candidate": ideal_value,
                "Status": "Ready" if is_ready else "Needs Work",
            }
            for feature, student, ideal_value, is_ready in rows
        ]
    )


def build_radar_values(profile: dict) -> pd.DataFrame:
    """Normalize profile and ideal values for radar chart comparison."""
    ideal = IDEAL_PROFILES[profile["domain_name"]]
    categories = {
        "Academics": (profile["cgpa"], ideal["cgpa"]),
        "Projects": (profile["projects"], ideal["projects"]),
        "Complexity": (profile["complexity"], ideal["complexity"]),
        "Certifications": (profile["certifications"], ideal["certifications"]),
        "Languages": (profile["languages"], ideal["languages"]),
        "Skills": (profile["skills"], ideal["skills"]),
        "Experience": (100 if profile["internship"] == "Yes" else 35, 100),
    }

    rows = []
    for category, (student_value, ideal_value) in categories.items():
        student_score = min((float(student_value) / float(ideal_value)) * 100, 100) if ideal_value else 0
        rows.append({"Category": category, "Student Profile": student_score, "Ideal Candidate": 100})
    return pd.DataFrame(rows)


def get_model_feature_importance(trained_model) -> pd.DataFrame:
    """Read feature importance directly from the trained Random Forest model."""
    importance_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(feature, feature) for feature in FEATURE_COLUMNS],
            "Importance": trained_model.feature_importances_,
        }
    )
    return importance_df.sort_values("Importance", ascending=False)


def unavailable_shap_explainability() -> dict:
    """Return a stable explainability payload when SHAP is unavailable."""
    return {
        "available": False,
        "top_positive": [],
        "top_negative": [],
        "explanation": "SHAP explainability is currently unavailable.",
        "contribution_df": pd.DataFrame(columns=["Feature", "SHAP Value", "Direction"]),
        "waterfall": None,
    }


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_trained_model):
    """Create and cache one SHAP explainer for the loaded Random Forest model."""
    if shap is None:
        return None

    try:
        return shap.Explainer(_trained_model)
    except Exception:
        try:
            return shap.TreeExplainer(_trained_model)
        except Exception:
            return None


def select_success_class_shap_output(shap_output, explainer) -> tuple[list[float], float]:
    """Extract class-1 SHAP values from modern and legacy SHAP outputs."""
    if isinstance(shap_output, list):
        class_index = 1 if len(shap_output) > 1 else 0
        values = shap_output[class_index][0]
        expected_value = getattr(explainer, "expected_value", 0.0)
        if isinstance(expected_value, (list, tuple)):
            base_value = expected_value[class_index]
        elif hasattr(expected_value, "ndim") and getattr(expected_value, "ndim", 0) > 0:
            base_value = expected_value[class_index] if len(expected_value) > class_index else expected_value[0]
        else:
            base_value = expected_value
        return [float(value) for value in values], float(base_value)

    values = shap_output.values
    base_values = shap_output.base_values

    if getattr(values, "ndim", 0) == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        selected_values = values[0, :, class_index]
        if getattr(base_values, "ndim", 0) == 2:
            selected_base = base_values[0, class_index]
        else:
            selected_base = base_values[class_index] if len(base_values) > class_index else base_values[0]
    else:
        selected_values = values[0]
        if getattr(base_values, "ndim", 0) == 2:
            selected_base = base_values[0, 1] if base_values.shape[1] > 1 else base_values[0, 0]
        elif getattr(base_values, "ndim", 0) == 1:
            selected_base = base_values[1] if len(base_values) > 1 else base_values[0]
        else:
            selected_base = base_values

    return [float(value) for value in selected_values], float(selected_base)


def build_shap_natural_language(positive_features: list[str], negative_features: list[str], probability: float) -> str:
    """Generate a concise explanation from SHAP positive and negative factors."""
    confidence_label = "high" if probability >= 75 else "moderate" if probability >= 50 else "low"
    positive_text = ", ".join(positive_features[:3]) if positive_features else "the available profile signals"
    negative_text = ", ".join(negative_features[:3]) if negative_features else "no major opposing factors"

    if negative_features:
        return (
            f"The model predicts a {confidence_label} internship success probability because "
            f"{positive_text} increased the prediction, while {negative_text} reduced the "
            "prediction confidence for this profile."
        )

    return (
        f"The model predicts a {confidence_label} internship success probability because "
        f"{positive_text} increased the prediction, with no major negative SHAP factors for this input."
    )


def generate_ai_explainability(result: dict, trained_model) -> dict:
    """Explain the current prediction using SHAP values for the user's input."""
    if shap is None:
        return unavailable_shap_explainability()

    explainer = get_shap_explainer(trained_model)
    if explainer is None:
        return unavailable_shap_explainability()

    input_df = result["input_df"]

    try:
        shap_output = explainer(input_df)
    except Exception:
        try:
            shap_output = explainer.shap_values(input_df)
        except Exception:
            return unavailable_shap_explainability()

    try:
        shap_values, base_value = select_success_class_shap_output(shap_output, explainer)
    except Exception:
        return unavailable_shap_explainability()

    contribution_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(feature, feature) for feature in FEATURE_COLUMNS],
            "SHAP Value": shap_values,
        }
    )
    contribution_df["Direction"] = contribution_df["SHAP Value"].map(lambda value: "Positive" if value >= 0 else "Negative")
    top_positive = (
        contribution_df[contribution_df["SHAP Value"] > 0]
        .sort_values("SHAP Value", ascending=False)
        .head(3)["Feature"]
        .tolist()
    )
    top_negative = (
        contribution_df[contribution_df["SHAP Value"] < 0]
        .sort_values("SHAP Value", ascending=True)
        .head(3)["Feature"]
        .tolist()
    )

    try:
        waterfall = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=input_df.iloc[0].tolist(),
            feature_names=[FEATURE_LABELS.get(feature, feature) for feature in FEATURE_COLUMNS],
        )
    except Exception:
        waterfall = None

    return {
        "available": True,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "explanation": build_shap_natural_language(top_positive, top_negative, result["probability"]),
        "contribution_df": contribution_df,
        "waterfall": waterfall,
    }


def rank_all_domains(profile: dict, label_encoder, trained_model) -> pd.DataFrame:
    """Predict success probability for every internship domain."""
    rows = []
    for domain_name in label_encoder.classes_:
        domain_profile = profile.copy()
        domain_profile["domain_name"] = domain_name
        domain_df = preprocess_input(domain_profile, label_encoder)
        rows.append({"Domain": domain_name, "Probability": predict_success(trained_model, domain_df)})

    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


# ---------------------------------------------------------------------------
# Reporting and Session State
# ---------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Create stable session keys so the app is safe before any prediction."""
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault("prediction_result", None)
    st.session_state.setdefault("prediction_history", [])


def reset_inputs() -> None:
    """Reset form fields and clear the current prediction view."""
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[key] = value
    st.session_state["prediction_result"] = None


def build_report_csv(result: dict) -> bytes:
    """Build a downloadable CSV report for the latest prediction."""
    report_rows = {
        "Generated At": result["timestamp"],
        "Target Internship Domain": result["profile"]["domain_name"],
        "Predicted Success Probability": f"{result['probability']:.2f}%",
        "Expected Probability After Improvement": f"{result['improved_probability']:.2f}%",
        "Potential Improvement": f"{result['improvement_gain']:.2f}%",
        "Domain Suitability": result["suitability"],
        "Domain Analysis": result["domain_analysis"],
        "Profile Strengths": "; ".join(result["strengths"]) or "No significant strengths identified",
        "Profile Gaps": "; ".join(result["gaps"]) or "No significant gaps found",
        "Recommended Improvements": "; ".join(result["recommendations"]),
    }

    buffer = StringIO()
    pd.DataFrame([report_rows]).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_pdf_report(result: dict) -> bytes:
    """Generate a professional PDF report for the latest prediction."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    def add_heading(text: str) -> None:
        story.append(Paragraph(text, styles["Heading2"]))
        story.append(Spacer(1, 8))

    def add_list(items: list[str]) -> None:
        if not items:
            story.append(Paragraph("None identified.", styles["BodyText"]))
            story.append(Spacer(1, 8))
            return
        for item in items:
            story.append(Paragraph(f"- {item}", styles["BodyText"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Internship Success Prediction Report", styles["Title"]))
    story.append(Paragraph(f"Generated At: {result['timestamp']}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    profile = result["profile"]
    summary_rows = [
        ["Target Internship Domain", profile["domain_name"]],
        ["Success Probability", f"{result['probability']:.2f}%"],
        ["Domain Suitability", result["suitability"]],
        ["Expected Probability After Improvement", f"{result['improved_probability']:.2f}%"],
        ["Probability Gain", f"{result['improvement_gain']:.2f}%"],
        ["Overall Profile Score", f"{result['profile_scores']['Overall Score']:.1f}/100"],
    ]
    summary_table = Table(summary_rows, colWidths=[210, 260])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f6fb")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9fb5bf")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    add_heading("Student Inputs")
    input_rows = [
        ["CGPA", f"{profile['cgpa']:.1f}"],
        ["Relevant Projects", profile["projects"]],
        ["Project Complexity", profile["complexity"]],
        ["Previous Internship", profile["internship"]],
        ["Certifications", profile["certifications"]],
        ["Programming Languages", profile["languages"]],
        ["Technical Skills", profile["skills"]],
    ]
    input_table = Table(input_rows, colWidths=[210, 260])
    input_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("PADDING", (0, 0), (-1, -1), 6)]))
    story.append(input_table)
    story.append(Spacer(1, 12))

    add_heading("AI Explainability")
    add_list([f"Top Positive Factor: {item}" for item in result["explainability"]["top_positive"]])
    add_list([f"Top Negative Factor: {item}" for item in result["explainability"]["top_negative"]])
    story.append(Paragraph(result["explainability"]["explanation"], styles["BodyText"]))
    story.append(Spacer(1, 12))

    add_heading("Candidate vs Ideal Candidate")
    comparison_rows = [result["comparison_df"].columns.tolist()] + result["comparison_df"].astype(str).values.tolist()
    comparison_table = Table(comparison_rows, colWidths=[135, 100, 120, 95])
    comparison_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dff3ea")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(comparison_table)
    story.append(Spacer(1, 12))

    add_heading("Profile Strengths")
    add_list(result["strengths"])
    add_heading("Profile Gaps")
    add_list(result["gaps"])
    add_heading("Recommended Improvements")
    add_list(result["recommendations"])

    doc.build(story)
    return buffer.getvalue()


def add_history_entry(result: dict) -> None:
    """Store recent predictions in session state for quick comparison."""
    history_entry = {
        "Time": result["timestamp"],
        "Domain": result["profile"]["domain_name"],
        "Current Probability": round(result["probability"], 2),
        "Expected Probability": round(result["improved_probability"], 2),
        "Suitability": result["suitability"],
    }

    st.session_state["prediction_history"].insert(0, history_entry)
    st.session_state["prediction_history"] = st.session_state["prediction_history"][:8]


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🎯 Internship Success Prediction System</h1>
            <p>AI-assisted readiness analysis for internship applications, built around your trained Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_summary(profile: dict) -> None:
    internship_label = "Experienced" if profile["internship"] == "Yes" else "No prior internship"
    st.markdown(
        f"""
        <div class="section-card">
            <h3 style="margin-top:0;">👤 Profile Summary</h3>
            <div class="summary-grid">
                <div class="summary-item"><div class="summary-label">CGPA</div><div class="summary-value">{profile["cgpa"]:.1f}/10</div></div>
                <div class="summary-item"><div class="summary-label">Projects</div><div class="summary-value">{profile["projects"]} projects</div></div>
                <div class="summary-item"><div class="summary-label">Complexity</div><div class="summary-value">{profile["complexity"]}/5</div></div>
                <div class="summary-item"><div class="summary-label">Target Domain</div><div class="summary-value">{profile["domain_name"]}</div></div>
                <div class="summary-item"><div class="summary-label">Internship</div><div class="summary-value">{internship_label}</div></div>
                <div class="summary-item"><div class="summary-label">Certifications</div><div class="summary-value">{profile["certifications"]}</div></div>
                <div class="summary-item"><div class="summary-label">Languages</div><div class="summary-value">{profile["languages"]}</div></div>
                <div class="summary-item"><div class="summary-label">Technical Skills</div><div class="summary-value">{profile["skills"]}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_inputs(label_encoder) -> dict:
    """Render all input widgets in the sidebar and return a validated profile."""
    with st.sidebar:
        st.title("⚙️ Profile Inputs")
        st.caption("Tune the profile, then run the model from the main panel.")

        st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, key="cgpa")
        st.number_input("Number of Relevant Projects", min_value=0, max_value=10, step=1, key="projects")

        if st.session_state["projects"] == 0:
            st.session_state["complexity"] = 0
            st.slider("Average Project Complexity", 0, 5, 0, disabled=True, key="complexity_display")
            st.info("Complexity is automatically set to 0 when projects are 0.")
        else:
            if st.session_state["complexity"] == 0:
                st.session_state["complexity"] = 3
            st.slider("Average Project Complexity", 1, 5, key="complexity")

        st.selectbox("Previous Internship Experience", ["No", "Yes"], key="internship")
        st.number_input("Relevant Certifications", min_value=0, max_value=10, step=1, key="certifications")
        st.number_input("Number of Programming Languages", min_value=0, max_value=10, step=1, key="languages")
        st.number_input("Number of Technical Skills", min_value=0, max_value=20, step=1, key="skills")
        st.selectbox("Target Internship Domain", list(label_encoder.classes_), key="domain_name")

        reset_clicked = st.button("↺ Reset Inputs", use_container_width=True)
        if reset_clicked:
            reset_inputs()
            st.rerun()

    raw_profile = {key: st.session_state[key] for key in DEFAULT_INPUTS}
    validated_profile, validation_messages = validate_inputs(raw_profile)

    for message in validation_messages:
        st.sidebar.info(message)

    return validated_profile


def render_probability_gauge(probability: float) -> None:
    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge" style="--value:{probability:.2f};">
                <div class="gauge-value">{probability:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_explainability(result: dict) -> None:
    st.subheader("🔍 AI Explainability (SHAP)")
    explainability = result["explainability"]

    if not explainability.get("available"):
        st.warning("SHAP explainability is currently unavailable.")
        return

    positive_col, negative_col = st.columns(2)

    with positive_col:
        st.markdown("**Top Positive Factors**")
        if explainability["top_positive"]:
            for factor in explainability["top_positive"]:
                st.success(f"✔ {factor}")
        else:
            st.info("No positive SHAP factors found for this input.")

    with negative_col:
        st.markdown("**Top Negative Factors**")
        if explainability["top_negative"]:
            for factor in explainability["top_negative"]:
                st.warning(f"✖ {factor}")
        else:
            st.info("No negative SHAP factors found for this input.")

    st.info(explainability["explanation"])

    contribution_df = explainability["contribution_df"].sort_values("SHAP Value", ascending=True)
    color_map = {"Positive": "#35d39d", "Negative": "#ff6b6b"}
    fig = px.bar(
        contribution_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map=color_map,
        text=contribution_df["SHAP Value"].map(lambda value: f"{value:+.3f}"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=390,
        margin=dict(l=20, r=20, t=20, b=28),
        legend_title_text="Contribution",
        xaxis_title="SHAP Contribution",
        yaxis_title="Feature",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_profile_scores(result: dict) -> None:
    st.subheader("Overall Profile Score")
    scores = result["profile_scores"]
    score_cols = st.columns([0.32, 0.68])

    with score_cols[0]:
        st.metric("Overall Score", f"{scores['Overall Score']:.1f}/100")

    with score_cols[1]:
        for label in [
            "Academic Score",
            "Projects Score",
            "Skills Score",
            "Certifications Score",
            "Professional Exposure score",
        ]:
            st.caption(f"{label}: {scores[label]:.1f}/100")
            st.progress(scores[label] / 100)


def render_candidate_comparison(result: dict) -> None:
    st.subheader("Candidate vs Ideal Candidate")
    comparison_df = result["comparison_df"]

    def highlight_status(row):
        if row["Status"] == "Needs Work":
            return ["background-color: #3a231f; color: #ffb4a8"] * len(row)
        return ["background-color: #163329; color: #9ff0cd"] * len(row)

    st.dataframe(
        comparison_df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    radar_df = result["radar_df"]
    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_df["Student Profile"],
            theta=radar_df["Category"],
            fill="toself",
            name="Student Profile",
            line_color="#37c7e6",
        )
    )
    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_df["Ideal Candidate"],
            theta=radar_df["Category"],
            fill="toself",
            name="Ideal Candidate",
            line_color="#35d39d",
            opacity=0.72,
        )
    )
    radar_fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=420,
        margin=dict(l=24, r=24, t=24, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(radar_fig, use_container_width=True)


def render_feature_importance(result: dict) -> None:
    st.subheader("Model Feature Importance")
    importance_df = result["feature_importance_df"].sort_values("Importance", ascending=True)
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=importance_df["Importance"].map(lambda value: f"{value:.3f}"),
        color="Importance",
        color_continuous_scale=["#263741", "#37c7e6", "#35d39d"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=12, b=22),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_domain_ranking(result: dict) -> None:
    st.subheader("Domain Ranking")
    ranking_df = result["domain_ranking_df"].sort_values("Probability", ascending=True)
    fig = px.bar(
        ranking_df,
        x="Probability",
        y="Domain",
        orientation="h",
        text=ranking_df["Probability"].map(lambda value: f"{value:.1f}%"),
        color="Probability",
        color_continuous_scale=["#3a231f", "#ffc857", "#35d39d"],
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Predicted Success Probability",
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(l=20, r=20, t=12, b=22),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: dict) -> None:
    """Render all prediction, analysis, recommendation, and report sections."""
    probability = result["probability"]
    improved_probability = result["improved_probability"]
    improvement_gain = result["improvement_gain"]
    status_label = result["suitability"]
    status_class = result["status_class"]
    status_icon = result["status_icon"]

    st.divider()
    st.subheader("📊 Prediction Results")

    metric_col, gauge_col = st.columns([1.35, 1])
    with metric_col:
        top_a, top_b, top_c = st.columns(3)
        top_a.metric("Success Probability", f"{probability:.2f}%")
        top_b.metric("After Improvement", f"{improved_probability:.2f}%", f"+{improvement_gain:.2f}%")
        top_c.metric("Target Domain", result["profile"]["domain_name"])

        st.progress(min(max(probability / 100, 0.0), 1.0), text="Current success probability")
        st.markdown(
            f'<span class="status-pill {status_class}">{status_icon} {status_label}</span>',
            unsafe_allow_html=True,
        )
        st.caption(result["domain_analysis"])

    with gauge_col:
        render_probability_gauge(probability)

    render_ai_explainability(result)
    render_profile_scores(result)
    render_candidate_comparison(result)
    render_feature_importance(result)
    render_domain_ranking(result)

    st.subheader("🧭 Domain Analysis")
    with st.expander("View domain suitability details", expanded=True):
        st.write(result["domain_analysis"])

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("✅ Profile Strengths")
        if result["strengths"]:
            for strength in result["strengths"]:
                st.success(strength)
        else:
            st.warning("No significant strengths identified yet.")

    with right_col:
        st.subheader("⚠️ Profile Gaps")
        if result["gaps"]:
            for gap in result["gaps"]:
                st.error(gap)
        else:
            st.success("No significant gaps found for this target domain.")

    st.subheader("💡 Recommended Improvements")
    with st.expander("Action plan", expanded=True):
        for index, recommendation in enumerate(result["recommendations"], start=1):
            st.info(f"{index}. {recommendation}")

    st.subheader("🚀 Expected Success Probability After Improvement")
    improvement_cols = st.columns(4)
    improvement_cols[0].metric("Current", f"{probability:.2f}%")
    improvement_cols[1].metric("Expected", f"{improved_probability:.2f}%")
    improvement_cols[2].metric("Potential Gain", f"+{improvement_gain:.2f}%")
    improvement_cols[3].metric("Improved Complexity", result["improved_profile"]["complexity"])
    st.caption("Current probability")
    st.progress(min(max(probability / 100, 0.0), 1.0))
    st.caption("Expected probability after improving weak features")
    st.progress(min(max(improved_probability / 100, 0.0), 1.0))

    with st.expander("Improved profile used for simulation"):
        display_profile = pd.DataFrame([result["improved_profile"]]).rename(
            columns={
                "cgpa": "CGPA",
                "projects": "Projects",
                "complexity": "Complexity",
                "internship": "Internship",
                "certifications": "Certifications",
                "languages": "Programming Languages",
                "skills": "Technical Skills",
                "domain_name": "Domain",
            }
        )
        st.dataframe(display_profile, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Prediction Report",
        data=build_pdf_report(result),
        file_name="internship_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_history() -> None:
    st.subheader("🕘 Prediction History")
    history = st.session_state["prediction_history"]

    if not history:
        st.caption("No predictions yet. Run the model to start a session history.")
        return

    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def run_prediction(profile: dict, label_encoder, trained_model) -> dict:
    """Run the complete prediction and analysis pipeline."""
    input_df = preprocess_input(profile, label_encoder)
    probability = predict_success(trained_model, input_df)
    suitability, status_class, status_icon = classify_suitability(probability)
    strengths = analyze_strengths(profile)
    gaps = analyze_gaps(profile)
    recommendations = recommend_improvements(profile, gaps)
    improved_profile, improved_df, improved_probability = simulate_profile_improvement(
        profile,
        label_encoder,
        trained_model,
    )

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "input_df": input_df,
        "probability": probability,
        "suitability": suitability,
        "status_class": status_class,
        "status_icon": status_icon,
        "domain_analysis": analyze_domain(profile),
        "strengths": strengths,
        "gaps": gaps,
        "recommendations": recommendations,
        "improved_profile": improved_profile,
        "improved_df": improved_df,
        "improved_probability": improved_probability,
        "improvement_gain": max(improved_probability - probability, 0.0),
    }
    result["profile_scores"] = calculate_profile_scores(profile)
    result["comparison_df"] = build_candidate_comparison(profile)
    result["radar_df"] = build_radar_values(profile)
    result["feature_importance_df"] = get_model_feature_importance(trained_model)
    result["domain_ranking_df"] = rank_all_domains(profile, label_encoder, trained_model)
    result["explainability"] = generate_ai_explainability(result, trained_model)
    return result


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    inject_custom_css()
    initialize_session_state()

    trained_model, label_encoder = load_artifacts()

    render_header()
    profile = render_sidebar_inputs(label_encoder)
    render_profile_summary(profile)

    predict_clicked = st.button("🔮 Predict Internship Success", type="primary", use_container_width=True)
    st.info("The trained Random Forest model stays unchanged. This dashboard only improves app structure and presentation.")

    if predict_clicked:
        result = run_prediction(profile, label_encoder, trained_model)
        st.session_state["prediction_result"] = result
        add_history_entry(result)

    result = st.session_state.get("prediction_result")
    if result is None:
        st.markdown(
            """
            <div class="section-card">
                <h3 style="margin-top:0;">Ready for Prediction</h3>
                <p style="color:#9caeb9;margin-bottom:0;">
                    Adjust the sidebar inputs and click Predict Internship Success to generate the probability,
                    strengths, gaps, recommendations, and improved-profile simulation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_result(result)

    render_history()


if __name__ == "__main__":
    main()
