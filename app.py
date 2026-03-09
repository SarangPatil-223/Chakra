import os
import sys
import json
import time
import threading
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import pydeck as pdk

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ingestion.firecrawl_pipeline import (
    run_pipeline,
    get_or_load_collection,
    _get_fallback_schemes,
)
from agent.fraud_guard import run_agent, generate_fraud_heatmap_data

logger = logging.getLogger(__name__)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YojanaAI — Government Scheme Finder",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Override (India.gov inspired) ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif !important;
}

.stApp {
    background: #f0f4f8;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── Top Nav Bar ── */
.top-nav {
    background: linear-gradient(135deg, #003366 0%, #004a9f 60%, #0066cc 100%);
    padding: 0;
    margin: -1rem -1rem 0 -1rem;
    box-shadow: 0 2px 12px rgba(0,51,102,0.4);
    position: sticky;
    top: 0;
    z-index: 999;
}

.nav-banner {
    background: #ff6600;
    color: white;
    text-align: center;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 0;
    letter-spacing: 0.5px;
}

.nav-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 24px;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.nav-emblem {
    width: 52px;
    height: 52px;
    background: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.nav-title { color: white; line-height: 1.2; }
.nav-title h1 { font-size: 1.4rem; font-weight: 800; margin: 0; letter-spacing: -0.3px; }
.nav-title p { font-size: 0.72rem; margin: 0; opacity: 0.85; font-weight: 400; }

.nav-links {
    display: flex;
    gap: 6px;
    align-items: center;
}

.nav-tag {
    background: rgba(255,255,255,0.15);
    color: white;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 500;
}

.nav-ai-badge {
    background: linear-gradient(135deg, #ff6600, #ff4500);
    color: white;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(255,100,0,0.4);
    animation: pulse-badge 2s infinite;
}

@keyframes pulse-badge {
    0%, 100% { box-shadow: 0 2px 8px rgba(255,100,0,0.4); }
    50% { box-shadow: 0 2px 16px rgba(255,100,0,0.7); }
}

/* ── Hero Section ── */
.hero-section {
    background: linear-gradient(135deg, #003366 0%, #0055a5 50%, #0077cc 100%);
    padding: 36px 24px;
    margin: 0 -1rem 24px -1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(255,255,255,0.05) 0%, transparent 60%);
    animation: shimmer 8s infinite;
}

@keyframes shimmer {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.hero-title {
    color: white;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    letter-spacing: -0.5px;
}

.hero-sub {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 8px 0 0 0;
    font-weight: 400;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.stat-pill {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 24px;
    padding: 8px 20px;
    color: white;
    font-weight: 600;
    font-size: 0.85rem;
    backdrop-filter: blur(8px);
}

.stat-pill span { font-size: 1.1rem; font-weight: 800; color: #ffd700; }

/* ── Scheme Cards ── */
.scheme-card {
    background: white;
    border-radius: 14px;
    padding: 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid #e8edf5;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
    height: 100%;
}

.scheme-card:hover {
    box-shadow: 0 8px 32px rgba(0,51,102,0.18);
    transform: translateY(-4px);
    border-color: #003366;
}

.card-header {
    background: linear-gradient(135deg, #003366, #0055a5);
    padding: 14px 16px;
}

.card-category-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.card-title {
    color: white;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.3;
}

.card-body { padding: 14px 16px; }

.card-summary {
    color: #4a5568;
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 0 0 12px 0;
}

.card-meta {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.meta-tag {
    background: #f0f4f8;
    color: #003366;
    border-radius: 8px;
    padding: 3px 8px;
    font-size: 0.68rem;
    font-weight: 500;
}

.eligibility-bar-wrap { margin-bottom: 12px; }
.elig-label { font-size: 0.72rem; color: #6b7280; font-weight: 500; margin-bottom: 4px; }
.elig-bar-bg {
    background: #e5e7eb;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.elig-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 1s ease;
}

.btn-check {
    display: block;
    width: 100%;
    background: linear-gradient(135deg, #003366, #0055a5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 0;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    text-decoration: none;
    transition: all 0.2s;
    letter-spacing: 0.3px;
}

.btn-check:hover {
    background: linear-gradient(135deg, #002244, #003d82);
    box-shadow: 0 4px 12px rgba(0,51,102,0.4);
    color: white;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}

.section-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #003366;
    margin: 0;
}

.section-divider {
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #003366, transparent);
    border-radius: 2px;
}

/* ── Chat (Assistant) ── */
.chat-container {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border: 1px solid #e8edf5;
    overflow: hidden;
}

.chat-header {
    background: linear-gradient(135deg, #003366, #0055a5);
    color: white;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.chat-header h3 { margin: 0; font-size: 1rem; font-weight: 700; }
.chat-status {
    font-size: 0.72rem;
    opacity: 0.8;
    background: rgba(255,255,255,0.15);
    padding: 2px 10px;
    border-radius: 10px;
}

.chat-body { padding: 16px 20px; min-height: 200px; max-height: 420px; overflow-y: auto; }

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 12px;
}

.msg-user .bubble {
    background: linear-gradient(135deg, #003366, #0055a5);
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    max-width: 75%;
    font-size: 0.85rem;
    line-height: 1.5;
}

.msg-bot {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    align-items: flex-start;
}

.bot-avatar {
    width: 32px;
    height: 32px;
    min-width: 32px;
    background: linear-gradient(135deg, #ff6600, #ff4500);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.msg-bot .bubble {
    background: #f0f4f8;
    color: #1a202c;
    border-radius: 4px 16px 16px 16px;
    padding: 10px 14px;
    max-width: 80%;
    font-size: 0.85rem;
    line-height: 1.6;
    border: 1px solid #e2e8f0;
}

.fraud-alert-bubble {
    background: linear-gradient(135deg, #fff1f0, #ffe4e4) !important;
    border: 2px solid #ff4444 !important;
    border-left: 4px solid #cc0000 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 2px solid #e8edf5;
}

section[data-testid="stSidebar"] .block-container {
    padding: 0 16px;
}

.sidebar-logo {
    background: linear-gradient(135deg, #003366, #0055a5);
    margin: -1rem -16px 0rem -16px;
    padding: 20px 16px;
    text-align: center;
    color: white;
}

.sidebar-logo h2 { margin: 0; font-size: 1.3rem; font-weight: 800; }
.sidebar-logo p { margin: 4px 0 0 0; font-size: 0.72rem; opacity: 0.8; }

.filter-section-title {
    color: #003366;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 16px 0 8px 0;
    padding-left: 4px;
}

/* ── Fraud Heatmap ── */
.heatmap-container {
    background: white;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* ── Streamlit widget overrides ── */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stTextInput label {
    font-weight: 600 !important;
    color: #003366 !important;
    font-size: 0.82rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #003366, #0055a5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #002244, #003d82) !important;
    box-shadow: 0 4px 12px rgba(0,51,102,0.4) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stChatInput"] > div { border: 2px solid #003366 !important; border-radius: 10px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: #003366;
}
.stTabs [aria-selected="true"] {
    background: #003366 !important;
    color: white !important;
    border-radius: 6px 6px 0 0!important;
}

/* ── Profile Form ── */
.profile-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e8edf5;
    margin-bottom: 16px;
}

/* ── Footer ── */
.gov-footer {
    background: #003366;
    color: rgba(255,255,255,0.7);
    text-align: center;
    padding: 20px;
    margin: 32px -1rem -1rem -1rem;
    font-size: 0.75rem;
    line-height: 1.8;
}

.gov-footer a { color: #66aaff; text-decoration: none; }

/* ── Animations ── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.fade-in { animation: fadeIn 0.4s ease forwards; }
</style>
""", unsafe_allow_html=True)



def init_state():
    defaults = {
        "schemes": [],
        "collection": None,
        "chat_history": [],
        "user_profile": {},
        "synced": False,
        "fraud_events": generate_fraud_heatmap_data(),
        "recording": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ════════════════════════════════════════════════════════════════
# Top Navigation Bar
# ════════════════════════════════════════════════════════════════
lang_map = {
    "🇬🇧 English": "en",
    "🇮🇳 हिंदी": "hi",
    "🌴 தமிழ்": "ta",
    "🟠 मराठी": "mr",
    "🌺 తెలుగు": "te",
    "🐟 বাংলা": "bn",
}

st.markdown("""
<div class="top-nav">
  <div class="nav-banner">🇮🇳 &nbsp; भारत सरकार &nbsp;|&nbsp; Government of India &nbsp;|&nbsp; YojanaAI — Powered by Firecrawl · Gemini · LangGraph</div>
  <div class="nav-main">
    <div class="nav-brand">
      <div class="nav-emblem">🏛️</div>
      <div class="nav-title">
        <h1>YojanaAI</h1>
        <p>सरकारी योजना खोजक &nbsp;|&nbsp; Government Scheme Finder</p>
      </div>
    </div>
    <div class="nav-links">
      <span class="nav-tag">📋 10 Schemes</span>
      <span class="nav-tag">🌐 8 Languages</span>
      <span class="nav-tag">🛡️ Fraud Guard</span>
      <span class="nav-ai-badge">✨ AI Powered</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <h2>🏛️ YojanaAI</h2>
      <p>Smart Scheme Finder for Every Citizen</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-section-title">🌐 Language</div>', unsafe_allow_html=True)
    selected_lang = st.selectbox("Select Language", list(lang_map.keys()), label_visibility="collapsed")

    st.markdown('<div class="filter-section-title">🔍 Scheme Filters</div>', unsafe_allow_html=True)
    cat_filter = st.multiselect(
        "Category",
        ["Farmers", "Education", "Women", "Health", "Senior Citizens", "Youth", "Business", "Housing", "General"],
        placeholder="All categories",
    )
    state_filter = st.selectbox(
        "State / UT",
        ["All India", "Andhra Pradesh", "Bihar", "Gujarat", "Karnataka", "Kerala", "Maharashtra",
         "Madhya Pradesh", "Punjab", "Rajasthan", "Tamil Nadu", "Uttar Pradesh", "West Bengal"],
    )
    income_filter = st.multiselect(
        "Income Level",
        ["Below Poverty Line", "Low Income", "Middle Income", "All"],
        placeholder="Any income",
    )

    st.markdown('<div class="filter-section-title">👤 My Profile</div>', unsafe_allow_html=True)
    with st.expander("Set Your Profile", expanded=False):
        occ = st.selectbox("Occupation", ["Farmer", "Student", "Business Owner", "Salaried", "Homemaker", "Senior Citizen", "Youth"])
        inc = st.selectbox("Annual Income", ["Below ₹1 Lakh", "₹1L–₹3L", "₹3L–₹6L", "₹6L–₹12L", "Above ₹12L"])
        age = st.slider("Age", 18, 80, 30)
        gender = st.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
        state_prf = st.text_input("State", "Maharashtra")
        if st.button("💾 Save Profile"):
            st.session_state.user_profile = {
                "occupation": occ,
                "income": inc,
                "age": age,
                "gender": gender,
                "state": state_prf,
            }
            st.success("Profile saved!")

    st.markdown('<div class="filter-section-title">🔄 Data Sync</div>', unsafe_allow_html=True)
    if st.button("🔄 Sync Live Schemes", use_container_width=True):
        with st.spinner("Syncing from india.gov.in via Firecrawl…"):
            progress = st.progress(0)
            status = st.empty()

            def cb(pct, msg):
                progress.progress(pct)
                status.text(msg)

            try:
                schemes, collection = run_pipeline(cb)
                st.session_state.schemes = schemes
                st.session_state.collection = collection
                st.session_state.synced = True
                st.success(f"✅ {len(schemes)} schemes synced!")
            except Exception as e:
                st.error(f"Sync failed: {e}")
                st.info("Loading offline data instead…")
                st.session_state.schemes = _get_fallback_schemes()

    if not st.session_state.schemes:
        st.session_state.schemes = _get_fallback_schemes()

    st.markdown("---")
    st.caption("🔒 Secured by Fraud-Guard AI\n📊 ChromaDB Vector Store\n🤖 Gemini 1.5 Flash + LangGraph")


# ════════════════════════════════════════════════════════════════
# Hero Section
# ════════════════════════════════════════════════════════════════
n_schemes = len(st.session_state.schemes)
n_cats = len({s.get("category") for s in st.session_state.schemes})

st.markdown(f"""
<div class="hero-section">
  <p class="hero-title">🏛️ Find Your Government Benefit</p>
  <p class="hero-sub">Discover schemes you're entitled to — in your language, instantly.</p>
  <div class="hero-stats">
    <div class="stat-pill"><span>{n_schemes}+</span> Active Schemes</div>
    <div class="stat-pill"><span>{n_cats}</span> Categories</div>
    <div class="stat-pill"><span>8</span> Languages</div>
    <div class="stat-pill"><span>AI</span> Fraud Protection</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Main Tabs
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📋 Browse Schemes", "🤖 AI Assistant + Voice", "🗺️ Fraud Heatmap"])


# ─────────────────────────────────────────────
# TAB 1: Scheme Grid
# ─────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header"><h2>📋 Available Schemes</h2><div class="section-divider"></div></div>', unsafe_allow_html=True)

    # Filter
    schemes = st.session_state.schemes
    if cat_filter:
        schemes = [s for s in schemes if s.get("category") in cat_filter]
    if income_filter:
        schemes = [s for s in schemes if s.get("income_level") in income_filter]
    if state_filter != "All India":
        schemes = [s for s in schemes if s.get("state") in ("All India", state_filter)]

    if not schemes:
        st.info("No schemes match your filters. Try widening the selection.")
    else:
        st.caption(f"Showing **{len(schemes)}** scheme(s)")

        # ── Eligibility score lookup
        profile = st.session_state.user_profile
        occ_key = profile.get("occupation", "").lower()
        inc_key = profile.get("income", "").lower()

        CAT_COLOR = {
            "Farmers": "#2d6a4f",
            "Education": "#1d3557",
            "Women": "#7b2d8b",
            "Health": "#c0392b",
            "Senior Citizens": "#6c5ce7",
            "Youth": "#0077cc",
            "Business": "#d35400",
            "Housing": "#2980b9",
            "General": "#003366",
        }

        CAT_ICON = {
            "Farmers": "🌾",
            "Education": "🎓",
            "Women": "♀️",
            "Health": "❤️",
            "Senior Citizens": "👴",
            "Youth": "⚡",
            "Business": "💼",
            "Housing": "🏠",
            "General": "🇮🇳",
        }

        CATEG_OCCUPATION_MAP = {
            "Farmer": "Farmers",
            "Student": "Education",
            "Business Owner": "Business",
            "Senior Citizen": "Senior Citizens",
            "Youth": "Youth",
        }

        def compute_score(scheme):
            base = 65
            occ_label = profile.get("occupation", "")
            if CATEG_OCCUPATION_MAP.get(occ_label) == scheme.get("category"):
                base += 22
            if "bpl" in inc_key or "1 lakh" in inc_key:
                if scheme.get("income_level") in ("Below Poverty Line", "Low Income"):
                    base += 8
            return min(base, 97)

        def score_to_color(s):
            if s >= 85:
                return "#16a34a"
            if s >= 65:
                return "#d97706"
            return "#dc2626"

        COLS_PER_ROW = 3
        for row_start in range(0, len(schemes), COLS_PER_ROW):
            row_schemes = schemes[row_start:row_start + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for col, scheme in zip(cols, row_schemes):
                cat = scheme.get("category", "General")
                color = CAT_COLOR.get(cat, "#003366")
                icon = CAT_ICON.get(cat, "🇮🇳")
                score = compute_score(scheme)
                sc = score_to_color(score)

                with col:
                    st.markdown(f"""
                    <div class="scheme-card fade-in">
                      <div class="card-header" style="background: linear-gradient(135deg, {color}, {color}cc);">
                        <div class="card-category-badge">{icon} {cat}</div>
                        <p class="card-title">{scheme.get("scheme_name", "Unknown Scheme")}</p>
                      </div>
                      <div class="card-body">
                        <p class="card-summary">{scheme.get("summary", scheme.get("benefits","")[:100])}</p>
                        <div class="card-meta">
                          <span class="meta-tag">📍 {scheme.get("state","All India")}</span>
                          <span class="meta-tag">💰 {scheme.get("income_level","All")}</span>
                        </div>
                        <div class="eligibility-bar-wrap">
                          <div class="elig-label">Your Eligibility Score: <strong style="color:{sc}">{score}%</strong></div>
                          <div class="elig-bar-bg">
                            <div class="elig-bar-fill" style="width:{score}%; background: linear-gradient(90deg, {sc}aa, {sc});"></div>
                          </div>
                        </div>
                        <a href="{scheme.get("application_link","#")}" target="_blank" class="btn-check">✅ Check Eligibility & Apply</a>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")  # Spacing


# ─────────────────────────────────────────────
# TAB 2: AI Assistant + Voice
# ─────────────────────────────────────────────
with tab2:
    col_chat, col_profile = st.columns([3, 1])

    with col_profile:
        st.markdown("### 👤 Profile Summary")
        pf = st.session_state.user_profile
        if pf:
            st.markdown(f"""
            <div class="profile-card">
              <p>👔 <strong>Occupation:</strong> {pf.get("occupation","—")}</p>
              <p>💰 <strong>Income:</strong> {pf.get("income","—")}</p>
              <p>🎂 <strong>Age:</strong> {pf.get("age","—")}</p>
              <p>⚧ <strong>Gender:</strong> {pf.get("gender","—")}</p>
              <p>📍 <strong>State:</strong> {pf.get("state","—")}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Set your profile in the sidebar to get personalized results.")

        st.markdown("### 🛡️ Fraud-Guard")
        st.markdown("""
        <div style="background:#fff8f0;border:1px solid #ff6600;border-radius:10px;padding:12px;font-size:0.8rem;">
          <b>🔴 Patterns Monitored:</b><br/>
          • PIN/OTP requests<br/>
          • Bribe/commission offers<br/>
          • Fake portal links<br/>
          • Threatening language<br/>
          • Fake prize/lottery claims
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💬 Try These:")
        example_queries = [
            "I am a farmer with 2 acres, what schemes can help me?",
            "मैं एक छात्र हूँ, मुझे scholarship चाहिए",
            "நான் ஒரு விவசாயி, என்னுடைய வருமானம் குறைவாக உள்ளது",
            "What health insurance is available for poor families?",
            "I have a small business, need a loan without collateral",
        ]
        for q in example_queries:
            if st.button(q[:45] + ("…" if len(q) > 45 else ""), key=f"ex_{hash(q)}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Fraud-Guard checking… then routing to AI…"):
                    result = run_agent(q, st.session_state.user_profile)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["response"],
                        "is_fraud": result.get("is_fraud", False),
                        "steps": result.get("intermediate_steps", []),
                    })
                st.rerun()

    with col_chat:
        st.markdown("""
        <div class="chat-container">
          <div class="chat-header">
            <span style="font-size:1.5rem">🤖</span>
            <div>
              <h3>SarkariBot — Your AI Assistant</h3>
              <span class="chat-status">🟢 Online · Fraud-Guard Active · Multilingual</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Render chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user fade-in">
                  <div class="bubble">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                fraud_class = "fraud-alert-bubble" if msg.get("is_fraud") else ""
                content = msg["content"].replace("\n", "<br>")
                st.markdown(f"""
                <div class="msg-bot fade-in">
                  <div class="bot-avatar">{"🚨" if msg.get("is_fraud") else "🤖"}</div>
                  <div class="bubble {fraud_class}">{content}</div>
                </div>""", unsafe_allow_html=True)

                # Show reasoning steps
                if msg.get("steps"):
                    with st.expander("🔍 Agent Reasoning"):
                        for step in msg["steps"]:
                            st.caption(step)

        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#9ca3af;">
              <div style="font-size:3rem">🏛️</div>
              <p style="font-size:1rem;font-weight:600;color:#003366;">SarkariBot is ready!</p>
              <p style="font-size:0.85rem">Ask about any scheme in <strong>any Indian language</strong>.</p>
              <p style="font-size:0.8rem;color:#ef4444;">⚠️ Try asking for a "bribe" or "PIN" — watch Fraud-Guard respond!</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Voice Input
        st.markdown("---")
        vcol1, vcol2 = st.columns([1, 3])
        with vcol1:
            voice_btn = st.button("🎤 Speak Query", use_container_width=True, key="voice_btn")
        with vcol2:
            st.caption("Click 🎤 to record 6 seconds of audio. Whisper will transcribe it.")

        if voice_btn:
            try:
                from agent.voice_input import record_and_transcribe
                with st.spinner("🎤 Recording 6 seconds… speak now!"):
                    transcript = record_and_transcribe(duration=6)
                if transcript:
                    st.success(f"📝 Transcribed: *{transcript}*")
                    st.session_state.chat_history.append({"role": "user", "content": transcript})
                    with st.spinner("Processing…"):
                        result = run_agent(transcript, st.session_state.user_profile)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["response"],
                            "is_fraud": result.get("is_fraud", False),
                            "steps": result.get("intermediate_steps", []),
                        })
                    st.rerun()
                else:
                    st.warning("No speech detected. Please try again.")
            except Exception as e:
                st.error(f"Voice input error: {e}")
                st.info("Ensure a microphone is connected and 'openai-whisper' + 'sounddevice' are installed.")

        # ── Text Input
        if user_query := st.chat_input("Type your question here (English, Hindi, Tamil, Marathi…)"):
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.spinner("🛡️ Fraud-Guard scanning… 🌐 Detecting language… 📋 Checking eligibility…"):
                result = run_agent(user_query, st.session_state.user_profile)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "is_fraud": result.get("is_fraud", False),
                "steps": result.get("intermediate_steps", []),
            })
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────
# TAB 3: Fraud Heatmap
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header"><h2>🗺️ Live Fraud Attempt Heatmap</h2><div class="section-divider"></div></div>', unsafe_allow_html=True)
    st.caption("Visualizing scam attempt patterns across Indian cities — helping governance accountability.")

    fraud_data = st.session_state.fraud_events
    df = pd.DataFrame(fraud_data)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Attempts", len(df), delta="+12 today", delta_color="inverse")
    k2.metric("High Risk Cities", df["city"].nunique())
    fraud_types = df["type"].value_counts()
    k3.metric("Top Fraud Type", fraud_types.index[0])
    k4.metric("Avg Severity", f"{df['weight'].mean():.1f}/10")

    # PyDeck Heatmap
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["lon", "lat"],
        get_weight="weight",
        radiusPixels=60,
        intensity=1.5,
        threshold=0.2,
        colorRange=[
            [0, 0, 255, 0],
            [0, 150, 255, 80],
            [0, 255, 200, 150],
            [255, 255, 0, 200],
            [255, 130, 0, 230],
            [255, 0, 0, 255],
        ],
    )

    view = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4.5, pitch=40)

    st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "{city}\nType: {type}\nSeverity: {weight}/10"},
    ))
    st.markdown("</div>", unsafe_allow_html=True)

    # Breakdown table
    st.markdown("### 📊 Fraud Attempt Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        by_type = df.groupby("type")["weight"].sum().sort_values(ascending=False).reset_index()
        by_type.columns = ["Fraud Type", "Total Severity Score"]
        st.dataframe(by_type, use_container_width=True, hide_index=True)
    with c2:
        by_city = df.groupby("city").agg(
            Attempts=("weight", "count"), Avg_Severity=("weight", "mean")
        ).sort_values("Attempts", ascending=False).reset_index()
        by_city["Avg_Severity"] = by_city["Avg_Severity"].round(1)
        st.dataframe(by_city, use_container_width=True, hide_index=True)

    st.info("🛡️ This data is used by Fraud-Guard to prioritize region-specific scam warnings. All data shown is synthetic for demo purposes.")


# ════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="gov-footer">
  <strong>🏛️ YojanaAI — Government Scheme Finder</strong><br/>
  Built for Chakravyuh 2.0 Hackathon &nbsp;|&nbsp; Powered by <strong>Firecrawl · Gemini 1.5 Flash · LangGraph · ChromaDB · Whisper</strong><br/>
  Data sourced from <a href="https://india.gov.in" target="_blank">india.gov.in</a> &nbsp;|&nbsp;
  Report Fraud: <strong>1930</strong> &nbsp;|&nbsp;
  <a href="https://cybercrime.gov.in" target="_blank">cybercrime.gov.in</a><br/>
  <em style="font-size:0.68rem">Disclaimer: This is a demonstration application. Always verify scheme details on official government portals.</em>
</div>
""", unsafe_allow_html=True)
