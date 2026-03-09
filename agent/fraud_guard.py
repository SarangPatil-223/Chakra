import os
import re
import json
import logging
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

from google import genai
from langgraph.graph import StateGraph, END
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "govt_schemes"

# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    user_input: str
    user_profile: dict
    detected_language: str
    is_fraud: bool
    fraud_reason: str
    fraud_severity: str          # "HIGH" | "MEDIUM" | "LOW"
    response: str
    matched_schemes: list[dict]
    eligibility_scores: list[dict]
    intermediate_steps: Annotated[list, lambda x, y: x + y]


# ─────────────────────────────────────────────
# Fraud Pattern Database
# ─────────────────────────────────────────────
FRAUD_PATTERNS = {
    "pin_otp": (
        r"\b(pin|otp|password|cvv|atm\s*pin|bank\s*pin|secret\s*code|passcode)\b",
        "Requesting banking credentials (PIN/OTP/CVV) — legitimate schemes NEVER ask for this.",
        "HIGH"
    ),
    "bribe": (
        r"\b(bribe|bakshish|commission|extra\s*fee|speed\s*money|cut|kickback|payment\s*for\s*approval)\b",
        "Bribe/commission detected — government approvals are free and follow due process.",
        "HIGH"
    ),
    "suspicious_link": (
        r"https?://(?!(?:www\.)?(india\.gov\.in|pmjay\.gov\.in|pmkisan\.gov\.in|scholarships\.gov\.in|pmjdy\.gov\.in|mudra\.org\.in|npscra\.nsdl\.co\.in|pmkvyofficial\.org|pmaymis\.gov\.in|pmfby\.gov\.in|wcd\.nic\.in))\S+",
        "Suspicious external link detected — only trust official .gov.in domains.",
        "HIGH"
    ),
    "threat": (
        r"\b(blackmail|threaten|expose|leak|report\s*you|arrest\s*you|legal\s*action\s*if\s*not)\b",
        "Threatening language detected — this is a potential extortion scam.",
        "HIGH"
    ),
    "fake_offer": (
        r"\b(guaranteed\s*approval|instant\s*money|lottery|won\s*prize|free\s*cash|double\s*your|get\s*rich)\b",
        "Fraudulent promise detected — no legitimate scheme guarantees instant money or prizes.",
        "MEDIUM"
    ),
    "personal_info_request": (
        r"\b(aadhar\s*number|pan\s*number|send\s*photo|send\s*document|whatsapp\s*your|share\s*your\s*bank)\b",
        "Unsolicited personal document request — share documents only on official portals.",
        "MEDIUM"
    ),
}

# ─────────────────────────────────────────────
# Helper: Gemini Client
# ─────────────────────────────────────────────
def _get_gemini():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured.")
    return genai.Client(api_key=GEMINI_API_KEY)


def _get_collection() -> Optional[chromadb.Collection]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Node 1: Fraud Detection
# ─────────────────────────────────────────────
def fraud_detection_node(state: AgentState) -> AgentState:
    """Pattern-based + AI-assisted fraud detection."""
    text = state["user_input"].lower()
    
    for pattern_name, (pattern, reason, severity) in FRAUD_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("FRAUD DETECTED [%s]: %s", pattern_name, reason)
            return {
                **state,
                "is_fraud": True,
                "fraud_reason": reason,
                "fraud_severity": severity,
                "response": _build_fraud_alert(reason, severity),
                "intermediate_steps": [f"🚨 Fraud pattern matched: {pattern_name}"],
            }

    # AI-layer: double-check with Gemini
    try:
        model = _get_gemini()
        prompt = f"""You are a government scam detection AI. Analyze this message for fraud indicators:
        
Message: "{state['user_input']}"

Look for: fake government agents, fee demands, credential harvesting, fake promises.
Reply in JSON: {{"is_fraud": true/false, "reason": "...", "severity": "HIGH/MEDIUM/LOW"}}
If not fraud: {{"is_fraud": false, "reason": "", "severity": "NONE"}}"""

        resp = model.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text[text.find("{"):text.rfind("}") + 1]
        result = json.loads(text)
        
        if result.get("is_fraud") and result.get("severity") in ("HIGH", "MEDIUM"):
            return {
                **state,
                "is_fraud": True,
                "fraud_reason": result.get("reason", "AI-detected fraud pattern"),
                "fraud_severity": result.get("severity", "MEDIUM"),
                "response": _build_fraud_alert(result.get("reason", ""), result.get("severity", "MEDIUM")),
                "intermediate_steps": ["🚨 AI Fraud Layer: suspicious content flagged"],
            }
    except Exception as e:
        logger.warning("AI fraud check failed: %s", e)

    return {
        **state,
        "is_fraud": False,
        "fraud_reason": "",
        "fraud_severity": "NONE",
        "intermediate_steps": ["✅ Fraud check passed"],
    }


def _build_fraud_alert(reason: str, severity: str) -> str:
    emoji = "🔴" if severity == "HIGH" else "🟡"
    return f"""
{emoji} **FRAUD ALERT — {severity} RISK**

⚠️ **Suspicious content detected in your message.**

**Reason:** {reason}

**What to do:**
- 🚫 Do NOT share any personal information, OTPs, or bank details with anyone.
- 📞 Report to **Cyber Crime Helpline: 1930** or visit [cybercrime.gov.in](https://cybercrime.gov.in)
- ✅ All genuine government schemes are **FREE** and applied on official `.gov.in` portals.
- 🏛️ Verify schemes at [india.gov.in](https://india.gov.in) only.

*This conversation has been flagged. Legitimate government officials never ask for fees or personal PINs.*
"""


# ─────────────────────────────────────────────
# Node 2: Language Manager
# ─────────────────────────────────────────────
LANGUAGE_PROMPTS = {
    "hindi": "आप एक सरकारी योजना सहायक हैं। सरल हिंदी में उत्तर दें।",
    "tamil": "நீங்கள் ஒரு அரசு திட்ட உதவியாளர். எளிய தமிழில் பதிலளிக்கவும்.",
    "marathi": "तुम्ही एक सरकारी योजना सहाय्यक आहात. सोप्या मराठीत उत्तर द्या.",
    "telugu": "మీరు ప్రభుత్వ పథకం సహాయకుడు. సరళమైన తెలుగులో సమాధానం ఇవ్వండి.",
    "bengali": "আপনি একজন সরকারি প্রকল্প সহায়তাকারী। সহজ বাংলায় উত্তর দিন।",
    "kannada": "ನೀವು ಸರ್ಕಾರಿ ಯೋಜನೆ ಸಹಾಯಕರು. ಸರಳ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "gujarati": "તમે સરકારી યોજના સહાયક છો. સરળ ગુજરાતીમાં જવાબ આપો.",
    "english": "You are a helpful Government Scheme Assistant. Respond in clear, simple English.",
}


def language_manager_node(state: AgentState) -> AgentState:
    """Detect user language and prepare language-aware context."""
    if state.get("is_fraud"):
        return state  # Skip if fraud detected

    try:
        model = _get_gemini()
        detect_prompt = f"""Detect the language of this text. Reply with ONLY one word from this list:
hindi, tamil, marathi, telugu, bengali, kannada, gujarati, english

Text: "{state['user_input']}"
"""
        resp = model.models.generate_content(model="gemini-1.5-flash", contents=detect_prompt)
        lang = resp.text.strip().lower()
        if lang not in LANGUAGE_PROMPTS:
            lang = "english"
    except Exception:
        lang = "english"

    logger.info("Detected language: %s", lang)
    return {
        **state,
        "detected_language": lang,
        "intermediate_steps": [f"🌐 Language detected: {lang}"],
    }


# ─────────────────────────────────────────────
# Node 3: Eligibility Logic
# ─────────────────────────────────────────────
def eligibility_node(state: AgentState) -> AgentState:
    """Query ChromaDB and compute eligibility scores using Gemini."""
    if state.get("is_fraud"):
        return state

    user_input = state["user_input"]
    profile = state.get("user_profile", {})
    lang = state.get("detected_language", "english")
    
    # 1. Semantic search in ChromaDB
    matched_schemes = []
    collection = _get_collection()
    if collection:
        try:
            results = collection.query(
                query_texts=[user_input],
                n_results=5,
                include=["metadatas", "distances"],
            )
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                scheme = dict(meta)
                scheme["_similarity_score"] = round(max(0, (1 - dist) * 100), 1)
                matched_schemes.append(scheme)
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)

    # 2. Gemini eligibility analysis
    try:
        model = _get_gemini()
        system_prompt = LANGUAGE_PROMPTS.get(lang, LANGUAGE_PROMPTS["english"])
        
        schemes_text = json.dumps(
            [{"name": s.get("scheme_name"), "eligibility": s.get("eligibility_criteria"), "benefits": s.get("benefits")}
             for s in matched_schemes[:3]], ensure_ascii=False
        )
        
        profile_text = json.dumps(profile, ensure_ascii=False) if profile else "Not provided"

        prompt = f"""{system_prompt}

User Query: {user_input}
User Profile: {profile_text}
Relevant Government Schemes: {schemes_text}

Task:
1. Understand the user's situation and needs.
2. For each scheme, estimate eligibility as a percentage (0-100%).
3. Explain in simple terms which schemes apply and why.
4. Give clear next steps to apply.
5. Be empathetic and encouraging — many users are first-time beneficiaries.

Format your response clearly with scheme names bolded."""

        resp = model.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        ai_response = resp.text

    except Exception as e:
        logger.error("Gemini eligibility analysis failed: %s", e)
        ai_response = _fallback_response(matched_schemes, lang)

    # 3. Compute eligibility scores
    eligibility_scores = _compute_eligibility_scores(matched_schemes, profile)

    return {
        **state,
        "matched_schemes": matched_schemes,
        "eligibility_scores": eligibility_scores,
        "response": ai_response,
        "intermediate_steps": [f"📋 Matched {len(matched_schemes)} schemes, computed eligibility"],
    }


def _compute_eligibility_scores(schemes: list[dict], profile: dict) -> list[dict]:
    """Heuristic eligibility scoring based on profile."""
    scores = []
    category_map = {
        "farmer": "Farmers",
        "student": "Education",
        "women": "Women",
        "health": "Health",
        "senior": "Senior Citizens",
        "business": "Business",
        "youth": "Youth",
    }

    occupation = profile.get("occupation", "").lower()
    income = profile.get("income", "").lower()

    for scheme in schemes:
        base_score = scheme.get("_similarity_score", 70)

        # Boost for category match
        for key, cat in category_map.items():
            if key in occupation and scheme.get("category") == cat:
                base_score = min(base_score + 15, 98)

        # Income alignment
        if "low income" in income or "bpl" in income:
            if scheme.get("income_level") in ("Below Poverty Line", "Low Income"):
                base_score = min(base_score + 10, 98)

        scores.append({
            "scheme_name": scheme.get("scheme_name", "Unknown"),
            "score": round(base_score),
            "category": scheme.get("category", "General"),
        })

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def _fallback_response(schemes: list[dict], lang: str) -> str:
    if not schemes:
        return "I found no matching schemes. Please try describing your situation in more detail."
    names = [s.get("scheme_name", "Unknown") for s in schemes[:3]]
    return f"Based on your query, these schemes may be relevant: {', '.join(names)}. Please visit india.gov.in for complete details."


# ─────────────────────────────────────────────
# Route Logic
# ─────────────────────────────────────────────
def route_after_fraud(state: AgentState) -> str:
    return END if state.get("is_fraud") else "language_manager"


# ─────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────
def build_agent() -> object:
    graph = StateGraph(AgentState)

    graph.add_node("fraud_detection", fraud_detection_node)
    graph.add_node("language_manager", language_manager_node)
    graph.add_node("eligibility_checker", eligibility_node)

    graph.set_entry_point("fraud_detection")
    graph.add_conditional_edges("fraud_detection", route_after_fraud, {END: END, "language_manager": "language_manager"})
    graph.add_edge("language_manager", "eligibility_checker")
    graph.add_edge("eligibility_checker", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_agent(user_input: str, user_profile: dict = None) -> AgentState:
    """Run the full LangGraph agent pipeline."""
    agent = get_agent()
    initial_state: AgentState = {
        "user_input": user_input,
        "user_profile": user_profile or {},
        "detected_language": "english",
        "is_fraud": False,
        "fraud_reason": "",
        "fraud_severity": "NONE",
        "response": "",
        "matched_schemes": [],
        "eligibility_scores": [],
        "intermediate_steps": [],
    }
    result = agent.invoke(initial_state)
    return result


# ─────────────────────────────────────────────
# Fraud Heatmap Data Generator
# ─────────────────────────────────────────────
def generate_fraud_heatmap_data() -> list[dict]:
    """Dummy georeferenced fraud attempt data for pydeck visualization."""
    import random
    random.seed(42)
    cities = [
        ("Mumbai", 19.0760, 72.8777),
        ("Delhi", 28.6139, 77.2090),
        ("Bangalore", 12.9716, 77.5946),
        ("Chennai", 13.0827, 80.2707),
        ("Hyderabad", 17.3850, 78.4867),
        ("Kolkata", 22.5726, 88.3639),
        ("Pune", 18.5204, 73.8567),
        ("Ahmedabad", 23.0225, 72.5714),
        ("Jaipur", 26.9124, 75.7873),
        ("Lucknow", 26.8467, 80.9462),
        ("Bhopal", 23.2599, 77.4126),
        ("Patna", 25.5941, 85.1376),
    ]
    points = []
    for city, lat, lon in cities:
        count = random.randint(3, 25)
        for _ in range(count):
            points.append({
                "city": city,
                "lat": lat + random.uniform(-0.3, 0.3),
                "lon": lon + random.uniform(-0.3, 0.3),
                "weight": random.randint(1, 10),
                "type": random.choice(["PIN Phishing", "Bribe Request", "Fake Portal", "OTP Fraud", "Fake Agent"]),
            })
    return points
