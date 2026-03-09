"""
Ingestion Pipeline: Firecrawl → Gemini 1.5 Flash → ChromaDB
Scrapes india.gov.in government schemes and stores structured JSON in ChromaDB.
"""

import os
import json
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types as genai_types
import chromadb
from chromadb.utils import embedding_functions
from firecrawl import FirecrawlApp

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SCHEMES_URL = "https://www.india.gov.in/my-government/schemes"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "govt_schemes"

SCHEMA_PROMPT = """
You are a Government Data Extraction AI. Given the following raw scraped markdown from india.gov.in,
extract ALL government scheme information and return a valid JSON array.

Each scheme object MUST follow this exact schema:
{
  "scheme_name": "Full official name of the scheme",
  "category": "One of: Farmers, Education, Women, Health, Senior Citizens, Youth, Business, Housing, General",
  "state": "All India or specific state name",
  "income_level": "Below Poverty Line / Low Income / Middle Income / All",
  "eligibility_criteria": "Concise eligibility requirements (2-3 sentences)",
  "benefits": "What the scheme provides (2-3 sentences)",
  "application_link": "URL or 'Visit official portal'",
  "summary": "One-sentence summary for UI cards"
}

Return ONLY the JSON array. No preamble, no explanation.

Raw Markdown:
{markdown}
"""


def get_firecrawl_client() -> FirecrawlApp:
    if not FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY not set in environment.")
    return FirecrawlApp(api_key=FIRECRAWL_API_KEY)


def get_gemini_model():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def scrape_schemes(app: FirecrawlApp, progress_callback=None) -> list[dict]:
    """Crawl india.gov.in schemes page and sub-links."""
    logger.info("Starting Firecrawl crawl of %s", SCHEMES_URL)

    if progress_callback:
        progress_callback(0.1, "Initiating Firecrawl crawl…")

    try:
        crawl_result = app.crawl_url(
            SCHEMES_URL,
            params={
                "crawlerOptions": {
                    "maxDepth": 2,
                    "limit": 20,
                    "includes": ["*/my-government/schemes*", "*/schemes/*"],
                },
                "pageOptions": {
                    "onlyMainContent": True,
                    "includeHtml": False,
                },
            },
        )
        pages = crawl_result.get("data", []) if isinstance(crawl_result, dict) else crawl_result
        logger.info("Crawled %d pages", len(pages))
        if progress_callback:
            progress_callback(0.4, f"Crawled {len(pages)} pages successfully.")
        return pages
    except Exception as e:
        logger.error("Firecrawl error: %s", e)
        if progress_callback:
            progress_callback(0.2, f"Firecrawl error: {e}. Using fallback data.")
        return []


def extract_schemes_with_gemini(model, pages: list[dict], progress_callback=None) -> list[dict]:
    """Use Gemini 1.5 Flash to parse raw markdown into structured scheme JSON."""
    all_schemes = []

    if not pages:
        logger.warning("No pages scraped. Returning fallback seed data.")
        return _get_fallback_schemes()

    combined_markdown = "\n\n---PAGE BREAK---\n\n".join(
        p.get("markdown", p.get("content", "")) for p in pages if p.get("markdown") or p.get("content")
    )

    if not combined_markdown.strip():
        return _get_fallback_schemes()

    # Chunk to avoid token limits
    chunk_size = 8000
    chunks = [combined_markdown[i:i + chunk_size] for i in range(0, len(combined_markdown), chunk_size)]
    logger.info("Processing %d text chunks with Gemini", len(chunks))

    for idx, chunk in enumerate(chunks[:5]):  # Limit to 5 chunks
        try:
            if progress_callback:
                progress_callback(0.5 + (idx / len(chunks)) * 0.3, f"Gemini parsing chunk {idx + 1}/{len(chunks)}…")

            prompt = SCHEMA_PROMPT.format(markdown=chunk)
            response = model.models.generate_content(
                model="gemini-1.5-flash", contents=prompt
            )
            text = response.text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text[text.find("["):text.rfind("]") + 1]

            schemes = json.loads(text)
            if isinstance(schemes, list):
                all_schemes.extend(schemes)
            time.sleep(0.5)  # Rate limit
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Chunk %d parsing error: %s", idx, e)
            continue

    return all_schemes if all_schemes else _get_fallback_schemes()


def _get_fallback_schemes() -> list[dict]:
    """Seed data for demo when API is unavailable."""
    return [
        {
            "scheme_name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
            "category": "Farmers",
            "state": "All India",
            "income_level": "Low Income",
            "eligibility_criteria": "Small and marginal farmers with cultivable land. Family must own land in their name. Annual income below ₹6 lakhs.",
            "benefits": "Direct income support of ₹6,000 per year in three equal installments of ₹2,000 each directly to bank accounts.",
            "application_link": "https://pmkisan.gov.in/",
            "summary": "₹6,000/year direct income support for small and marginal farmers.",
        },
        {
            "scheme_name": "PM Fasal Bima Yojana",
            "category": "Farmers",
            "state": "All India",
            "income_level": "All",
            "eligibility_criteria": "All farmers including sharecroppers and tenant farmers growing notified crops.",
            "benefits": "Comprehensive crop insurance at very low premium rates (2% for Kharif, 1.5% for Rabi crops) against natural calamities.",
            "application_link": "https://pmfby.gov.in/",
            "summary": "Low-premium crop insurance against natural disasters and weather events.",
        },
        {
            "scheme_name": "National Scholarship Portal (NSP)",
            "category": "Education",
            "state": "All India",
            "income_level": "Low Income",
            "eligibility_criteria": "Students from minority communities, SC/ST/OBC categories studying in Class 1 to PhD. Family income below ₹2.5 lakhs per year.",
            "benefits": "Scholarships ranging from ₹1,000 to ₹20,000 per year directly credited to student bank accounts.",
            "application_link": "https://scholarships.gov.in/",
            "summary": "Comprehensive scholarship portal for minority, SC/ST, OBC students.",
        },
        {
            "scheme_name": "Beti Bachao Beti Padhao",
            "category": "Women",
            "state": "All India",
            "income_level": "All",
            "eligibility_criteria": "Girl child from birth. All families regardless of income eligible for Sukanya Samriddhi account.",
            "benefits": "Financial education support, Sukanya Samriddhi account with 8.2% interest, media campaigns for girl child protection.",
            "application_link": "https://wcd.nic.in/bbbp-schemes",
            "summary": "Empowering girl child through education and financial security schemes.",
        },
        {
            "scheme_name": "Ayushman Bharat PM-JAY",
            "category": "Health",
            "state": "All India",
            "income_level": "Below Poverty Line",
            "eligibility_criteria": "Bottom 40% of population as per SECC 2011 data. No age, family size, or gender restriction.",
            "benefits": "Health cover up to ₹5 lakh per family per year for secondary and tertiary hospitalization at empanelled hospitals.",
            "application_link": "https://pmjay.gov.in/",
            "summary": "₹5 lakh health insurance cover for 10 crore+ vulnerable families.",
        },
        {
            "scheme_name": "PM Awas Yojana (Urban)",
            "category": "Housing",
            "state": "All India",
            "income_level": "Low Income",
            "eligibility_criteria": "Urban residents without pucca house. EWS (income < ₹3L), LIG (₹3L-₹6L), MIG-I (₹6L-₹12L) categories.",
            "benefits": "Interest subsidy on home loans: 6.5% for EWS/LIG, 4% for MIG-I, 3% for MIG-II. Central assistance of ₹1.5 lakh for EWS.",
            "application_link": "https://pmaymis.gov.in/",
            "summary": "Affordable housing with interest subsidies for urban low-income families.",
        },
        {
            "scheme_name": "Pradhan Mantri Mudra Yojana",
            "category": "Business",
            "state": "All India",
            "income_level": "All",
            "eligibility_criteria": "Non-farm micro/small enterprise. Can be partnership, proprietorship, or individual. No collateral required for Shishu and Kishor loans.",
            "benefits": "Collateral-free loans: Shishu (up to ₹50K), Kishor (₹50K-₹5L), Tarun (₹5L-₹10L) for business expansion.",
            "application_link": "https://mudra.org.in/",
            "summary": "Collateral-free business loans up to ₹10 lakh for micro entrepreneurs.",
        },
        {
            "scheme_name": "Atal Pension Yojana",
            "category": "Senior Citizens",
            "state": "All India",
            "income_level": "Low Income",
            "eligibility_criteria": "Indian citizens aged 18-40 years with a savings bank account. Not an income taxpayer.",
            "benefits": "Guaranteed pension of ₹1,000 to ₹5,000 per month after age 60 based on contribution. Government co-contributes 50% or ₹1,000 per year.",
            "application_link": "https://www.npscra.nsdl.co.in/",
            "summary": "Guaranteed monthly pension plan for unorganized sector workers.",
        },
        {
            "scheme_name": "PM Kaushal Vikas Yojana",
            "category": "Youth",
            "state": "All India",
            "income_level": "All",
            "eligibility_criteria": "Youth aged 15-45 years. School/college dropouts or unemployed youth. Indian citizen.",
            "benefits": "Free skill training in 300+ job roles. Certificate recognized nationally. Monetary reward on successful certification.",
            "application_link": "https://www.pmkvyofficial.org/",
            "summary": "Free skill training and certification for 10 million youth in 300+ trades.",
        },
        {
            "scheme_name": "Pradhan Mantri Jan Dhan Yojana",
            "category": "General",
            "state": "All India",
            "income_level": "Below Poverty Line",
            "eligibility_criteria": "Any Indian citizen without a bank account. No minimum balance required. Available for people above 10 years of age.",
            "benefits": "Zero balance account, RuPay debit card, ₹2 lakh accident insurance, ₹30,000 life insurance, and overdraft up to ₹10,000.",
            "application_link": "https://pmjdy.gov.in/",
            "summary": "Zero-balance bank accounts with insurance & overdraft for the unbanked.",
        },
    ]


def store_in_chromadb(schemes: list[dict], progress_callback=None) -> chromadb.Collection:
    """Store structured scheme data in ChromaDB using SentenceTransformers."""
    if progress_callback:
        progress_callback(0.85, "Initializing ChromaDB…")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete and recreate for fresh sync
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []
    for i, scheme in enumerate(schemes):
        doc_text = (
            f"{scheme.get('scheme_name', '')} {scheme.get('category', '')} "
            f"{scheme.get('eligibility_criteria', '')} {scheme.get('benefits', '')} "
            f"{scheme.get('state', '')} {scheme.get('income_level', '')}"
        )
        documents.append(doc_text)
        metadatas.append({k: str(v) for k, v in scheme.items()})
        ids.append(f"scheme_{i:04d}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Stored %d schemes in ChromaDB.", len(schemes))

    if progress_callback:
        progress_callback(1.0, f"✅ Successfully indexed {len(schemes)} schemes!")

    return collection


def run_pipeline(progress_callback=None) -> tuple[list[dict], chromadb.Collection]:
    """Full pipeline: Firecrawl → Gemini → ChromaDB."""
    logger.info("=== Starting Ingestion Pipeline ===")

    try:
        fc_app = get_firecrawl_client()
        pages = scrape_schemes(fc_app, progress_callback)
    except Exception as e:
        logger.warning("Firecrawl unavailable: %s. Using fallback data.", e)
        if progress_callback:
            progress_callback(0.4, "Using offline seed data (Firecrawl key missing).")
        pages = []

    try:
        model = get_gemini_model()
        schemes = extract_schemes_with_gemini(model, pages, progress_callback)
    except Exception as e:
        logger.warning("Gemini unavailable: %s. Using fallback schemes.", e)
        schemes = _get_fallback_schemes()

    collection = store_in_chromadb(schemes, progress_callback)
    logger.info("=== Pipeline Complete: %d schemes stored ===", len(schemes))
    return schemes, collection


def get_or_load_collection() -> Optional[chromadb.Collection]:
    """Retrieve existing ChromaDB collection without re-scraping."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        return None


if __name__ == "__main__":
    def console_progress(pct, msg):
        print(f"[{pct*100:.0f}%] {msg}")

    schemes, col = run_pipeline(console_progress)
    print(f"\nDone! {len(schemes)} schemes indexed.")
    print("Sample scheme:", json.dumps(schemes[0], indent=2))
