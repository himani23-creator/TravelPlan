import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings as LCEmbeddings

# Custom embeddings — avoids langchain-huggingface which conflicts with langchain-core 1.x
class STEmbeddings(LCEmbeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
from langchain_groq import ChatGroq

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 RAG Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient header */
.tp-header {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a3d62 50%, #0d1b2e 100%);
    border-radius: 16px;
    padding: 32px 40px;
    text-align: center;
    margin-bottom: 24px;
    border: 1px solid #1a4a70;
}
.tp-header h1 { color: #e8f4ff; font-size: 2.4rem; font-weight: 800;
    margin: 0 0 8px; letter-spacing: -0.5px; }
.tp-header p  { color: #7ba7cc; font-size: 1rem; margin: 0; }
.tp-badge {
    display: inline-block; background: #1e3a5f; color: #5cb8ff;
    border: 1px solid #2a5080; border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; margin-top: 10px;
    font-weight: 600;
}

/* Result card */
.tp-result {
    background: #0d1b2e;
    border: 1px solid #1a4a70;
    border-radius: 14px;
    padding: 28px 32px;
    color: #cdd9ef;
    line-height: 1.8;
}

/* Source badge */
.tp-source {
    display: inline-block; background: #0a2a45; color: #5cb8ff;
    border: 1px solid #1a4a70; border-radius: 20px;
    padding: 4px 14px; font-size: 0.76rem; margin: 3px 3px 0 0;
    font-weight: 600;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0d1b2e;
    border-right: 1px solid #1a3a5c;
}
section[data-testid="stSidebar"] label { color: #7ba7cc !important; font-weight: 600; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select {
    background: #162032 !important;
    color: #e8f4ff !important;
    border: 1px solid #1a4a70 !important;
    border-radius: 8px !important;
}

/* Streamlit button */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important;
    padding: 12px 24px !important; font-size: 1rem !important;
    width: 100% !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Knowledge Base ─────────────────────────────────────────────────────────────
TRAVEL_KB = [
    {"content": """Paris Travel Guide: Eiffel Tower best at dusk, book 2 months ahead.
Louvre needs 4+ hours, closed Tuesdays. Montmartre for crepes at Rue Lepic.
Seine River cruise (Bateaux Mouches) €15. Best foodie areas: 3rd (Le Marais), 11th.
Museum Pass €52/2 days covers 50+ museums. Hidden gem: Palais Royal gardens (free).""",
     "metadata": {"destination": "Paris", "type": "general_guide", "source": "travel_blog"}},

    {"content": """Paris Food & Dining: Breakfast at Du Pain et des Idées bakery (19th arr), opens 7am.
Lunch: Bouillon Pigalle classic French under €15. Dinner: Bistrot Paul Bert (11th) for steak-frites.
Markets: Marché d'Aligre cheapest produce. Wine bars: Le Verre Volé, Septime Cave.
Avoid Champs-Élysées restaurants — overcharge by 40%.""",
     "metadata": {"destination": "Paris", "type": "food", "source": "reddit_travel"}},

    {"content": """Paris Budget 2024: Hotel 3-star €90-130/night. Hostel €30-50/night.
Metro carnet 10 tickets €17.35. Day pass €8.65.
State museums FREE first Sunday each month.
Budget traveller daily: €80-120. Mid-range: €150-250.""",
     "metadata": {"destination": "Paris", "type": "budget", "source": "nomadic_matt"}},

    {"content": """Tokyo Essential: Shibuya Crossing best from Starbucks 2nd floor.
Senso-ji Temple: arrive before 8am. TeamLab Borderless: book 2 months ahead.
Tsukiji Outer Market: best sushi breakfast at 6-7am. Akihabara: 5-7 floors electronics/anime.
Shinjuku Golden Gai: 200+ tiny bars, cover charge ¥500-1000. Day trips: Nikko, Hakone, Kamakura.""",
     "metadata": {"destination": "Tokyo", "type": "general_guide", "source": "lonely_planet"}},

    {"content": """Tokyo Food: Ramen at Ichiran (tonkotsu) and Fuunji (tsukemen).
Conveyor belt sushi: Uobei Shibuya from ¥110. Izakaya: Yurakucho cheap yakitori ¥150/skewer.
7-Eleven and Lawson onigiri (¥150) legitimately delicious.
Depachika: Isetan Shinjuku basement 50+ gourmet stalls.""",
     "metadata": {"destination": "Tokyo", "type": "food", "source": "japan_travel_blog"}},

    {"content": """Tokyo Budget: IC Card Suica/Pasmo load ¥3000-5000.
JR Pass 7-day ¥50,000 worth it for day trips. Capsule hotel ¥3000-5000/night.
Cash culture: withdraw from 7-Eleven ATMs. NEVER tip — considered rude.
Daily spend budget: ¥8000-12000. Mid-range: ¥20000-40000.""",
     "metadata": {"destination": "Tokyo", "type": "budget", "source": "reddit_japantravel"}},

    {"content": """Bali Experiences: Ubud — Tegallalang rice terraces at sunrise.
Seminyak: Ku De Ta beach club for sunset cocktails. Canggu: Echo Beach surfing, La Brisa.
Uluwatu: clifftop Kecak dance at sunset IDR 150000. Temple etiquette: wear sarong.
Best waterfalls: Sekumpul (north Bali) and Tukad Cepung (near Ubud).""",
     "metadata": {"destination": "Bali", "type": "general_guide", "source": "travel_blog"}},

    {"content": """Bali Budget 2024: Scooter rental IDR 60000-80000/day (~$4-5).
Private driver IDR 350000-500000/day. Budget villa with pool Canggu $25-50/night.
Warung meals IDR 25000-50000 ($1.50-3). Visa: 30-day free most nationalities.
Tourist tax $10 arrival fee 2024. Budget daily: $30-50, mid-range: $80-150.""",
     "metadata": {"destination": "Bali", "type": "budget", "source": "nomadic_matt"}},

    {"content": """New York City: Central Park bike rental $15/hour, 6-mile loop.
The Met Museum pay-what-you-wish ($0-$30). Brooklyn Bridge walk free, best views from DUMBO.
High Line Park free. Staten Island Ferry free, best Statue of Liberty views.
Hidden gems: The Morgan Library, Tenement Museum, Governors Island.""",
     "metadata": {"destination": "New York", "type": "general_guide", "source": "nyc_guide"}},

    {"content": """NYC Food: Best pizza Di Fara (Brooklyn, $5/slice), Joe's Pizza Greenwich Village.
Bagels: Absolute Bagels Upper West Side. Chinatown: Joe's Shanghai soup dumplings.
Smorgasburg: weekends in Williamsburg, 100+ food vendors.
Budget eats: Halal Guys 53rd & 6th, Gray's Papaya hot dogs.""",
     "metadata": {"destination": "New York", "type": "food", "source": "eater_ny"}},

    {"content": """Universal Travel Tips: Roll clothes, packing cubes, leave 20% space.
Essential: 20000mAh power bank, universal adapter, travel insurance (World Nomads / SafetyWing).
Cards: Charles Schwab debit (no ATM fees), Chase Sapphire (no foreign fees).
Apps: Google Maps offline, Google Translate camera, XE Currency.""",
     "metadata": {"destination": "general", "type": "tips", "source": "travel_guide"}},

    {"content": """Solo Travel: Safest cities Tokyo, Lisbon, Medellin.
Meeting people: hostel common rooms, walking tours (tip-based).
Female solo travel: Japan, Iceland, Portugal, New Zealand safest.
Digital nomad hubs: Chiang Mai, Medellin, Lisbon, Tbilisi — under $2000/month.""",
     "metadata": {"destination": "general", "type": "solo_travel", "source": "reddit_solotravel"}},
]

RAG_SYSTEM = """\
You are an expert travel planner AI. Create personalized, actionable itineraries.

## Retrieved Travel Knowledge
{context}

---
## Instructions
Using ONLY the retrieved context:
1. Create a detailed day-by-day itinerary
2. Include specific restaurant names, neighbourhoods, insider tips
3. Provide realistic time estimates and logistics
4. Add budget breakdown from retrieved cost info
5. Flag warnings and things to avoid
6. Mark [Research needed] if context doesn't cover something

Format with clear headings: ## Overview | ## Day-by-Day | ## Food & Dining | ## Budget | ## Pro Tips | ## Warnings\
"""

# ── Cached vectorstore (builds once per session) ───────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading knowledge base & embeddings (first time only ~60s)...")
def get_vectorstore():
    embeddings = STEmbeddings("all-MiniLM-L6-v2")
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in TRAVEL_KB]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='tp-header'>
  <h1>🌍 RAG Travel Planner</h1>
  <p>5-Layer Retrieval-Augmented Generation · Semantic search over real travel knowledge</p>
  <span class='tp-badge'>LangChain 1.2.15 · Groq Llama 3.3 70B · ChromaDB · 100% Free</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Trip Settings")

    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key from console.groq.com — no credit card needed",
    )
    st.caption("👆 Get free key at [console.groq.com](https://console.groq.com)")
    st.divider()

    destination = st.text_input("📍 Destination", value="Paris")

    duration = st.slider("📅 Duration (days)", min_value=1, max_value=14, value=4)

    budget = st.selectbox(
        "💰 Budget Level",
        ["budget", "mid-range", "luxury"],
        index=1,
    )
    travel_style = st.selectbox(
        "🎯 Travel Style",
        ["cultural", "adventure", "food", "relaxation", "food and culture", "food and adventure"],
        index=4,
    )
    group_type = st.selectbox(
        "👥 Group Type",
        ["solo", "couple", "family", "group"],
        index=1,
    )
    interests_txt = st.text_input(
        "✨ Special Interests (comma-separated)",
        value="local food, museums, wine bars",
    )
    st.divider()

    plan_btn = st.button("🗺️ Plan My Trip!", use_container_width=True)

    st.markdown("---")
    st.markdown("**📍 Destinations in KB:**")
    st.markdown("Paris · Tokyo · Bali · New York · General Tips")

# ── Main: Results ──────────────────────────────────────────────────────────────
if plan_btn:
    if not groq_key.strip():
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        try:
            # Load vectorstore (cached after first call)
            vs = get_vectorstore()

            with st.spinner("🤖 Generating your personalised itinerary..."):
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=2048,
                    groq_api_key=groq_key.strip(),
                )

                retriever = vs.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4},
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", RAG_SYSTEM),
                    ("human", "{input}"),
                ])

                def fmt(docs):
                    return "\n\n".join(d.page_content for d in docs)

                chain = (
                    RunnableParallel(
                        context=retriever | fmt,
                        raw_docs=retriever,
                        input=RunnablePassthrough(),
                    )
                    | RunnableParallel(
                        answer=prompt | llm | StrOutputParser(),
                        context=lambda x: x["raw_docs"],
                    )
                )

                interests = [i.strip() for i in interests_txt.split(",") if i.strip()]
                query = (
                    f"Plan a {duration}-day {budget} {travel_style} trip to "
                    f"{destination} for {group_type} traveller interested in "
                    f"{', '.join(interests) if interests else 'general sightseeing'}. "
                    "Include restaurants, attractions, budget breakdown, insider tips."
                )

                result = chain.invoke(query)

            # ── Display itinerary ──────────────────────────────────────────────
            st.success("✅ Itinerary generated!")

            col1, col2, col3 = st.columns(3)
            col1.metric("📍 Destination", destination)
            col2.metric("📅 Duration", f"{duration} days")
            col3.metric("💰 Budget", budget.title())

            st.markdown("---")
            st.markdown(result["answer"])

            # ── Source badges ──────────────────────────────────────────────────
            sources = result.get("context", [])
            if sources:
                st.markdown("---")
                st.markdown("#### 📚 Knowledge Sources Retrieved")
                badges = " ".join(
                    f"<span class='tp-source'>📍 {d.metadata.get('destination')} · {d.metadata.get('type')}</span>"
                    for d in sources
                )
                st.markdown(badges, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    # Placeholder when no query yet
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #3a5a7a;'>
      <div style='font-size:4rem; margin-bottom:16px'>✈️</div>
      <h3 style='color:#5cb8ff; margin-bottom:8px'>Ready to plan your trip?</h3>
      <p style='font-size:1rem'>
        Fill in your trip details in the sidebar<br>
        and click <strong style='color:#5cb8ff'>🗺️ Plan My Trip!</strong>
      </p>
      <br>
      <p style='font-size:0.85rem; color:#2a4a6a'>
        Powered by ChromaDB semantic search · Groq Llama 3.3 70B · LangChain 1.2.15 LCEL
      </p>
    </div>
    """, unsafe_allow_html=True)
