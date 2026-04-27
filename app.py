import solara
import os
import warnings
import threading
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from typing import List, Dict, Any
from dataclasses import dataclass

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
Share itinerary with family, check in daily.
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

Format with headings: Overview | Day-by-Day | Food & Dining | Budget | Pro Tips | Warnings\
"""

# ── Global singletons (lazy-init) ──────────────────────────────────────────────
_embeddings = None
_vectorstore = None

def _get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in TRAVEL_KB]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        _vectorstore = Chroma.from_documents(documents=chunks, embedding=_embeddings)
    return _vectorstore

# ── Reactive state ─────────────────────────────────────────────────────────────
groq_key      = solara.reactive("")
destination   = solara.reactive("Paris")
duration      = solara.reactive(4)
budget        = solara.reactive("mid-range")
style         = solara.reactive("food and culture")
group         = solara.reactive("couple")
interests_txt = solara.reactive("local food, museums, wine bars")
itinerary     = solara.reactive("")
sources       = solara.reactive([])
loading       = solara.reactive(False)
status_msg    = solara.reactive("")
error_msg     = solara.reactive("")

# ── Core logic ─────────────────────────────────────────────────────────────────
def run_rag():
    if not groq_key.value.strip():
        error_msg.value = "⚠️ Please enter your Groq API key first."
        return

    loading.value   = True
    error_msg.value = ""
    itinerary.value = ""
    sources.value   = []

    try:
        status_msg.value = "⏳ Loading knowledge base & embeddings..."
        vs = _get_vectorstore()

        status_msg.value = "🤖 Initialising Groq LLM..."
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2048,
            groq_api_key=groq_key.value.strip(),
        )

        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        prompt    = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM),
            ("human", "{input}"),
        ])

        def fmt(docs): return "\n\n".join(d.page_content for d in docs)

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

        interests = [i.strip() for i in interests_txt.value.split(",") if i.strip()]
        query = (
            f"Plan a {duration.value}-day {budget.value} {style.value} trip to "
            f"{destination.value} for {group.value} traveller interested in "
            f"{', '.join(interests) if interests else 'general sightseeing'}. "
            "Include restaurants, attractions, budget breakdown, insider tips."
        )

        status_msg.value = "🔍 Retrieving context & generating itinerary..."
        result = chain.invoke(query)

        itinerary.value = result["answer"]
        sources.value   = [
            {"destination": d.metadata.get("destination"),
             "type": d.metadata.get("type"),
             "source": d.metadata.get("source")}
            for d in result.get("context", [])
        ]
        status_msg.value = "✅ Done!"

    except Exception as exc:
        error_msg.value  = f"❌ {exc}"
        status_msg.value = ""
    finally:
        loading.value = False
def plan_trip():
    t = threading.Thread(target=run_rag, daemon=True)
    t.start()

CSS = """
body { background: #0f1117; font-family: 'Inter', sans-serif; }

.tp-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d3b5e 50%, #1a1f35 100%);
    border-bottom: 1px solid #2a3a5c;
    padding: 28px 40px 20px;
    text-align: center;
}
.tp-header h1 { font-size: 2.2rem; font-weight: 800; color: #e8f4ff;
    letter-spacing: -0.5px; margin: 0 0 6px; }
.tp-header p  { color: #7ba7cc; font-size: 0.95rem; margin: 0; }

.tp-badge {
    display: inline-block; background: #1e3a5f; color: #5cb8ff;
    border: 1px solid #2a5080; border-radius: 20px;
    padding: 3px 12px; font-size: 0.75rem; margin-top: 8px;
}

.tp-panel {
    background: #161b2e; border: 1px solid #242d45;
    border-radius: 14px; padding: 22px 20px; margin: 16px;
}
.tp-panel-title {
    font-size: 0.8rem; font-weight: 700; color: #5cb8ff;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px;
}

.tp-btn {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #fff !important; font-weight: 700 !important;
    border-radius: 10px !important; padding: 14px !important;
    font-size: 1rem !important; width: 100% !important;
    border: none !important; cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.tp-btn:hover { opacity: 0.88 !important; }
.tp-btn:disabled { opacity: 0.5 !important; cursor: not-allowed !important; }

.tp-result {
    background: #161b2e; border: 1px solid #242d45;
    border-radius: 14px; padding: 28px 30px; margin: 16px;
    color: #cdd9ef; line-height: 1.75;
}
.tp-result h2, .tp-result h3 { color: #5cb8ff; }

.tp-source-badge {
    display: inline-block; background: #0d2d4a; color: #5cb8ff;
    border: 1px solid #1a4a70; border-radius: 20px;
    padding: 4px 14px; font-size: 0.75rem; margin: 4px 4px 0 0;
}

.tp-status {
    color: #7ba7cc; font-size: 0.9rem;
    text-align: center; padding: 10px 0;
}
.tp-error {
    background: #2d1a1a; color: #ff6b6b;
    border: 1px solid #5a2a2a; border-radius: 10px; padding: 12px 16px;
    font-size: 0.9rem; margin: 12px 16px;
}
.tp-placeholder {
    text-align: center; padding: 60px 30px; color: #3a4a6a;
}
.tp-placeholder-icon { font-size: 3rem; margin-bottom: 12px; }
.tp-placeholder p { font-size: 0.95rem; }

.v-input__slot { background: #1e2640 !important; border-radius: 8px !important; }
.v-label { color: #7ba7cc !important; font-size: 0.82rem !important; }
.v-select__selection, .v-text-field input { color: #cdd9ef !important; }
"""


# ── Components ─────────────────────────────────────────────────────────────────
@solara.component
def Header():
    solara.HTML("div", unsafe_innerHTML="""
        <div class='tp-header'>
          <h1>🌍 RAG Travel Planner</h1>
          <p>5-Layer Retrieval-Augmented Generation · Powered by Groq + ChromaDB</p>
          <span class='tp-badge'>LangChain 1.2.15 · Llama 3.3 70B · Free</span>
        </div>
    """)

@solara.component
def InputPanel():
    solara.HTML("div", unsafe_innerHTML="<div class='tp-panel-title'>🗺️ Trip Details</div>")

    solara.InputText(
        label="🔑 Groq API Key (free at console.groq.com)",
        value=groq_key, password=True,
        style="margin-bottom:12px",
    )
    solara.InputText(
        label="📍 Destination",
        value=destination,
        style="margin-bottom:12px",
    )
    solara.SliderInt(
        label=f"📅 Duration: {duration.value} days",
        value=duration, min=1, max=14,
    )
    solara.Select(
        label="💰 Budget Level",
        value=budget,
        values=["budget", "mid-range", "luxury"],
        style="margin-bottom:8px",
    )
    solara.Select(
        label="🎯 Travel Style",
        value=style,
        values=["cultural", "adventure", "food", "relaxation",
                "food and culture", "food and adventure"],
        style="margin-bottom:8px",
    )
    solara.Select(
        label="👥 Group Type",
        value=group,
        values=["solo", "couple", "family", "group"],
        style="margin-bottom:8px",
    )
    solara.InputText(
        label="✨ Special Interests (comma-separated)",
        value=interests_txt,
        style="margin-bottom:16px",
    )
    solara.Button(
        label="🗺️  Plan My Trip!",
        on_click=plan_trip,
        disabled=loading.value,
        class_="tp-btn",
        color="primary",
        style="width:100%;font-weight:700;",
    )

@solara.component
def SourceBadges():
    if not sources.value:
        return
    badges_html = "".join(
        f"<span class='tp-source-badge'>📍 {s['destination']} · {s['type']}</span>"
        for s in sources.value
    )
    solara.HTML("div", unsafe_innerHTML=f"<div style='margin-top:18px'>"
                f"<p style='color:#5cb8ff;font-size:0.8rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>"
                f"📚 Retrieved Sources</p>{badges_html}</div>")

@solara.component
def ResultPanel():
    if loading.value:
        solara.ProgressLinear(indeterminate=True, color="#1565c0")
        solara.HTML("p", unsafe_innerHTML=f"<p class='tp-status'>{status_msg.value}</p>")

    if error_msg.value:
        solara.HTML("div", unsafe_innerHTML=f"<div class='tp-error'>{error_msg.value}</div>")

    if itinerary.value:
        solara.Markdown(itinerary.value)
        SourceBadges()
    elif not loading.value and not error_msg.value:
        solara.HTML("div", unsafe_innerHTML="""
            <div class='tp-placeholder'>
              <div class='tp-placeholder-icon'>✈️</div>
              <p>Fill in your trip details and click<br>
              <strong style='color:#5cb8ff'>🗺️ Plan My Trip!</strong> to generate<br>
              your AI-powered itinerary.</p>
            </div>
        """)


# ── Main Page ──────────────────────────────────────────────────────────────────
@solara.component
def Page():
    solara.Title("🌍 RAG Travel Planner")
    solara.Style(CSS)

    Header()

    with solara.Row(style="align-items:flex-start; gap:0; margin:0"):
        # ── Left: Inputs ──────────────────────────────────────────────
        with solara.Column(style="width:340px; min-width:300px; flex-shrink:0"):
            with solara.Column(classes=["tp-panel"]):
                InputPanel()

        # ── Right: Results ────────────────────────────────────────────
        with solara.Column(style="flex:1; min-width:0"):
            with solara.Column(classes=["tp-result"]):
                ResultPanel()
