
import uuid
import streamlit as st

from app.assistant import ask_bankmate


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BankMate | Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# =========================================================
# DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>

/* Global */
.stApp {
    background: #080F1C;
    color: #EAF1FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

header[data-testid="stHeader"] {
    background: rgba(8, 15, 28, 0.85);
}

/* Typography */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3 {
    color: #F4F7FC !important;
    letter-spacing: -0.7px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D192A;
    border-right: 1px solid #1C3048;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem;
}

/* Brand */
.brand {
    font-size: 1.65rem;
    font-weight: 800;
    color: #F6F9FF;
    letter-spacing: -1.2px;
}

.brand span {
    color: #52B8FF;
}

.brand-sub {
    color: #8297B3;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 5px;
}

/* Sidebar sections */
.side-heading {
    color: #7188A5;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin: 26px 0 12px;
}

.side-item {
    color: #C0CEE0;
    font-size: 0.88rem;
    padding: 10px 11px;
    border-radius: 9px;
    margin-bottom: 5px;
    background: transparent;
}

.side-item.active {
    background: #172E49;
    color: #70C5FF;
    border: 1px solid #254B70;
}

.side-divider {
    border-top: 1px solid #203149;
    margin: 23px 0;
}

/* Top header */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}

.eyebrow {
    color: #60BFFF;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 2px;
}

.status {
    color: #6CE0B5;
    background: rgba(48, 190, 139, 0.10);
    border: 1px solid rgba(48, 190, 139, 0.25);
    border-radius: 30px;
    padding: 7px 12px;
    font-size: 0.72rem;
    font-weight: 700;
}

/* Welcome hero */
.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 85% 20%, rgba(44, 142, 226, 0.23), transparent 30%),
        linear-gradient(125deg, #142A45, #102039 65%, #142B48);
    border: 1px solid #274666;
    border-radius: 22px;
    padding: 38px 40px;
    margin-bottom: 28px;
}

.hero-kicker {
    color: #69C2FF;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.15;
    margin: 17px 0 13px;
    max-width: 650px;
}

.hero p {
    color: #B5C8DF;
    max-width: 660px;
    font-size: 0.98rem;
    line-height: 1.8;
}

.hero-tag {
    display: inline-block;
    margin-top: 15px;
    border: 1px solid rgba(87, 195, 255, 0.28);
    background: rgba(54, 155, 220, 0.12);
    color: #8AD3FF;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 0.73rem;
}

/* Section headings */
.section-heading {
    color: #7BC8FF;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 1.9px;
    margin: 28px 0 13px;
}

/* Capability cards */
.capability {
    background: linear-gradient(145deg, #14243A, #101D30);
    border: 1px solid #263C57;
    border-radius: 16px;
    padding: 20px;
    min-height: 158px;
}

.cap-icon {
    font-size: 1.4rem;
    margin-bottom: 15px;
}

.cap-title {
    color: #F0F5FC;
    font-size: 0.94rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.cap-desc {
    color: #91A5BF;
    font-size: 0.79rem;
    line-height: 1.65;
}

/* Buttons */
.stButton > button {
    background: #142943;
    border: 1px solid #2A4767;
    color: #DDEBFA;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #1B3C5F;
    border-color: #54B6F5;
    color: #FFFFFF;
}

/* Primary new chat button */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(110deg, #176DB6, #258FD5);
    border: 1px solid #398FCE;
    color: white;
}

/* Chat */
div[data-testid="stChatMessage"] {
    background: #111F32;
    border: 1px solid #263A53;
    border-radius: 16px;
    padding: 17px;
    margin-bottom: 14px;
}

div[data-testid="stChatInput"] {
    background: #122239;
    border: 1px solid #315273;
    border-radius: 15px;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #52B8FF;
    box-shadow: 0 0 0 2px rgba(82, 184, 255, 0.12);
}

.stChatInput textarea {
    color: #F2F7FC !important;
}

/* Source expanders */
div[data-testid="stExpander"] {
    background: #101F32;
    border: 1px solid #29425F;
    border-radius: 12px;
}

/* Footer */
.footer {
    color: #647D9B;
    font-size: 0.72rem;
    text-align: center;
    line-height: 1.8;
    padding-top: 25px;
}

.muted {
    color: #8499B3;
    font-size: 0.8rem;
}

/* Remove excessive decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">🏦 Bank<span>Mate</span></div>
    <div class="brand-sub">Intelligence platform</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-divider"></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="side-heading">WORKSPACE</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="side-item active">✦ &nbsp; AI Assistant</div>
    <div class="side-item">👤 &nbsp; Customer Intelligence</div>
    <div class="side-item">▥ &nbsp; Transaction Analytics</div>
    <div class="side-item">▤ &nbsp; Policy Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-divider"></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="side-heading">AI INFRASTRUCTURE</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="muted" style="line-height:2.5">
        🟢 &nbsp; Assistant available<br>
        ⚡ &nbsp; LangChain orchestration<br>
        ✧ &nbsp; Groq LLM<br>
        ◈ &nbsp; ChromaDB retrieval
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-divider"></div>',
                unsafe_allow_html=True)

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("""
    <div class="footer">
        BANKMATE INTELLIGENCE<br>
        Synthetic demonstration environment
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

left, right = st.columns([4, 1])

with left:
    st.markdown(
        '<div class="eyebrow">BANKING · INTELLIGENCE · AUTOMATION</div>',
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        '<div class="status">● SYSTEM READY</div>',
        unsafe_allow_html=True
    )


# =========================================================
# WELCOME EXPERIENCE
# =========================================================

if not st.session_state.messages:

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">YOUR AI-POWERED BANKING WORKSPACE</div>
        <h1>Clarity behind<br>every transaction.</h1>
        <p>
            Meet your intelligent banking assistant. Explore customer
            information, understand transaction activity and retrieve
            policy-backed answers through natural conversation.
        </p>
        <div class="hero-tag">✦ &nbsp; Intelligent answers. Evidence-backed insights.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">EXPLORE YOUR WORKSPACE</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    cards = [
        (
            "👤",
            "Customer intelligence",
            "Retrieve customer profiles and explore individual customer information."
        ),
        (
            "▥",
            "Transaction analytics",
            "Analyze debit, credit and net transaction flow with calculated metrics."
        ),
        (
            "📚",
            "Policy intelligence",
            "Find relevant policy evidence with traceable document references."
        )
    ]

    for column, card in zip([c1, c2, c3], cards):
        with column:
            st.markdown(f"""
            <div class="capability">
                <div class="cap-icon">{card[0]}</div>
                <div class="cap-title">{card[1]}</div>
                <div class="cap-desc">{card[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">START A CONVERSATION</div>',
        unsafe_allow_html=True
    )

    prompts = [
        "Show me the profile of customer C101.",
        "What are C101's transaction metrics?",
        "What should a customer do about an unauthorized transaction?",
        "What happens when a refund is approved?"
    ]

    col1, col2 = st.columns(2)

    for index, prompt in enumerate(prompts):
        column = col1 if index % 2 == 0 else col2

        with column:
            if st.button(
                prompt,
                key=f"prompt_{index}",
                use_container_width=True
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()


# =========================================================
# RESPONSE RENDERER
# =========================================================

def render_assistant_response(response):

    source_header = "Sources (fictional BankMate training policies):"

    answer, separator, source_text = response.partition(source_header)

    st.markdown(answer.strip())

    if separator and source_text.strip():

        with st.expander("📚  Supporting policy references"):

            st.caption(
                "Retrieved fictional training documents supporting this answer."
            )

            for line in source_text.strip().splitlines():
                line = line.strip()

                if line.startswith("- "):
                    st.markdown(f"- {line[2:]}")


# =========================================================
# CONVERSATION HISTORY
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":
        avatar = "👤"
    else:
        avatar = "🏦"

    with st.chat_message(message["role"], avatar=avatar):

        if message["role"] == "assistant":
            render_assistant_response(message["content"])
        else:
            st.markdown(message["content"])


# =========================================================
# CHAT ENGINE
# =========================================================

typed_prompt = st.chat_input(
    "Ask BankMate about customers, transactions or banking policies..."
)

prompt = st.session_state.pending_prompt or typed_prompt

if prompt:

    st.session_state.pending_prompt = None

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏦"):

        with st.spinner("BankMate is analyzing your request..."):

            try:

                response = ask_bankmate(
                    prompt,
                    session_id=st.session_state.session_id
                )

                render_assistant_response(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception:
                error_message = (
                    "I couldn't process that request right now. "
                    "Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    BANKMATE · AI BANKING INTELLIGENCE<br>
    Powered by LangChain · Groq · ChromaDB · Streamlit<br>
    Demonstration environment using synthetic customer data
    and fictional training policies.
</div>
""", unsafe_allow_html=True)
