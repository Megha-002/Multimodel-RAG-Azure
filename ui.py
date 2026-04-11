import streamlit as st
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ===============================
# CUSTOM CSS — CHARCOAL + ORANGE THEME
# ===============================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Main background — charcoal */
    .stApp {
        background-color: #1a1a1a;
        color: #e8e0d5;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #222222;
        border-right: 1px solid #333333;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown {
        color: #c8bfb0;
    }

    /* User message — warm orange right aligned */
    .user-message {
        background: linear-gradient(135deg, #e8681a, #f5892a);
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 68%;
        color: #ffffff;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(232, 104, 26, 0.25);
    }

    /* Bot message — dark card left aligned */
    .bot-message {
        background-color: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-left: 3px solid #e8681a;
        padding: 14px 18px;
        border-radius: 4px 18px 18px 18px;
        margin: 8px auto 8px 0;
        max-width: 68%;
        color: #e8e0d5;
        font-size: 15px;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: #2a2a2a;
        color: #e8e0d5;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 15px;
        font-family: 'Inter', sans-serif;
    }

    .stTextInput > div > div > input:focus {
        border: 1px solid #e8681a;
        box-shadow: 0 0 0 2px rgba(232, 104, 26, 0.15);
    }

    .stTextInput > div > div > input::placeholder {
        color: #666666;
    }

    /* Buttons */
    .stButton > button {
        background-color: #e8681a;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #f5892a;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(232, 104, 26, 0.3);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #333333;
        margin: 12px 0;
    }

    /* Latency badge */
    .latency-badge {
        background-color: rgba(232, 104, 26, 0.1);
        border: 1px solid rgba(232, 104, 26, 0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        color: #e8681a;
        display: inline-block;
        margin-top: 6px;
        font-family: 'Inter', sans-serif;
    }

    /* Source badge */
    .source-badge {
        background-color: rgba(180, 160, 120, 0.1);
        border: 1px solid rgba(180, 160, 120, 0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        color: #b4a078;
        display: inline-block;
        margin: 2px;
        font-family: 'Inter', sans-serif;
    }

    /* Spinner color */
    .stSpinner > div {
        border-top-color: #e8681a !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    ::-webkit-scrollbar-thumb {
        background: #444444;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #e8681a;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# QUERY FASTAPI FUNCTION
# ===============================

def query_rag(question):
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ===============================
# RECORD VOICE FUNCTION
# ===============================

def record_voice():
    DURATION = 5
    SAMPLE_RATE = 16000
    st.info("🎤 Recording for 5 seconds... Speak now!")
    import numpy as np
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, SAMPLE_RATE, audio)
    return temp_file.name

# ===============================
# MAIN UI LAYOUT
# ===============================

# Header
st.markdown("""
<div style='text-align: center; padding: 28px 0 16px 0;'>
    <h1 style='
        color: #e8e0d5;
        font-size: 2em;
        font-weight: 600;
        font-family: Inter, sans-serif;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    '>
        Enterprise Knowledge Assistant
    </h1>
    <p style='
        color: #666666;
        font-size: 13px;
        font-family: Inter, sans-serif;
    '>
        Powered by LLaMA 3 &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; MiniLM
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0;'>
        <p style='color: #e8681a; font-size: 13px; font-weight: 600;
                  font-family: Inter, sans-serif; margin-bottom: 12px;'>
            ⚙️ SYSTEM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**LLM:** LLaMA 3")
    st.markdown("**Embeddings:** MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Speech:** Whisper Base")

    st.markdown("---")

    st.markdown("""
    <p style='color: #e8681a; font-size: 13px; font-weight: 600;
              font-family: Inter, sans-serif; margin-bottom: 12px;'>
        📁 KNOWLEDGE BASE
    </p>
    """, unsafe_allow_html=True)

    st.markdown("- HR Policy Documents")
    st.markdown("- Finance SOPs")
    st.markdown("- Scanned Invoices")
    st.markdown("- Process Diagrams")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# Chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align: center; padding: 40px 0; color: #444444;
                    font-family: Inter, sans-serif; font-size: 14px;'>
            Ask me anything about company policies, documents, or procedures.
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 &nbsp;{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message">🤖 &nbsp;{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if "latency" in msg:
                st.markdown(
                    f'<div class="latency-badge">⏱ {msg["latency"]}s</div>',
                    unsafe_allow_html=True
                )
            if "sources" in msg:
                for source in set(msg["sources"]):
                    st.markdown(
                        f'<div class="source-badge">📎 {os.path.basename(source)}</div>',
                        unsafe_allow_html=True
                    )

st.markdown("---")

# Input area
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_input = st.text_input(
        "Ask anything...",
        placeholder="e.g. What is the leave policy?",
        label_visibility="collapsed"
    )

with col2:
    send_clicked = st.button("Send 🚀")

with col3:
    voice_clicked = st.button("🎤 Voice")

# ===============================
# HANDLE TEXT QUERY
# ===============================

if send_clicked and user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        result = query_rag(user_input)

    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", "No answer returned"),
            "latency": result.get("latency_seconds"),
            "sources": result.get("sources", [])
        })

    st.rerun()

# ===============================
# HANDLE VOICE QUERY
# ===============================

if voice_clicked:
    audio_path = record_voice()

    with open(audio_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/speech",
            files={"file": ("audio.wav", f, "audio/wav")}
        )

    os.remove(audio_path)

    if response.status_code == 200:
        data = response.json()
        transcribed = data.get("transcribed_text", "")

        st.session_state.messages.append({
            "role": "user",
            "content": f"🎤 {transcribed}"
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("answer", "No answer returned"),
            "latency": data.get("latency_seconds"),
            "sources": data.get("sources", [])
        })

        st.rerun()
    else:
        st.error("Voice query failed. Make sure FastAPI is running.")