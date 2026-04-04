import streamlit as st
import requests
import sounddevice as sd
import numpy as np
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
# CUSTOM CSS — GRADIENT THEME
# ===============================

st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Chat message — user */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        color: white;
        font-size: 15px;
    }

    /* Chat message — bot */
    .bot-message {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #667eea;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 70%;
        color: white;
        font-size: 15px;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid #667eea;
        border-radius: 25px;
        padding: 10px 20px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 8px 24px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: scale(1.02);
    }

    /* Flowchart container */
    .flow-container {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }

    /* Step — inactive */
    .step-inactive {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 12px;
    }

    /* Step — active */
    .step-active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        color: white;
        font-size: 12px;
        font-weight: bold;
        box-shadow: 0 0 12px rgba(102,126,234,0.6);
    }

    /* Step — done */
    .step-done {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        color: white;
        font-size: 12px;
        font-weight: bold;
    }

    /* Arrow */
    .arrow {
        text-align: center;
        color: #667eea;
        font-size: 18px;
        padding: 2px 0;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(15,12,41,0.9);
    }

    /* Latency badge */
    .latency-badge {
        background: rgba(102,126,234,0.2);
        border: 1px solid #667eea;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #667eea;
        display: inline-block;
        margin-top: 6px;
    }

    /* Sources badge */
    .source-badge {
        background: rgba(56,239,125,0.1);
        border: 1px solid #38ef7d;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #38ef7d;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# ===============================
# FLOWCHART FUNCTION
# ===============================

def show_flowchart(active_step=None, done_steps=[]):
    steps = [
        ("💬", "User Query"),
        ("🔢", "Embedding"),
        ("🔍", "Similarity Search"),
        ("📄", "Chunk Retrieval"),
        ("🦙", "LLaMA 3"),
        ("✅", "Response")
    ]

    st.markdown('<div class="flow-container">', unsafe_allow_html=True)
    st.markdown("**⚡ RAG Pipeline**", unsafe_allow_html=True)

    cols = st.columns(len(steps) * 2 - 1)

    for i, (icon, label) in enumerate(steps):
        col_idx = i * 2
        with cols[col_idx]:
            if i in done_steps:
                st.markdown(
                    f'<div class="step-done">{icon}<br>{label}</div>',
                    unsafe_allow_html=True
                )
            elif i == active_step:
                st.markdown(
                    f'<div class="step-active">{icon}<br>{label}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="step-inactive">{icon}<br>{label}</div>',
                    unsafe_allow_html=True
                )

        # Arrow between steps
        if i < len(steps) - 1:
            with cols[col_idx + 1]:
                st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# QUERY FASTAPI FUNCTION
# ===============================

def query_rag(question, flowchart_placeholder):
    import time

    # Step 0 — User Query
    with flowchart_placeholder:
        show_flowchart(active_step=0, done_steps=[])
    time.sleep(0.5)

    # Step 1 — Embedding
    with flowchart_placeholder:
        show_flowchart(active_step=1, done_steps=[0])
    time.sleep(0.5)

    # Step 2 — Similarity Search
    with flowchart_placeholder:
        show_flowchart(active_step=2, done_steps=[0, 1])
    time.sleep(0.5)

    # Step 3 — Chunk Retrieval
    with flowchart_placeholder:
        show_flowchart(active_step=3, done_steps=[0, 1, 2])
    time.sleep(0.5)

    # Step 4 — LLaMA 3 (actual API call happens here)
    with flowchart_placeholder:
        show_flowchart(active_step=4, done_steps=[0, 1, 2, 3])
        st.markdown(
            "<p style='color:rgba(255,255,255,0.4); font-size:12px; text-align:center'>⏳ LLaMA 3 is generating response...</p>",
            unsafe_allow_html=True
        )

    # Call FastAPI
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question},
            timeout=120
        )
        data = response.json()

        # Step 5 — Response
        with flowchart_placeholder:
            show_flowchart(active_step=5, done_steps=[0, 1, 2, 3, 4])
        time.sleep(0.3)

        # All done
        with flowchart_placeholder:
            show_flowchart(active_step=None, done_steps=[0, 1, 2, 3, 4, 5])

        return data

    except Exception as e:
        return {"error": str(e)}

# ===============================
# RECORD VOICE FUNCTION
# ===============================

def record_voice():
    DURATION = 5
    SAMPLE_RATE = 16000

    st.info("🎤 Recording for 5 seconds... Speak now!")
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
<div style='text-align: center; padding: 20px 0'>
    <h1 style='background: linear-gradient(135deg, #667eea, #764ba2, #38ef7d);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-size: 2.5em;
               font-weight: 800;'>
        🤖 Enterprise RAG Chatbot
    </h1>
    <p style='color: rgba(255,255,255,0.5); font-size: 14px;'>
        Powered by LLaMA 3 · ChromaDB · MiniLM
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Info")
    st.markdown("**LLM:** LLaMA 3 (Ollama)")
    st.markdown("**Embeddings:** MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Speech:** Whisper Base")
    st.markdown("---")
    st.markdown("### 📁 Knowledge Base")
    st.markdown("- HR Policy PDFs")
    st.markdown("- Finance Documents")
    st.markdown("- Scanned Invoices")
    st.markdown("- Process Diagrams")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Flowchart placeholder — always visible
flowchart_placeholder = st.empty()
with flowchart_placeholder:
    show_flowchart(active_step=None, done_steps=[])

st.markdown("---")

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if "latency" in msg:
                st.markdown(
                    f'<div class="latency-badge">⏱️ {msg["latency"]}s</div>',
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
        result = query_rag(user_input, flowchart_placeholder)

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
# HANDLE TEXT QUERY
# ===============================

if send_clicked and user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    result = query_rag(user_input, flowchart_placeholder)

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