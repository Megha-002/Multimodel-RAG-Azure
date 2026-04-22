import hashlib

import streamlit as st
from audiorecorder import audiorecorder
from api_client import query_rag, transcribe_voice
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Policy Assistant",
    page_icon="💼",
    layout="wide",
)

# ── CSS: layout + mobile-friendly + menu button tweak ──────
st.markdown("""
<style>
    /* ── Center content with max-width on all screens ── */
    .main .block-container {
        max-width: 860px;
        margin: 0 auto;
        padding: 1.5rem 1rem 2rem 1rem;
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #ffffff;
        padding: 10px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 78%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.55;
        word-break: break-word;
    }
    .bot-bubble {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4);
        color: #ffffff;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 78%;
        font-size: 0.95rem;
        line-height: 1.55;
        word-break: break-word;
        border: none;
    }
    .sources-box {
        border-left: 3px solid #6366f1;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        opacity: 0.7;
        max-width: 78%;
        margin-top: 2px;
    }

    /* ── Input bar ── */
    .stTextInput input {
        border-radius: 24px !important;
        padding: 10px 18px !important;
        font-size: 0.95rem !important;
    }

    /* ── Send button ── */
    div[data-testid="stHorizontalBlock"] .stButton button {
        border-radius: 24px !important;
        padding: 9px 20px !important;
        font-weight: 600 !important;
        width: 100%;
    }

    /* ── Replace three-dots menu with gear/settings icon ── */
    [data-testid="main-menu-button"] > div > svg {
        display: none;
    }
    [data-testid="main-menu-button"]::after {
        content: "⚙️";
        font-size: 1.25rem;
        line-height: 1;
    }
    [data-testid="main-menu-button"] {
        border-radius: 50% !important;
        background: rgba(255,255,255,0.08) !important;
        padding: 4px !important;
    }

    /* ── Mobile tweaks ── */
    @media (max-width: 640px) {
        .main .block-container {
            padding: 0.75rem 0.5rem 1.5rem 0.5rem;
        }
        .user-bubble, .bot-bubble, .sources-box {
            max-width: 92%;
            font-size: 0.9rem;
        }
        .chat-header h1 { font-size: 1.4rem !important; }
        .chat-header p  { font-size: 0.8rem !important; }
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="chat-header" style="text-align:center;padding:8px 0 4px 0;">
    <h1 style="font-size:1.8rem;font-weight:700;margin-bottom:2px;">💼 Employee Policy Assistant</h1>
    <p style="font-size:0.85rem;opacity:0.6;margin-top:0;">
        Ask anything about HR policies, benefits, forms, or workplace guidelines.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Session State ──────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_audio_key" not in st.session_state:
    st.session_state.last_audio_key = None
if "pending_input" not in st.session_state: 
    st.session_state.pending_input = ""

# ── Chat History Display ───────────────────────────────────
for chat in st.session_state.chat_history:
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin:6px 0;">
        <div class="user-bubble">🧑 {chat['question']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;justify-content:flex-start;margin:6px 0;">
        <div class="bot-bubble">💼 {chat['answer']}</div>
    </div>""", unsafe_allow_html=True)

    if chat.get("sources"):
        sources_text = " | ".join(set(chat["sources"]))
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-start;">
            <div class="sources-box">📄 Sources: {sources_text}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

st.divider()

# ── Voice Recorder ─────────────────────────────────────────
audio = audiorecorder(
    start_prompt="🎙️ Record",
    stop_prompt="⏹️ Stop",
    show_visualizer=False,
    key="mic",
)

if len(audio) > 0:
    # FIX 1: Export explicitly as WAV.
    # pydub's default export format is mp3 — sending mp3 bytes labelled as
    # audio/wav causes Groq Whisper to silently fail or reject the request.
    audio_bytes = audio.export(format="wav").read()

    # md5 hash reliably detects a genuinely new recording (unlike len())
    audio_key = hashlib.md5(audio_bytes).hexdigest()

    if st.session_state.last_audio_key != audio_key:
        st.session_state.last_audio_key = audio_key

        with st.spinner("🎧 Transcribing your voice..."):
            transcript = transcribe_voice(audio_bytes, "recorded_audio.wav")

        if transcript:
            st.session_state.pending_input = transcript
            st.rerun()
        else:
            st.warning("⚠️ Transcription failed — please try again or type your question.")

# ── Input Row ──────────────────────────────────────────────

# ✅ Set BEFORE the widget renders — this is allowed
if st.session_state.pending_input:
    st.session_state["chat_input"] = st.session_state.pending_input
    st.session_state.pending_input = ""   # clear staging var

col_input, col_send = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "Message",
        placeholder="e.g. How do I apply for leave?",
        label_visibility="collapsed",
        key="chat_input"          # NO value= here
    )

with col_send:
    send = st.button("Send ➤", use_container_width=True)

# ── Send Logic ─────────────────────────────────────────────
if send and question.strip():
    with st.spinner("🔍 Finding answer..."):
        result = query_rag(question)

    st.session_state.chat_history.append({
        "question": question,
        "answer":   result.get("answer", "Sorry, I could not find an answer."),
        "sources":  result.get("sources", []),
    })

    # Clear the input box via its widget key
    st.session_state.pending_input = ""   
    st.rerun()