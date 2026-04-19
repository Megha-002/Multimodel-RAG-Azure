import streamlit as st
import os
from audiorecorder import audiorecorder
from api_client import query_rag, transcribe_voice
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── Page Setup ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multimodal RAG Assistant")
st.markdown("Ask questions from your documents using **text or voice**.")
st.divider()

# ── Session State ──────────────────────────────────────────
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "answer" not in st.session_state:
    st.session_state["answer"] = ""
if "sources" not in st.session_state:
    st.session_state["sources"] = []

# ── Voice Input Section ────────────────────────────────────
st.subheader("🎙️ Speak Your Question")
audio = audiorecorder("⏺️ Click to Record", "⏹️ Click to Stop")

if len(audio) > 0:
    # Play back what was recorded
    st.audio(audio.export().read(), format="audio/wav")

    if st.button("📝 Transcribe & Use as Question"):
        with st.spinner("Transcribing via Groq Whisper..."):
            audio_bytes = audio.export().read()
            transcript = transcribe_voice(audio_bytes, "recorded_audio.wav")

        if transcript:
            st.success(f"Heard: **{transcript}**")
            st.session_state["question"] = transcript
        else:
            st.error("Transcription failed. Please try again.")

st.divider()

# ── Text Input Section ─────────────────────────────────────
st.subheader("💬 Or Type Your Question")
question = st.text_input(
    "Type your question here:",
    value=st.session_state["question"],
    placeholder="e.g. What is the refund policy?"
)

if st.button("🔍 Get Answer") and question:
    with st.spinner("Searching documents and generating answer..."):
        result = query_rag(question)

    st.session_state["answer"] = result.get("answer", "")
    st.session_state["sources"] = result.get("sources", [])

st.divider()

# ── Answer Section ─────────────────────────────────────────
if st.session_state["answer"]:
    st.subheader("✅ Answer")
    st.write(st.session_state["answer"])

    if st.session_state["sources"]:
        st.subheader("📄 Sources")
        for source in set(st.session_state["sources"]):
            st.markdown(f"- `{source}`")