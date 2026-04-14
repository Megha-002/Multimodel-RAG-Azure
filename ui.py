import streamlit as st
import requests
import os

# ===============================
# CONFIG
# ===============================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ===============================
# SESSION STATE
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# API CALL FUNCTION
# ===============================

def query_rag(question):
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ===============================
# UI HEADER
# ===============================

st.title("🤖 Enterprise Knowledge Assistant")
st.caption("Powered by LLaMA 3 · ChromaDB · MiniLM")

# ===============================
# SIDEBAR
# ===============================

with st.sidebar:
    st.markdown("### ⚙️ System Info")
    st.markdown("- LLM: LLaMA 3 (Groq)")
    st.markdown("- Embeddings: MiniLM")
    st.markdown("- Vector DB: ChromaDB")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ===============================
# CHAT DISPLAY
# ===============================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if "latency" in msg:
                st.caption(f"⏱ {msg['latency']} sec")

            if "sources" in msg and msg["sources"]:
                st.caption("📎 Sources:")
                for src in set(msg["sources"]):
                    st.caption(f"- {os.path.basename(src)}")

# ===============================
# USER INPUT
# ===============================

user_input = st.chat_input("Ask about company policies, documents...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API
    with st.spinner("Thinking..."):
        result = query_rag(user_input)

    if "error" in result:
        with st.chat_message("assistant"):
            st.error(result["error"])
    else:
        answer = result.get("answer", "No answer returned")
        latency = result.get("latency_seconds")
        sources = result.get("sources", [])

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "latency": latency,
            "sources": sources
        })

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"⏱ {latency} sec")

            if sources:
                st.caption("📎 Sources:")
                for src in set(sources):
                    st.caption(f"- {os.path.basename(src)}")