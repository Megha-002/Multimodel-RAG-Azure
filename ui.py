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

# 🔥 IMPORTANT — dynamic API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ===============================
# CUSTOM UI (same theme retained)
# ===============================

st.markdown("""
<style>
.stApp { background-color: #1a1a1a; color: #e8e0d5; }
.user-message {
    background: linear-gradient(135deg, #e8681a, #f5892a);
    padding: 12px; border-radius: 15px; margin: 8px 0;
    text-align: right; color: white;
}
.bot-message {
    background-color: #2a2a2a;
    padding: 12px; border-radius: 15px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# API CALL
# ===============================

def query_rag(question):
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=120
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

# ===============================
# HEADER
# ===============================

st.markdown("""
<h1 style='text-align:center;'>Enterprise Knowledge Assistant</h1>
<p style='text-align:center;color:gray;'>RAG • LLaMA3 • ChromaDB</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# CHAT HISTORY
# ===============================

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{msg['content']}</div>", unsafe_allow_html=True)

        if "latency" in msg:
            st.caption(f"⏱ {msg['latency']}s")

# ===============================
# INPUT
# ===============================

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input("Ask something...", label_visibility="collapsed")

with col2:
    send_clicked = st.button("Send 🚀")

# ===============================
# HANDLE QUERY
# ===============================

if send_clicked and user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        result = query_rag(user_input)

    if "error" in result:
        st.error(result["error"])
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", "No answer"),
            "latency": result.get("latency_seconds")
        })

    st.rerun()

# ===============================
# CLEAR CHAT
# ===============================

if st.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()