import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

API_URL = os.getenv("API_URL", "http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Enterprise Knowledge Assistant")

def query_rag(question):
    try:
        res = requests.post(f"{API_URL}/query", json={"question": question}, timeout=120)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        result = query_rag(user_input)

    if "error" in result:
        st.error(result["error"])
    else:
        answer = result["answer"]
        latency = result.get("latency_seconds")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"⏱ {latency}s")