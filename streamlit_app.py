"""Streamlit front end. Deploy this file to Streamlit Community Cloud."""
import os
import time

import requests
import streamlit as st

# On Streamlit Cloud, secrets come from the dashboard. Copy them to environment variables.
try:
    for key in ("GROQ_API_KEY", "HF_TOKEN", "LLM_MODEL", "API_URL", "MAX_DISTANCE"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # no secrets file locally; .env is used instead
# add api url
API_URL = os.getenv("API_URL", "").rstrip("/")
#Set the page layout 
st.set_page_config(page_title="Bank Support Assistant", page_icon="🏦")
st.title("Bank Customer Support Assistant")
st.caption("Answers come only from the bank customer guide. "
           "Never share your PIN, card number or one-time password.")
#Html for question asking in ui
question = st.text_area("Your question", max_chars=1000,
                        placeholder="e.g. How do I block my debit card?")

if st.button("Ask"):
    q = question.strip()
    data, error = None, None
    started = time.perf_counter()
    if not q:
        error = "Please enter a question."
    else:
        try:
            with st.spinner("Searching the guide..."):
                if API_URL:
                    resp = requests.post(f"{API_URL}/ask", json={"question": q}, timeout=60)
                    if resp.status_code == 422:
                        error = "Please check your question (1 to 1000 characters)."
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                else:
                    from support_rag import get_rag
                    data = get_rag().ask(q)
        except requests.RequestException:
            error = "The answer service is unreachable right now. Please try again later."
        except Exception:
            error = "The assistant is temporarily unavailable. Please try again later."
    #error thrown
    if error:
        st.warning(error)
    #otherwise give data 
    elif data:
        st.subheader("Answer")
        st.write(data["answer"])
        if data.get("status") == "answered" and data.get("sources"):
            st.caption("Sources: " + ", ".join(data["sources"]))
        st.caption(f"Response time: {(time.perf_counter() - started) * 1000:.0f} ms")
