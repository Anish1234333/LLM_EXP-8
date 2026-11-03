# Bank Customer Support RAG (Experiment 7)

FastAPI service (`app.py`) and Streamlit UI (`streamlit_app.py`) for a RAG assistant over a bank customer guide.

## Run locally
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env             # then add your HF_TOKEN
    uvicorn app:app --reload         # API docs at http://127.0.0.1:8000/docs
    streamlit run streamlit_app.py   # UI at http://localhost:8501

## Test
    pytest -v

## Notes
- The LLM is a cloud model through Hugging Face Inference Providers (free quota applies).
- Never commit `.env` or share API keys.
