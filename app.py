# app.py - Neurix MVP Demo using Streamlit + Hugging Face Inference API (requests) + Supabase

import uuid
import streamlit as st
import requests
from supabase import create_client
import pyvis.network as net
from collections import Counter

# ---------------------- Configuration ----------------------
config = st.secrets.get("general", st.secrets)
HF_TOKEN = config.get("HF_TOKEN")
SUPABASE_URL = config.get("SUPABASE_URL")
SUPABASE_KEY = config.get("SUPABASE_KEY")

if not HF_TOKEN or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❗ Missing one of HF_TOKEN, SUPABASE_URL, SUPABASE_KEY in Streamlit secrets.")
    st.stop()

# Setup Supabase client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# HF Inference API headers & endpoint
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

# ---------------------- Helper Functions ----------------------

def summarize(text: str) -> str:
    """Summarize input text via Hugging Face Inference API using requests."""
    try:
        payload = {"inputs": text}
        response = requests.post(SUMMARIZATION_URL, headers=HF_HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # data format: [{'summary_text': '...'}]
        return data[0].get('summary_text', '').strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Summarization API error: {e}")
        # Fallback: return original text or truncated
        return text if len(text) < 200 else text[:200] + "..."


def extract_keys(text: str, top_k: int = 8) -> list[str]:
    """Naive keyword extraction: top frequent words longer than 4 chars."""
    words = [w.strip('.,!?:;"\'') for w in text.lower().split()]
    stopwords = set([
        "that","with","this","about","after","before","where","which","while",
        "there","their","would","could","should","your","from","have","just",
        "they","them","what","when"
    ])
    candidates = [w for w in words if len(w) > 4 and w not in stopwords]
    freq = Counter(candidates)
    return [w for w,_ in freq.most_common(top_k)]

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix MVP — Digital Brain Demo")

# Tabs for note vs file
tab1, tab2 = st.tabs(["📝 Write Note", "📂 Upload File"])
with tab1:
    user_text = st.text_area("Enter your note:", height=200)
with tab2:
    uploaded = st.file_uploader("Upload PDF/DOCX/TXT or image", type=["pdf","docx","txt","png","jpg"])

if st.button("▶️ Process to Create Node"):
    # Read content
    if uploaded:
        if uploaded.type.startswith("image/"):
            text = "OCR not implemented - placeholder"
        else:
            raw = uploaded.read()
            try:
                text = raw.decode("utf-8")
            except:
                text = str(raw)
    else:
        text = user_text

    # Summarize input
    with st.spinner("⏳ Summarizing..."):
        summary = summarize(text)
    # Extract keys
    with st.spinner("🔑 Extracting keys..."):
        keys = extract_keys(summary)

    # Create node object
    node = {
        "id": str(uuid.uuid4()),
        "summary": summary,
        "content": text,
        "keys": keys,
        "metadata": {
            "created_at": st.session_state.get("now", ""),
            "source": uploaded.name if uploaded else "user_note"
        },
        "is_public": False
    }
    st.subheader("Generated Node")
    st.json(node)

    # Save node in session
    st.session_state.setdefault("nodes", []).append(node)

    # Insert into Supabase
    sb.table("nodes").insert(node).execute()
    st.success("Node saved to Supabase!")

# Display knowledge graph if nodes exist
nodes = st.session_state.get("nodes", [])
if nodes:
    g = net.Network(height="600px", width="100%", notebook=True)
    # Add nodes
    for n in nodes:
        g.add_node(n["id"], label=n["summary"][:30] + "...")
    # Add edges for shared keys
    for i, ni in enumerate(nodes):
        for nj in nodes[i+1:]:
            if set(ni["keys"]) & set(nj["keys"]):
                g.add_edge(ni["id"], nj["id"])
    g.show("graph.html")
    st.subheader("🔗 Knowledge Graph")
    html = open("graph.html", "r", encoding="utf-8").read()
    st.components.v1.html(html, height=600)

# Footer
st.markdown("---")
st.write("**Note:** This is a demo MVP. Further enhancements will include OCR, authentication, and advanced analytics.")
