# app.py - Neurix MVP Demo using Streamlit + Hugging Face InferenceApi + Supabase

import uuid
import streamlit as st
from huggingface_hub import InferenceApi
from supabase import create_client
import pyvis.network as net

# ---------------------- Configuration ----------------------
# Load secrets from Streamlit (local .streamlit/secrets.toml or Cloud UI)
config = st.secrets.get("general", st.secrets)
HF_TOKEN = config.get("HF_TOKEN")
SUPABASE_URL = config.get("SUPABASE_URL")
SUPABASE_KEY = config.get("SUPABASE_KEY")

if not HF_TOKEN or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❗ Missing one of HF_TOKEN, SUPABASE_URL, SUPABASE_KEY in Streamlit secrets.")
    st.stop()

# Initialize clients
summarizer = InferenceApi(repo_id="sshleifer/distilbart-cnn-12-6", token=HF_TOKEN)
keywordizer = InferenceApi(repo_id="pszemraj/keyword-extractor", token=HF_TOKEN)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------- Helper Functions ----------------------

def summarize(text: str) -> str:
    """Summarize input text via Hugging Face InferenceApi."""
    # Call InferenceApi with text as positional arg
    out = summarizer(text)
    # out is a list of strings (summaries)
    return out[0].strip()


def extract_keys(text: str, top_k: int = 8) -> list[str]:
    """Extract keywords via Hugging Face keyword-extraction InferenceApi."""
    out = keywordizer(text)
    # model returns a comma-separated string of keywords
    keys = out[0].split(", ")
    return keys[:top_k]

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
    # Render and display graph
    g.show("graph.html")
    st.subheader("🔗 Knowledge Graph")
    html = open("graph.html", "r", encoding="utf-8").read()
    st.components.v1.html(html, height=600)

# Footer
st.markdown("---")
st.write("**Note:** This is a demo MVP. Further enhancements will include OCR, authentication, and advanced analytics.")
