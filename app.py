# app.py - Neurix MVP Demo using Streamlit + Hugging Face + Supabase
import os
import uuid
import streamlit as st
from huggingface_hub import InferenceClient
from supabase import create_client
import pyvis.network as net

# ---------------------- Configuration ----------------------
# Secrets from .streamlit/secrets.toml or Streamlit Cloud
HF_TOKEN = st.secrets['HF_TOKEN']
SUPABASE_URL = st.secrets['SUPABASE_URL']
SUPABASE_KEY = st.secrets['SUPABASE_KEY']

# Initialize clients
hf_client = InferenceClient(token=HF_TOKEN)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------- Helper Functions ----------------------

def summarize(text: str) -> str:
    """Summarize input text via Hugging Face inference API"""
    res = hf_client.summarization(model="facebook/bart-large-cnn", inputs=text)
    return res[0]['summary_text']


def extract_keys(text: str, top_k: int = 8) -> list[str]:
    """Extract keywords via a HF keyword-extraction model"""
    res = hf_client.text_generation(
        model="pszemraj/keyword-extractor",
        inputs=text,
        parameters={"max_new_tokens": top_k * 2}
    )
    keys = res[0]['generated_text'].split(", ")
    return keys[:top_k]

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix MVP — Digital Brain Demo")

# Tabs for note vs file
tab1, tab2 = st.tabs(["📝 Write Note", "📂 Upload File"]);
with tab1:
    user_text = st.text_area("Enter your note:", height=200)
with tab2:
    uploaded = st.file_uploader("Upload PDF/DOCX/TXT or image", type=["pdf","docx","txt","png","jpg"])

if st.button("▶️ Process to Create Node"):
    # Read content
    if uploaded:
        if uploaded.type.startswith("image/"):
            text = st.text("OCR not implemented - placeholder")
        else:
            raw = uploaded.read()
            try:
                text = raw.decode('utf-8')
            except:
                text = str(raw)
    else:
        text = user_text

    # Summarize
    with st.spinner("⏳ Summarizing..."):
        summary = summarize(text)
    # Extract keys
    with st.spinner("🔑 Extracting keys..."):
        keys = extract_keys(summary)

    # Create node
    node = {
        'id': str(uuid.uuid4()),
        'summary': summary,
        'content': text,
        'keys': keys,
        'metadata': {
            'created_at': st.session_state.get('now', ''),
            'source': uploaded.name if uploaded else 'user_note'
        },
        'is_public': False
    }
    st.subheader("Generated Node")
    st.json(node)

    # Save node locally in session
    st.session_state.setdefault('nodes', []).append(node)

    # Insert into Supabase
    sb.table('nodes').insert(node).execute()
    st.success("Node saved to Supabase!")

# Display graph if nodes exist
nodes = st.session_state.get('nodes', [])
if nodes:
    g = net.Network(height="600px", width="100%", notebook=True)
    # add nodes
    for n in nodes:
        g.add_node(n['id'], label=n['summary'][:30] + '...')
    # add edges on shared keys
    for i, ni in enumerate(nodes):
        for nj in nodes[i+1:]:
            if set(ni['keys']) & set(nj['keys']):
                g.add_edge(ni['id'], nj['id'])
    g.show('graph.html')
    st.subheader("🔗 Knowledge Graph")
    html = open('graph.html', 'r', encoding='utf-8').read()
    st.components.v1.html(html, height=600)

# Instructions and footer
st.markdown("---")
st.write("**Note:** This is a demo MVP. Further enhancements will include OCR, authentication, and advanced analytics.")
