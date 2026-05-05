import streamlit as st  # UI framework
import PyPDF2  # PDF text extraction
import requests  # HTTP requests
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter  # text chunking
from qdrant_client import QdrantClient  # vector database client
from qdrant_client.models import VectorParams, Distance
from duckduckgo_search import DDGS  # fallback search
from bs4 import BeautifulSoup  # HTML parsing
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer  # local embeddings
from openai import OpenAI  # OpenRouter-compatible client
import time

# Streamlit session state initialization
if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract clean text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            return text
        else:
            st.warning(f"Failed to fetch content from {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None

# Wrapper for safe URL content fetching
def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None

# Generate embeddings using local model
def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()