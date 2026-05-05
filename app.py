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

# Crawl and collect internal URLs from a website
def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    clean_url = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    urls.add(clean_url)
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls

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

# Extract text from uploaded PDFs
def process_uploaded_pdfs(uploaded_files):
    pdf_list = []
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    content += extracted
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return pdf_list


# Main pipeline: chunk, embed, and store in Qdrant
def process_and_index_documents(uploaded_files, web_urls=None, chunk_size=500, crawl_website=False):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    all_chunks = []
    doc_metadata = []

    # Process PDFs
    if uploaded_files:
        all_documents = process_uploaded_pdfs(uploaded_files)
        for doc in all_documents:
            chunks = text_splitter.split_text(doc["content"])
            all_chunks.extend(chunks)
            for _ in chunks:
                doc_metadata.append({"filename": doc["filename"], "source": "pdf_dataset"})

    # Process websites
    if web_urls:
        urls = [url.strip() for url in web_urls.split(",")]

        if crawl_website:
            all_urls = set()
            progress_bar = st.progress(0)
            progress_text = st.empty()
            for i, base_url in enumerate(urls):
                progress_text.text(f"Crawling website: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                progress_bar.progress((i + 1) / len(urls))
            urls = list(all_urls)
            progress_text.text(f"Found {len(urls)} unique URLs")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, url in enumerate(urls):
            progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)
            if content is not None:
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)
                for _ in chunks:
                    doc_metadata.append({"url": url, "source": "web_content"})
            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)

        progress_text.empty()
        progress_bar.empty()

    if not all_chunks:
        st.error("No content to process.")
        return None, None

    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = get_embeddings(all_chunks)

    # Store in Qdrant vector DB
    client = QdrantClient("http://localhost:6333")
    collection_name = "agent_rag_index"
    VECTOR_SIZE = 384

    with st.spinner("Creating vector database..."):
        try:
            client.delete_collection(collection_name)
        except:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [{"content": chunk, "metadata": metadata} for chunk, metadata in zip(all_chunks, doc_metadata)]

        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )

    st.success(f"Indexed {len(all_chunks)} chunks successfully!")
    return client, collection_name

# Call OpenRouter model for response generation
def ask_ai(prompt, api_key):
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# RAG pipeline: retrieve → decide → answer
def answer_question(question, client, collection_name, api_key, top_k=3):

    if not question.strip():
        st.warning("Enter a question.")
        return

    # Vector search
    def search(text: str):
        query_embedding = get_embeddings(text)[0]
        return client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)

    # Format retrieved chunks
    def format_docs(docs):
        formatted = []
        for doc in docs:
            if doc.payload["metadata"]["source"] == "pdf_dataset":
                src = f"\nSource: PDF {doc.payload['metadata']['filename']}"
            else:
                src = f"\nSource: Web {doc.payload['metadata']['url']}"
            formatted.append(doc.payload["content"] + src)
        return "\n\n".join(formatted)

    results = search(question)
    context = format_docs(results)

    # Agent decision step
    decision_prompt = f"""
Return 1 if answer exists in context, else 0.

Context:
{context}

Question:
{question}
"""

    decision = ask_ai(decision_prompt, api_key).strip()

    # If found → answer from context
    if decision == "1":
        final_prompt = f"""
Answer using ONLY this context:

{context}

Question:
{question}
"""
        answer = ask_ai(final_prompt, api_key)
        st.markdown(answer)

    # Else → fallback search
    else:
        results = DDGS().text(question, max_results=5)
        online_context = "\n\n".join(doc["body"] for doc in results)

        final_prompt = f"""
Answer using this context:

{online_context}

Question:
{question}
"""
        answer = ask_ai(final_prompt, api_key)
        st.markdown(answer)


# ---------------- UI ----------------

st.set_page_config(page_title="Agentic RAG System", layout="wide")

st.title("Agentic RAG System")

# API key input
api_key = st.text_input("Enter OpenRouter API Key:", type="password")
if api_key:
    st.session_state.api_key = api_key

# File upload
uploaded_files = st.file_uploader("Upload PDFs:", accept_multiple_files=True, type=["pdf"])

# Website input
web_urls = st.text_input("Enter website URLs:")
crawl_website = st.checkbox("Crawl entire website")

# Process button
if st.button("Process Documents"):
    if not st.session_state.get("api_key"):
        st.error("Enter API key first.")
    else:
        st.session_state.client, st.session_state.collection_name = process_and_index_documents(
            uploaded_files, web_urls, crawl_website=crawl_website
        )

# Question answering
if st.session_state.client and st.session_state.collection_name:
    question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        answer_question(
            question,
            st.session_state.client,
            st.session_state.collection_name,
            st.session_state.api_key,
        )
        
elif uploaded_files or web_urls:
    st.warning("Please process and index the documents first.")
else:
    st.info("Upload PDFs or provide website URLs to get started.")