import streamlit as st
import time
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer

from services.content_loader import process_uploaded_pdfs, get_all_urls, fetch_url_content
from services.vector_store import init_and_upload_vector_store, search_vector_store
from services.llm_service import ask_ai
from utils.prompts import DECISION_PROMPT, FINAL_PROMPT_DOCS, FINAL_PROMPT_WEB

# Load embedding model globally to avoid reloading
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_query_for_retrieval(query: str) -> str:
    # Remove phrases that shouldn't affect semantic search
    patterns_to_remove = [
        r"(?i)\bin\s+\d+\s+words\b",
        r"(?i)\baround\s+\d+\s+words\b",
        r"(?i)\bfor\s+\d+\s+marks\b",
        r"(?i)\bbriefly\b",
        r"(?i)\bin\s+detail\b",
        r"(?i)\bwith\s+headings\b",
        r"(?i)\bwith\s+subheadings\b",
        r"(?i)\bwith\s+proper\s+subheadings\b",
        r"(?i)\bexplain\s+clearly\b",
        r"(?i)\bwith\s+examples\b",
        r"(?i)\bstep-by-step\b",
        r"(?i)\belaborate\b",
        r"(?i)\bsummarize\b",
        r"(?i)\bkey\s+points\b",
        r"(?i)\band\s+key\s+points\b",
        r"(?i)\bbullet\s+points\b",
        r"(?i)\btable\s+format\b",
    ]
    
    cleaned_query = query
    for pattern in patterns_to_remove:
        cleaned_query = re.sub(pattern, "", cleaned_query)
        
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
    return cleaned_query if cleaned_query else query

def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()

def process_and_index_documents(uploaded_files, web_urls=None, chunk_size=1000, crawl_website=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
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
    collection_name = "agent_rag_index"
    client = init_and_upload_vector_store(collection_name, all_chunks, doc_metadata, embeddings, vector_size=384)

    st.success(f"Indexed {len(all_chunks)} chunks successfully!")
    return client, collection_name

def answer_question(question, client, collection_name, api_key, top_k=8, source_type="pdf"):
    if not question.strip():
        st.warning("Enter a question.")
        return

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

    # Vector search
    retrieval_query = clean_query_for_retrieval(question)
    query_embedding = get_embeddings(retrieval_query)[0]
    results = search_vector_store(client, collection_name, query_embedding, top_k)
    context = format_docs(results)

    # Agent decision step
    decision_prompt = DECISION_PROMPT.format(context=context, question=question)
    decision_raw = ask_ai(decision_prompt, api_key)
    decision = "1" if "1" in decision_raw else "0"

    # If found → answer from context
    if decision == "1":
        if source_type == "both":
            st.info("Answer found in your sources")
        elif source_type == "web":
            st.info("Answer found from website content")
        else:
            st.info("Answer found in your documents")
        final_prompt = FINAL_PROMPT_DOCS.format(context=context, question=question)
        answer = ask_ai(final_prompt, api_key)
        st.markdown(answer)

    # Else → fallback (SAFE)
    else:
        st.info("Not found in documents → searching online...")

        try:
            time.sleep(2)  # avoid rate limit
            ddgs_results = DDGS().text(question, max_results=5)
            online_context = "\n\n".join(doc["body"] for doc in ddgs_results)

            final_prompt = FINAL_PROMPT_WEB.format(context=online_context, question=question)
            answer = ask_ai(final_prompt, api_key)
            st.markdown(answer)

        except Exception:
            st.warning("Search rate-limited. Try again in a few seconds.")
