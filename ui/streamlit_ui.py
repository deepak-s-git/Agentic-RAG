import streamlit as st
from services.rag_pipeline import process_and_index_documents, answer_question

def render_ui():
    # Streamlit session state initialization
    if "client" not in st.session_state:
        st.session_state.client = None
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None

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
