import streamlit as st
from services.rag_pipeline import process_and_index_documents, answer_question

def render_ui():
    st.set_page_config(page_title="Agentic RAG", layout="wide")

    # Minimal, Elegant, Premium Dark Theme CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Backgrounds & Fonts */
    html, body, [class*="css"], .stApp, p, h1, h2, h3, h4, h5, h6, span, div, input, button, textarea {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stApp {
        background-color: #212121;
        color: #ECECEC;
    }
    
    /* Hide top header and footer for a cleaner app look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2F2F2F;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ECECEC !important;
    }
    h1 {
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-weight: 500;
        font-size: 1rem;
        color: #B4B4B4 !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Thin elegant separators */
    hr {
        border-top: 1px solid #2F2F2F;
        margin: 1.5rem 0;
    }
    
    /* Button Styling (Clean, Flat, Minimal) */
    .stButton > button {
        background-color: #ECECEC;
        border: 1px solid #ECECEC;
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 500;
        font-size: 14px;
        transition: background-color 0.2s ease, border-color 0.2s ease;
        width: 100%;
        box-shadow: none;
    }
    .stButton > button, .stButton > button * {
        color: #171717 !important;
    }
    .stButton > button:hover {
        background-color: #D4D4D4;
        border-color: #D4D4D4;
    }
    .stButton > button:hover, .stButton > button:hover * {
        color: #171717 !important;
    }
    
    /* Hide "Press Enter to apply" instruction */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* Ensure chat input background matches theme and remove red focus */
    .stChatInputContainer {
        border-radius: 16px;
        border: 1px solid #404040;
        box-shadow: none;
    }
    .stChatInputContainer:focus-within {
        border-color: #737373 !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: transparent;
        border: 1px dashed #404040;
        border-radius: 6px;
        padding: 1rem;
        transition: border-color 0.2s ease, background-color 0.2s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #737373;
        background-color: #2A2A2A;
    }
    
    /* Alerts / Infos */
    .stAlert {
        background-color: #2F2F2F;
        border: 1px solid #404040;
        color: #ECECEC;
        border-radius: 6px;
        font-size: 14px;
    }
    
    /* Chat messages layout */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1rem 0;
        margin: 0;
    }
    [data-testid="chat-message-user"] {
        background-color: transparent !important;
    }
    [data-testid="chat-message-assistant"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Session State Init
    if "client" not in st.session_state:
        st.session_state.client = None
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your documents today?"}]

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("<h2>Configuration</h2>", unsafe_allow_html=True)
        api_key = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...")
        if api_key:
            st.session_state.api_key = api_key
            
        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h2>Data Sources</h2>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload Documents (PDF)", accept_multiple_files=True, type=["pdf"])
        
        web_urls = st.text_input("Website URLs (comma separated)", placeholder="https://example.com")
        crawl_website = st.checkbox("Crawl entire website recursively")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        button_label = "Process Documents"
        has_pdfs = bool(uploaded_files)
        has_urls = bool(web_urls and web_urls.strip())
        
        if has_pdfs and has_urls:
            button_label = "Process Sources"
        elif has_urls:
            button_label = "Crawl Website"
            
        if st.button(button_label):
            if not st.session_state.get("api_key"):
                st.error("Please enter an API key first.")
            else:
                st.session_state.client, st.session_state.collection_name = process_and_index_documents(
                    uploaded_files, web_urls, crawl_website=crawl_website
                )
                if st.session_state.client:
                    if has_pdfs and has_urls:
                        st.session_state.source_type = "both"
                    elif has_urls:
                        st.session_state.source_type = "web"
                    else:
                        st.session_state.source_type = "pdf"

    # Main Chat Area
    st.markdown("<h1>Agentic RAG</h1>", unsafe_allow_html=True)
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat Input
    if question := st.chat_input("Message Agentic RAG..."):
        if not st.session_state.get("api_key"):
            st.warning("Please enter your OpenRouter API Key in the sidebar.")
        elif not st.session_state.client or not st.session_state.collection_name:
            st.warning("Please upload and process documents in the sidebar first.")
        else:
            # Show user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
                
            # Show assistant message and generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    
                    # Capture the markdown output from answer_question without modifying backend
                    captured_answer = []
                    original_markdown = st.markdown
                    
                    def captured_markdown(body, *args, **kwargs):
                        captured_answer.append(body)
                        original_markdown(body, *args, **kwargs)
                        
                    st.markdown = captured_markdown
                    
                    answer_question(
                        question,
                        st.session_state.client,
                        st.session_state.collection_name,
                        st.session_state.api_key,
                        source_type=st.session_state.get("source_type", "pdf")
                    )
                    
                    st.markdown = original_markdown
                    
                    if captured_answer:
                        # Append the last rendered markdown to history (which is the actual answer)
                        st.session_state.messages.append({"role": "assistant", "content": captured_answer[-1]})
