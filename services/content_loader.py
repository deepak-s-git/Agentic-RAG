import streamlit as st
import requests
import PyPDF2
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Optional

def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                # Ignore fragments, js, and query params for cleaner crawling
                if url.startswith("#") or url.startswith("javascript:"):
                    continue
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    # Drop the fragment and query for cleaner indexing
                    clean_url = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    if not clean_url.endswith((".png", ".jpg", ".jpeg", ".pdf", ".css", ".js", ".svg")):
                        urls.add(clean_url.rstrip("/"))
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Decompose noisy UI elements to extract only documentation content
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "meta", "svg", "form", "button"]):
                element.decompose()
                
            # Attempt to target the main article body
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("body") or soup
            
            # Use newline separators to preserve paragraph structure
            text = main_content.get_text(separator="\n\n")
            
            # Clean up whitespace while preserving paragraphs
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            
            return text
        else:
            st.warning(f"Failed to fetch content from {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None

def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None

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
