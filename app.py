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