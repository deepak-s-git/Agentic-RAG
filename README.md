# Agentic RAG System (PDF + Website + Fallback Search)

An Agentic Retrieval-Augmented Generation (RAG) system that answers questions using PDFs, website content, and fallback web search through intelligent routing.

---

## Features

* PDF-based question answering
* Website content extraction and processing
* Agent-based decision logic for source selection
* Semantic search using Qdrant vector database
* Local embeddings using SentenceTransformers
* Fallback search using DuckDuckGo
* Context-grounded responses to minimize hallucination

---

## System Architecture

```
User Query → Vector Search (Qdrant) → Context Retrieved → LLM Decision  
        ↓                              ↓  
     Found                        Not Found  
        ↓                              ↓  
   Answer from docs           Web Search → LLM Answer  
```

---

## Tech Stack

Python, Streamlit, Qdrant, SentenceTransformers, OpenRouter API, BeautifulSoup, PyPDF2, Docker

---

## Installation and Setup

### Prerequisites

* Python 3.10 (recommended for compatibility)
* Docker installed and running
* OpenRouter API Key

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Agentic-RAG.git
cd Agentic-RAG
```

---

### 2. Install Docker (if not installed)

On macOS (Homebrew):

```bash
brew install --cask docker
```

Open Docker Desktop and ensure it is running.

---

### 3. Pull Qdrant Image

```bash
docker pull qdrant/qdrant
```

---

### 4. Start Qdrant Container

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Keep this terminal running while using the application.

---

### 5. Verify Qdrant

Open in browser:

```
http://localhost:6333/dashboard
```

---

### 6. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### 7. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 8. Run the Application

Open a new terminal:

```bash
source venv/bin/activate
streamlit run app.py
```

Access the app at:

```
http://localhost:8501
```

---

## API Key Setup

Generate an API key from:

https://openrouter.ai/keys

Enter the key in the application UI.

---

## Usage

1. Enter API key
2. Upload PDF files or provide website URLs
3. Click "Process Documents"
4. Ask questions based on the indexed content

---

## Agent Workflow

1. Search within PDF content
2. If not found, search website content
3. If still not found, perform fallback web search
4. Generate final answer using LLM

---

## Notes

* DuckDuckGo may rate limit frequent requests
* Ensure Docker is running before starting the app
* Python 3.10 is recommended for compatibility with ML libraries

---

## Future Improvements

* Chat history support
* Source citation display
* Deployment to cloud platforms
* Integration with production-grade search APIs

---

## Author

Deepak S
BTech Computer Science Engineering

---

## Support

If you find this project useful, consider starring the repository.
