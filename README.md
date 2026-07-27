# MultimodalRAG

**MultimodalRAG** is an advanced Retrieval-Augmented Generation (RAG) system designed to process and query PDF documents containing **text, images, and tables**. It leverages **multimodal embeddings**, **semantic retrieval**, and **Large Language Models (LLMs)** via Groq to deliver accurate, source-grounded answers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

* **Multi-PDF Upload** with automatic preprocessing
* **Smart Extraction** of text, images, and tabular data
* **Advanced Semantic Chunking** using hybrid strategies
* **Multimodal Embeddings** (text and image) via CLIP/BGE
* **Vector Indexing** with Qdrant
* **Context-Aware Retrieval** using vector similarity
* **LLM-Driven Response Generation** (GPT-4o, Claude, LLaMA via Groq API)
* **Modern Web Interface** powered by Streamlit
* **Table Interpretation** with caption/context extraction
* **OCR and Image Analysis** with automatic captioning
* **Integrated Monitoring & Performance Metrics**
* **Containerized Deployment** with Docker & Docker Compose

---

## Technology Stack

| Component  | Technology                 | Description                     |
| ---------- | -------------------------- | ------------------------------- |
| Frontend   | Streamlit                  | Interactive web UI              |
| Vector DB  | Qdrant                     | High-performance vector search  |
| Embeddings | CLIP / BGE / OpenAI        | Multimodal encoding models      |
| LLMs       | Groq API                   | Access to GPT, Claude, LLaMA    |
| Parsing    | PyMuPDF / Tesseract        | Text/table/OCR extraction       |
| Pipeline   | LangChain                  | RAG orchestration framework     |
| CI/CD      | Pre-commit, GitHub Actions | Dev workflow automation         |

---

## Data Flow

<img width="3684" height="1789" alt="diagram" src="https://github.com/user-attachments/assets/3efae443-9a70-4c81-af10-ee66da3eadab" />

---


## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd multimodalrag
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the full stack (App + Qdrant)

```bash
docker-compose up -d
```

### 4. Open the app

```bash
open http://localhost:8501
```

---

## Local Setup (Alternative)

```bash
pip install -r requirements.txt
docker run -d -p 6333:6333 qdrant/qdrant
streamlit run streamlit_app/Home.py
```

---

## Pre-Launch Checklist

* [ ] Python 3.8+ installed
* [ ] Docker & Docker Compose available
* [ ] `.env` configured with valid GROQ\_API\_KEY
* [ ] Ports 8501 (Streamlit) and 6333 (Qdrant) available

---

## Functional Test Scenarios

1. **Upload** a sample PDF file
2. **Textual query**: "What is this document about?"
3. **Image query**: "Show me related diagrams or illustrations"
4. **Table query**: "What data is shown in the tables?"

---

## Troubleshooting

**Qdrant not reachable**

```bash
docker ps
curl http://localhost:6333/health
```

**GROQ API not responding**

```bash
cat .env | grep GROQ_API_KEY
```

**Port conflict**

```bash
streamlit run streamlit_app/Home.py --server.port=8502
```

---

## Developer Setup

### Docker-based Setup

```bash
cp .env.example .env
# Add GROQ_API_KEY
docker-compose up -d
```

### Makefile-based Setup

```bash
make setup-dev       # Install dependencies & pre-commit hooks
make qdrant-start    # Launch Qdrant
make run             # Launch Streamlit App
```

---

## Makefile Commands Reference

```bash
make help             # List all commands
make setup-dev        # Full local dev setup
make run              # Launch the app
make reindex          # Re-index all documents
make evaluate         # Run automatic evaluation
make benchmark        # Run benchmark analysis
make clean            # Clean temporary files
make docker-build     # Build the Docker image
make ci               # Run CI pipeline
```
Other commands include `lint`, `format`, `check-all`, `bandit`, and more for code quality checks can be found in the Makefile or by running `make help`.

---

## Project Structure

```
multimodalrag/
├── src/                  
│   ├── config.py         
│   ├── core/             
│   ├── llm/              
│   ├── pipeline/         
│   └── utils/            
├── data/                 
│   ├── models/           
│   └── raw/              
├── scripts/              
├── streamlit_app/        
├── logs/                 
```

---

## Workflow

1. **Upload PDF** via UI
2. **Automatic Processing** and semantic chunking
3. **Multimodal Indexing** (text + image)
4. **Query** using text, image or hybrid inputs
5. **Response Generation** using LLMs

---

## Running Indexer Manually

```bash
make reindex
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Documentation

* [Quick Start Guide](docs/GUIDA_AVVIO.md)
* [Makefile Command Reference](#makefile-commands-reference)
* [Benchmarks](docs/BENCHMARKS.md) 

---
