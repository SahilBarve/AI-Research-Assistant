# AI Research Assistant

An AI-powered research assistant that lets you upload documents, search through them, and ask questions about their content.

I built this project to explore how a practical **Retrieval-Augmented Generation (RAG)** system can be designed from the ground up, including document processing, hybrid retrieval, reranking, source citations, local LLM inference, evaluation, Docker deployment, and CI/CD.

The application is currently deployed on **AWS EC2**.

---

## What can it do?

The main idea is simple:

**Upload your documents → search them → ask questions → get answers with sources.**

The application currently supports:

* Uploading PDF, DOCX, TXT, and Markdown files
* Document cleaning and chunking
* Semantic search using embeddings
* Keyword search using BM25
* Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Question answering using a local LLM
* Source/page citations in responses
* Conversation history
* Document search and management
* Retrieval evaluation
* Docker-based deployment
* AWS deployment
* Automated CI/CD with GitHub Actions

---

## Why I built it

I wanted to go beyond a basic "chat with PDF" application and understand what actually happens inside a RAG system.

The project gave me an opportunity to work on different parts of an AI application:

* Building REST APIs with FastAPI
* Designing a retrieval pipeline
* Comparing dense and lexical retrieval
* Implementing hybrid search
* Measuring the effect of reranking
* Working with vector databases
* Running an LLM locally with Ollama
* Building a React frontend
* Containerizing the application
* Deploying the system on AWS
* Setting up CI/CD

The goal was to build something that resembles the architecture of a real application rather than just a single Python script.

---

# How it works

At a high level, the system follows this pipeline:

```text
                  Document Upload
                        │
                        ▼
                Document Processing
                        │
                        ▼
                   Text Chunking
                        │
                        ▼
                   Embeddings
                    /       \
                   /         \
                  ▼           ▼
              Qdrant        BM25
              Dense         Keyword
             Search         Search
                  \           /
                   \         /
                    ▼       ▼
                   RRF Fusion
                       │
                       ▼
                Candidate Chunks
                       │
                       ▼
                Cross-Encoder
                  Reranking
                       │
                       ▼
                 Top Results
                       │
                       ▼
                Context Builder
                       │
                       ▼
                    Ollama
                       │
                       ▼
              Answer + Citations
```

When a question is asked, the system doesn't simply send the question directly to the LLM.

It first finds relevant pieces of the uploaded documents and then gives those pieces to the LLM as context. This helps keep the generated answer grounded in the available documents.

---

# Retrieval system

One of the main parts of the project was experimenting with different retrieval approaches.

### Dense retrieval

I use:

```text
BAAI/bge-small-en-v1.5
```

to generate 384-dimensional embeddings.

These embeddings are stored in **Qdrant** and are used for semantic similarity search.

### BM25

I also implemented BM25 for keyword-based retrieval.

This is useful when exact words, names, technical terms, or phrases matter.

### Hybrid retrieval

Instead of relying on only one retrieval method, the system combines the dense and BM25 results using **Reciprocal Rank Fusion (RRF)**.

The current implementation uses:

```text
RRF k = 60
```

This gives the system both semantic and lexical retrieval capabilities.

---

# Reranking

After hybrid retrieval, the candidate chunks are passed through a cross-encoder:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker looks at the question and each candidate chunk together and produces a more detailed relevance score.

I also tested different numbers of candidates before reranking to understand the trade-off between retrieval quality and latency.

---

# Evaluation

Rather than assuming that one retrieval method was better, I created a small labeled benchmark and compared the approaches.

The benchmark contains **12 queries**.

### Results

| Method        | Precision@5 |   Recall@5 |     NDCG@5 |        MRR |
| ------------- | ----------: | ---------: | ---------: | ---------: |
| Dense         |      0.1167 |     0.1944 |     0.1720 |     0.2795 |
| BM25          |      0.1333 |     0.2118 |     0.2197 |     0.4338 |
| RRF           |      0.1333 |     0.2049 |     0.2097 |     0.3709 |
| Reranker — 10 |      0.1667 |     0.2674 |     0.2982 |     0.4861 |
| Reranker — 15 |  **0.2000** | **0.3229** | **0.3765** | **0.6528** |
| Reranker — 20 |  **0.2000** | **0.3229** |     0.3729 | **0.6528** |

I also measured reranking latency:

| Candidates | Average latency |
| ---------: | --------------: |
|         10 |        196.9 ms |
|         15 |        469.8 ms |
|         20 |        745.7 ms |

In this benchmark, using 15 candidates produced the same Precision@5, Recall@5, and MRR as 20 candidates while taking less time.

These numbers are specific to this benchmark and aren't meant to represent performance on every dataset.

---

# Answers with sources

One of the things I wanted to avoid was having the LLM generate an answer without showing where the information came from.

Retrieved chunks keep information about:

* Document
* Page
* Chunk ID
* Text

The context builder passes this information to the LLM, and responses use citation references to point back to the retrieved sources.

The application also validates the citations generated by the model.

---

# Frontend

The frontend is built with React and TypeScript.

It currently has separate sections for:

* **Dashboard**
* **Documents**
* **Research / Chat**
* **Search**
* **Conversations**
* **Evaluation**

The frontend communicates with the FastAPI backend through `/api/v1`.

For production, Nginx serves the React application and forwards API requests to FastAPI.

---

# Architecture

The production setup currently looks like this:

```text
                         Internet
                            │
                            ▼
                    AWS EC2 Instance
                            │
                            ▼
                     Nginx + React
                         Port 80
                            │
                            │ /api/
                            ▼
                       FastAPI
                       Port 8000
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
         PostgreSQL       Qdrant        Ollama
                                      qwen2.5:3b
```

The internal services communicate through the Docker network.

Only the frontend/Nginx entry point is exposed publicly.

---

# Tech Stack

### Backend

* Python
* FastAPI
* Pydantic
* SQLAlchemy
* Alembic

### RAG / AI

* LangChain
* Sentence Transformers
* Hugging Face Transformers
* Qdrant
* BM25
* Ollama

### Frontend

* React
* TypeScript
* Vite
* Nginx

### Infrastructure

* Docker
* Docker Compose
* AWS EC2
* GitHub Actions

### Databases

* PostgreSQL
* Qdrant

---

# Project Structure

```text
AI-Research-Assistant/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── exceptions/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   └── services/
│   │       ├── chunkers/
│   │       ├── document_processors/
│   │       ├── embeddings/
│   │       ├── llm/
│   │       └── retrieval/
│   │
│   ├── tests/
│   ├── alembic/
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── nginx/
│   ├── Dockerfile
│   └── package.json
│
├── docs/
├── docker-compose.yml
├── document_chunks.snapshot
├── README.md
└── .gitignore
```

---

# Running locally

### Backend

```bash
cd backend

python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker

From the project root:

```bash
docker compose up -d --build
```

Check the services:

```bash
docker compose ps
```

---

# Testing

The backend has an automated test suite that runs through GitHub Actions.

To run it locally:

```bash
cd backend
python -m pytest -v
```

The CI pipeline also checks:

* Backend tests
* Frontend production build
* Docker image builds

---

# CI/CD

The project uses GitHub Actions for both CI and CD.

Every push to `main` goes through the CI workflow.

```text
git push
   │
   ▼
GitHub Actions
   │
   ├── Backend tests
   ├── Frontend build
   └── Docker builds
           │
           ▼
       CI passes
           │
           ▼
      CD workflow
           │
           ▼
       AWS EC2
           │
           ├── Pull latest code
           └── Rebuild containers
```

The deployment uses GitHub repository secrets for the EC2 connection details.

---

# AWS Deployment

The application is currently running on an AWS EC2 instance using Docker Compose.

The deployment process is:

```text
Local development
       │
       ▼
   git push
       │
       ▼
    GitHub
       │
       ▼
   CI workflow
       │
       ▼
   CD workflow
       │
       ▼
    AWS EC2
       │
       ▼
 Docker Compose
```

This means a successful push to `main` can automatically update the deployed application.

---

# Things I would improve next

There are still several areas I would improve if I continued developing the project:

* Persistent conversation memory using PostgreSQL
* Authentication and user accounts
* HTTPS and a custom domain
* More extensive end-to-end testing
* Larger retrieval evaluation datasets
* Background processing for large document uploads
* Better monitoring and logging
* GPU-based inference for faster LLM/reranker workloads
* More advanced context compression

---

# What I learned

This project ended up being much more than implementing a chatbot.

Some of the main things I worked with were:

* Designing a modular FastAPI backend
* Building a complete RAG pipeline
* Combining semantic and keyword retrieval
* Evaluating retrieval quality with actual metrics
* Understanding the latency/quality trade-off of reranking
* Working with vector databases
* Running LLMs locally
* Connecting a React frontend to a backend API
* Containerizing multiple services
* Deploying an AI application on AWS
* Setting up automated CI/CD

It also helped me understand that getting a RAG system working is only one part of building the application. Testing, deployment, observability, and maintaining a clean architecture are equally important.

---

## Author

**Sahil Barve**

B.E. Computer Engineering
D.Y. Patil College of Engineering, Pune

GitHub: [SahilBarve](https://github.com/SahilBarve)
