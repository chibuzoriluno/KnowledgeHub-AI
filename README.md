# KnowledgeHub AI

> A production-oriented Retrieval-Augmented Generation (RAG) system for intelligent document question answering.

## Overview

KnowledgeHub AI is an end-to-end AI engineering project that demonstrates the design and implementation of a modular Retrieval-Augmented Generation (RAG) system using open-source technologies.

The application allows users to upload documents, preprocess and chunk their content, generate vector embeddings, store and retrieve relevant information from a vector database, combine dense and lexical retrieval, rerank retrieved evidence, and generate grounded answers using a locally hosted Large Language Model (LLM) served through Ollama.

The project emphasizes software engineering, modular architecture, evaluation, testing, retrieval quality, provenance, and deployment readiness rather than notebook-based experimentation.

---

## Current Features

* Document upload through a FastAPI REST API
* UTF-8 text validation for uploaded documents
* Sentence-aware text chunking with overlap
* Dense vector embedding generation
* ChromaDB vector storage and similarity search
* BM25 lexical retrieval
* Hybrid dense + lexical retrieval
* Reciprocal Rank Fusion (RRF)
* Dense-distance relevance and out-of-domain gating
* BGE cross-encoder reranking
* Automatic BM25 refresh when the indexed corpus changes
* Document-level retrieval filtering
* Configurable top-k retrieval
* Configurable retrieval distance threshold
* Hybrid retrieval used by both `/search` and `/rag`
* Retrieval scoring and source metadata
* Retrieval-Augmented Generation using a local Gemma 2B model through Ollama
* Grounded generation using retrieved document context
* Source information returned with generated answers
* HTTP error handling for LLM service failures
* Structured application logging
* Automated test suite using pytest
* Retrieval evaluation with Hit@1, Hit@3, Mean Reciprocal Rank, and Recall@3
* Out-of-domain query rejection evaluation
* Controlled evaluation corpus covering multiple document formats
* Corpus structure and provenance planning for future multi-format ingestion

---

## Retrieval Architecture

The selected Phase 8 retrieval architecture combines semantic retrieval, lexical retrieval, rank fusion, relevance gating, and cross-encoder reranking.

```text
Query
  |
  +----------------------+
  |                      |
  v                      v
Dense Retrieval       BM25 Retrieval
  |                      |
  +----------+-----------+
             |
             v
   Dense Relevance / OOD Gate
             |
             v
     Reciprocal Rank Fusion
             |
             v
      Top 10 Candidates
             |
             v
   BAAI/bge-reranker-base
             |
             v
        Final Top-k
             |
             v
         RAG Context
             |
             v
      Gemma 2B via Ollama
```

Dense retrieval provides semantic similarity, BM25 provides lexical relevance, and Reciprocal Rank Fusion combines the two ranked result sets without directly comparing incompatible scoring systems.

The dense retrieval threshold acts as the out-of-domain relevance gate. If no dense result satisfies the configured threshold, the query is rejected rather than allowing BM25-only results to pass through.

The fused candidate set is then reranked using `BAAI/bge-reranker-base`, which evaluates the query and candidate chunk together and produces the final relevance ordering.

---

## Retrieval Evaluation

KnowledgeHub AI includes a controlled retrieval-evaluation framework under:

```text
app/evaluation/
```

The current evaluation set contains:

* 20 in-domain questions
* 5 out-of-domain questions
* paraphrased queries
* ambiguous queries
* realistic distractor documents
* overlapping technical topics
* multi-document relevance cases

Measured metrics include:

* Hit@1
* Hit@3
* Mean Reciprocal Rank (MRR)
* Document Recall@3
* Out-of-domain rejection
* Retrieval latency
* Retrieval + reranking latency

Several retrieval architectures were evaluated during Phase 8:

```text
Dense retrieval
        ↓
BM25 lexical retrieval
        ↓
Dense + BM25 + RRF
        ↓
Dense-gated hybrid retrieval
        ↓
Dense + BM25 + RRF + BGE reranking
```

The final selected architecture uses:

```text
Dense candidate pool: 10
BM25 candidate pool: 10
RRF candidate pool sent to reranker: 10
Final top-k: 3
Dense max distance: 1.3
RRF k: 60
Reranker: BAAI/bge-reranker-base
```

### Final controlled-corpus results

```text
In-domain Hit@1:          90.0%
In-domain Hit@3:         100.0%
Mean Reciprocal Rank:     0.9500
Mean document Recall@3:   97.5%
Out-of-domain rejection: 100.0%
```

These results are specific to the project's controlled evaluation corpus and should not be interpreted as general retrieval-performance claims.

The reranker improved retrieval quality over the gated hybrid baseline, particularly in top-rank accuracy and overall ranking quality.

---

## Evaluation Corpus

A controlled multi-format corpus is maintained under:

```text
app/evaluation/corpus/
```

The corpus currently includes examples representing:

* TXT
* Markdown
* PDF
* DOCX
* PPTX
* XLSX
* CSV
* HTML
* scanned-document scenarios
* visual/diagram scenarios

Corpus inventory:

```text
app/evaluation/corpus_manifest.json
```

Observed corpus structure:

```text
app/evaluation/corpus_structure.json
```

Retrieval evaluation cases:

```text
app/evaluation/retrieval_eval.json
```

The corpus is used to establish repeatable evaluation, provenance requirements, and future ingestion requirements before production multi-format document handling is introduced.

---

## Retrieval Evaluation Tools

The evaluation package includes separate evaluators for the major retrieval stages:

```text
app/evaluation/retrieval_evaluator.py
app/evaluation/bm25_evaluator.py
app/evaluation/hybrid_evaluator.py
app/evaluation/reranked_hybrid_evaluator.py
```

These allow dense, lexical, hybrid, and reranked retrieval approaches to be compared independently.

This provides a reproducible engineering trail showing why the final retrieval architecture was selected.

---

## Automated Testing

The project uses pytest for automated verification.

Current test coverage includes:

* API behavior
* document upload validation
* chunking behavior
* dense retrieval behavior
* BM25 lexical retrieval
* BM25 index refresh behavior
* document-level retrieval filtering
* Reciprocal Rank Fusion
* hybrid retrieval
* out-of-domain gating
* BGE reranking logic
* reranked hybrid retrieval
* RAG service behavior
* retrieval evaluation metrics
* source metadata behavior
* empty-result behavior

Current test status:

```text
48 passed
```

---

## Technology Stack

* Python
* FastAPI
* Pydantic
* Sentence Transformers
* ChromaDB
* rank-bm25
* BAAI/bge-reranker-base
* Ollama
* Gemma 2B
* httpx
* pytest
* Docker
* Git
* GitHub

---

## Current AI Models

### Dense Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used to generate 384-dimensional embeddings for semantic retrieval through ChromaDB.

### Cross-Encoder Reranker

```text
BAAI/bge-reranker-base
```

Used as a second-stage reranker after dense retrieval, BM25 retrieval, and Reciprocal Rank Fusion.

The reranker dynamically uses CUDA when available and otherwise runs on CPU.

### Generation Model

```text
gemma2:2b
```

Hosted locally through Ollama and used to generate answers from retrieved document context.

---

## API Endpoints

### Root

```text
GET /
```

Returns application status information.

### Health Check

```text
GET /health
```

Returns service health information.

### Document Upload

```text
POST /documents/upload
```

Currently supports UTF-8 `.txt` documents.

Uploaded documents are:

```text
validated
   ↓
normalized
   ↓
sentence-aware chunked
   ↓
embedded
   ↓
stored in ChromaDB
```

Production multi-format ingestion is planned for Phase 9.

### Search

```text
POST /search
```

Uses the production retrieval architecture:

```text
Dense + BM25
      ↓
Dense relevance gate
      ↓
RRF
      ↓
BGE reranking
      ↓
Final ranked results
```

Search results include:

* chunk ID
* document ID
* chunk index
* chunk text
* dense distance where available
* final reranker score

### RAG

```text
POST /rag
```

Uses the same production retrieval pipeline as `/search`, then supplies the retrieved evidence to Gemma through Ollama.

The model is instructed to answer only from retrieved context and not invent unsupported facts.

---

## Project Structure

```text
KnowledgeHub-AI/
│
├── app/
│   ├── api/
│   │
│   ├── core/
│   │
│   ├── evaluation/
│   │   ├── corpus/
│   │   ├── corpus_manifest.json
│   │   ├── corpus_structure.json
│   │   ├── retrieval_eval.json
│   │   ├── retrieval_evaluator.py
│   │   ├── bm25_evaluator.py
│   │   ├── hybrid_evaluator.py
│   │   └── reranked_hybrid_evaluator.py
│   │
│   ├── ingestion/
│   │
│   ├── models/
│   │
│   ├── retrieval/
│   │
│   ├── services/
│   │   ├── bm25_service.py
│   │   ├── chunk_service.py
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── generation_service.py
│   │   ├── hybrid_retrieval_service.py
│   │   ├── rag_service.py
│   │   ├── reranked_hybrid_retrieval_service.py
│   │   ├── reranker_service.py
│   │   ├── retrieval_service.py
│   │   ├── rrf_service.py
│   │   ├── text_service.py
│   │   └── vector_service.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   └── vector_db/
│
├── docs/
├── scripts/
├── tests/
├── docker/
├── README.md
├── requirements.txt
└── pytest.ini
```

---

## Development Status

### Completed

**Phase 1 — Repository and application foundation**

Project structure, configuration, FastAPI application, health endpoints, and development environment.

**Phase 2 — Document ingestion**

Document upload, text processing, chunking, embedding, and vector storage.

**Phase 3 — Retrieval**

Semantic vector search, configurable top-k retrieval, distance filtering, and document-level filtering.

**Phase 4 — Local generation**

Integration with Ollama and the locally hosted Gemma 2B model.

**Phase 5 — End-to-end RAG**

Grounded question answering using retrieved document context and source reporting.

**Phase 6 — API hardening and testing**

Structured errors, logging, automated API/service tests, and production-oriented reliability improvements.

**Phase 7 — Retrieval evaluation**

Controlled retrieval benchmarking, sentence-aware chunking, out-of-domain rejection testing, evaluation tooling, and a controlled multi-format corpus.

**Phase 8 — Retrieval architecture and stress evaluation**

Expanded evaluation set, realistic distractor corpus, retrieval latency measurement, BM25 lexical retrieval, dense + lexical hybrid retrieval, Reciprocal Rank Fusion, out-of-domain relevance gating, BGE cross-encoder reranking, candidate-pool experiments, and comparative retrieval benchmarking.

Phase 8 resulted in the selection of the following production retrieval architecture:

```text
Dense + BM25
      ↓
Dense relevance gate
      ↓
RRF
      ↓
Top 10 candidates
      ↓
BAAI/bge-reranker-base
      ↓
Final top-k
```

The selected architecture achieved:

```text
Hit@1:          90.0%
Hit@3:         100.0%
MRR:             0.9500
Recall@3:        97.5%
OOD rejection: 100.0%
```

on the project's controlled evaluation set.

---

### Next

**Phase 9 — Multi-format production ingestion**

Production ingestion support for formats such as:

* PDF
* DOCX
* PPTX
* PPT
* XLSX
* CSV
* Markdown
* HTML
* additional structured and semi-structured formats

Phase 9 will focus on extracting text and structured content from multiple document types while preserving document provenance.

---

### Planned

**Phase 10 — Document understanding**

Tables, OCR, images, diagrams, figures, visual document understanding, and richer provenance metadata.

Potential metadata will include:

* page number
* slide number
* sheet name
* table identifier
* figure identifier
* document section
* source location

Multimodal or vision-language models may be introduced if evaluation demonstrates that visual semantics cannot be adequately captured through extraction and OCR alone.

**Phase 11 — Security and authorization**

Authentication, authorization, document permissions, security boundaries, document-level access control, and multi-tenant architecture.

**Phase 12 — Scale and deployment**

Large-volume ingestion, retrieval performance optimization, observability, containerized deployment, production-scale stress testing, GPU-aware inference deployment, and infrastructure optimization.

**Phase 13 — Productization**

Final API improvements, documentation, deployment guidance, portfolio presentation, production-readiness review, and product packaging.

---

## Design Goals

KnowledgeHub AI is being developed around the following engineering principles:

* Modular architecture
* Clear separation of concerns
* Local and open-source AI components
* Reproducible evaluation
* Grounded generation
* Explicit retrieval quality measurement
* Hybrid retrieval
* Evidence-based architecture selection
* Provenance-aware document processing
* Automated testing
* Production-oriented error handling
* Scalable architecture
* Deployment portability
* Security-aware system design

---

## Production Direction

KnowledgeHub AI is designed as a production-oriented RAG backend rather than a local-only demonstration.

The current local development environment uses CPU inference where necessary, but model services are designed so production deployment can take advantage of GPU-capable infrastructure.

Future deployment targets may include containerized cloud infrastructure such as AWS, Azure, or similar environments.

The retrieval and reranking pipeline is designed to remain modular so individual components can be optimized or replaced independently as scale and deployment requirements evolve.

---

## License

MIT License
