# KnowledgeHub AI

> A production-oriented Retrieval-Augmented Generation (RAG) system for intelligent document question answering.

## Overview

KnowledgeHub AI is an end-to-end AI engineering project that demonstrates the design and implementation of a modular Retrieval-Augmented Generation (RAG) system using open-source technologies.

The application allows users to upload documents, preprocess and chunk their content, generate vector embeddings, store and retrieve relevant information from a vector database, and generate grounded answers using a locally hosted Large Language Model (LLM) served through Ollama.

The project emphasizes software engineering, modular architecture, evaluation, testing, retrieval quality, and deployment rather than notebook-based experimentation.

---

## Current Features

* Document upload through a FastAPI REST API
* UTF-8 text validation for uploaded documents
* Sentence-aware text chunking with overlap
* Dense vector embedding generation
* ChromaDB vector storage and similarity search
* Document-level retrieval filtering
* Configurable top-k retrieval
* Configurable retrieval distance threshold
* Retrieval-Augmented Generation using a local Gemma 2B model through Ollama
* Grounded generation using retrieved document context
* Source information returned with generated answers
* HTTP error handling for LLM service failures
* Structured application logging
* Automated test suite using pytest
* Retrieval evaluation with Hit@1 and Hit@3
* Out-of-domain query rejection evaluation
* Controlled evaluation corpus covering multiple document formats
* Corpus structure and provenance planning for future multi-format ingestion

---

## Retrieval Evaluation

The project includes a controlled retrieval evaluation pipeline under:

```text
app/evaluation/
```

The current baseline evaluation contains:

* 9 in-domain questions
* 1 out-of-domain question
* Hit@1 measurement
* Hit@3 measurement
* Out-of-domain rejection measurement

Current controlled-corpus baseline:

```text
In-domain Hit@1: 100%
In-domain Hit@3: 100%
Out-of-domain rejection: 100%
```

These results are specific to the current controlled evaluation corpus and should not be interpreted as general retrieval-performance claims.

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

The corpus is being used to establish repeatable evaluation and provenance requirements before production multi-format ingestion is implemented.

---

## Automated Testing

The project uses pytest for automated verification.

Current test coverage includes:

* API behavior
* document upload validation
* chunking behavior
* retrieval service behavior
* RAG service behavior
* retrieval evaluation metrics
* document-level filtering

Current test status:

```text
25 passed
```

---

## Technology Stack

* Python
* FastAPI
* Pydantic
* Sentence Transformers
* ChromaDB
* Ollama
* Gemma 2B
* httpx
* pytest
* Docker
* Git
* GitHub

---

## Project Structure

```text
KnowledgeHub-AI/

│
├── app/
│   ├── api/
│   ├── core/
│   ├── evaluation/
│   │   ├── corpus/
│   │   ├── corpus_manifest.json
│   │   ├── corpus_structure.json
│   │   ├── retrieval_eval.json
│   │   └── retrieval_evaluator.py
│   ├── ingestion/
│   ├── models/
│   ├── retrieval/
│   ├── services/
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
└── requirements.txt
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

### In Progress

**Phase 8 — Retrieval architecture and stress evaluation**

Planned work includes:

* larger evaluation datasets
* ambiguous and distractor queries
* larger corpus stress testing
* retrieval latency measurements
* richer provenance evaluation
* duplicate and repeated-content testing
* hybrid lexical + dense retrieval
* reciprocal rank fusion (RRF)
* reranking
* retrieval-quality comparison across approaches

### Planned

**Phase 9 — Multi-format production ingestion**

Support for production ingestion of formats such as PDF, DOCX, PPTX, XLSX, CSV, Markdown, HTML, and additional document types.

**Phase 10 — Document understanding**

Tables, OCR, images, diagrams, figures, and richer document provenance.

**Phase 11 — Security and authorization**

Authentication, authorization, document permissions, security boundaries, and multi-tenant access control.

**Phase 12 — Scale and deployment**

Large-volume ingestion, performance optimization, observability, containerized deployment, and production-scale testing.

**Phase 13 — Productization**

Final API/documentation improvements, deployment guidance, portfolio presentation, and production-readiness review.

---

## Design Goals

KnowledgeHub AI is being developed around the following engineering principles:

* Modular architecture
* Clear separation of concerns
* Local and open-source AI components
* Reproducible evaluation
* Grounded generation
* Explicit retrieval quality measurement
* Provenance-aware document processing
* Automated testing
* Production-oriented error handling
* Scalable architecture

---

## License

MIT License
