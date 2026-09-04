# Computer Networking & Infrastructure Manual

**Document ID:** networking.md

**Author:** Apex Software Consulting Infrastructure Team

**Target System:** MedRAG (Medical Retrieval-Augmented Generation & Clinical Decision Support)

**Classification:** Internal Technical Reference / Client Onboarding Asset

---

## 1. Executive Summary & Networking Strategic Context

### 1.1 Architecture & Scope

At Apex Software Consulting, our deployment of **MedRAG** for HealthPulse Technologies requires a high-performance, fault-tolerant, and secure networking infrastructure. Because MedRAG handles real-time clinical queries, semantic similarity lookups across high-dimensional vector spaces, and distributed multi-modal document ingestion, networking cannot be an afterthought.

The underlying network topology must satisfy three primary mandates:

* **Strict HIPAA & Security Compliance:** End-to-end encryption in transit (TLS 1.3) and isolation of clinical payload traffic from public internet egress.
* **Low Retrieval Latency:** Sub-10ms inter-service communication between vector database clusters, document parsers, and LLM orchestration gateways.
* **Resilient Multi-Tenant Isolation:** Dynamic network policy enforcement using Virtual Private Clouds (VPC), private subnets, and Kubernetes NetworkPolicies.

This manual serves as the technical networking blueprint for `networking.md` within the MedRAG evaluation corpus and production deployment manifests.

---

## 2. Enterprise Network Topology & VPC Design

```
                     ┌───────────────────────────────────────────┐
                     │          Internet / Client Access         │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │           Cloudflare WAF / CDN            │
                     └─────────────────────┬─────────────────────┘
                                           │
 ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
 │ HealthPulse Production VPC (10.100.0.0/16)                                        │
 │                                                                                   │
 │  ┌─────────────────────────────────────────────────────────────────────────────┐  │
 │  │ Public Ingress Subnet (10.100.1.0/24)                                        │  │
 │  │                                                                             │  │
 │  │    ┌───────────────────────────┐         ┌──────────────────────────────┐   │  │
 │  │    │  Application Load Balancer│         │  NAT Gateway (Egress Only)   │   │  │
 │  │    └─────────────┬─────────────┘         └──────────────┬───────────────┘   │  │
 │  └──────────────────┼──────────────────────────────────────┼───────────────────┘  │
 │                     │                                      │                      │
 │  ┌──────────────────┼──────────────────────────────────────┼───────────────────┐  │
 │  │ App Tier Private Subnet (10.100.10.0/24)                │                   │  │
 │  │                  │                                      │                   │  │
 │  │                  ▼                                      │                   │  │
 │  │    ┌───────────────────────────┐                        │                   │  │
 │  │    │ FastAPI Gateway Cluster   │                        │                   │  │
 │  │    └─────────────┬─────────────┘                        │                   │  │
 │  └──────────────────┼──────────────────────────────────────┼───────────────────┘  │
 │                     │                                      │                      │
 │  ┌──────────────────┼──────────────────────────────────────┼───────────────────┐  │
 │  │ Data Tier Isolated Subnet (10.100.20.0/24)              │                   │  │
 │  │                  │                                      │                   │  │
 │  │                  ▼                                      ▼                   │  │
 │  │    ┌───────────────────────────┐         ┌──────────────────────────────┐   │  │
 │  │    │  Qdrant Vector Cluster    │         │  External Medical API / LLM  │   │  │
 │  │    │  (gRPC / Port 6334)       │         │  Endpoints (TLS 1.3 Outbound)│   │  │
 │  │    └───────────────────────────┘         └──────────────────────────────┘   │  │
 │  └─────────────────────────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────────────────────┘

```

### 2.1 Subnet Partitioning & CIDR Allocation

To maintain strict privilege separation, the MedRAG deployment within HealthPulse Technologies operates inside a dedicated Virtual Private Cloud (VPC) spanned across three Availability Zones (AZs).

* **Public Subnets (`10.100.1.0/24`, `10.100.2.0/24`):** Hosts Application Load Balancers (ALBs) and NAT Gateways. Direct compute workloads are prohibited here.
* **Application Subnets (`10.100.10.0/24`, `10.100.11.0/24`):** Hosts stateless FastAPI services, ingestion worker pods, and cross-encoder reranking microservices.
* **Data Layer Subnets (`10.100.20.0/24`, `10.100.21.0/24`):** Fully isolated subnets housing Qdrant vector database nodes, PostgreSQL metadata stores, and Redis cache clusters. No direct Internet access is permitted.

---

## 3. High-Performance Protocol Selection: gRPC vs. REST

MedRAG segregates network protocol usage depending on whether communications target external client applications or internal inter-service node paths.

### 3.1 External Tier: REST & WebSockets

For external client applications (e.g., electronic health record UI integrations, web portals), MedRAG exposes HTTPS REST endpoints and WebSockets for real-time streaming LLM output.

* **Protocol:** HTTP/2 over TLS 1.3
* **Payload Format:** JSON
* **Use Case:** Initial user query submission, session creation, status polling, and token-by-token streaming response delivery.

### 3.2 Internal Tier: High-Throughput gRPC

For internal service-to-service calls (e.g., application gateway to Qdrant vector database or cross-encoder rerankers), HTTP/1.1 JSON overhead imposes unacceptable latency. Apex Software Consulting standardizes internal communications on **gRPC over HTTP/2**.

```
[ FastAPI App Gateway ] ─── gRPC (HTTP/2 + Protocol Buffers) ───► [ Qdrant Vector DB Node ]

```

#### Key Technical Advantages:

1. **Binary Serialization:** Protocol Buffers (`.proto`) significantly reduce payload serialization overhead compared to string-based JSON.
2. **HTTP/2 Multiplexing:** Multiple requests are sent over a single TCP connection, eliminating TCP handshakes and head-of-line blocking.
3. **Low Overhead Vector Streaming:** High-dimensional floating-point array payloads (e.g., 1536-dim embeddings) are transferred natively as packed binary fields.

---

## 4. Network Security & Zero-Trust Policies

### 4.1 In-Transit Encryption (TLS 1.3)

All network communications—both external and intra-cluster (pod-to-pod)—require Transport Layer Security (TLS 1.3). Inter-pod mutual TLS (mTLS) is enforced using an Istio Service Mesh, guaranteeing:

* Mutual identity verification between microservices via short-lived X.509 certificates.
* Transparent packet encryption without requiring application code changes.

### 4.2 Kubernetes NetworkPolicies

To enforce a strict Zero-Trust network architecture, default deny rules are applied across all namespaces. Communication is explicitly allowed only along verified paths:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-app-to-vector-db
  namespace: medrag-prod
spec:
  podSelector:
    matchLabels:
      app: qdrant-vector-db
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: medrag-api-gateway
    ports:
    - protocol: TCP
      port: 6334

```

---

## 5. Provenance Tracking & Metadata Integration in `networking.md`

In accordance with **Phase 8 (Step 8.9)** of the MedRAG implementation roadmap, this document functions as controlled corpus asset `#2` within `app/evaluation/corpus_manifest.json`.

```json
{
  "document_id": "networking",
  "filename": "networking.md",
  "format": "md",
  "category": "text",
  "content_types": ["text"]
}

```

### 5.1 Controlled Facts & Citation Locations

To validate MedRAG's semantic search and exact-location provenance attribution, `networking.md` contains the following deterministic test points:

* **Fact 1:** "HealthPulse Production VPC uses CIDR block 10.100.0.0/16."
*Expected Provenance:* `{"document_id": "networking", "section": "2.1 Subnet Partitioning & CIDR Allocation"}`
* **Fact 2:** "Inter-service vector communications utilize gRPC on TCP port 6334."
*Expected Provenance:* `{"document_id": "networking", "section": "3.2 Internal Tier: High-Throughput gRPC"}`
* **Fact 3:** "Service mesh mTLS enforcement is handled via Istio."
*Expected Provenance:* `{"document_id": "networking", "section": "4.1 In-Transit Encryption (TLS 1.3)"}`

---

## 6. Execution & Verification

To verify that `networking.md` is registered and indexed correctly alongside `ml_overview.txt` in the evaluation manifest, execute the following script from the root directory:

```bash
python -c "import json; data=json.load(open('app/evaluation/corpus_manifest.json', encoding='utf-8')); docs={d['document_id']: d['filename'] for d in data['documents'] if d['category']=='text'}; print('Text Category Corpus:', docs)"

```

#### Expected System Output:

```text
Text Category Corpus: {'ml_overview': 'ml_overview.txt', 'networking': 'networking.md'}

```