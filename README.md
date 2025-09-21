# EL-CAP
Evidence-Linked, Context-Aware Molecular Prioritization

**EL-CAP** is an intelligent platform that transforms the way pharmaceutical and chemical companies discover and prioritize new molecules.  
By combining **graph neural networks (GNN)** for molecular structure analysis with **transformer-based NLP** for understanding scientific context, EL-CAP delivers **evidence-linked, context-aware rankings** of potential drug candidates.  

Whether you start with a **research question**, a **candidate molecule**, or **both**, EL-CAP can:  
- 🚀 Search millions of compounds in seconds using a **vector database** (FAISS / ChromaDB)  
- 📚 Link every result to **real-world evidence**: assays, publications, patents  
- 🧪 Predict **affinity**, **ADMET**, **novelty**, and **safety risks** through multi-task learning  
- 📊 Generate a single **composite score** to guide decision-making  

**From massive chemical libraries to ranked, evidence-backed shortlists — in one step.**
EL-CAP is a two-stage **retrieval** → **scoring** system with multimodal encoders (GNN for molecules, Transformer for text), a vector store for fast candidate recall, and multi-task heads plus a meta-model for final prioritization.
## Architecture review
```mermaid

flowchart TD

%% ========== DATA INGESTION ==========
subgraph A[Data Sources]
  A1[ChEMBL / BindingDB] 
  A2[PubMed / Patents] 
  A3[ELN / Internal Data]
end

A --> B[Ingestion & Curation (Airflow/Prefect)]
B --> C1[Molecule Standardization (RDKit)]
B --> C2[Assay Normalization (units, targets, endpoints)]

%% ========== ENCODERS ==========
subgraph D[Feature Extraction Encoders]
  D1[GNN Encoder\n(Graph Isomorphism Network / MPNN)\n+ SSL Pretraining\n+ ECFP fingerprints]
  D2[Transformer Encoder\n(SciBERT / PubMedBERT)\n+ MLM domain tuning\n+ Slot Extraction: target, assay, units]
end

C1 --> D1
C2 --> D2

D1 --> E1[Molecular Embeddings h_mol (ℝ^512)]
D2 --> E2[Text Embeddings h_txt (ℝ^768)]

%% ========== VECTOR STORES ==========
subgraph F[Storage Layers]
  F1[Vector Store (FAISS IVF-HNSW-PQ)\nMol index: {mol_id, h_mol}]
  F2[Text Passage Index (FAISS)\nEvidence index: {doc_id, h_txt}]
  F3[Tabular Store (DuckDB/Postgres)\nAssay schema, labels, flags]
end

E1 --> F1
E2 --> F2
C2 --> F3

%% ========== QUERY PIPELINE ==========
subgraph Q[Query Handling]
  Q1[User Query: text and/or molecule]
  Q2[Mol Preproc → h_mol*]
  Q3[Text Encode → h_query]
  Q4[Mol+Text Fusion\nLinear proj / light MLP]
end

Q1 --> Q2
Q1 --> Q3
Q2 --> Q4
Q3 --> Q4

%% Retrieval
Q3 --> R1[Text-only Retrieval\nANN search on {h_mol}]
Q2 --> R2[Mol-only Retrieval\nANN search on {h_mol}]
Q4 --> R3[Mol+Text Retrieval\nANN search on {h_mol}]

R1 --> S1[Top-K Candidates]
R2 --> S1
R3 --> S1

%% ========== RERANK / SCORING ==========
subgraph S[Scoring Pipeline]
  S1 --> S2[Cross-Encoder (Late Interaction / ColBERT-style)\nToken-level match text↔atoms]
  S2 --> S3[Fusion Layer\n[h_mol, h_txt, features] → MLP]
  S3 --> S4[Multi-Task Heads\nAffinity, ADMET, Risk, Novelty]
  S4 --> S5[Meta-Model (LightGBM w/ monotonicity)\nComposite Priority Score]
end

%% ========== OUTPUT ==========
S5 --> O1[Ranked List]
S5 --> O2[Evidence Linking\nRetrieve passages from Text Index]
F2 --> O2
F3 --> O1

O1 --> U1[User Dashboard / API]
O2 --> U1

```

  # Search Modes

EL-CAP exposes three complementary retrieval modes that feed the same scoring pipeline.

---

## 1) Text-Only (context → molecules)

**Input:** free-form requirements (target, assay conditions, TPP constraints)  
**Query vector:** `h_query = NLP(text)`  
**Index queried:** molecular vectors `h_mol` (vector store)  

**Use cases:** early triage from literature/TPP; no starting structure  
**Pros:** broad recall from context; fastest ideation  
**Cons:** semantic gap (text ↔ structure) without good contrastive pretraining  

**Flow:**  
text → NLP → h_query → ANN(h_query, {h_mol}) → top-K molecules → scoring → rank

---

## 2) Mol-Only (molecule → analogs/evidence)

**Input:** candidate SMILES/SDF  
**Query vector:** `h_mol* = GNN(mol)`  
**Index queried:** molecular vectors (analogs) and/or text vectors (evidence)  

**Use cases:** lead optimization, IP proximity, explainability  
**Pros:** robust for scaffold-centric workflows  
**Cons:** lacks explicit context; mitigated at scoring stage via fusion with optional text  

**Flow:**  
mol → GNN → h_mol* → ANN(h_mol*, {h_mol}) → top-K → (optional ANN to texts) → scoring → rank


---

## 3) Mol+Text (joint context)

**Input:** molecule + requirements (target/assay/ADMET rules)  
**Query vector:** `h_qry = g(h_mol*, h_query)` via concat/MLP or light cross-attention  
**Index queried:** molecular (and optionally text) vectors using `h_qry`  

**Use cases:** context-aware analog search; best Recall@K under constraints  
**Pros:** highest relevance; aligns structure with requirements before scoring  
**Cons:** most complex training (needs positive (mol, context) pairs and hard negatives)  

**Flow:**  
mol → GNN → h_mol*
text → NLP → h_query
h_qry = g(h_mol*, h_query) → ANN(h_qry) → top-K → scoring → rank

---

## Retrieval → Scoring bridge (common to all modes)

- **Retrieval output:** top-K `mol_id` (+ cached `h_mol`, metadata, candidate evidence)  
- **Scoring input:** `[h_mol, h_query, structured features]` → Fusion → Heads → Meta-Model  
- **Output:** composite score + evidence cards (linked assay snippets, patents, PAINS/Lipinski/hERG/CYP flags)  

---

## KPIs by stage

- **Retrieval:** Recall@K, NDCG@K, latency p95  
- **Scoring:** RMSE (affinity), ROC-AUC/PR-AUC (ADMET/risk), calibration (ECE), cost/1k queries  

---

## Practical defaults

- **Embedding dim:** 384–768; cosine similarity with L2-normalized vectors  
- **ANN:** HNSW or IVF-PQ (GPU) depending on scale; K=200–1000 for rerank window  
- **Fusion:** start with concat + MLP; upgrade to cross-attention for localized effects  
- **Meta-model:** linear regression for interpretability; LightGBM for nonlinearity if needed



