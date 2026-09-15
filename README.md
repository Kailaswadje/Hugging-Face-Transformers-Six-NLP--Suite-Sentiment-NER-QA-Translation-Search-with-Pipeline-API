# 🤗 Hugging Face Transformers — Six NLP Tasks, Suite sentiment, NER, QA Translation search with pipeline API

A hands-on survey of **six core NLP tasks** — sentiment classification, summarization, named entity recognition, question answering, translation, and semantic search — each solved with a **pretrained transformer model downloaded straight from Hugging Face Hub**, no fine-tuning, no training loop, no API key. Every task is tested against real, messy text: an actual internship acceptance email, a Bollywood movie review, and a Wikipedia-style company profile.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Pipeline%20API-FFD21E)
![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Semantic%20Search-FF6F00)
![NLP](https://img.shields.io/badge/NLP-Multi--Task-green)

---

## 📌 Overview

Hugging Face's `pipeline()` function is the fastest path from "I have text" to "I have an NLP result" — one line loads a pretrained model, tokenizer, and post-processing logic together, ready to run. This notebook uses that single function across six genuinely different tasks, proving how much capability comes bundled into a well-designed abstraction, all running **locally with open-weights models** rather than through a hosted API.

---

## 🔬 Task 1 — Sentiment / Text Classification

```python
classifier = pipeline(task="text-classification")
classifier(review)   # → [{'label': 'POSITIVE', 'score': 0.99...}]
```
Tested against a genuinely positive movie review (a Baahubali review), a rewritten negative version of the *same* review, and a neutral throwaway sentence ("I went out in the evening to the garden") — a clean three-point sanity check across the full sentiment spectrum, not just a single happy-path example.

---

## 🔬 Task 2 — Text Summarization

```python
summarizer = pipeline(task="summarization", max_length=62)
summarizer(editorial)
```
Run against a long-form cricket World Cup editorial and the internship email, first with an explicit `max_length=62` constraint, then again with the pipeline's own default length — a direct comparison of how much the summary length parameter actually changes the output.

---

## 🔬 Task 3 — Named Entity Recognition (NER)

```python
ner_tagger = pipeline("ner")
ner_tagger("Apple is planning to buy Samsung")
```
Three short, deliberately ambiguous test sentences probe whether the model can tell **"Apple" the company** from **"apple" the fruit** based on surrounding context alone — before scaling up to tagging entities across the full movie review and the internship email.

---

## 🔬 Task 4 — Extractive Question Answering

```python
reader = pipeline("question-answering")
reader(question="When are we going to receive an offer letter?", context=email)
```
Questions are answered by **extracting the exact span of text** from the given context — no generation, no hallucination risk. Tested against the internship email and a Reliance Industries company profile, including one **deliberately unanswerable question** ("give me the revenue details of microsoft organisation?" against a passage about Reliance) — an honest test of whether the model can recognize when an answer genuinely isn't present in the context, the same failure-testing discipline used throughout this series' RAG notebooks.

---

## 🔬 Task 5 — Machine Translation

```python
translate = pipeline("translation_en_to_fr")
translate(email, max_length=1000)
```
The full internship email translated end-to-end from English to French in a single call — demonstrating that `pipeline()`'s one-line pattern scales to sequence-to-sequence generation tasks just as easily as classification tasks.

---

## 🔬 Task 6 — Semantic Search with Sentence Embeddings

This is the notebook's most substantial section, building a small semantic search engine from scratch:

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(documents)   # 10 documents → (10, 384) embedding matrix
```

### Cosine Similarity + Top-K Retrieval
```python
new_text_embedding = model.encode("What is Artificial Intelligence?")
cos_scores = util.pytorch_cos_sim(new_text_embedding, document_embeddings)[0]
top_results = torch.topk(cos_scores, k=1)
```
Every document is embedded once into a `(10, 384)` matrix; any new query is embedded on demand and compared against all ten via cosine similarity — the same core mechanic behind every vector-store retriever built elsewhere in this series, here implemented **from first principles with raw tensors**, no FAISS or Chroma involved.

### Wrapped as a Reusable Function
```python
def semantic_search_engine(query, embedder_model):
    query_embedding = embedder_model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k=1)
    idx = top_results.indices[0]
    return documents[idx]

semantic_search_engine("Tell me what is yoga", model)      # → correctly finds the yoga document
semantic_search_engine("Tell me about galaxy", model)       # → correctly finds the Milky Way document
```
Two test queries — phrased nothing like the stored documents' exact wording — both retrieve the correct match, proving genuine semantic (not keyword) matching from a hand-rolled ~15-line retrieval function.

---

## 🗂️ Repository Structure

```
huggingface-transformers-nlp-tasks/
├── GenAI_Hugging_Face_Transformer_Models.ipynb   # Main notebook
├── requirements.txt                                # Dependencies
├── .gitignore                                      # Keeps model cache and secrets out of git
├── .env.example                                    # Template for optional environment variables
└── README.md                                       # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- No API key required — every model used is public on Hugging Face Hub

### Installation

```bash
git clone https://github.com/Kailaswadje/huggingface-transformers-nlp-tasks.git
cd huggingface-transformers-nlp-tasks

pip install -r requirements.txt

jupyter notebook GenAI_Hugging_Face_Transformer_Models.ipynb
```

> 💡 First run downloads several pretrained models (classification, summarization, NER, QA, translation, and the sentence embedding model) — expect a few minutes and a few hundred MB of disk usage on first execution. Subsequent runs use the local Hugging Face cache.

---

## 🧠 Key Takeaways

- **`pipeline()` is an entire NLP task in one function call** — model selection, tokenization, inference, and output formatting are all bundled, making it the fastest way to prototype an NLP capability before deciding whether it needs customization
- **Extractive QA cannot hallucinate an answer** — because it selects a text span rather than generating free text, testing it with an unanswerable question is a genuinely informative failure case, not just a formality
- **NER context-sensitivity is a real, testable property** — "Apple" resolving differently in "Apple is good for health" vs. "Apple is planning to buy Samsung" shows the model using surrounding words, not a lookup table
- **Semantic search doesn't require a vector database** — a `SentenceTransformer` model, cosine similarity, and `torch.topk` are enough to build genuine semantic retrieval from scratch, the same principle FAISS and Chroma package at production scale elsewhere in this series
- **All of this runs with open-weights, locally-cached models** — no API key, no per-call billing, a useful contrast to the hosted-API-based notebooks earlier in the series

---

## 🔮 Possible Extensions

- [ ] Swap the default models for task-specific fine-tuned checkpoints (e.g. a finance-tuned sentiment model)
- [ ] Scale the semantic search section to hundreds of documents and benchmark retrieval speed against FAISS
- [ ] Add a zero-shot classification pipeline for label sets defined at inference time
- [ ] Combine the QA and semantic search sections into a minimal RAG system built entirely on `transformers` + `sentence-transformers`, no LangChain required

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this made the Hugging Face pipeline API click for you, consider giving it a star!
