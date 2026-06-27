# 🧠 AI Document Assistant

A production-quality RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, **ChromaDB**, and **sentence-transformers**. Upload PDFs or paste text, ask questions, and get accurate answers grounded in your documents — with page-level citations and confidence scores.

## ✨ Features

- Multi-PDF upload (drag & drop) and paste-text input
- OCR fallback for scanned/image-only PDFs (requires Tesseract)
- Chunking via `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap)
- Local embeddings via `sentence-transformers/all-MiniLM-L6-v2` (no API cost)
- Persistent vector storage in ChromaDB with duplicate-embedding prevention
- Chat interface with conversation history, source citations, page numbers, and confidence scores
- Copy / download answers, export full chat as Markdown or JSON
- Supports both **OpenAI** and **Google Gemini** as the LLM backend
- Bonus tools: document summary, key points, quiz generator, flashcards, answer translation, keyword search, token usage & estimated cost tracking
- Clear database / delete individual documents
- Dark-mode compatible UI

## 📁 Project Structure

```
project/
├── app.py                 # Main entry point (home page + pipeline bootstrap)
├── config.py               # Centralized settings loaded from .env
├── requirements.txt
├── .env.example
├── pages/
│   ├── upload.py            # Dedicated upload & document management page
│   ├── chat.py               # Dedicated chat + bonus tools page
│   └── settings.py           # Settings, usage stats, DB management
├── rag/
│   ├── loader.py              # PDF/text extraction (+ OCR)
│   ├── splitter.py            # Chunking with page metadata
│   ├── embeddings.py          # sentence-transformers wrapper
│   ├── vector_store.py        # ChromaDB persistence layer
│   ├── retriever.py            # Similarity search + confidence
│   ├── llm.py                   # OpenAI / Gemini provider abstraction
│   └── chain.py                  # RAG orchestration + bonus features
├── utils/
│   └── helpers.py                  # Session-state & export utilities
├── data/uploads/                    # Uploaded PDFs are saved here
└── vector_db/                        # ChromaDB persistent storage
```

## 🚀 Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - `LLM_PROVIDER` to `openai` or `gemini`
   - `OPENAI_API_KEY` (if using OpenAI) or `GOOGLE_API_KEY` (if using Gemini)

4. **(Optional) Enable OCR** for scanned PDFs — install Tesseract:
   ```bash
   # macOS
   brew install tesseract
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   ```
   If `tesseract` isn't on your PATH, set `TESSERACT_CMD` in `.env` to its full path.

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

   The first run will download the embedding model (`all-MiniLM-L6-v2`, ~90MB) from Hugging Face — this requires internet access once; it's cached locally afterward.

## 🧩 How It Works (RAG Pipeline)

1. User uploads a PDF or pastes text
2. Text is extracted (with OCR fallback for image-based PDFs) and cleaned
3. Text is split into 1000-character chunks with 200-character overlap
4. Each chunk is embedded using `all-MiniLM-L6-v2`
5. Embeddings + metadata (source, page number) are stored in ChromaDB
6. User asks a question → the question is embedded
7. ChromaDB performs similarity search to retrieve the top-K most relevant chunks
8. Retrieved chunks are inserted into a strict prompt template that forbids hallucination
9. The LLM (OpenAI or Gemini) generates an answer
10. The answer is displayed with source citations, page numbers, and a confidence score

## ⚠️ Troubleshooting

| Problem | Likely Cause |
|---|---|
| "No API key configured" | Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in `.env` |
| "Could not open PDF" | File may be corrupted or password-protected |
| "No extractable text found" | Scanned PDF with OCR unavailable — install Tesseract |
| Slow first run | Embedding model is downloading (one-time, ~90MB) |
| Answers seem ungrounded | Lower temperature, increase Top K, or re-check chunking |

## 📝 Notes on Cost Estimates

Token/cost figures shown in the app use approximate public pricing tables and are intended as directional guidance only — refer to your provider's billing dashboard for exact charges.
