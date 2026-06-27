# 🧠 AI Document Assistant

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Google-Gemini_API-blue?logo=google)
![ChromaDB](https://img.shields.io/badge/Vector%20Database-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

A production-ready **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **Google Gemini**, **ChromaDB**, and **Sentence Transformers**. Upload PDF documents or paste text, then interact with your knowledge base through a conversational AI interface that provides **context-aware answers**, **source citations**, **page references**, and **confidence scores**.

---

## 🚀 Live Demo

🌐 **Try the application**

https://ai-assistant-alk4f2bdikbrxmvwchbcq9.streamlit.app/

---

## 📸 Preview

> Add screenshots after uploading them to an `assets/` folder.

```text
assets/
├── home.png
├── upload.png
├── chat.png
└── settings.png
```

```markdown
![Home](assets/home.png)

![Chat](assets/chat.png)
```

---

# ✨ Features

### 📄 Document Processing

* Upload multiple PDF documents
* Paste custom text for indexing
* OCR support for scanned PDFs using Tesseract
* Automatic text extraction with PyMuPDF

### 🧠 Retrieval-Augmented Generation (RAG)

* Recursive text chunking
* Semantic embeddings using **all-MiniLM-L6-v2**
* Persistent ChromaDB vector database
* Duplicate document detection
* Fast similarity search

### 💬 AI Chat

* Conversational interface
* Source citations
* Page references
* Confidence scores
* Conversation history

### 🎁 Bonus Features

* Document summarization
* Key point extraction
* Quiz generation
* Flashcard generation
* Keyword search
* Multi-language translation
* Token usage tracking
* Cost estimation
* Copy & download responses
* Export conversations (Markdown / JSON)

---

# 🏗 System Architecture

```text
                    User
                      │
                      ▼
             Streamlit Web Interface
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     Upload PDFs             Paste Text
          │
          ▼
   PyMuPDF + OCR Extraction
          │
          ▼
   Recursive Text Splitter
          │
          ▼
 SentenceTransformer Embeddings
          │
          ▼
     Chroma Vector Database
          │
          ▼
   Semantic Similarity Search
          │
          ▼
   Relevant Context Retrieval
          │
          ▼
      Google Gemini API
          │
          ▼
  Context-Aware AI Response
```

---

# 🧩 Tech Stack

| Category        | Technology                 |
| --------------- | -------------------------- |
| Frontend        | Streamlit                  |
| Framework       | LangChain                  |
| LLM             | Google Gemini API / OpenAI |
| Vector Database | ChromaDB                   |
| Embeddings      | sentence-transformers      |
| PDF Processing  | PyMuPDF                    |
| OCR             | Tesseract OCR              |
| Translation     | Deep Translator            |
| Language        | Python                     |

---

# 📂 Project Structure

```text
project/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── README.md
│
├── pages/
│   ├── upload.py
│   ├── chat.py
│   └── settings.py
│
├── rag/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   └── chain.py
│
├── utils/
│   └── helpers.py
│
├── data/
│   └── uploads/
│
└── vector_db/
```

---

# 🔄 RAG Workflow

1. Upload one or more PDF documents.
2. Extract text using **PyMuPDF**.
3. Apply OCR when scanned pages are detected.
4. Split text into overlapping chunks.
5. Generate vector embeddings using **all-MiniLM-L6-v2**.
6. Store embeddings inside **ChromaDB**.
7. User submits a question.
8. Retrieve the most relevant document chunks.
9. Send retrieved context to the selected LLM.
10. Generate an accurate answer with citations.

---

# ⚙ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file.

```env
LLM_PROVIDER=gemini

GOOGLE_API_KEY=your_google_api_key

OPENAI_API_KEY=your_openai_api_key
```

---

## 5. Run the Application

```bash
streamlit run app.py
```

---

# 📦 Core Dependencies

* Streamlit
* LangChain
* LangChain Google GenAI
* ChromaDB
* Sentence Transformers
* PyMuPDF
* pytesseract
* Pillow
* Deep Translator
* Pandas
* NumPy

---

# 💡 How Retrieval-Augmented Generation Works

```text
PDF
 │
 ▼
Extract Text
 │
 ▼
Chunk Documents
 │
 ▼
Generate Embeddings
 │
 ▼
Store in ChromaDB
 │
 ▼
User Question
 │
 ▼
Similarity Search
 │
 ▼
Relevant Chunks
 │
 ▼
Gemini / OpenAI
 │
 ▼
Final Answer
```

---

# ⚠ Troubleshooting

| Problem           | Solution                                                    |
| ----------------- | ----------------------------------------------------------- |
| API key missing   | Configure `.env` correctly                                  |
| Empty PDF         | Ensure the PDF contains readable text                       |
| Scanned PDF       | Install Tesseract OCR                                       |
| Slow first launch | Embedding model downloads on first run                      |
| Weak answers      | Increase retrieval Top-K or upload better-quality documents |

---

# 📈 Future Improvements

* Multi-user authentication
* Cloud vector database
* Hybrid semantic + keyword search
* Streaming AI responses
* Document versioning
* Citation highlighting
* Support for DOCX, PPTX, CSV, and Excel
* Docker deployment
* Mobile-responsive UI

---

# 🛠 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Google Gemini API
* OpenAI Integration
* LangChain
* ChromaDB
* Vector Embeddings
* Semantic Search
* Prompt Engineering
* Streamlit Development
* OCR Processing
* Python Application Development

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Jaihind Murari**

* GitHub: https://github.com/yourusername
* LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.

It helps others discover the project and motivates future improvements.
