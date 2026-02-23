# 📚 RAG Chatbot with LangGraph Memory (Flask + Groq + Chroma)

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions from **PDF documents** and **YouTube videos**, built using **LangChain**, **LangGraph memory**, **Chroma vector database**, and **Groq LLM** with a **Flask web interface**.

This project demonstrates how to combine document retrieval with conversational memory to create an intelligent assistant that remembers previous context during a session.

---

# 🚀 Features

✅ PDF document ingestion
✅ YouTube audio transcription (Whisper)
✅ Vector database with ChromaDB
✅ HuggingFace embeddings
✅ Groq LLM (fast inference)
✅ LangGraph conversation memory
✅ Flask web UI
✅ Source document & page reference display
✅ Modular architecture (config + ingest + app)

---

# 🧠 How It Works (Architecture)

1. **Ingestion Phase**

   * Load PDFs and YouTube audio
   * Convert speech → text
   * Split into chunks
   * Generate embeddings
   * Store in Chroma vector database

2. **Chat Phase**

   * User asks a question
   * Relevant chunks retrieved from vector DB
   * Context + question sent to LLM
   * LangGraph maintains conversation memory
   * Response returned with references

---

# 📂 Project Structure

```
RAG-CHATBOT/
│── app.py                  # Flask chatbot app
│── ingest.py               # Document ingestion pipeline
│── config.py               # Configuration settings
│── .env                    # Environment variables template
│── README.md
│
├── data/                   # PDF documents
├── docs/
│   ├── youtube/            # Downloaded YouTube audio
│   └── chroma/             # Vector database
│
└── templates/
    └── index.html          # Flask UI
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

## 3️⃣ Install Dependencies

If using pip:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

---

# 🔐 Environment Variables

Create `.env` file in project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can copy from:

```
.env.example
```

---

# 📥 System Requirements

⚠️ Required for YouTube transcription:

Install **FFmpeg**

Windows:
https://ffmpeg.org/download.html

After installation → add to PATH.

---

# 📊 Step 1: Document Ingestion

Run once to create vector database.

```bash
python ingest.py
```

This will:

* Process PDFs from `data/`
* Download & transcribe YouTube video
* Create embeddings
* Store in `docs/chroma/`

---

# 💬 Step 2: Run Chatbot

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 🧾 Configuration

Edit `config.py`:

```python
YOUTUBE_VIDEO_URL = "your_url"
PDF_SOURCE_DIRECTORY = "data"
CHROMA_PERSIST_DIRECTORY = "docs/chroma"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
CHUNK_SIZE = 2028
CHUNK_OVERLAP = 250
```

---

# 🧠 Memory System (LangGraph)

The chatbot uses:

```
MemorySaver()
```

This enables:

✅ Conversation history
✅ Context continuity
✅ Session-based memory

Each session uses a `thread_id` to maintain state.

---

# 🔎 Retrieval Process

1. Similarity search (Top K = 3)
2. Context extraction
3. Prompt construction
4. LLM response generation
5. Source references returned

---

# 🤖 Model Used

LLM: **Llama 3.1 8B Instant (Groq)**
Embeddings: **multilingual-e5-large**
Speech-to-Text: **Faster-Whisper**

---

# 🖥️ API Endpoint

POST `/chat`

Request:

```json
{
  "message": "What is machine learning?"
}
```

Response:

```json
{
  "response": "Answer text..."
}
```

---

# 📌 Example Use Cases

* Academic question answering
* Research assistant
* Lecture video Q&A
* Knowledge base chatbot
* Internal company documentation bot

---

# ⚡ Performance Tips (Low RAM PCs)

If using 4GB RAM laptop:

Change embedding model:

```python
all-MiniLM-L6-v2
```

Use smaller Whisper model:

```
small
```

---

# 🛠️ Future Improvements

* User authentication
* Chat history database
* Docker deployment
* Streaming responses
* Multi-document upload UI
* Cloud vector database

---

# 🐞 Troubleshooting

### Chroma DB not loading

Make sure:

```
docs/chroma/
```

exists and ingestion completed.

---

### Whisper errors

Install FFmpeg properly.

---

### API key error

Check `.env` file.

---

# 🙌 Acknowledgements

* LangChain
* LangGraph
* Groq
* HuggingFace
* ChromaDB

---

❗ Important Setup Instructions
1️⃣ Add Your YouTube Video Link

Open config.py and update the YouTube URL:

YOUTUBE_VIDEO_URL: str = "PASTE_YOUR_YOUTUBE_LINK_HERE"



2️⃣ Add PDF Files

Place all your PDF documents inside the data/ folder.

Project structure example:

data/
   book.pdf
   notes.pdf
   research_paper.pdf

The ingestion script will automatically read all PDFs from this folder.