# RAG Chatbot Backend

Production-ready RAG (Retrieval-Augmented Generation) chatbot backend for Physical AI & Humanoid Robotics documentation.

## Tech Stack

- **Python 3.9+**
- **FastAPI** - Web framework
- **Gemini API** - Embeddings (text-embedding-3-small) + Chat (Gemini .5 flash)
- **Qdrant Cloud** - Vector database
- **Neon Serverless Postgres** - Conversation storage
- **SQLAlchemy** - ORM with async support
- **Pydantic** - Data validation

## Prerequisites

1. **Python 3.9+**
2. **UV Package Manager** (recommended) or pip
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **External Services**:
   - GEMINI API key: https://aistudio.google.com/
   - Qdrant Cloud cluster: https://cloud.qdrant.io (free tier)
   - Neon Postgres database: https://neon.tech (free tier)

## Setup

### 1. Install Dependencies

Using UV (recommended):
```bash
cd backend
uv sync
```

Using pip:
```bash
cd backend
pip install -e .
```

### 2. Configure Environment Variables

Copy the example env file:
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```env
# GEMINI_API
GEMINI_API_KEY=sk-your-key-here

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key

# Neon Postgres
DATABASE_URL=postgresql+asyncpg://user:password@hostname/database

# Application (optional, defaults provided)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### 3. Apply Database Schema

Apply the Postgres schema to your Neon database:

```bash
# Connect to your Neon database
psql "postgresql://user:password@hostname/database"

# Run the schema file
\i specs/011-rag-chatbot-integration/contracts/database-schema.sql
```

Or use a database client to execute the SQL from:
`specs/011-rag-chatbot-integration/contracts/database-schema.sql`

### 4. Index Documentation

Index the documentation files to Qdrant:

```bash
cd backend
python scripts/index_docs.py --docs-dir ../book/docs
```

This will:
- Process all markdown/MDX files
- Generate embeddings
- Upload to Qdrant vector database

Expected output:
```
INFO - Found 50 markdown files
INFO - Processing file.md: 3 chunks
INFO - Indexed file.md (3 chunks)
...
INFO - Indexing complete: 50 files, 150 chunks
```

## Running the Backend

### Development Mode

```bash
cd backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger UI)
- **Health**: http://localhost:8000/api/health

### Production Mode

```bash
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### GET /api/health
Check health of all services (Qdrant, Postgres, GEMINI).

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "qdrant": "up",
    "postgres": "up",
    "geminiapi": "up"
  }
}
```

### POST /api/chat
Send a chat message and get AI response with sources.

**Request:**
```json
{
  "message": "What is Physical AI?",
  "session_id": "optional-uuid",
  "selected_text": "optional selected text from page"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "message": "Physical AI refers to...",
  "sources": [
    {
      "title": "Introduction to Physical AI",
      "file_path": "intro/physical-ai.md",
      "relevance_score": 0.92,
      "excerpt": "Physical AI is..."
    }
  ],
  "timestamp": "2025-12-01T12:00:00Z"
}
```

### GET /api/sessions/{session_id}/history
Get conversation history for a session.

**Response:**
```json
{
  "session_id": "uuid",
  "messages": [
    {
      "message_id": "uuid",
      "role": "user",
      "content": "What is ROS2?",
      "created_at": "2025-12-01T12:00:00Z"
    },
    {
      "message_id": "uuid",
      "role": "assistant",
      "content": "ROS2 is...",
      "context_used": {...},
      "created_at": "2025-12-01T12:00:01Z"
    }
  ]
}
```

## Testing

### Manual Testing

1. **Health Check:**
   ```bash
   curl http://localhost:8000/api/health
   ```

2. **Chat Query:**
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is Physical AI?"}'
   ```

3. **Session History:**
   ```bash
   curl http://localhost:8000/api/sessions/{session_id}/history
   ```

### Unit Tests (Coming Soon)

```bash
cd backend
pytest tests/ --cov=src --cov-report=html
```

## Deployment

### Option 1: Railway

1. Create new project: https://railway.app
2. Connect GitHub repository
3. Configure environment variables
4. Deploy

### Option 2: Render

1. Create new web service: https://render.com
2. Connect GitHub repository
3. Build command: `pip install -e .`
4. Start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Configure environment variables
6. Deploy

### Environment Variables for Production

Ensure these are set in your deployment platform:
- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `DATABASE_URL`
- `CORS_ORIGINS` (include your production frontend URL)
- `APP_ENV=production`

## Troubleshooting

### Qdrant Connection Failed
- Verify `QDRANT_URL` is correct (should be HTTPS)
- Check API key is valid
- Ensure collection exists (run indexing script)

### Postgres Connection Failed
- Verify `DATABASE_URL` format: `postgresql+asyncpg://user:password@host/db`
- Ensure database schema is applied
- Check connection limits (Neon free tier: 100 connections)

### GEMINI API Errors
- Verify API key is valid
- Check account has credits
- Monitor rate limits

### No Search Results
- Run indexing script to populate Qdrant
- Verify documents were processed correctly
- Check `CHUNK_SIZE` and `SIMILARITY_THRESHOLD` settings

## Project Structure

```
backend/
├── src/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/              # Pydantic models
│   │   ├── chat.py
│   │   └── document.py
│   ├── services/            # Business logic
│   │   ├── embedding.py
│   │   ├── vector_store.py
│   │   ├── llm.py
│   │   ├── conversation.py
│   │   └── rag_service.py
│   ├── api/                 # API endpoints
│   │   ├── health.py
│   │   ├── chat.py
│   │   └── sessions.py
│   └── utils/               # Utilities
│       ├── markdown.py
│       └── sanitization.py
├── scripts/
│   └── index_docs.py        # Indexing script
├── tests/                   # Tests (to be implemented)
├── pyproject.toml           # Dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## License

Part of the Physical AI & Humanoid Robotics documentation project.