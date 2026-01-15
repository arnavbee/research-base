from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import asyncio

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="arXiv AI Research Q&A API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Pydantic Models ==========

class PaperSource(BaseModel):
    model_config = ConfigDict(extra="ignore")
    arxiv_id: str
    title: str
    authors: List[str]
    relevance_score: float
    arxiv_url: str
    abstract_snippet: Optional[str] = None

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    answer: str
    sources: List[PaperSource]
    timestamp: str
    query: str

class Paper(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str
    arxiv_url: str
    pdf_url: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class StatsResponse(BaseModel):
    total_papers: int
    today_added: int
    last_sync: Optional[str]
    categories: List[str]
    sync_status: str

class SyncLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str
    completed_at: Optional[str] = None
    papers_added: int = 0
    papers_failed: int = 0
    status: str = "running"
    error_message: Optional[str] = None

# ========== ChromaDB & Embedding Setup ==========

import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="/app/backend/chroma_data")
collection = chroma_client.get_or_create_collection(
    name="arxiv_papers",
    metadata={"hnsw:space": "cosine"}
)

# Initialize embedding model (will be loaded on first use)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
    return embedding_model

def generate_embedding(text: str) -> List[float]:
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

# ========== arXiv Paper Fetching ==========

import arxiv

async def fetch_arxiv_papers(categories: List[str], max_results: int = 50) -> List[dict]:
    """Fetch recent papers from arXiv"""
    papers = []
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    try:
        search = arxiv.Search(
            query=category_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client()
        results = client.results(search)
        
        for result in results:
            paper = {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title.replace("\n", " ").strip(),
                "authors": [author.name for author in result.authors],
                "abstract": result.summary.replace("\n", " ").strip(),
                "categories": list(result.categories),
                "published_date": result.published.isoformat(),
                "arxiv_url": result.entry_id,
                "pdf_url": result.pdf_url
            }
            papers.append(paper)
            
    except Exception as e:
        logger.error(f"Error fetching arXiv papers: {e}")
        
    return papers

async def ingest_paper_to_vector_db(paper: dict) -> bool:
    """Add a paper to ChromaDB vector database"""
    try:
        # Check if paper already exists
        existing = collection.get(ids=[paper["arxiv_id"]])
        if existing and existing['ids']:
            return False
            
        # Create embedding text from title and abstract
        embedding_text = f"{paper['title']} {paper['abstract']}"
        embedding = generate_embedding(embedding_text)
        
        # Add to ChromaDB
        collection.add(
            ids=[paper["arxiv_id"]],
            embeddings=[embedding],
            metadatas=[{
                "title": paper["title"],
                "authors": ", ".join(paper["authors"][:5]),
                "abstract": paper["abstract"][:500],
                "categories": ", ".join(paper["categories"]),
                "published_date": paper["published_date"],
                "arxiv_url": paper["arxiv_url"],
                "pdf_url": paper["pdf_url"]
            }],
            documents=[embedding_text[:1000]]
        )
        
        # Also store in MongoDB for full metadata
        paper_doc = Paper(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper["authors"],
            abstract=paper["abstract"],
            categories=paper["categories"],
            published_date=paper["published_date"],
            arxiv_url=paper["arxiv_url"],
            pdf_url=paper["pdf_url"]
        )
        await db.papers.update_one(
            {"arxiv_id": paper["arxiv_id"]},
            {"$set": paper_doc.model_dump()},
            upsert=True
        )
        
        return True
    except Exception as e:
        logger.error(f"Error ingesting paper {paper.get('arxiv_id', 'unknown')}: {e}")
        return False

# ========== RAG with Claude ==========

from emergentintegrations.llm.chat import LlmChat, UserMessage

async def generate_answer_with_claude(query: str, context_papers: List[dict]) -> str:
    """Generate answer using Claude with paper context"""
    try:
        api_key = os.environ.get("EMERGENT_LLM_KEY")
        if not api_key:
            return "Error: EMERGENT_LLM_KEY not configured"
        
        # Build context from retrieved papers
        context_parts = []
        for i, paper in enumerate(context_papers, 1):
            context_parts.append(
                f"[Paper {i}]\n"
                f"Title: {paper.get('title', 'N/A')}\n"
                f"Authors: {paper.get('authors', 'N/A')}\n"
                f"Abstract: {paper.get('abstract', 'N/A')[:800]}\n"
            )
        
        context = "\n\n".join(context_parts)
        
        system_message = """You are an AI research assistant specializing in machine learning and artificial intelligence papers from arXiv. 
Your role is to answer questions about AI research based on the provided paper abstracts and summaries.
Be concise, accurate, and cite the relevant papers when appropriate.
If the provided papers don't contain enough information to answer the question, say so clearly."""

        prompt = f"""Based on the following research papers from arXiv, please answer this question:

Question: {query}

Research Papers Context:
{context}

Please provide a comprehensive but concise answer based on these papers. Reference specific papers when making claims."""

        chat = LlmChat(
            api_key=api_key,
            session_id=f"arxiv-qa-{uuid.uuid4()}",
            system_message=system_message
        ).with_model("anthropic", "claude-sonnet-4-5-20250929")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating answer with Claude: {e}")
        return f"I apologize, but I encountered an error while generating the answer. Please try again. Error: {str(e)}"

# ========== API Endpoints ==========

@api_router.get("/")
async def root():
    return {"message": "arXiv AI Research Q&A API", "version": "1.0.0"}

@api_router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """Search papers and get AI-generated answer"""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Search ChromaDB for similar papers
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["metadatas", "distances", "documents"]
        )
        
        # Process results
        sources = []
        context_papers = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, arxiv_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 1.0
                
                # Convert distance to similarity score (cosine distance to similarity)
                relevance_score = round((1 - distance) * 100, 1)
                
                source = PaperSource(
                    arxiv_id=arxiv_id,
                    title=metadata.get('title', 'Unknown'),
                    authors=metadata.get('authors', '').split(', ')[:3],
                    relevance_score=relevance_score,
                    arxiv_url=metadata.get('arxiv_url', f'https://arxiv.org/abs/{arxiv_id}'),
                    abstract_snippet=metadata.get('abstract', '')[:200] + "..."
                )
                sources.append(source)
                
                context_papers.append({
                    'title': metadata.get('title', ''),
                    'authors': metadata.get('authors', ''),
                    'abstract': metadata.get('abstract', '')
                })
        
        # Generate answer with Claude
        if context_papers:
            answer = await generate_answer_with_claude(query, context_papers)
        else:
            answer = "I couldn't find any relevant papers in the database to answer your question. Try triggering a sync to fetch the latest papers from arXiv."
        
        return SearchResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@api_router.get("/papers/recent")
async def get_recent_papers(limit: int = 20):
    """Get recently added papers"""
    papers = await db.papers.find(
        {},
        {"_id": 0}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    
    return papers

@api_router.get("/papers/{arxiv_id}")
async def get_paper(arxiv_id: str):
    """Get paper by arXiv ID"""
    paper = await db.papers.find_one({"arxiv_id": arxiv_id}, {"_id": 0})
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@api_router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    total_papers = await db.papers.count_documents({})
    
    # Count papers added today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_added = await db.papers.count_documents({
        "created_at": {"$gte": today_start.isoformat()}
    })
    
    # Get last sync info
    last_sync_log = await db.sync_logs.find_one(
        {"status": "completed"},
        {"_id": 0},
        sort=[("completed_at", -1)]
    )
    
    last_sync = None
    if last_sync_log and last_sync_log.get("completed_at"):
        last_sync = last_sync_log["completed_at"]
    
    # Get sync status
    running_sync = await db.sync_logs.find_one({"status": "running"}, {"_id": 0})
    sync_status = "syncing" if running_sync else "idle"
    
    categories = os.environ.get("ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL,cs.CV,cs.NE").split(",")
    
    return StatsResponse(
        total_papers=total_papers,
        today_added=today_added,
        last_sync=last_sync,
        categories=categories,
        sync_status=sync_status
    )

@api_router.post("/sync/trigger")
async def trigger_sync(background_tasks: BackgroundTasks):
    """Trigger manual paper sync"""
    # Check if sync is already running
    running_sync = await db.sync_logs.find_one({"status": "running"})
    if running_sync:
        return {"status": "already_running", "message": "A sync is already in progress"}
    
    # Create sync log
    sync_log = SyncLog(
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running"
    )
    await db.sync_logs.insert_one(sync_log.model_dump())
    
    # Run sync in background
    background_tasks.add_task(run_paper_sync, sync_log.id)
    
    return {"status": "started", "sync_id": sync_log.id, "message": "Sync started in background"}

async def run_paper_sync(sync_id: str):
    """Run the paper sync process"""
    papers_added = 0
    papers_failed = 0
    
    try:
        categories = os.environ.get("ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL,cs.CV,cs.NE").split(",")
        
        logger.info(f"Starting sync for categories: {categories}")
        
        # Fetch papers from arXiv
        papers = await fetch_arxiv_papers(categories, max_results=50)
        logger.info(f"Fetched {len(papers)} papers from arXiv")
        
        # Ingest each paper
        for paper in papers:
            success = await ingest_paper_to_vector_db(paper)
            if success:
                papers_added += 1
            else:
                papers_failed += 1
        
        # Update sync log
        await db.sync_logs.update_one(
            {"id": sync_id},
            {"$set": {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "papers_added": papers_added,
                "papers_failed": papers_failed,
                "status": "completed"
            }}
        )
        
        logger.info(f"Sync completed: {papers_added} added, {papers_failed} skipped/failed")
        
    except Exception as e:
        logger.error(f"Sync error: {e}")
        await db.sync_logs.update_one(
            {"id": sync_id},
            {"$set": {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "papers_added": papers_added,
                "papers_failed": papers_failed,
                "status": "failed",
                "error_message": str(e)
            }}
        )

@api_router.get("/sync/status")
async def get_sync_status():
    """Get current sync status"""
    running_sync = await db.sync_logs.find_one({"status": "running"}, {"_id": 0})
    if running_sync:
        return {"status": "syncing", "sync": running_sync}
    
    last_sync = await db.sync_logs.find_one(
        {},
        {"_id": 0},
        sort=[("started_at", -1)]
    )
    
    return {"status": "idle", "last_sync": last_sync}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
