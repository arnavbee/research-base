# arXiv AI Research Q&A System - PRD

## Original Problem Statement
Build a full-stack web application that automatically monitors arXiv's AI research papers, ingests them daily into a vector database, and provides an AI-powered Q&A interface where users can ask natural language questions about the latest research.

## User Personas
1. **AI Researchers** - Need quick access to latest research findings
2. **ML Engineers** - Looking for implementation details and techniques
3. **Academics** - Seeking literature for papers and citations
4. **Students** - Learning about cutting-edge AI developments

## Core Requirements (Static)
- ✅ Automated paper ingestion from arXiv (cs.AI, cs.LG, cs.CL, cs.CV, cs.NE)
- ✅ Vector database storage with semantic search (ChromaDB)
- ✅ AI-powered Q&A with RAG (Claude Sonnet via Emergent LLM key)
- ✅ Real-time statistics dashboard
- ✅ Recent papers panel
- ✅ Source citations with relevance scores
- ✅ Dark theme UI with Obsidian Intelligence design

## What's Been Implemented (January 2026)

### Backend (FastAPI)
- `/api/search` - RAG-powered semantic search with Claude answers
- `/api/papers/recent` - Get latest ingested papers
- `/api/stats` - System statistics (total papers, sync status)
- `/api/sync/trigger` - Manual sync trigger
- `/api/sync/status` - Check sync progress
- ChromaDB for vector storage
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- MongoDB for paper metadata

### Frontend (React)
- Statistics cards (Total Papers, Today Added, Last Sync, Categories)
- Glassmorphism search interface with example questions
- AI answer display with formatted text
- Source citations with relevance scores and arXiv links
- Recent papers panel (scrollable, 20 papers)
- Responsive dark theme design

## Technology Stack
- **Backend**: FastAPI, Python 3.11
- **Frontend**: React, Tailwind CSS, shadcn/ui
- **Vector DB**: ChromaDB (self-hosted)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude Sonnet via Emergent LLM key
- **Database**: MongoDB
- **Paper Source**: arXiv API

## Prioritized Backlog

### P0 (Critical) - Completed ✅
- [x] Paper ingestion pipeline
- [x] Vector search functionality
- [x] AI Q&A with Claude
- [x] Frontend dashboard

### P1 (High Priority) - Future
- [ ] Scheduled daily sync (cron job)
- [ ] PDF full-text extraction
- [ ] Error handling improvements

### P2 (Medium Priority) - Future
- [ ] User accounts for saved queries
- [ ] Email notifications for topics
- [ ] Advanced filters (date, authors)
- [ ] Export literature reviews

### P3 (Low Priority) - Future
- [ ] Paper recommendations
- [ ] Multi-language support
- [ ] Trending topics dashboard

## Next Tasks
1. Set up scheduled daily sync job
2. Add PDF text extraction for fuller context
3. Implement caching for frequently asked questions
4. Add user authentication for personalized features
