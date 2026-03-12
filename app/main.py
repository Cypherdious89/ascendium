"""
Astro Conversational Insight Agent — FastAPI Application.

Multi-turn conversational AI service with:
- Session-based memory (sliding window)
- Intent-aware RAG retrieval (FAISS + sentence-transformers)
- Zodiac derivation from birth date
- Hindi/English language toggle
- OpenAI / Stub LLM abstraction
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.models import ChatRequest, ChatResponse
from app.profile import enrich_profile
from app.session import SessionManager
from app.rag.indexer import KnowledgeIndexer
from app.rag.retriever import KnowledgeRetriever
from app.rag.intent import classify_intent, Intent
from app.llm.client import get_llm_client, build_system_prompt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
session_manager = SessionManager(max_history=10)
indexer: KnowledgeIndexer = None
retriever: KnowledgeRetriever = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build FAISS index from data/ directory."""
    global indexer, retriever, llm_client

    logger.info("Starting up — building knowledge index...")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    indexer = KnowledgeIndexer(data_dir=data_dir)
    num_chunks = indexer.load_all_documents()
    indexer.build_index()

    retriever = KnowledgeRetriever(indexer, similarity_threshold=0.25, top_k=5)

    llm_client = get_llm_client()

    logger.info(f"Ready! Indexed {num_chunks} knowledge chunks.")

    yield

    logger.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Astro Conversational Insight Agent",
    description="Multi-turn astrology AI with RAG-powered knowledge retrieval",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "index_size": indexer.index.ntotal if indexer and indexer.index else 0,
        "llm_mode": "gemini" if os.getenv("GEMINI_API_KEY") else "stub",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — handles multi-turn astrology conversations.

    Pipeline:
    1. Get/create session → derive zodiac
    2. Classify intent → decide if retrieval is needed
    3. If retrieval: query FAISS → get relevant chunks
    4. Build LLM prompt with context
    5. Generate response
    6. Store turn in session
    7. Return structured response
    """
    try:
        # ── 1. Session & Profile ──────────────────────────────────────────
        profile_dict = request.user_profile.model_dump()
        enriched_profile = enrich_profile(profile_dict)
        zodiac = enriched_profile["zodiac"]

        session = session_manager.get_or_create_session(
            session_id=request.session_id,
            user_profile=enriched_profile,
        )

        # ── 2. Intent Classification ─────────────────────────────────────
        should_retrieve, intent = classify_intent(request.message)

        # ── 3. Handle special intents ─────────────────────────────────────
        if intent == Intent.SUMMARY:
            summary = session_manager.get_session_summary(request.session_id)
            if summary:
                session_manager.add_turn(
                    request.session_id, "user", request.message
                )
                session_manager.add_turn(
                    request.session_id, "assistant", summary,
                    retrieval_used=False,
                )
                return ChatResponse(
                    response=summary,
                    zodiac=zodiac,
                    context_used=[],
                    retrieval_used=False,
                )

        # ── 4. RAG Retrieval (if needed) ──────────────────────────────────
        retrieved_context = ""
        context_sources = []

        if should_retrieve and retriever:
            # Enhance query with zodiac for better relevance
            enhanced_query = f"{zodiac} {request.message}"
            results = retriever.retrieve(enhanced_query)

            if results:
                retrieved_context = retriever.format_context(results)
                context_sources = retriever.get_context_sources(results)

        # ── 5. Build LLM Prompt ───────────────────────────────────────────
        language = request.user_profile.preferred_language
        system_prompt = build_system_prompt(
            zodiac=zodiac,
            user_profile=enriched_profile,
            retrieved_context=retrieved_context,
            language=language,
        )

        conversation_history = session_manager.get_history(request.session_id)

        # ── 6. Generate Response ──────────────────────────────────────────
        response_text = llm_client.generate(
            user_message=request.message,
            system_message=system_prompt,
            conversation_history=conversation_history,
        )

        # ── 7. Store Turn in Session ──────────────────────────────────────
        session_manager.add_turn(
            request.session_id, "user", request.message
        )
        session_manager.add_turn(
            request.session_id, "assistant", response_text,
            retrieval_used=should_retrieve and bool(context_sources),
            context_sources=context_sources,
        )

        # ── 8. Return Response ────────────────────────────────────────────
        return ChatResponse(
            response=response_text,
            zodiac=zodiac,
            context_used=context_sources,
            retrieval_used=should_retrieve and bool(context_sources),
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )
