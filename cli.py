"""
Interactive CLI for the Astro Conversational Insight Agent.

Runs a multi-turn conversation loop:
1. Collects user birth details once at the start
2. Loops: takes a question → shows response → asks "continue? y/n"
3. Maintains full session memory across turns
"""

import os
import sys
import json
import logging

# Suppress noisy logs for CLI experience
# Suppress noisy warnings before any imports
logging.basicConfig(level=logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
load_dotenv()

from app.profile import enrich_profile
from app.session import SessionManager
from app.rag.indexer import KnowledgeIndexer
from app.rag.retriever import KnowledgeRetriever
from app.rag.intent import classify_intent, Intent
from app.llm.client import get_llm_client, build_system_prompt


def setup():
    """Initialize RAG index and LLM client."""
    print("🔮 Initializing Astro Agent...")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    indexer = KnowledgeIndexer(data_dir=data_dir)
    num_chunks = indexer.load_all_documents()

    # Suppress BertModel LOAD REPORT and progress bars during model loading
    import io
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        indexer.build_index()
    finally:
        sys.stdout = _real_stdout

    retriever = KnowledgeRetriever(indexer, similarity_threshold=0.25, top_k=5)
    llm_client = get_llm_client()
    session_manager = SessionManager(max_history=10)

    print(f"✅ Ready! Indexed {num_chunks} knowledge chunks.\n")
    return retriever, llm_client, session_manager


def collect_user_profile():
    """Collect user birth details interactively."""
    print("=" * 55)
    print("  🌟 Astro Conversational Insight Agent 🌟")
    print("=" * 55)
    print("\nLet me know a bit about you first:\n")

    name = input("  Your name: ").strip() or "User"
    birth_date = input("  Date of birth (YYYY-MM-DD): ").strip()
    birth_time = input("  Time of birth (HH:MM, or press Enter to skip): ").strip() or None
    birth_place = input("  Place of birth (or press Enter to skip): ").strip() or None

    lang = input("  Preferred language (en/hi) [en]: ").strip().lower()
    if lang not in ("en", "hi"):
        lang = "en"

    profile = {
        "name": name,
        "birth_date": birth_date,
        "birth_time": birth_time,
        "birth_place": birth_place,
        "preferred_language": lang,
    }

    enriched = enrich_profile(profile)
    print(f"\n  ✨ Your zodiac sign: {enriched['zodiac']}")
    if enriched.get("age"):
        print(f"  🎂 Age: {enriched['age']}")

    return enriched


def handle_message(message, enriched_profile, session_id, retriever, llm_client, session_manager):
    """Process a single user message through the full pipeline."""
    zodiac = enriched_profile["zodiac"]

    # 1. Intent classification
    should_retrieve, intent = classify_intent(message)

    # 2. Handle summary intent directly from session
    if intent == Intent.SUMMARY:
        summary = session_manager.get_session_summary(session_id)
        if summary:
            session_manager.add_turn(session_id, "user", message)
            session_manager.add_turn(session_id, "assistant", summary, retrieval_used=False)
            return summary, [], False

    # 3. RAG retrieval (if needed)
    retrieved_context = ""
    context_sources = []

    if should_retrieve:
        enhanced_query = f"{zodiac} {message}"
        results = retriever.retrieve(enhanced_query)
        if results:
            retrieved_context = retriever.format_context(results)
            context_sources = retriever.get_context_sources(results)

    # 4. Build LLM prompt
    language = enriched_profile.get("preferred_language", "en")
    system_prompt = build_system_prompt(
        zodiac=zodiac,
        user_profile=enriched_profile,
        retrieved_context=retrieved_context,
        language=language,
    )

    conversation_history = session_manager.get_history(session_id)

    # 5. Generate response
    response_text = llm_client.generate(
        user_message=message,
        system_message=system_prompt,
        conversation_history=conversation_history,
    )

    # 6. Store turn in session
    retrieval_used = should_retrieve and bool(context_sources)
    session_manager.add_turn(session_id, "user", message)
    session_manager.add_turn(
        session_id, "assistant", response_text,
        retrieval_used=retrieval_used,
        context_sources=context_sources,
    )

    return response_text, context_sources, retrieval_used


def main():
    retriever, llm_client, session_manager = setup()
    enriched_profile = collect_user_profile()
    zodiac = enriched_profile["zodiac"]

    session_id = "cli-session"
    session_manager.get_or_create_session(session_id, enriched_profile)

    turn_number = 0

    print("-" * 55)
    print(f"  Ask me anything about your career, love,")
    print(f"  spirituality, or planetary influences!")
    print("-" * 55)

    while True:
        turn_number += 1
        print(f"\n[Turn {turn_number}]")

        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n🙏 Namaste! May the stars guide your path. Goodbye!")
            break

        if not message:
            print("  (Please type a question)")
            turn_number -= 1
            continue

        # Process the message
        print("\n🔭 Thinking...\n")
        response, sources, retrieval_used = handle_message(
            message, enriched_profile, session_id,
            retriever, llm_client, session_manager,
        )

        # Display response
        print(f"🌟 Astro Guide ({zodiac}):")
        print(f"  {response}\n")

        # Show metadata
        if sources:
            print(f"  📚 Sources: {', '.join(sources)}")
        print(f"  🔍 Retrieval used: {'Yes' if retrieval_used else 'No'}")

        # Continue checkpoint
        print()
        try:
            cont = input("Want to continue? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cont = "n"

        if cont != "y":
            print("\n🙏 Namaste! May the stars guide your path. Goodbye!")
            break

    # Show session summary on exit
    print(f"\n📊 Session stats: {turn_number} turn(s) in this conversation.")


if __name__ == "__main__":
    main()
