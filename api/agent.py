"""Pydantic AI agent for VIC - the voice of Vic Keegan.

Dual-path architecture:
- Fast path: Immediate response within 2 seconds
- Enrichment path: Background context building for better follow-ups
"""

import os
import asyncio
import httpx
from pydantic_ai import Agent
from pydantic import ValidationError
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .models import (
    ValidatedVICResponse,
    DeclinedResponse,
    SearchResults,
    FastVICResponse,
    EnrichedVICResponse,
    ExtractedEntity,
    EntityConnection,
    SuggestedTopic,
)
from .tools import search_articles, get_user_memory
from .agent_deps import VICAgentDeps
from .agent_config import get_fast_agent, get_enriched_agent, VIC_SYSTEM_PROMPT, SAFE_TOPIC_CLUSTERS

# Lazy-loaded agent instance (legacy)
_vic_agent: Optional[Agent] = None


# =============================================================================
# Session Context Store - In-memory storage for enrichment data
# =============================================================================

@dataclass
class SessionContext:
    """Enrichment context for a conversation session."""
    entities: list[ExtractedEntity] = field(default_factory=list)
    connections: list[EntityConnection] = field(default_factory=list)
    suggestions: list[SuggestedTopic] = field(default_factory=list)
    topics_discussed: list[str] = field(default_factory=list)
    last_response: str = ""
    enrichment_complete: bool = False
    # Name spacing: track turns since name was used
    turns_since_name_used: int = 0
    name_used_in_greeting: bool = False
    # Track the last suggestion VIC made for affirmation handling
    last_suggested_topic: str = ""


# Affirmation patterns - user confirming a suggestion
AFFIRMATION_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please", "aye",
    "absolutely", "definitely", "certainly", "indeed", "alright", "right",
}

AFFIRMATION_PHRASES = {
    "go on", "tell me more", "tell me", "go ahead", "yes please", "sure thing",
    "of course", "i'd like that", "i would like that", "sounds good", "sounds great",
    "let's do it", "let's hear it", "why not", "i'm interested", "please do",
}


# LRU-style session store (max 100 sessions)
_session_contexts: dict[str, SessionContext] = {}
MAX_SESSIONS = 100


def get_session_context(session_id: Optional[str]) -> SessionContext:
    """Get or create session context."""
    if not session_id:
        return SessionContext()

    if session_id not in _session_contexts:
        # Evict oldest if at capacity
        if len(_session_contexts) >= MAX_SESSIONS:
            oldest_key = next(iter(_session_contexts))
            del _session_contexts[oldest_key]
        _session_contexts[session_id] = SessionContext()

    return _session_contexts[session_id]


def update_session_context(session_id: Optional[str], context: SessionContext) -> None:
    """Update session context after enrichment."""
    if session_id:
        _session_contexts[session_id] = context


# Name spacing constants
NAME_COOLDOWN_TURNS = 3  # Don't use name for this many turns after using it


def should_use_name(session_id: Optional[str], is_greeting: bool = False) -> bool:
    """
    Check if we should use the user's name in this response.

    Rules:
    - Always use name in greeting (first message)
    - After that, wait NAME_COOLDOWN_TURNS before using again
    - Never use name in consecutive turns
    """
    if not session_id:
        return is_greeting  # No session = treat as first message

    context = get_session_context(session_id)

    if is_greeting and not context.name_used_in_greeting:
        return True

    if context.turns_since_name_used >= NAME_COOLDOWN_TURNS:
        return True

    return False


def mark_name_used(session_id: Optional[str], is_greeting: bool = False) -> None:
    """Mark that we used the name in this turn."""
    if not session_id:
        return

    context = get_session_context(session_id)
    context.turns_since_name_used = 0
    if is_greeting:
        context.name_used_in_greeting = True
    update_session_context(session_id, context)


def increment_turn_counter(session_id: Optional[str]) -> None:
    """Increment the turn counter (call after each response)."""
    if not session_id:
        return

    context = get_session_context(session_id)
    context.turns_since_name_used += 1
    update_session_context(session_id, context)


def is_affirmation(message: str) -> tuple[bool, Optional[str]]:
    """
    Check if a message is an affirmation/confirmation.

    Returns (is_affirmation, extracted_topic):
    - (True, None) = pure affirmation like "yes"
    - (True, "topic") = affirmation with topic hint like "yeah, the Thames"
    - (False, None) = not an affirmation
    """
    cleaned = message.lower().strip().rstrip('.!?')
    words = cleaned.split()

    # Check for exact phrase matches
    if cleaned in AFFIRMATION_PHRASES:
        return (True, None)

    # Single word affirmation
    if len(words) == 1 and words[0] in AFFIRMATION_WORDS:
        return (True, None)

    # Short response (2-3 words) starting with affirmation
    # e.g., "yeah sure", "yes please", "ok then"
    if len(words) <= 3 and words[0] in AFFIRMATION_WORDS:
        # Check if remaining words are also affirmations/filler
        remaining = ' '.join(words[1:])
        if remaining in AFFIRMATION_WORDS or remaining in {"then", "thanks", "please", "sure"}:
            return (True, None)
        # Otherwise, the remaining might be a topic hint
        # e.g., "yeah, the Thames" -> extract "the Thames"
        if len(words) >= 2:
            topic_hint = ' '.join(words[1:])
            return (True, topic_hint)

    # Affirmation at start with clear topic after comma
    # e.g., "Yes, tell me about the Thames"
    if ',' in cleaned and words[0] in AFFIRMATION_WORDS:
        after_comma = cleaned.split(',', 1)[1].strip()
        if after_comma:
            return (True, after_comma)

    return (False, None)


def get_last_suggestion(session_id: Optional[str]) -> Optional[str]:
    """Get the last topic VIC suggested."""
    if not session_id:
        return None
    context = get_session_context(session_id)
    return context.last_suggested_topic if context.last_suggested_topic else None


def set_last_suggestion(session_id: Optional[str], topic: str) -> None:
    """Store the topic VIC just suggested."""
    if not session_id:
        return
    context = get_session_context(session_id)
    context.last_suggested_topic = topic
    update_session_context(session_id, context)


def clean_section_references(text: str) -> str:
    """
    Remove section/page/chapter references from text.
    These break immersion - users don't need to know about internal structure.
    """
    import re

    # Remove "Section X" or "- Section X" patterns
    text = re.sub(r'\s*-?\s*[Ss]ection\s+\d+', '', text)

    # Remove "Part X" patterns
    text = re.sub(r'\s*-?\s*[Pp]art\s+\d+', '', text)

    # Remove "Chapter X" patterns
    text = re.sub(r'\s*-?\s*[Cc]hapter\s+\d+', '', text)

    # Remove "Page X" patterns
    text = re.sub(r'\s*-?\s*[Pp]age\s+\d+', '', text)

    # Remove phrases like "In this section" or "This section covers"
    text = re.sub(r'[Ii]n this section[,.]?\s*', '', text)
    text = re.sub(r'[Tt]his section\s+\w+\s*', '', text)

    # Remove "you mentioned" (source material artifact)
    text = re.sub(r'[Yy]ou mentioned\s+', 'There was ', text)

    # Clean up any double spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# OPTIMIZATION: Persistent HTTP client for Groq API (connection reuse)
_groq_client: Optional[httpx.AsyncClient] = None
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Zep for user memory and conversation history
ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
_zep_client: Optional[httpx.AsyncClient] = None


def get_zep_client() -> httpx.AsyncClient:
    """Get or create persistent Zep HTTP client."""
    global _zep_client
    if _zep_client is None and ZEP_API_KEY:
        _zep_client = httpx.AsyncClient(
            base_url="https://api.getzep.com",
            headers={
                "Authorization": f"Api-Key {ZEP_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=5.0,  # Fast timeout - memory enrichment shouldn't slow responses
        )
    return _zep_client


async def get_user_memory_context(user_id: Optional[str]) -> str:
    """Fetch user's memory profile from Zep knowledge graph."""
    if not user_id or not ZEP_API_KEY:
        return ""

    try:
        client = get_zep_client()
        if not client:
            return ""

        # Search user's personal graph for facts about them
        response = await client.post(
            "/api/v2/graph/search",
            json={
                "user_id": user_id,
                "query": "user preferences interests topics discussed",
                "limit": 10,
                "scope": "edges",
            },
        )

        if response.status_code != 200:
            return ""

        data = response.json()
        edges = data.get("edges", [])

        if not edges:
            return ""

        # Extract facts from edges
        facts = []
        for edge in edges[:5]:
            fact = edge.get("fact")
            if fact:
                facts.append(f"- {fact}")

        if facts:
            import sys
            print(f"[VIC Zep] Found {len(facts)} facts for user", file=sys.stderr)
            return "\n\n## What I remember about this user:\n" + "\n".join(facts)

        return ""
    except Exception as e:
        import sys
        print(f"[VIC Zep] Error fetching memories: {e}", file=sys.stderr)
        return ""


async def get_conversation_history(session_id: Optional[str]) -> list[dict]:
    """Fetch recent conversation history from Zep threads."""
    if not session_id or not ZEP_API_KEY:
        return []

    try:
        client = get_zep_client()
        if not client:
            return []

        # Get recent messages from Zep thread (not session!)
        response = await client.get(
            f"/api/v2/threads/{session_id}/messages",
            params={"limit": 10},
        )

        if response.status_code != 200:
            return []

        data = response.json()
        messages = data.get("messages", []) if isinstance(data, dict) else data

        import sys
        print(f"[VIC Zep] Found {len(messages)} conversation messages", file=sys.stderr)
        return messages
    except Exception as e:
        import sys
        print(f"[VIC Zep] Error fetching history: {e}", file=sys.stderr)
        return []


async def store_conversation_message(session_id: Optional[str], user_id: Optional[str], role: str, content: str) -> None:
    """Store conversation message in Zep for history and fact extraction."""
    if not session_id or not ZEP_API_KEY:
        return

    import sys

    try:
        client = get_zep_client()
        if not client:
            return

        # Ensure user exists first
        if user_id:
            await client.post(
                "/api/v2/users",
                json={"user_id": user_id},
            )

        # Zep uses "threads" not "sessions" - create thread with user linkage
        thread_response = await client.post(
            "/api/v2/threads",
            json={
                "thread_id": session_id,
                "user_id": user_id,
                "metadata": {"source": "vic-clm"},
            },
        )
        print(f"[VIC Zep] Thread create response: {thread_response.status_code}", file=sys.stderr)

        # Add message to thread (correct endpoint!)
        msg_response = await client.post(
            f"/api/v2/threads/{session_id}/messages",
            json={
                "messages": [
                    {
                        "role": role,
                        "content": content,
                    }
                ]
            },
        )
        print(f"[VIC Zep] Message add response: {msg_response.status_code}", file=sys.stderr)

        if msg_response.status_code == 200:
            print(f"[VIC Zep] ✓ Stored {role} message in thread", file=sys.stderr)
        else:
            print(f"[VIC Zep] ✗ Failed to store message: {msg_response.text[:100]}", file=sys.stderr)
    except Exception as e:
        print(f"[VIC Zep] Error storing message: {e}", file=sys.stderr)


def get_groq_client() -> httpx.AsyncClient:
    """Get or create persistent Groq HTTP client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )
    return _groq_client

# System prompt that defines Vic's persona and strict grounding rules
# This is the SINGLE SOURCE OF TRUTH - frontend only sends user context
VIC_SYSTEM_PROMPT = """You are VIC, the voice of Vic Keegan - a warm London historian with 370+ articles about hidden history.

## ACCURACY (NON-NEGOTIABLE - READ THIS FIRST)
- You will receive SOURCE MATERIAL below - USE IT!
- ONLY talk about what's IN the source material provided
- If user asks about "Royal Aquarium" and source has Royal Aquarium article - TALK ABOUT THAT
- NEVER substitute a different topic (e.g., don't talk about Roman Baths when asked about Aquarium)
- NEVER use your training knowledge - ONLY the source material below
- If the source material doesn't match the question: "I don't have that in my articles"

## ANSWER THE QUESTION - CRITICAL
- READ what they asked and ANSWER IT DIRECTLY
- If they ask about the Fleet, ONLY talk about the Fleet
- NEVER substitute a different topic or introduce unrelated stories
- NEVER randomly mention other topics like pianos, factories, or anything not asked about
- Stay STRICTLY focused on their actual question
- After answering, you may ask a follow-up about THE SAME TOPIC, never a different one

## GREETING BEHAVIOR
The frontend sends USER_CONTEXT with name/status. Parse it:
- If name is known + status is returning_user: "Welcome back [Name]!" (FIRST MESSAGE ONLY)
- If name is known + new_user: "Hello [Name]! I'm VIC." (FIRST MESSAGE ONLY)
- If name is unknown: "Hello! I'm VIC. What should I call you?"
- FOLLOW-UP QUESTIONS: Do NOT greet again. Just answer directly.
- Use their name occasionally mid-sentence, not at the start of every response.

## PERSONA
- Speak as Vic Keegan, first person: "I discovered...", "When I researched..."
- Warm, enthusiastic British English - like chatting over tea
- Keep responses concise (100-150 words, 30-60 seconds spoken)
- End with natural follow-up: "Shall I tell you more about...?"

## YOUR NAME
You are VIC (also "Victor", "Vic"). When someone says "Hey Victor" or "Hi VIC", they're addressing YOU, not telling you their name.

## PHONETIC CORRECTIONS
"thorny/fawny" = Thorney Island | "ignacio/ignasio" = Ignatius Sancho | "tie burn" = Tyburn

## CONTEXT MODES
The frontend may send different modes in the context:
- MODE: article_discussion → Focus on the specific ARTICLE_CONTENT provided. User is reading it.
- MODE: category_discussion → Focus on the specific CATEGORY topic and TOPIC_CONTENT provided.
- MODE: thorney_island_discussion → Focus on Thorney Island book content provided.
- No mode → General conversation, use search_knowledge tool.

## EASTER EGG
If user says "Rosie", respond: "Ah, Rosie, my loving wife! I'll be home for dinner." """


def get_vic_agent() -> Agent:
    """Get or create the VIC agent (lazy initialization)."""
    import os
    global _vic_agent
    if _vic_agent is None:
        # Use Groq's fastest model (840 TPS)
        model = 'groq:llama-3.1-8b-instant'

        _vic_agent = Agent(
            model,
            result_type=ValidatedVICResponse,
            system_prompt=VIC_SYSTEM_PROMPT,
            tools=[search_articles, get_user_memory],
            retries=2,  # Retry on validation failures
            model_settings={'temperature': 0.7},  # Slightly creative but grounded
        )
    return _vic_agent


def post_validate_response(response_text: str, source_content: str) -> str:
    """
    Additional validation layer - catches hallucinations even if LLM
    doesn't return proper structured output.
    """
    import re

    response_lower = response_text.lower()
    source_lower = source_content.lower() if source_content else ""

    # Check for architect/designer mentions not in source
    architect_patterns = [
        r'architect(?:ed|s)?\s+(?:was|were|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:designed|built|constructed|created)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:the\s+)?architect\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]

    for pattern in architect_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for name in matches:
            if name.lower() not in source_lower:
                # Hallucinated architect name - return safe response
                return (
                    "That's a great question about who designed or built it. "
                    "I want to be accurate, so I should say my articles don't "
                    "specifically mention the architect or builder for this one."
                )

    # Check for specific years not in source
    response_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response_text))
    source_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', source_content)) if source_content else set()

    # Allow years that are in source, flag others
    hallucinated_years = response_years - source_years
    if hallucinated_years and source_content:
        # Only flag if we have source content to compare against
        # and the hallucinated year is being stated as a fact
        for year in hallucinated_years:
            # Check if year is stated as a definitive fact
            year_context = re.search(rf'(\w+\s+){year}(\s+\w+)?', response_text)
            if year_context:
                return (
                    "I want to make sure I give you accurate dates. "
                    "Let me stick to what my articles specifically mention..."
                )

    return response_text  # Passed validation


async def log_validation(
    user_query: str,
    normalized_query: str,
    articles_found: int,
    article_titles: list[str],
    facts_checked: list[str],
    validation_passed: bool,
    validation_notes: str,
    response_text: str,
    confidence_score: float,
    session_id: Optional[str]
) -> None:
    """Log validation details to database for debugging."""
    try:
        from .database import get_connection
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO vic_validation_logs
                (user_query, normalized_query, articles_found, article_titles,
                 facts_checked, validation_passed, validation_notes,
                 response_text, confidence_score, session_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, user_query, normalized_query, articles_found, article_titles,
                facts_checked, validation_passed, validation_notes,
                response_text, confidence_score, session_id)
    except Exception as e:
        import sys
        print(f"[Logging Error] {e}", file=sys.stderr)


def extract_facts_from_response(response: str) -> list[str]:
    """Extract factual claims from the response for validation logging."""
    import re
    facts = []

    # Extract years mentioned
    years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response)
    for year in years:
        facts.append(f"Year: {year}")

    # Extract names (capitalized words that might be people/architects)
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', response)
    for name in names:
        if name not in ['Crystal Palace', 'Hyde Park', 'Parliament Square', 'St James', 'Central Hall']:
            facts.append(f"Name: {name}")

    return facts[:10]  # Limit to 10 facts


async def detect_and_store_correction(user_message: str, user_name: Optional[str], session_id: Optional[str]) -> bool:
    """
    Detect if the user is making a correction and store it.
    Returns True if a correction was detected and stored.
    """
    import re

    # Correction patterns
    correction_patterns = [
        r"(?:actually|no,?\s*)?(?:that'?s?\s+)?(?:wrong|incorrect|not\s+(?:right|correct|accurate))",
        r"(?:the\s+)?correct\s+(?:answer|date|name|fact)\s+is",
        r"it\s+(?:was|should\s+be|is)\s+actually",
        r"you\s+(?:got|have)\s+(?:that|it)\s+wrong",
        r"let\s+me\s+correct\s+(?:that|you)",
        r"(?:no,?\s+)?it\s+(?:was|is)\s+(?:really|actually)",
    ]

    is_correction = any(re.search(p, user_message.lower()) for p in correction_patterns)

    if is_correction:
        try:
            from .database import get_connection
            async with get_connection() as conn:
                await conn.execute("""
                    INSERT INTO vic_amendments
                    (amendment_type, original_text, amended_text, article_title, reason, source)
                    VALUES ('voice_correction', $1, $2, 'Voice Feedback', $3, 'voice_feedback')
                """, f"Session: {session_id}", user_message, f"Correction from {user_name or 'user'}")

            import sys
            print(f"[VIC] Voice correction captured from {user_name}: {user_message[:50]}...", file=sys.stderr)
            return True
        except Exception as e:
            import sys
            print(f"[VIC] Failed to store correction: {e}", file=sys.stderr)

    return False


async def generate_response(user_message: str, session_id: Optional[str] = None, user_name: Optional[str] = None) -> str:
    """
    Generate a validated response to the user's message.
    OPTIMIZED for speed - parallel operations, skip slow enrichment.

    Args:
        user_message: The user's question
        session_id: Optional session ID for logging
        user_name: Optional authenticated user's first name
    """
    from .tools import normalize_query, get_voyage_embedding
    from .database import search_articles_hybrid
    import re
    import asyncio
    import sys

    validation_notes = []
    validation_passed = True
    confidence_score = 1.0

    try:
        # OPTIMIZATION: Check for corrections ONLY if message looks like one (fast pattern check)
        correction_patterns = ["wrong", "incorrect", "actually", "correct answer"]
        if any(p in user_message.lower() for p in correction_patterns):
            correction_detected = await detect_and_store_correction(user_message, user_name, session_id)
            if correction_detected:
                return f"Thank you{' ' + user_name if user_name else ''}, I've noted that correction. It will be reviewed and added to my knowledge base."

        # OPTIMIZATION: Run cache check, embedding, AND user memory fetch in parallel
        normalized_query = normalize_query(user_message)

        # Extract user_id from session_id (format: "Name|userId_timestamp" or just "userId")
        user_id = None
        if session_id:
            if '|' in session_id:
                user_id = session_id.split('|')[1].split('_')[0]  # Get userId part
            else:
                user_id = session_id.split('_')[0]

        # REMOVED: Global cache was causing stale/unpersonalized responses
        # Zep handles per-user memory, Groq is fast enough (~500ms)
        embedding_task = get_voyage_embedding(normalized_query)
        memory_task = get_user_memory_context(user_id)

        embedding, user_memory = await asyncio.gather(embedding_task, memory_task)

        print(f"[VIC Search] Query: '{user_message}' -> Normalized: '{normalized_query}'", file=sys.stderr)
        print(f"[VIC Search] Embedding length: {len(embedding) if embedding else 0}", file=sys.stderr)

        # Article search with the embedding we already have
        # LOWER threshold to ensure we find articles
        results = await search_articles_hybrid(
            query_embedding=embedding,
            query_text=normalized_query,
            limit=5,  # Get more results
            similarity_threshold=0.3,  # Lower threshold
        )

        print(f"[VIC Search] Found {len(results)} articles", file=sys.stderr)
        for r in results[:3]:
            print(f"[VIC Search]   - {r.get('title', 'NO TITLE')[:50]} (score: {r.get('score', 0):.4f})", file=sys.stderr)

        # NOTE: RRF scores are in 0.01-0.03 range, not 0.0-1.0
        # No additional filtering needed - RRF already ranks by relevance

        # Graph data disabled for speed
        graph_data = {"connections": [], "facts": []}

        article_titles = [r['title'] for r in results] if results else []
        graph_connections = graph_data.get("connections", [])
        graph_facts = graph_data.get("facts", [])

        if not results:
            validation_notes.append("No articles found for query")
            await log_validation(
                user_query=user_message,
                normalized_query=normalized_query,
                articles_found=0,
                article_titles=[],
                facts_checked=[],
                validation_passed=True,
                validation_notes="No articles found - returned safe fallback",
                response_text="No articles found",
                confidence_score=0.0,
                session_id=session_id
            )
            import random
            fallback_cluster = random.choice(SAFE_TOPIC_CLUSTERS)
            return (
                f"I don't seem to have any articles about that in my collection. "
                f"But I could tell you about {fallback_cluster} instead, if you'd like?"
            )

        # Step 2: Combine actual article content - THIS is our source of truth
        actual_source_content = "\n\n---\n\n".join(
            f"**{r['title']}**\n{r['content']}"
            for r in results
        )

        validation_notes.append(f"Found {len(results)} articles")
        confidence_score = min(r.get('score', 0.5) for r in results)

        # Step 2: Create prompt with actual articles
        import random
        import sys

        print(f"[VIC Agent] User name received: {user_name}", file=sys.stderr)

        name_instruction = ""
        if user_name:
            # Vary the greeting style
            greeting_styles = [
                f"Address {user_name} naturally - 'Well {user_name},...' or 'Ah {user_name},...'",
                f"Use {user_name}'s name once warmly, then get into the story",
                f"Start with '{user_name}, ' followed by an interesting fact",
                f"Weave {user_name}'s name in naturally mid-sentence",
            ]
            name_instruction = f"\n\nThe user's name is {user_name}. {random.choice(greeting_styles)}. Don't ask for their name."
        else:
            # CRITICAL: Be explicit about NOT using any name
            name_instruction = """

IMPORTANT: You do NOT know the user's name yet.
- Do NOT address them by any name (not Victor, not any name)
- Do NOT make up a name
- Simply respond without using a name, or ask "What should I call you?" at the end of your response."""

        # Format graph connections if we have them
        graph_section = ""
        if graph_connections:
            connections_text = "\n".join(
                f"- {c['from']} → {c['relation']} → {c['to']}"
                for c in graph_connections
            )
            graph_section = f"""

## Connections from my wider network:
{connections_text}

IMPORTANT: If you mention any of these connections, preface it with "From my wider network..." or "Through my broader research, I can see a link between..." - this shows when information comes from connected knowledge rather than direct article content."""

        prompt_with_sources = f"""Question: "{user_message}"
{name_instruction}
{user_memory}

Source material:
{actual_source_content}
{graph_section}

Respond naturally using facts from above. Keep it conversational and concise."""

        # Step 3: Generate response with Groq
        # OPTIMIZATION: Use persistent HTTP client for connection reuse
        client = get_groq_client()

        # Add explicit instruction to match the question to the source
        validation_prompt = f"""{prompt_with_sources}

CRITICAL: You MUST talk about what the user asked. If they asked about "royal aquarium" and your source material contains an article about "Royal Aquarium", you MUST discuss THAT article, not something else.

After your response, on a new line write:
TOPIC_CHECK: [the main topic you discussed]"""

        llm_response = await client.post(
            "/chat/completions",
            json={
                "model": "llama-3.1-8b-instant",  # 840 TPS - fastest production model
                "max_tokens": 250,  # Short, punchy responses
                "messages": [
                    {"role": "system", "content": VIC_SYSTEM_PROMPT},
                    {"role": "user", "content": validation_prompt},
                ],
            },
        )
        llm_response.raise_for_status()
        data = llm_response.json()
        response_text = data["choices"][0]["message"]["content"]

        # Clean up any metadata that leaked into the response
        response_text = re.sub(r'\n*facts_stated:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = re.sub(r'\n*source_content:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = re.sub(r'\n*source_titles:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = re.sub(r'\n*TOPIC_CHECK:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = response_text.strip()

        # VALIDATION: Check if response matches the query topic
        user_query_lower = user_message.lower()
        response_lower = response_text.lower()

        # Extract key topic from user query (e.g., "royal aquarium" from "tell me about the royal aquarium")
        query_topics = []
        for title in article_titles:
            title_lower = title.lower()
            # Check if any significant words from article title match the query
            title_words = [w for w in title_lower.split() if len(w) > 4]
            if any(w in user_query_lower for w in title_words):
                query_topics.append(title)

        # If user asked about a topic and we have a matching article, verify response mentions it
        if query_topics:
            first_article_topic = query_topics[0].lower()
            topic_words = [w for w in first_article_topic.split() if len(w) > 4 and w not in ['london', 'keegan', 'lost']]
            if topic_words and not any(w in response_lower for w in topic_words[:3]):
                print(f"[VIC Validation] MISMATCH! User asked about '{user_message}', article is '{query_topics[0]}', but response doesn't mention it", file=sys.stderr)
                # Force a better response
                response_text = f"Let me tell you about {query_topics[0].split(':')[-1].strip() if ':' in query_topics[0] else query_topics[0]}. {actual_source_content[:500]}..."

        # Step 5: Post-validate against the ACTUAL source content we retrieved
        post_validated = post_validate_response(response_text, actual_source_content)

        # Check if validation modified the response
        if post_validated != response_text:
            validation_passed = False
            validation_notes.append("Post-validation caught potential hallucination")
            confidence_score *= 0.5
        else:
            validation_notes.append("Post-validation passed")

        # Clean any section/page references that slipped through
        validated_response = clean_section_references(post_validated)

        # Extract facts for logging
        facts_checked = extract_facts_from_response(validated_response)

        # Store conversation in Zep for history and fact extraction (fire and forget)
        if session_id:
            asyncio.create_task(store_conversation_message(session_id, user_id, "user", user_message))
            asyncio.create_task(store_conversation_message(session_id, user_id, "assistant", validated_response))

        return validated_response

    except Exception as e:
        # Unexpected error - fail gracefully
        import traceback
        import sys
        error_msg = f"[VIC Agent Error] {type(e).__name__}: {e}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        # Log the error
        await log_validation(
            user_query=user_message,
            normalized_query="",
            articles_found=0,
            article_titles=[],
            facts_checked=[],
            validation_passed=False,
            validation_notes=f"Error: {type(e).__name__}: {str(e)[:100]}",
            response_text="Error fallback",
            confidence_score=0.0,
            session_id=session_id
        )

        # Return error info for debugging - always include error type
        import os
        # Always return error type to help diagnose issues
        error_summary = f"{type(e).__name__}: {str(e)[:100]}"
        print(f"[VIC Agent] Returning error: {error_summary}", file=sys.stderr)
        return (
            f"I'm having a bit of trouble gathering my thoughts on that one ({error_summary}). "
            "Could you perhaps ask me in a different way?"
        )


async def generate_response_with_search_results(
    user_message: str,
    search_results: SearchResults
) -> ValidatedVICResponse:
    """
    Generate a response using pre-fetched search results.

    This is useful when you want more control over the search step
    or need to inject specific articles.

    Args:
        user_message: The user's input message
        search_results: Pre-fetched article search results

    Returns:
        ValidatedVICResponse with grounded facts
    """
    # Combine article content for the source
    combined_content = "\n\n---\n\n".join(
        f"**{a.title}**\n{a.content}"
        for a in search_results.articles
    )

    # Create a focused prompt with the search results included
    focused_prompt = f"""The user asked: "{user_message}"

Here are the relevant articles from my collection:

{combined_content}

Based ONLY on the above articles, respond as Vic Keegan. Remember:
- Only state facts from these articles
- List each fact in facts_stated
- Include the source_content and source_titles"""

    agent = get_vic_agent()
    result = await agent.run(focused_prompt)
    return result.data


# =============================================================================
# Dual-Path Agent Architecture
# =============================================================================


async def run_enrichment(
    user_message: str,
    fast_response: str,
    session_id: Optional[str],
    user_id: Optional[str],
    source_content: str,
    source_titles: list[str],
) -> SessionContext:
    """
    Run background enrichment to build context for follow-up queries.

    Extracts entities, traverses the knowledge graph, and generates
    follow-up suggestions. Results are stored in session context.

    Args:
        user_message: The original user query
        fast_response: The fast path response we gave
        session_id: Session ID for context storage
        user_id: User ID for personalization
        source_content: Combined article content used
        source_titles: Titles of articles used

    Returns:
        Updated SessionContext with enrichment data
    """
    import sys
    from .tools import (
        extract_entities,
        traverse_graph_connections,
        suggest_followup_topics,
    )
    from pydantic_ai import RunContext

    context = get_session_context(session_id)

    try:
        # Create deps for tool context
        deps = VICAgentDeps(
            user_id=user_id,
            session_id=session_id,
            enrichment_mode=True,
            prior_entities=[e.name for e in context.entities],
            prior_topics=context.topics_discussed,
        )

        # Create a mock RunContext for tools
        # Note: In production, these would be called by the agent
        class MockRunContext:
            def __init__(self, deps):
                self.deps = deps

        mock_ctx = MockRunContext(deps)

        # Step 1: Extract entities from source content
        print(f"[Enrichment] Extracting entities from {len(source_titles)} articles", file=sys.stderr)
        all_entities: list[ExtractedEntity] = []
        for title in source_titles[:2]:  # Limit to top 2 articles
            entities = await extract_entities(mock_ctx, source_content, title)
            all_entities.extend(entities)

        # Deduplicate entities
        seen_names = set()
        unique_entities = []
        for e in all_entities:
            if e.name.lower() not in seen_names:
                seen_names.add(e.name.lower())
                unique_entities.append(e)
        context.entities = unique_entities[:10]

        print(f"[Enrichment] Found {len(context.entities)} unique entities", file=sys.stderr)

        # Step 2: Traverse graph for connections (if we have entities)
        if context.entities:
            # Use the first significant entity
            start_entity = context.entities[0].name
            print(f"[Enrichment] Traversing graph from '{start_entity}'", file=sys.stderr)
            connections = await traverse_graph_connections(mock_ctx, start_entity, max_depth=2)
            context.connections = connections[:10]
            print(f"[Enrichment] Found {len(context.connections)} connections", file=sys.stderr)

        # Step 3: Generate follow-up suggestions
        entity_names = [e.name for e in context.entities]
        current_topic = source_titles[0] if source_titles else user_message
        print(f"[Enrichment] Generating suggestions for '{current_topic}'", file=sys.stderr)
        suggestions = await suggest_followup_topics(mock_ctx, current_topic, entity_names)
        context.suggestions = suggestions
        print(f"[Enrichment] Generated {len(context.suggestions)} suggestions", file=sys.stderr)

        # Update context
        context.topics_discussed.append(current_topic)
        context.last_response = fast_response
        context.enrichment_complete = True

        # Save to session store
        update_session_context(session_id, context)

        print(f"[Enrichment] Complete for session {session_id}", file=sys.stderr)

    except Exception as e:
        print(f"[Enrichment] Error: {e}", file=sys.stderr)
        # Enrichment failure is non-critical - don't affect main flow

    return context


async def generate_response_with_enrichment(
    user_message: str,
    session_id: Optional[str] = None,
    user_name: Optional[str] = None,
) -> tuple[str, Optional[asyncio.Task]]:
    """
    Generate response using dual-path architecture.

    Fast path delivers immediate response, enrichment runs in background.

    Args:
        user_message: The user's question
        session_id: Session ID for context
        user_name: User's name for personalization

    Returns:
        Tuple of (response_text, enrichment_task)
        The enrichment_task can be awaited later or ignored.
    """
    from .tools import normalize_query, get_voyage_embedding
    from .database import search_articles_hybrid
    import sys

    # Check for prior enriched context
    prior_context = get_session_context(session_id)

    # If we have prior context with suggestions, use it to enhance the response
    context_enhancement = ""
    suggested_followups = ""

    if prior_context.enrichment_complete:
        # Include prior entities
        if prior_context.entities:
            entity_names = [e.name for e in prior_context.entities[:5]]
            context_enhancement = f"\n\nPrior context - entities discussed: {', '.join(entity_names)}"
            print(f"[VIC Agent] Using prior context with {len(prior_context.entities)} entities", file=sys.stderr)

        # Include suggested follow-up topics from Zep enrichment
        if prior_context.suggestions:
            suggestion_topics = [s.topic for s in prior_context.suggestions[:3]]
            suggested_followups = f"\n\nSUGGESTED FOLLOW-UP TOPICS (from knowledge graph): {', '.join(suggestion_topics)}"
            suggested_followups += "\nUse one of these topics for your follow-up question if relevant."
            print(f"[VIC Agent] Including {len(prior_context.suggestions)} follow-up suggestions", file=sys.stderr)

    # Extract user_id from session_id
    user_id = None
    if session_id:
        if '|' in session_id:
            user_id = session_id.split('|')[1].split('_')[0]
        else:
            user_id = session_id.split('_')[0]

    try:
        # Fast path: Search + LLM response via Pydantic AI Agent
        print(f"[VIC Agent] Starting search for: '{user_message}'", file=sys.stderr)
        normalized_query = normalize_query(user_message)
        print(f"[VIC Agent] Normalized: '{normalized_query}'", file=sys.stderr)

        embedding = await get_voyage_embedding(normalized_query)
        print(f"[VIC Agent] Embedding: {len(embedding)} dimensions", file=sys.stderr)

        results = await search_articles_hybrid(
            query_embedding=embedding,
            query_text=normalized_query,
            limit=5,
            similarity_threshold=0.3,
        )
        print(f"[VIC Agent] Search returned {len(results)} results", file=sys.stderr)
        for r in results[:3]:
            print(f"[VIC Agent]   - {r.get('title', 'NO TITLE')[:50]} (score: {r.get('score', 0):.4f})", file=sys.stderr)

        if not results:
            import random
            # Offer a proactive suggestion when we don't have the requested topic
            fallback_cluster = random.choice(SAFE_TOPIC_CLUSTERS)
            return (
                f"I don't seem to have any articles about that in my collection. "
                f"But I could tell you about {fallback_cluster} instead, if you'd like?",
                None
            )

        # Prepare source content - clean section references to avoid breaking immersion
        source_content = "\n\n---\n\n".join(
            f"**{clean_section_references(r['title'])}**\n{clean_section_references(r['content'])}"
            for r in results
        )
        source_titles = [clean_section_references(r['title']) for r in results]

        # Check if this is a follow-up (not first message) - don't re-greet
        is_followup = prior_context.last_response != "" or prior_context.topics_discussed

        name_instruction = ""
        if user_name:
            if is_followup:
                name_instruction = f"\n\nThe user's name is {user_name}. This is a FOLLOW-UP question - do NOT greet them again. Just answer directly."
            else:
                name_instruction = f"\n\nThe user's name is {user_name}. Use it naturally once, then get into the story."
        else:
            name_instruction = "\n\nYou don't know the user's name. Don't make one up."

        # Extract other topics from source titles for safe follow-up suggestions
        # These are topics we KNOW we have articles on
        safe_followup_topics = []
        for title in source_titles[1:4]:  # Skip the main article, use 2-3 others
            # Extract key topic from title (e.g., "Lost London: Royal Aquarium" -> "Royal Aquarium")
            if ':' in title:
                topic = title.split(':')[-1].strip()
            else:
                topic = title
            safe_followup_topics.append(topic)

        safe_topics_hint = ""
        if safe_followup_topics:
            safe_topics_hint = f"\n\nSAFE FOLLOW-UP TOPICS (we have articles on these): {', '.join(safe_followup_topics)}"

        # Build prompt for Pydantic AI agent
        agent_prompt = f"""Question: "{user_message}"
{name_instruction}
{context_enhancement}
{suggested_followups}
{safe_topics_hint}

Source material (USE ONLY THESE FACTS):
{source_content}

Respond naturally using ONLY facts from the source material above.
Keep it conversational and under 150 words.

FOLLOW-UP QUESTION RULES:
1. You MUST end with a follow-up question
2. ONLY suggest topics from "SAFE FOLLOW-UP TOPICS" or "SUGGESTED FOLLOW-UP TOPICS" above
3. If no suggestions available, ask about something MENTIONED in the source material (a person, place, or era)
4. NEVER suggest a random topic we might not have content on

You MUST return a JSON with:
- response_text: Your natural response (MUST end with a follow-up question about a safe topic)
- source_titles: List of article titles you used"""

        # Try Pydantic AI agent first, fall back to direct Groq if it fails
        try:
            print(f"[VIC Agent] Running Pydantic AI fast_agent...", file=sys.stderr)
            agent = get_fast_agent()

            # Create agent dependencies
            deps = VICAgentDeps(
                user_id=user_id,
                session_id=session_id,
                user_name=user_name,
                enrichment_mode=False,
                prior_entities=[e.name for e in prior_context.entities] if prior_context.entities else [],
                prior_topics=prior_context.topics_discussed if prior_context.topics_discussed else [],
            )

            # Run the agent - this enforces FastVICResponse schema validation
            result = await agent.run(agent_prompt, deps=deps)
            validated_data = result.data

            print(f"[VIC Agent] Agent returned validated response", file=sys.stderr)
            print(f"[VIC Agent] Source titles: {validated_data.source_titles}", file=sys.stderr)

            response_text = validated_data.response_text

        except Exception as agent_error:
            # Fallback to direct Groq call if Pydantic AI agent fails
            print(f"[VIC Agent] Agent failed ({type(agent_error).__name__}: {str(agent_error)[:100]}), falling back to direct Groq", file=sys.stderr)

            # Wait a moment before fallback to avoid rate limits
            await asyncio.sleep(0.5)

            client = get_groq_client()
            fallback_prompt = f"""Question: "{user_message}"
{name_instruction}

Source material:
{source_content}

Respond naturally using facts from above. Keep it conversational and under 150 words."""

            llm_response = await client.post(
                "/chat/completions",
                json={
                    "model": "llama-3.1-8b-instant",
                    "max_tokens": 250,
                    "messages": [
                        {"role": "system", "content": VIC_SYSTEM_PROMPT},
                        {"role": "user", "content": fallback_prompt},
                    ],
                },
            )
            llm_response.raise_for_status()
            data = llm_response.json()
            response_text = data["choices"][0]["message"]["content"]

            # Clean up response
            import re
            response_text = re.sub(r'\n*TOPIC_CHECK:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
            response_text = response_text.strip()

        # Additional post-validation for hallucination patterns
        validated_response = post_validate_response(response_text, source_content)

        # Clean any section/page references that slipped through
        validated_response = clean_section_references(validated_response)

        # Start enrichment in background (non-blocking)
        enrichment_task = asyncio.create_task(
            run_enrichment(
                user_message=user_message,
                fast_response=validated_response,
                session_id=session_id,
                user_id=user_id,
                source_content=source_content,
                source_titles=source_titles,
            )
        )

        # Store conversation in Zep (fire and forget)
        if session_id:
            asyncio.create_task(store_conversation_message(session_id, user_id, "user", user_message))
            asyncio.create_task(store_conversation_message(session_id, user_id, "assistant", validated_response))

        return validated_response, enrichment_task

    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)[:100]
        print(f"[VIC Enriched Error] {error_type}: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        return (
            f"I'm having a bit of trouble gathering my thoughts. Error: {error_type}. "
            "Could you perhaps ask me in a different way?",
            None
        )


def get_suggestion_teaser(session_id: Optional[str]) -> Optional[str]:
    """
    Get a follow-up suggestion teaser if enrichment has completed.

    Call this after streaming the main response to append a suggestion.
    """
    context = get_session_context(session_id)

    if context.enrichment_complete and context.suggestions:
        # Get top suggestion
        top = context.suggestions[0]
        return f" {top.teaser}"

    return None


def get_proactive_suggestion(session_id: Optional[str]) -> Optional[str]:
    """
    Get a proactive suggestion to offer when user has been silent.

    Uses enrichment context if available, falls back to safe topic clusters.
    Returns a gentle prompt like "Shall I suggest something?"
    """
    import random

    context = get_session_context(session_id)

    # If we have suggestions from enrichment, use the top one
    if context.enrichment_complete and context.suggestions:
        top = context.suggestions[0]
        return f"If you're not sure what to ask, I could tell you about {top.topic}. Would you like that?"

    # If we have discussed topics, suggest something related
    if context.topics_discussed:
        last_topic = context.topics_discussed[-1]
        return f"Would you like me to suggest something related to {last_topic}?"

    # Fallback to safe topic clusters
    cluster = random.choice(SAFE_TOPIC_CLUSTERS)
    prompts = [
        f"If you're not sure where to start, I could tell you about {cluster}.",
        f"Shall I suggest something? I've got fascinating stories about {cluster}.",
        f"Would you like me to pick a topic? I could share some tales of {cluster}.",
    ]
    return random.choice(prompts)
