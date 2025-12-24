"""Pydantic AI agent for VIC - the voice of Vic Keegan."""

import os
import httpx
from pydantic_ai import Agent
from pydantic import ValidationError

from .models import ValidatedVICResponse, DeclinedResponse, SearchResults
from .tools import search_articles, get_user_memory

# Lazy-loaded agent instance
_vic_agent: Agent | None = None

# OPTIMIZATION: Persistent HTTP client for Anthropic API (connection reuse)
_anthropic_client: httpx.AsyncClient | None = None
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Supermemory for user profiles
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
_supermemory_client: httpx.AsyncClient | None = None


def get_supermemory_client() -> httpx.AsyncClient:
    """Get or create persistent Supermemory HTTP client."""
    global _supermemory_client
    if _supermemory_client is None and SUPERMEMORY_API_KEY:
        _supermemory_client = httpx.AsyncClient(
            base_url="https://api.supermemory.ai",
            headers={
                "Authorization": f"Bearer {SUPERMEMORY_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=5.0,  # Fast timeout - memory enrichment shouldn't slow responses
        )
    return _supermemory_client


async def get_user_memory_context(user_id: str | None) -> str:
    """Fetch user's memory profile from Supermemory for personalization."""
    if not user_id or not SUPERMEMORY_API_KEY:
        return ""

    try:
        client = get_supermemory_client()
        if not client:
            return ""

        # Search for user's past interactions
        response = await client.post(
            "/v4/search",
            json={
                "q": "interests topics articles viewed searches",
                "containerTags": [user_id],
                "limit": 5,
            },
        )

        if response.status_code != 200:
            return ""

        data = response.json()
        memories = data.get("results", [])

        if not memories:
            return ""

        # Format memories into context
        memory_items = []
        for m in memories[:5]:
            content = m.get("content", "")
            if content:
                memory_items.append(f"- {content}")

        if memory_items:
            import sys
            print(f"[VIC Supermemory] Found {len(memory_items)} memories for user", file=sys.stderr)
            return "\n\n## What I remember about this user:\n" + "\n".join(memory_items)

        return ""
    except Exception as e:
        import sys
        print(f"[VIC Supermemory] Error fetching memories: {e}", file=sys.stderr)
        return ""


async def store_conversation_topics(user_id: str | None, topics: list[str]) -> None:
    """Store conversation topics in Supermemory for future reference."""
    if not user_id or not topics or not SUPERMEMORY_API_KEY:
        return

    try:
        client = get_supermemory_client()
        if not client:
            return

        content = f"User discussed: {', '.join(topics)}"

        await client.post(
            "/v3/documents",
            json={
                "content": content,
                "containerTag": user_id,
                "metadata": {
                    "type": "conversation_topic",
                    "topics": topics,
                    "source": "vic_clm",
                },
            },
        )
        import sys
        print(f"[VIC Supermemory] Stored topics: {topics}", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"[VIC Supermemory] Error storing topics: {e}", file=sys.stderr)


def get_anthropic_client() -> httpx.AsyncClient:
    """Get or create persistent Anthropic HTTP client."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=15.0,
        )
    return _anthropic_client

# System prompt that defines Vic's persona and strict grounding rules
VIC_SYSTEM_PROMPT = """You are VIC, the voice of Vic Keegan - a warm London historian.

## How to speak
- Warm, enthusiastic, conversational British English
- First person: "I discovered...", "I've always been fascinated by..."
- Like chatting to a friend over tea
- Keep responses concise - 2-3 sentences per point, not essays
- NO meta-references like "my articles say" or "according to my research" - just share the facts naturally

## Accuracy rules
- ONLY state facts from the provided articles
- If something isn't mentioned (architect, date, designer), say "I'm not sure about that" and move on
- Never guess or infer - if you don't know, you don't know

## Style
- Be warm but get to the point
- Don't repeat the user's question back
- End with a natural follow-up like "Shall I tell you more about...?" only if relevant"""


def get_vic_agent() -> Agent:
    """Get or create the VIC agent (lazy initialization)."""
    import os
    global _vic_agent
    if _vic_agent is None:
        # Use Google Gemini if GOOGLE_API_KEY is set, otherwise Anthropic
        if os.environ.get("GOOGLE_API_KEY"):
            model = 'google-gla:gemini-2.0-flash'
        else:
            model = 'anthropic:claude-3-5-haiku-20241022'

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
    session_id: str | None
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


async def detect_and_store_correction(user_message: str, user_name: str | None, session_id: str | None) -> bool:
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


async def generate_response(user_message: str, session_id: str | None = None, user_name: str | None = None) -> str:
    """
    Generate a validated response to the user's message.
    OPTIMIZED for speed - parallel operations, skip slow enrichment.

    Args:
        user_message: The user's question
        session_id: Optional session ID for logging
        user_name: Optional authenticated user's first name
    """
    from .tools import normalize_query, get_voyage_embedding
    from .database import search_articles_hybrid, get_cached_response, cache_response
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

        cache_task = get_cached_response(user_message)
        embedding_task = get_voyage_embedding(normalized_query)
        memory_task = get_user_memory_context(user_id)

        # Wait for all - if cache hits, we still have embedding ready for next time
        cached, embedding, user_memory = await asyncio.gather(cache_task, embedding_task, memory_task)

        if cached:
            print(f"[VIC Cache] HIT for '{user_message}'", file=sys.stderr)
            return cached["response"]

        # Article search with the embedding we already have
        results = await search_articles_hybrid(
            query_embedding=embedding,
            query_text=normalized_query,
            limit=2,
            similarity_threshold=0.3,
        )

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
            return (
                "I don't seem to have any articles about that in my collection. "
                "Is there something else about London's history I can help you with?"
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

        # Step 3: Generate response with direct Anthropic call (faster)
        # OPTIMIZATION: Use persistent HTTP client for connection reuse
        client = get_anthropic_client()
        llm_response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,  # Short, punchy responses
                "system": VIC_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt_with_sources}],
            },
        )
        llm_response.raise_for_status()
        data = llm_response.json()
        response_text = data["content"][0]["text"]

        # Clean up any metadata that leaked into the response
        response_text = re.sub(r'\n*facts_stated:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = re.sub(r'\n*source_content:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = re.sub(r'\n*source_titles:.*$', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        response_text = response_text.strip()

        # Step 5: Post-validate against the ACTUAL source content we retrieved
        validated_response = post_validate_response(response_text, actual_source_content)

        # Check if validation modified the response
        if validated_response != response_text:
            validation_passed = False
            validation_notes.append("Post-validation caught potential hallucination")
            confidence_score *= 0.5
        else:
            validation_notes.append("Post-validation passed")

        # Extract facts for logging
        facts_checked = extract_facts_from_response(validated_response)

        # Cache the response for future queries (fire and forget)
        try:
            await cache_response(user_message, validated_response, article_titles)
        except Exception as cache_err:
            print(f"[VIC Cache] Write error: {cache_err}", file=sys.stderr)

        # Store topics in Supermemory for future personalization (fire and forget)
        if user_id and article_titles:
            asyncio.create_task(store_conversation_topics(user_id, article_titles))

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

        # Return error info in dev mode for debugging
        import os
        if os.environ.get("DEBUG"):
            return f"Error: {type(e).__name__}: {str(e)[:200]}"
        return (
            "I'm having a bit of trouble gathering my thoughts on that one. "
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
