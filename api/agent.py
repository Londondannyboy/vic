"""Pydantic AI agent for VIC - the voice of Vic Keegan."""

import os
import httpx
from pydantic_ai import Agent
from pydantic import ValidationError

from .models import ValidatedVICResponse, DeclinedResponse, SearchResults
from .tools import search_articles, get_user_memory

# Lazy-loaded agent instance
_vic_agent: Agent | None = None

# OPTIMIZATION: Persistent HTTP client for Groq API (connection reuse)
_groq_client: httpx.AsyncClient | None = None
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Zep for user memory and conversation history
ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
_zep_client: httpx.AsyncClient | None = None


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


async def get_user_memory_context(user_id: str | None) -> str:
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


async def get_conversation_history(session_id: str | None) -> list[dict]:
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


async def store_conversation_message(session_id: str | None, user_id: str | None, role: str, content: str) -> None:
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
