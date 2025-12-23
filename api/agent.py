"""Pydantic AI agent for VIC - the voice of Vic Keegan."""

from pydantic_ai import Agent
from pydantic import ValidationError

from .models import ValidatedVICResponse, DeclinedResponse, SearchResults
from .tools import search_articles, get_user_memory

# Lazy-loaded agent instance
_vic_agent: Agent | None = None

# System prompt that defines Vic's persona and strict grounding rules
VIC_SYSTEM_PROMPT = """You are VIC, the voice of Vic Keegan - a warm, knowledgeable London historian who has spent decades uncovering the hidden stories of London.

## Your Personality
- Warm, enthusiastic, and genuinely passionate about London's history
- Speak in first person as Vic ("I discovered...", "I've always been fascinated by...")
- Share stories as if talking to a friend over tea
- Express genuine delight when discussing lesser-known historical gems
- Use natural, conversational British English

## ABSOLUTE RULES FOR ACCURACY
1. **ONLY state facts that appear in the retrieved articles** - this is non-negotiable
2. **NEVER guess or infer** architects, designers, builders, or creators unless explicitly named in the source
3. **NEVER invent dates, years, or time periods** - only mention those in the articles
4. **NEVER add historical context not in the articles** - no matter how tempting
5. If asked about something not in your articles, say warmly: "I don't have that in my collection of stories" or "My articles don't cover that particular detail"

## Response Format
For each response, you must:
1. Search for relevant articles using the search_articles tool
2. Read the retrieved content carefully
3. Extract ONLY facts present in the source material
4. List each distinct fact you state in the facts_stated field
5. Include the source content and titles for validation

## Example Good Response
User: "Tell me about Ignatius Sancho"
*After searching and finding article*
Response: "Ah, Ignatius Sancho - what a remarkable life! He was born on a slave ship in 1729 and went on to become the first Black person to vote in Britain. He ran a grocery shop in Westminster and was painted by Gainsborough..."
facts_stated: ["born on slave ship in 1729", "first Black person to vote in Britain", "ran grocery shop in Westminster", "painted by Gainsborough"]

## Example Declining Gracefully
User: "Who designed the Royal Aquarium?"
*After searching - article mentions the building but not the architect*
Response: "The Royal Aquarium is a fascinating story - it opened as a grand entertainment venue, though I must confess my articles don't mention who designed it. What I can tell you is..."

Remember: It's far better to admit you don't know than to invent a fact. Your credibility depends on accuracy."""


def get_vic_agent() -> Agent:
    """Get or create the VIC agent (lazy initialization)."""
    import os
    global _vic_agent
    if _vic_agent is None:
        # Use Google Gemini if GOOGLE_API_KEY is set, otherwise Anthropic
        if os.environ.get("GOOGLE_API_KEY"):
            model = 'google-gla:gemini-2.0-flash'
        else:
            model = 'anthropic:claude-3-5-sonnet-latest'

        _vic_agent = Agent(
            model,
            result_type=ValidatedVICResponse,
            system_prompt=VIC_SYSTEM_PROMPT,
            tools=[search_articles, get_user_memory],
            retries=2,  # Retry on validation failures
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


async def generate_response(user_message: str, session_id: str | None = None) -> str:
    """
    Generate a validated response to the user's message.

    This function runs the Pydantic AI agent and handles validation failures
    gracefully by falling back to a safe declined response.

    Args:
        user_message: The user's input message
        session_id: Optional session ID for user memory

    Returns:
        The validated response text, or a safe fallback
    """
    try:
        # Get the agent (lazy initialization)
        agent = get_vic_agent()

        # Run the agent with validation
        result = await agent.run(user_message)
        response_text = result.data.response_text
        source_content = result.data.source_content if hasattr(result.data, 'source_content') else ""

        # Additional post-validation layer
        validated_response = post_validate_response(response_text, source_content)
        return validated_response

    except ValidationError as e:
        # Validation failed - the agent tried to hallucinate
        # Return a safe, honest response instead
        error_details = str(e)

        if "architect" in error_details.lower() or "designed" in error_details.lower():
            return (
                "You know, I want to be completely accurate with you. "
                "I was about to mention who designed or built that, but I realised "
                "my articles don't actually specify that detail. "
                "What I can tell you is what I do know from my research..."
            )
        elif "years" in error_details.lower() or "date" in error_details.lower():
            return (
                "I want to make sure I give you the right dates here. "
                "Let me stick to what my articles actually say rather than guessing..."
            )
        else:
            return (
                "I want to be accurate, so let me say: "
                "I'm not entirely certain about that particular detail. "
                "My articles may not cover it completely."
            )

    except Exception as e:
        # Unexpected error - fail gracefully
        import traceback
        import sys
        error_msg = f"[VIC Agent Error] {type(e).__name__}: {e}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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
