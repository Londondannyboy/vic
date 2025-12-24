"""Tools for the Pydantic AI agent - article search and user memory."""

import os
import httpx
from pydantic_ai import RunContext

from .models import SearchResults, ArticleResult
from .database import search_articles_hybrid

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL = "voyage-2"
ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
LOST_LONDON_GRAPH_ID = "lost-london"

# OPTIMIZATION: Persistent HTTP clients for connection reuse
_voyage_client: httpx.AsyncClient | None = None


def get_voyage_client() -> httpx.AsyncClient:
    """Get or create persistent Voyage HTTP client."""
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = httpx.AsyncClient(
            base_url="https://api.voyageai.com",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
    return _voyage_client


# Phonetic corrections for voice input - matching lost.london
PHONETIC_CORRECTIONS: dict[str, str] = {
    # Names
    "ignacio": "ignatius",
    "ignasio": "ignatius",
    "ignacius": "ignatius",
    # Places
    "thorny": "thorney",
    "fawny": "thorney",
    "fauny": "thorney",
    "forney": "thorney",
    "tie burn": "tyburn",
    "tieburn": "tyburn",
    "aquarim": "aquarium",
    "aquariam": "aquarium",
    "royale": "royal",
    "cristal": "crystal",
    "crystle": "crystal",
    "shakespear": "shakespeare",
    "shakespere": "shakespeare",
    "westmister": "westminster",
    "white hall": "whitehall",
    "parliment": "parliament",
    "tems": "thames",
    "devils acre": "devil's acre",
    "devil acre": "devil's acre",
    # Additional corrections
    "voxhall": "vauxhall",
    "vox hall": "vauxhall",
    "southwork": "southwark",
    "grenwich": "greenwich",
    "wolwich": "woolwich",
    "bermondsy": "bermondsey",
    "holbourn": "holborn",
    "aldwich": "aldwych",
    "chisick": "chiswick",
    "dulwitch": "dulwich",
}


def normalize_query(query: str) -> str:
    """Apply phonetic corrections to normalize the query."""
    import re
    normalized = query.lower().strip()

    for wrong, correct in PHONETIC_CORRECTIONS.items():
        # Use word boundary matching for accuracy
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        normalized = pattern.sub(correct, normalized)

    return normalized


async def get_voyage_embedding(text: str) -> list[float]:
    """Generate embedding using Voyage AI with persistent client."""
    client = get_voyage_client()
    response = await client.post(
        "/v1/embeddings",
        json={
            "model": VOYAGE_MODEL,
            "input": text,
            "input_type": "query",
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


async def search_zep_graph(query: str) -> dict:
    """
    Search Zep knowledge graph for entity connections.

    Returns facts and relationships that span multiple articles.
    This enriches responses with connections the user might not know about.
    """
    if not ZEP_API_KEY:
        return {"facts": [], "connections": []}

    try:
        async with httpx.AsyncClient() as client:
            # Search for edges (relationships between entities)
            edge_response = await client.post(
                f"https://api.getzep.com/api/v2/graph/{LOST_LONDON_GRAPH_ID}/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "limit": 5,
                    "scope": "edges",
                    "reranker": "rrf",
                },
                timeout=5.0,  # Fast timeout - enrichment shouldn't slow us down
            )

            facts = []
            connections = []

            if edge_response.status_code == 200:
                data = edge_response.json()
                edges = data.get("edges", [])

                for edge in edges:
                    # Extract fact
                    if edge.get("fact"):
                        facts.append(edge["fact"])

                    # Extract connection
                    source = edge.get("source_node_name")
                    target = edge.get("target_node_name")
                    relation = edge.get("relation")
                    if source and target and relation:
                        connections.append({
                            "from": source,
                            "relation": relation,
                            "to": target
                        })

            return {"facts": facts[:3], "connections": connections[:3]}

    except Exception as e:
        import sys
        print(f"[Zep Graph] Error: {e}", file=sys.stderr)
        return {"facts": [], "connections": []}


async def search_articles(ctx: RunContext[None], query: str) -> SearchResults:
    """
    Search Lost London articles using hybrid vector + keyword search.

    This tool searches the knowledge base for articles relevant to the query.
    Results are ranked by semantic similarity and keyword matching.

    Args:
        query: The user's question or topic to search for

    Returns:
        SearchResults containing matching articles and the normalized query
    """
    # Normalize query with phonetic corrections
    normalized_query = normalize_query(query)

    # Get embedding
    embedding = await get_voyage_embedding(normalized_query)

    # Search database
    results = await search_articles_hybrid(
        query_embedding=embedding,
        query_text=normalized_query,
        limit=5,
        similarity_threshold=0.45,
    )

    articles = [
        ArticleResult(
            id=r["id"],
            title=r["title"],
            content=r["content"],
            score=r["score"],
        )
        for r in results
    ]

    return SearchResults(articles=articles, query=normalized_query)


async def get_user_memory(ctx: RunContext[None], user_id: str) -> dict:
    """
    Retrieve user facts and preferences from Zep Cloud knowledge graph.

    Zep automatically extracts facts and entities from conversations,
    so this retrieves what Zep knows about the user.

    Args:
        user_id: The unique identifier for the user

    Returns:
        Dictionary containing user facts and memories from the graph
    """
    if not ZEP_API_KEY or not user_id:
        return {"facts": [], "memories": []}

    async with httpx.AsyncClient() as client:
        try:
            # Search user's personal graph for facts about them
            response = await client.post(
                "https://api.getzep.com/api/v2/graph/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "user_id": user_id,
                    "query": "user preferences interests name",
                    "limit": 10,
                    "scope": "edges",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Extract facts from edges
            facts = []
            for edge in data.get("edges", []):
                if edge.get("fact"):
                    facts.append(edge["fact"])

            return {
                "facts": facts,
                "memories": data.get("edges", []),
            }
        except Exception:
            # Fail gracefully - memory is optional
            return {"facts": [], "memories": []}


async def save_user_message(user_id: str, message: str, role: str = "user") -> bool:
    """
    Add a message to the user's Zep memory graph.

    Zep automatically extracts facts and entities from messages,
    so no explicit "remember" logic is needed.

    Args:
        user_id: The unique identifier for the user
        message: The message content
        role: Either "user" or "assistant"

    Returns:
        True if saved successfully, False otherwise
    """
    if not ZEP_API_KEY or not user_id:
        return False

    async with httpx.AsyncClient() as client:
        try:
            # First ensure user exists
            await client.post(
                "https://api.getzep.com/api/v2/users",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"user_id": user_id},
                timeout=10.0,
            )
        except Exception:
            pass  # User may already exist

        try:
            # Add message to user's graph
            response = await client.post(
                "https://api.getzep.com/api/v2/graph",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "user_id": user_id,
                    "type": "message",
                    "data": f"{role}: {message}",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False


async def search_knowledge_graph(query: str, limit: int = 10) -> list[dict]:
    """
    Search the Lost London knowledge graph via Zep.

    This provides relationship-aware search across articles.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of relevant edges/relationships from the graph
    """
    if not ZEP_API_KEY:
        return []

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.getzep.com/api/v2/graph/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "graph_id": LOST_LONDON_GRAPH_ID,
                    "query": query,
                    "limit": limit,
                    "scope": "edges",
                    "reranker": "rrf",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("edges", [])
        except Exception:
            return []
