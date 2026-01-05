"""Validated Interests API for Human-in-the-Loop Zep Storage.

Only stores interests to Zep after user confirmation (human-in-the-loop).
This ensures the knowledge graph contains only validated, user-approved facts.
"""

import os
import sys
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
_zep_client: Optional[httpx.AsyncClient] = None


def get_zep_client() -> Optional[httpx.AsyncClient]:
    """Get or create persistent Zep HTTP client."""
    global _zep_client
    if _zep_client is None and ZEP_API_KEY:
        _zep_client = httpx.AsyncClient(
            base_url="https://api.getzep.com",
            headers={
                "Authorization": f"Api-Key {ZEP_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
    return _zep_client


class ValidatedInterestRequest(BaseModel):
    """Request to store a validated interest in Zep."""
    userId: str
    fact: str
    articleId: Optional[int] = None
    articleTitle: Optional[str] = None
    validated: bool = True


class PendingInterestRequest(BaseModel):
    """Request to create a pending interest for human confirmation."""
    userId: str
    topic: str
    articleId: Optional[int] = None
    articleTitle: Optional[str] = None
    articleSlug: Optional[str] = None
    source: str = "conversation"


@router.post("/api/store-validated-interest")
async def store_validated_interest(request: ValidatedInterestRequest):
    """
    Store a validated interest in Zep.

    Called by the frontend after user confirms an interest.
    Only validated interests are stored in Zep for personalization.
    """
    if not request.validated:
        raise HTTPException(status_code=400, detail="Interest must be validated")

    if not ZEP_API_KEY:
        print("[Validated Interests] No ZEP_API_KEY, skipping storage", file=sys.stderr)
        return {"success": True, "stored": False, "reason": "No Zep API key"}

    try:
        client = get_zep_client()
        if not client:
            return {"success": True, "stored": False, "reason": "No Zep client"}

        # Ensure user exists
        await client.post(
            "/api/v2/users",
            json={"user_id": request.userId},
        )

        # Add validated fact to user's graph
        # Format: "User is confirmed interested in [Article Title]"
        fact_text = request.fact
        if request.articleTitle and "interested in" not in fact_text.lower():
            fact_text = f"User has confirmed interest in {request.articleTitle}"

        response = await client.post(
            f"/api/v2/users/{request.userId}/facts",
            json={
                "facts": [fact_text],
                "metadata": {
                    "validated": True,
                    "article_id": request.articleId,
                    "article_title": request.articleTitle,
                    "source": "human_validated",
                },
            },
        )

        if response.status_code == 200:
            print(f"[Validated Interests] ✓ Stored validated interest for {request.userId}: {request.articleTitle}", file=sys.stderr)
            return {"success": True, "stored": True}
        else:
            print(f"[Validated Interests] ✗ Zep error: {response.status_code} - {response.text[:100]}", file=sys.stderr)
            return {"success": True, "stored": False, "reason": f"Zep error: {response.status_code}"}

    except Exception as e:
        print(f"[Validated Interests] Error storing to Zep: {e}", file=sys.stderr)
        return {"success": True, "stored": False, "reason": str(e)}


@router.post("/api/create-pending-interest")
async def create_pending_interest(request: PendingInterestRequest):
    """
    Create a pending interest for human confirmation.

    Called by the CLM when a user discusses a topic.
    The interest is stored as "pending" until the user confirms it.
    """
    try:
        # Call the frontend API to create the pending interest
        frontend_url = os.environ.get("FRONTEND_URL", "https://lost.london")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{frontend_url}/api/interests",
                json={
                    "userId": request.userId,
                    "topic": request.topic,
                    "articleId": request.articleId,
                    "articleTitle": request.articleTitle,
                    "articleSlug": request.articleSlug,
                    "source": request.source,
                },
            )

            if response.status_code == 200:
                print(f"[Pending Interest] Created pending interest for {request.userId}: {request.topic}", file=sys.stderr)
                return {"success": True}
            else:
                print(f"[Pending Interest] Frontend error: {response.status_code}", file=sys.stderr)
                return {"success": False, "reason": f"Frontend error: {response.status_code}"}

    except Exception as e:
        print(f"[Pending Interest] Error: {e}", file=sys.stderr)
        return {"success": False, "reason": str(e)}
