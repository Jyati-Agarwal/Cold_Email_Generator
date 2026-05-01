import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

from auth.gmail_oauth import get_authorization_url, exchange_code_for_tokens, revoke_token
from auth.token_store import save_tokens, delete_tokens, is_gmail_connected
from services.gmail_service import create_draft

router = APIRouter()

@router.get("/auth/gmail/connect")
def connect_gmail(user_id: str):
    """
    Redirects the browser directly to the Google OAuth2 consent screen.
    Works both from frontend JS (window.location.href) and direct browser URL.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    try:
        auth_url = get_authorization_url(user_id)
        # Redirect the browser directly to Google's OAuth consent screen
        return RedirectResponse(url=auth_url, status_code=302)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/auth/gmail/callback")
def gmail_callback(code: str, state: str):
    """
    Handles the OAuth callback, exchanges code for tokens, and redirects to frontend.
    """
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    
    if not code or not state:
        return RedirectResponse(f"{frontend_url}/gmail-error?reason=invalid_code")
        
    try:
        tokens = exchange_code_for_tokens(code)
        save_tokens(state, tokens) # state is the user_id
        return RedirectResponse(f"{frontend_url}/gmail-connected")
    except Exception as e:
        print(f"Callback error: {e}")
        import traceback
        traceback.print_exc()
        return RedirectResponse(f"{frontend_url}/gmail-error?reason=exchange_failed")

@router.get("/auth/gmail/status")
def get_gmail_status(user_id: str):
    """
    Checks if the user has connected their Gmail account.
    """
    print(f"Checking status for user: {user_id}")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    try:
        connected = is_gmail_connected(user_id)
        print(f"User {user_id} connected: {connected}")
        return {"connected": connected}
    except Exception as e:
        print(f"Error checking status: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class DisconnectRequest(BaseModel):
    user_id: str

@router.post("/auth/gmail/disconnect")
def disconnect_gmail(request: DisconnectRequest):
    """
    Revokes the OAuth token and deletes tokens from DB.
    """
    from auth.token_store import get_tokens
    
    tokens = get_tokens(request.user_id)
    if tokens and tokens.get("refresh_token"):
        revoke_token(tokens["refresh_token"])
    elif tokens and tokens.get("access_token"):
        revoke_token(tokens["access_token"])
        
    delete_tokens(request.user_id)
    return {"status": "disconnected"}

class DraftRequest(BaseModel):
    user_id: str
    recipient: str
    subject: str
    body: str

@router.post("/email/save-draft")
def save_draft(request: DraftRequest):
    """
    Saves the provided email content as a draft in the user's connected Gmail account.
    """
    if not is_gmail_connected(request.user_id):
        raise HTTPException(status_code=400, detail="Gmail not connected. Please connect your Gmail first.")
        
    result = create_draft(
        user_id=request.user_id,
        recipient=request.recipient,
        subject=request.subject,
        body=request.body
    )
    
    if result["status"] == "error":
        msg = result["message"]
        if msg == "Gmail not connected":
            raise HTTPException(status_code=400, detail="Gmail not connected")
        elif msg == "Session expired, reconnect Gmail":
            raise HTTPException(status_code=401, detail="Session expired, please reconnect Gmail")
        else:
            raise HTTPException(status_code=500, detail=msg)
            
    return result
