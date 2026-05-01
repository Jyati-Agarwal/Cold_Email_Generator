import base64
import os
import datetime
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth.token_store import get_tokens, save_tokens
from auth.gmail_oauth import refresh_access_token, _get_client_config

class GmailDraftError(Exception):
    """Custom error for draft creation failures."""
    pass

def get_valid_credentials(user_id: str) -> Credentials:
    """
    Fetch tokens, refresh if needed, and return a Credentials object.
    """
    tokens = get_tokens(user_id)
    if not tokens:
        raise GmailDraftError("Gmail not connected")
        
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_at_str = tokens.get("expires_at")
    
    expires_at = None
    if expires_at_str:
        if expires_at_str.endswith('Z'):
            expires_at_str = expires_at_str[:-1] + '+00:00'
        expires_at = datetime.datetime.fromisoformat(expires_at_str)
        
        # Ensure it has timezone info for comparison
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=datetime.timezone.utc)
            
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Check if expired or expiring within 5 minutes
    if expires_at and (expires_at - now).total_seconds() < 300:
        if not refresh_token:
            raise GmailDraftError("Session expired, reconnect Gmail")
            
        try:
            new_tokens = refresh_access_token(refresh_token)
            # Update tokens dictionary
            tokens.update(new_tokens)
            save_tokens(user_id, tokens)
            access_token = tokens["access_token"]
        except Exception as e:
            raise GmailDraftError("Session expired, reconnect Gmail")
            
    config = _get_client_config()["web"]
    
    return Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri=config["token_uri"],
        client_id=config["client_id"],
        client_secret=config["client_secret"]
    )

def create_draft(user_id: str, recipient: str, subject: str, body: str) -> dict:
    """Create a draft email in the user's Gmail account."""
    try:
        creds = get_valid_credentials(user_id)
        service = build("gmail", "v1", credentials=creds)

        message = EmailMessage()
        message.set_content(body)
        message["To"] = recipient
        message["From"] = "me"
        message["Subject"] = subject

        # Encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"message": {"raw": encoded_message}}
        
        # Call the Gmail API to create the draft
        draft = service.users().drafts().create(userId="me", body=create_message).execute()
        
        return {
            "status": "success",
            "draft_id": draft["id"],
            "message": "Draft saved successfully."
        }

    except GmailDraftError as e:
        return {"status": "error", "message": str(e)}
    except HttpError as error:
        return {"status": "error", "message": f"Draft creation failed: {error}"}
    except Exception as e:
        return {"status": "error", "message": f"Draft creation failed: {str(e)}"}
