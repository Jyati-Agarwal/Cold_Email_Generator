import os
import requests
from google_auth_oauthlib.flow import Flow
from typing import Dict, Any

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
REDIRECT_URI = os.environ.get("GMAIL_REDIRECT_URI", "http://localhost:8000/api/auth/gmail/callback")

class OAuthError(Exception):
    """Custom exception for OAuth related errors."""
    pass

def _get_client_config() -> Dict[str, Any]:
    """Helper to build client config dict from env vars."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise OAuthError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in environment variables.")
        
    return {
        "web": {
            "client_id": client_id,
            "project_id": "cold-email-generator", # Placeholder, not strictly needed for the flow
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": [REDIRECT_URI]
        }
    }

def get_authorization_url(user_id: str) -> str:
    """
    Build the Google OAuth2 authorization URL.
    Pass user_id as the state parameter so we can identify the user on callback.
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        raise OAuthError("GOOGLE_CLIENT_ID must be set in environment variables.")
        
    import urllib.parse
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "state": user_id,
        "prompt": "consent"
    }
    query_string = urllib.parse.urlencode(params)
    return f"https://accounts.google.com/o/oauth2/v2/auth?{query_string}"

def exchange_code_for_tokens(code: str) -> dict:
    """
    Exchange the authorization code for access_token, refresh_token, and expires_at.
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise OAuthError("Missing client ID or secret for token exchange.")
        
    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI
            }
        )
        
        if response.status_code != 200:
            raise OAuthError(f"Token exchange failed: {response.text}")
            
        data = response.json()
        
        import datetime
        expires_in = data.get("expires_in", 3600)
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)).isoformat()
        
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_at": expires_at
        }
    except requests.RequestException as e:
        raise OAuthError(f"Network error during token exchange: {e}")

def refresh_access_token(refresh_token: str) -> dict:
    """
    POST to https://oauth2.googleapis.com/token with grant_type=refresh_token
    Return updated access_token and expires_at
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise OAuthError("Missing client ID or secret for token refresh.")

    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
        )
        
        if response.status_code != 200:
            raise OAuthError(f"Token refresh failed: {response.text}")
            
        data = response.json()
        
        import datetime
        expires_in = data.get("expires_in", 3600)
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)).isoformat()
        
        return {
            "access_token": data["access_token"],
            "expires_at": expires_at
        }
    except requests.RequestException as e:
        raise OAuthError(f"Network error during token refresh: {e}")

def revoke_token(token: str) -> None:
    """
    POST to https://oauth2.googleapis.com/revoke
    Used when user disconnects Gmail
    """
    try:
        requests.post(
            "https://oauth2.googleapis.com/revoke",
            params={"token": token},
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
    except requests.RequestException as e:
        # Ignore network errors on revoke, token is already deleted locally
        pass
