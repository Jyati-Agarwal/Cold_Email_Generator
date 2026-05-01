import os
import datetime
from typing import Dict, Optional
from pymongo import MongoClient
from cryptography.fernet import Fernet

# Initialize MongoDB client
def get_tokens_collection():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    print(f"Connecting to MongoDB with URI: {mongo_uri}")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client.email_automation
        return db.gmail_tokens
    except Exception as e:
        print(f"CRITICAL: Failed to connect to MongoDB. Exception: {e}")
        return None

tokens_collection = get_tokens_collection()

def _get_fernet() -> Fernet:
    key = os.environ.get("TOKEN_ENCRYPTION_KEY")
    if not key:
        raise ValueError("TOKEN_ENCRYPTION_KEY environment variable is not set.")
    return Fernet(key.encode())

def save_tokens(user_id: str, tokens: Dict[str, str]) -> None:
    """
    Upsert into `gmail_tokens` by `user_id`.
    Encrypt access_token and refresh_token before storing.
    Store expires_at as a UTC datetime.
    """
    if tokens_collection is None:
        raise RuntimeError("MongoDB connection is not established.")
        
    f = _get_fernet()
    
    encrypted_access = f.encrypt(tokens["access_token"].encode()).decode()
    encrypted_refresh = None
    if tokens.get("refresh_token"):
        encrypted_refresh = f.encrypt(tokens["refresh_token"].encode()).decode()
        
    expires_at_str = tokens.get("expires_at")
    if isinstance(expires_at_str, str):
        # Handle 'Z' suffix for UTC if present
        if expires_at_str.endswith('Z'):
            expires_at_str = expires_at_str[:-1] + '+00:00'
        expires_at = datetime.datetime.fromisoformat(expires_at_str)
    else:
        expires_at = expires_at_str # Already datetime or None

    update_doc = {
        "access_token": encrypted_access,
        "expires_at": expires_at,
        "updated_at": datetime.datetime.utcnow()
    }
    
    # Only update refresh_token if provided (refresh flow doesn't always return one)
    if encrypted_refresh:
        update_doc["refresh_token"] = encrypted_refresh
        
    tokens_collection.update_one(
        {"user_id": user_id},
        {"$set": update_doc},
        upsert=True
    )

def get_tokens(user_id: str) -> Optional[Dict[str, str]]:
    """
    Fetch and decrypt tokens for a user.
    Return None if not found.
    """
    if tokens_collection is None:
        return None
        
    doc = tokens_collection.find_one({"user_id": user_id})
    if not doc:
        return None
        
    f = _get_fernet()
    
    try:
        access_token = f.decrypt(doc["access_token"].encode()).decode()
        refresh_token = None
        if doc.get("refresh_token"):
            refresh_token = f.decrypt(doc["refresh_token"].encode()).decode()
            
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": doc.get("expires_at").isoformat() if doc.get("expires_at") else None
        }
    except Exception as e:
        print(f"Failed to decrypt tokens for user {user_id}: {e}")
        return None

def delete_tokens(user_id: str) -> None:
    """
    Remove tokens document for a user.
    """
    if tokens_collection is not None:
        tokens_collection.delete_one({"user_id": user_id})

def is_gmail_connected(user_id: str) -> bool:
    """
    Return True if a valid token document exists for user_id.
    """
    if tokens_collection is None:
        return False
    # Simply check if the document exists
    return tokens_collection.count_documents({"user_id": user_id}, limit=1) > 0
