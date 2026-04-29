from fastapi import APIRouter
from pydantic import BaseModel
from services.gmail_service import create_draft

router = APIRouter()

class DraftRequest(BaseModel):
    recipient: str
    subject: str
    body: str

@router.post("/draft")
def save_draft(request: DraftRequest):
    """
    Saves the provided email content as a draft in the user's connected Gmail account.
    """
    result = create_draft(
        recipient=request.recipient,
        subject=request.subject,
        body=request.body
    )
    
    if result["status"] == "error":
        return {"error": result["message"]}, 400
        
    return result
