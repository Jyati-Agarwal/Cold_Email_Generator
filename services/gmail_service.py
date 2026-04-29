import os
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]

def authenticate_gmail():
    """Shows basic usage of the Gmail API.
    Logs the user in and returns the credentials.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError("credentials.json not found. Please download it from Google Cloud Console.")
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            # This will open a browser window for OAuth
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
            
    return creds

def create_draft(recipient: str, subject: str, body: str) -> dict:
    """Create a draft email in the user's Gmail account."""
    try:
        creds = authenticate_gmail()
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
            "message": "Draft saved successfully in your Gmail account."
        }

    except HttpError as error:
        return {"status": "error", "message": f"An error occurred: {error}"}
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
         return {"status": "error", "message": f"Authentication failed: {str(e)}"}
