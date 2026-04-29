from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Email Automation API",
    description="Backend for the AI-powered cold email generator",
    version="0.1.0",
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Allow the Next.js dev server (port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cold-email-generator-khaki.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

from api.input_handler import router as input_router
from api.gmail_handler import router as gmail_router

app.include_router(input_router, prefix="/api")
app.include_router(gmail_router, prefix="/api")

@app.get("/")
def root():
    """Root endpoint — confirms the API is alive."""
    return {"message": "Email Automation API is running 🚀"}


@app.get("/health")
def health_check():
    """
    Health-check endpoint.
    The frontend pings this to verify the backend is reachable.
    """
    return {
        "status": "ok",
        "service": "email-automation-backend",
        "version": "0.1.0",
    }
