from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import traceback
import time

from services.resume_parser import parse_resume, ParseError
from services.job_parser import parse_job_description, JobParseError
from services.web_search import research_company, WebSearchError
from services.context_builder import build_context, ContextBuildError
from services.email_generator import generate_email, EmailGenerationError

router = APIRouter()


@router.post("/process")
async def process_inputs(
    resume: UploadFile = File(..., description="User Resume (PDF)"),
    job_description_text: Optional[str] = Form(None, description="Job description text"),
    job_description_image: Optional[UploadFile] = File(None, description="Job description image"),
):
    """
    Full pipeline: resume + JD → parse → search → context → email.
    Returns all intermediate results so the frontend can display them.
    """
    pipeline_log = []

    def log_step(name: str, status: str, duration_ms: int):
        pipeline_log.append({"step": name, "status": status, "duration_ms": duration_ms})

    result = {
        "resume_data": None,
        "job_data": None,
        "search_data": None,
        "context": None,
        "email": None,
        "pipeline_log": pipeline_log,
    }

    try:
        # ── Validate inputs ──────────────────────────────────────────────
        if not resume.filename or not resume.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Resume must be a PDF file."})

        if not job_description_text and not job_description_image:
            return JSONResponse(
                status_code=400,
                content={"error": "Please provide either job description text or an image."},
            )

        # ── Step 1: Parse Resume ─────────────────────────────────────────
        t0 = time.time()
        try:
            resume_bytes = await resume.read()
            resume_data = parse_resume(resume_bytes)
            result["resume_data"] = resume_data
            log_step("parse_resume", "success", int((time.time() - t0) * 1000))
        except (ParseError, Exception) as e:
            log_step("parse_resume", "failed", int((time.time() - t0) * 1000))
            return JSONResponse(status_code=400, content={"error": f"Resume parsing failed: {e}", "pipeline_log": pipeline_log})

        # ── Step 2: Parse Job Description ────────────────────────────────
        t0 = time.time()
        try:
            if job_description_text:
                job_data = parse_job_description(text=job_description_text)
            else:
                img_bytes = await job_description_image.read()
                job_data = parse_job_description(
                    image_bytes=img_bytes,
                    image_mime_type=job_description_image.content_type,
                )
            result["job_data"] = job_data
            log_step("parse_job_description", "success", int((time.time() - t0) * 1000))
        except (JobParseError, Exception) as e:
            log_step("parse_job_description", "failed", int((time.time() - t0) * 1000))
            return JSONResponse(status_code=400, content={"error": f"JD parsing failed: {e}", "pipeline_log": pipeline_log})

        # ── Step 3: Web Search ───────────────────────────────────────────
        t0 = time.time()
        search_data = {"company_summary_sources": [], "contact_sources": [], "found_emails": [], "search_ran": False}
        try:
            company_name = job_data.get("company_name", "")
            role = job_data.get("role", "General Application")
            has_email = job_data.get("has_explicit_email", False)
            if company_name:
                search_data = research_company(company_name, role, has_email)
            result["search_data"] = search_data
            log_step("research_company", "success", int((time.time() - t0) * 1000))
        except (WebSearchError, Exception) as e:
            result["search_data"] = search_data
            log_step("research_company", "failed", int((time.time() - t0) * 1000))
            # Continue with degraded data — don't crash the pipeline

        # ── Step 4: Build Context ────────────────────────────────────────
        t0 = time.time()
        try:
            context = build_context(resume_data, job_data, search_data)
            result["context"] = context
            log_step("build_context", "success", int((time.time() - t0) * 1000))
        except (ContextBuildError, Exception) as e:
            log_step("build_context", "failed", int((time.time() - t0) * 1000))
            return JSONResponse(status_code=500, content={"error": f"Context build failed: {e}", "pipeline_log": pipeline_log})

        # ── Step 5: Generate Email ───────────────────────────────────────
        t0 = time.time()
        try:
            email_result = generate_email(context)
            result["email"] = email_result
            log_step("generate_email", "success", int((time.time() - t0) * 1000))
        except (EmailGenerationError, Exception) as e:
            log_step("generate_email", "failed", int((time.time() - t0) * 1000))
            return JSONResponse(status_code=500, content={"error": f"Email generation failed: {e}", "pipeline_log": pipeline_log})

        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"CRITICAL ERROR IN /process:\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})
