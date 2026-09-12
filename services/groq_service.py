import base64
import json
import re
import threading
import time
from typing import Dict, Any, List

from groq import Groq
from services.pdf_service import extract_pages_with_text, render_pdf_pages

GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "student_name": {"type": "string"},
        "roll_no": {"type": "string"},
        "score": {"type": "number"},
        "percentage": {"type": "number"},
        "grade": {"type": "string"},
        "confidence": {"type": "number"},
        "feedback": {"type": "string"},
        "question_results": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "marks_awarded": {"type": "number"},
                "max_marks": {"type": "number"},
                "status": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["question", "marks_awarded", "max_marks", "status", "reason"],
            "additionalProperties": False,
        }},
    },
    "required": ["student_name", "roll_no", "score", "percentage", "grade", "confidence", "feedback", "question_results"],
    "additionalProperties": False,
}

GRADING_SYSTEM = """You are GradeFlow AI, an AI-assisted examination marking engine.
You receive an official question paper, official answer key, and ONE student's complete answer sheet.
All pages supplied for the student belong to the same student.
Identify name and roll number only from the student submission. Grade only against the supplied references.
Award justified partial credit. Never exceed maximum marks. Be conservative with unclear handwriting.
confidence must be between 0 and 1. Return only valid JSON matching the schema."""

# Separate locks keep OCR conservative while allowing grading to use a small amount
# of concurrency. The app defaults to 2 grading workers, but OCR remains serialized.
_VISION_LOCK = threading.Lock()
_GRADE_LOCK = threading.Lock()

# A tiny spacing delay prevents bursts after a successful call. This is not a
# replacement for provider limits; 429 responses still use the server retry delay.
VISION_GAP_SECONDS = 1.0
GRADE_GAP_SECONDS = 0.25
_last_vision_call = 0.0
_last_grade_call = 0.0


def _wait_for_gap(kind: str):
    global _last_vision_call, _last_grade_call
    gap = VISION_GAP_SECONDS if kind == "vision" else GRADE_GAP_SECONDS
    last = _last_vision_call if kind == "vision" else _last_grade_call
    remaining = gap - (time.monotonic() - last)
    if remaining > 0:
        time.sleep(remaining)


def _mark_call(kind: str):
    global _last_vision_call, _last_grade_call
    now = time.monotonic()
    if kind == "vision":
        _last_vision_call = now
    else:
        _last_grade_call = now


def _sleep_for_rate_limit(exc: Exception, default_seconds: float = 8.0) -> None:
    message = str(exc)
    match = re.search(r"try again in ([0-9.]+)s", message, flags=re.IGNORECASE)
    seconds = float(match.group(1)) if match else default_seconds
    time.sleep(min(max(seconds + 1.0, 2.0), 90.0))


def _call_with_retries(fn, attempts: int = 4):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "429" not in message and "rate_limit" not in message and "rate limit" not in message:
                raise
            if attempt < attempts - 1:
                _sleep_for_rate_limit(exc)
    raise last_error


def _vision_ocr_page(client: Groq, model: str, image_bytes: bytes, page_number: int) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    content = [
        {"type": "text", "text": (
            "Transcribe this exam answer-sheet page. Keep student name, roll number, "
            "question numbers, options, equations and written answers. Do not grade. "
            "Use [unclear] only when necessary. Plain text only."
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
    ]

    def request():
        _wait_for_gap("vision")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_completion_tokens=500,
        )
        _mark_call("vision")
        return response

    with _VISION_LOCK:
        response = _call_with_retries(request, attempts=4)

    return f"\n--- PAGE {page_number} ---\n{response.choices[0].message.content or ''}"


def vision_ocr_pdf(api_key: str, pdf_bytes: bytes, model: str, max_pages: int = 30) -> str:
    """OCR only pages that have little/no selectable text; preserve all page order."""
    page_info = extract_pages_with_text(pdf_bytes, min_chars=25)
    if page_info:
        ocr_pages = [p["page"] for p in page_info if p["needs_ocr"] and p["page"] <= max_pages]
        local_parts = [f"\n--- PAGE {p['page']} ---\n{p['text']}" for p in page_info if p["page"] <= max_pages and not p["needs_ocr"]]
    else:
        ocr_pages = list(range(1, max_pages + 1))
        local_parts = []

    if not ocr_pages:
        return "\n".join(local_parts).strip()

    images = render_pdf_pages(pdf_bytes, max_pages=max_pages, scale=0.60, page_numbers=ocr_pages)
    client = Groq(api_key=api_key)
    ocr_map = {}
    for item in images:
        ocr_map[item["page"]] = _vision_ocr_page(client, model, item["bytes"], item["page"])

    combined = []
    max_seen = max([p["page"] for p in page_info if p["page"] <= max_pages] + ocr_pages, default=0)
    for page_number in range(1, max_seen + 1):
        match = next((p for p in page_info if p["page"] == page_number), None)
        if match and not match["needs_ocr"]:
            combined.append(f"\n--- PAGE {page_number} ---\n{match['text']}")
        elif page_number in ocr_map:
            combined.append(ocr_map[page_number])
    return "\n".join(combined).strip()


def _grade_request(client, model, question_paper, answer_key, student_text, max_marks):
    """Grade one complete student submission using Groq JSON Object Mode.

    JSON Object Mode is intentionally used instead of strict json_schema mode
    because it is more compatible across Groq models and avoids schema-validation
    failures such as json_validate_failed.
    """
    prompt = f"""
Grade ONE student's complete exam submission.

All pages in the STUDENT ANSWERS belong to the same student.
Do not treat pages as separate students.

QUESTION PAPER:
{question_paper[:60000]}

OFFICIAL ANSWER KEY:
{answer_key[:60000]}

STUDENT ANSWERS:
{student_text[:90000]}

MAXIMUM EXAM MARKS:
{max_marks}

Return ONLY a valid JSON object. Do not use Markdown. Do not add any text before or after the JSON.

Use exactly these top-level keys:
student_name, roll_no, score, percentage, grade, confidence, feedback, question_results

question_results must be a JSON array. Each item must contain:
question, marks_awarded, max_marks, status, reason

Rules:
- Identify student_name and roll_no only from the student's submission.
- If either is unavailable, use "Unknown".
- Grade the complete submission across all pages.
- Do not double-count an answer.
- Award justified partial credit.
- Never award more than the maximum marks.
- score must be between 0 and {max_marks}.
- percentage must be between 0 and 100.
- confidence must be between 0 and 1.
- If handwriting or OCR is unclear, lower confidence and explain it in feedback.
- status should normally be one of: correct, partial, incorrect, unanswered, unclear.
- Keep feedback concise.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are GradeFlow AI. Return only a valid JSON object. "
                    "Never return Markdown or explanatory text outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=2600,
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)

def grade_student_paper(api_key: str, grading_model: str, question_paper: str,
                        answer_key: str, student_text: str, max_marks: int) -> Dict[str, Any]:
    client = Groq(api_key=api_key)

    def request():
        _wait_for_gap("grade")
        result = _grade_request(client, grading_model, question_paper, answer_key, student_text, max_marks)
        _mark_call("grade")
        return result

    # Grading calls are protected from bursts and still retry provider 429s.
    with _GRADE_LOCK:
        result = _call_with_retries(request, attempts=4)

    score = max(0.0, min(float(result.get("score", 0)), float(max_marks)))
    result["score"] = round(score, 2)
    result["total_marks"] = int(max_marks)
    result["percentage"] = round(score / max_marks * 100, 2) if max_marks else 0
    confidence = float(result.get("confidence", 0))
    result["confidence"] = max(0.0, min(confidence, 1.0))
    return result
