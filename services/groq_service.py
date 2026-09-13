import base64
import json
import re
import threading
import time
from typing import Any, Dict

from groq import Groq

from services.pdf_service import extract_pages_with_text, render_pdf_pages

_VISION_LOCK = threading.Lock()
_GRADE_LOCK = threading.Lock()
VISION_GAP_SECONDS = 1.0
GRADE_GAP_SECONDS = 0.35
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
    if kind == "vision":
        _last_vision_call = time.monotonic()
    else:
        _last_grade_call = time.monotonic()


def _sleep_for_rate_limit(exc: Exception, default_seconds: float = 8.0):
    message = str(exc)
    match = re.search(r"try again in ([0-9.]+)s", message, flags=re.I)
    seconds = float(match.group(1)) if match else default_seconds
    time.sleep(min(max(seconds + 1.0, 2.0), 90.0))


def _call_with_retries(fn, attempts: int = 4):
    """Retry short-lived rate limits, but NEVER hammer a daily-token limit."""
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            message = str(exc).lower()

            # TPD is a daily model-token ceiling. If Groq gives a short
            # "try again in ..." interval, wait once and retry automatically.
            if "tokens per day" in message or "tpd" in message:
                match = re.search(r"try again in ([0-9]+)m(?:([0-9.]+)s)?", message, flags=re.I)
                if attempt == 0 and match:
                    minutes = float(match.group(1))
                    seconds = float(match.group(2) or 0)
                    wait_seconds = min(max(minutes * 60 + seconds + 2.0, 5.0), 360.0)
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    "Groq daily token limit is still exhausted for this model. "
                    "Groq currently provides Qwen 3.6 and Qwen 3.8 as the Vision models; "
                    "this patch cannot bypass the daily quota. Wait for the reset or "
                    "use a higher Groq tier. Original error: " + str(exc)
                ) from exc

            is_rate = (
                "429" in message
                or "rate limit" in message
                or "rate_limit" in message
            )
            if not is_rate or attempt == attempts - 1:
                raise
            _sleep_for_rate_limit(exc)
    raise last


def _vision_ocr_page(client: Groq, model: str, image_bytes: bytes, page_number: int) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    content = [
        {
            "type": "text",
            "text": (
                "Transcribe this exam answer-sheet page exactly enough for grading. "
                "Keep student name, roll number, question numbers, options, equations, "
                "calculations and written answers. Do not grade. Do not summarize. "
                "Use [unclear] only when necessary. Plain text only."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        },
    ]

    def request():
        _wait_for_gap("vision")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_completion_tokens=350,
        )
        _mark_call("vision")
        return response

    with _VISION_LOCK:
        response = _call_with_retries(request)

    return f"\n--- PAGE {page_number} ---\n{response.choices[0].message.content or ''}"


def vision_ocr_pdf(api_key: str, pdf_bytes: bytes, model: str, max_pages: int = 30, fallback_model: str = "qwen/qwen3.8-27b") -> str:
    """Use local text extraction for digital pages and Vision only for scanned pages."""
    page_info = extract_pages_with_text(pdf_bytes, min_chars=25)
    if not page_info:
        return ""

    selected = [p for p in page_info if p["page"] <= max_pages]
    ocr_pages = [p["page"] for p in selected if p["needs_ocr"]]
    local_pages = {p["page"]: p["text"] for p in selected if not p["needs_ocr"]}

    ocr_map = {}
    if ocr_pages:
        images = render_pdf_pages(
            pdf_bytes,
            max_pages=max_pages,
            scale=0.60,
            page_numbers=ocr_pages,
        )
        client = Groq(api_key=api_key)
        active_model = model
        for item in images:
            try:
                ocr_map[item["page"]] = _vision_ocr_page(
                    client, active_model, item["bytes"], item["page"]
                )
            except RuntimeError as exc:
                # If the primary Vision model has exhausted its daily token quota,
                # try the alternate Vision model once. This is a model-level fallback,
                # not a retry against the exhausted quota.
                if "daily token limit reached" in str(exc).lower() and fallback_model and active_model != fallback_model:
                    active_model = fallback_model
                    ocr_map[item["page"]] = _vision_ocr_page(
                        client, active_model, item["bytes"], item["page"]
                    )
                else:
                    raise

    combined = []
    for page in range(1, max((p["page"] for p in selected), default=0) + 1):
        if page in local_pages:
            combined.append(f"\n--- PAGE {page} ---\n{local_pages[page]}")
        elif page in ocr_map:
            combined.append(ocr_map[page])

    return "\n".join(combined).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON even if a model accidentally wraps it in Markdown."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def _grading_prompt(question_paper: str, answer_key: str, student_text: str, max_marks: int) -> str:
    return f"""
You are GradeFlow AI, an examination marking engine.

Grade ONE student's complete answer submission. Every page in STUDENT ANSWERS belongs to the same student.

QUESTION PAPER:
{question_paper[:60000]}

OFFICIAL ANSWER KEY:
{answer_key[:60000]}

STUDENT ANSWERS:
{student_text[:90000]}

TOTAL MAXIMUM MARKS: {max_marks}

Return ONLY one valid JSON object with these top-level keys:
student_name, roll_no, score, percentage, grade, confidence, feedback, question_results

question_results is an array. Each item MUST contain:
question, student_answer, correct_answer, marks_awarded, max_marks, status, reason

IMPORTANT:
- student_answer MUST contain the student's actual answer copied from STUDENT ANSWERS.
- correct_answer MUST contain the corresponding answer from OFFICIAL ANSWER KEY.
- Do NOT leave student_answer or correct_answer empty or omit these fields.
- If the answer cannot be determined, write "Unclear from submission" instead.

Rules:
- Identify student_name and roll_no only from the student submission. Use "Unknown" if unavailable.
- Treat all pages as one student and do not double-count answers.
- Grade only against the supplied question paper and answer key.
- Award partial marks whenever the student's answer is partly correct.
- Full marks only when the answer is fully correct.
- Award 0 marks when the answer is completely incorrect or unanswered.
- For a partly correct answer, award a reasonable score between 0 and the question maximum.
- Never exceed the maximum marks for any question.
- score must equal the sum of all question marks_awarded.
- score must be 0 to {max_marks}.
- For partial answers, award a reasonable amount between 0 and the question's maximum marks.
- Never exceed the maximum marks for any question.
- score must equal the sum of all question marks_awarded.
- score must be 0 to {max_marks}.
- percentage must be 0 to 100.
- confidence must be 0 to 1.
- status should be correct, partial, incorrect, unanswered, or unclear.
- If OCR/handwriting is unclear, reduce confidence and mention it briefly in feedback.
- No Markdown and no text outside JSON.
"""


def _grade_request(client: Groq, model: str, prompt: str, max_marks: int, strict_json: bool):
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. You are an accurate exam grader."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=2600,
    )
    if strict_json:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def grade_student_paper(
    api_key: str,
    grading_model: str,
    question_paper: str,
    answer_key: str,
    student_text: str,
    max_marks: int,
) -> Dict[str, Any]:
    client = Groq(api_key=api_key)
    prompt = _grading_prompt(question_paper, answer_key, student_text, max_marks)

    def request():
        _wait_for_gap("grade")
        try:
            # First attempt: JSON Object Mode for predictable machine-readable output.
            response = _grade_request(client, grading_model, prompt, max_marks, True)
        except Exception as first_error:
            # Some model/account combinations reject or fail JSON validation. Retry once
            # without response_format; we still parse and validate the returned JSON locally.
            message = str(first_error).lower()
            if "json_validate_failed" not in message and "failed to validate json" not in message:
                raise
            time.sleep(1.0)
            response = _grade_request(client, grading_model, prompt, max_marks, False)
        _mark_call("grade")
        return response

    with _GRADE_LOCK:
        response = _call_with_retries(request)

    content = response.choices[0].message.content or "{}"
    try:
        result = _extract_json(content)
    except Exception as exc:
        raise ValueError(f"Grading model returned invalid JSON: {exc}") from exc

    score = max(0.0, min(float(result.get("score", 0)), float(max_marks)))
    result["score"] = round(score, 2)
    result["total_marks"] = int(max_marks)
    result["percentage"] = round(score / max_marks * 100, 2) if max_marks else 0
    confidence = float(result.get("confidence", 0))
    result["confidence"] = max(0.0, min(confidence, 1.0))
    return result
