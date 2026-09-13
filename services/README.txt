GradeFlow AI - 503 capacity patch

Replace ONLY services/groq_service.py in your GitHub repository with this file.

Fixes:
- A final 503/over-capacity response is converted to a controlled RuntimeError so the existing fallback-model logic can actually activate.
- Faster exponential backoff for temporary 503 capacity errors.
- Automatically chooses the other Qwen Vision model if the fallback is accidentally configured to the same model.
- No service_tier setting is used.

Do not replace the rest of the repository.
