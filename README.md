# GradeFlow AI

AI-assisted exam checking for separate student PDFs.

## Core rule

**1 PDF = 1 student.** A student PDF can contain **2 or more pages**. All pages in that PDF are treated as one complete submission.

## Processing

1. Upload question paper PDF.
2. Upload official answer key PDF.
3. Upload 100+ separate student PDFs.
4. Digital pages are extracted locally.
5. Only scanned/handwritten pages use Vision OCR.
6. All pages for one student are combined.
7. The complete submission is graded once.
8. Results appear progressively and are exported to Excel.

## Reliability

- Vision OCR is serialized to reduce token bursts.
- Groq 429 responses are retried using the provider's retry time when available.
- JSON Object Mode is used for grading.
- If JSON validation fails, grading automatically retries once without `response_format` and parses JSON locally.
- A failed student does not stop the remaining batch.
