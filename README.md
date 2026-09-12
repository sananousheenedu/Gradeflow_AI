# GradeFlow AI

GradeFlow AI is a Streamlit + Groq bulk exam checker for scanned and handwritten student answer sheets.

## Workflow
1. Upload one question paper PDF.
2. Upload one official answer key PDF.
3. Upload 100+ separate student answer-sheet PDFs.
4. Each PDF is treated as exactly one student, even when it contains 2 or more pages.
5. Scanned or handwritten pages are processed with Vision OCR, one page at a time.
6. All pages belonging to that PDF are combined into one student submission.
7. GradeFlow extracts the student's name and roll number, grades the paper, and keeps each student's result separate.
8. The app builds one consolidated result sheet with Roll No, Student Name, Score, Percentage, Grade, Confidence, Review, and File.
9. Download Excel and JSON feedback.

## Multi-page student PDFs
A student PDF can contain 2, 3, 5, 8, or more pages. The pages are OCR'd and combined before grading, so the pages are never treated as separate students.

## Batch reliability
- Each student PDF is processed independently, so one failed paper does not stop the batch.
- Vision requests are serialized and retried when Groq rate limits occur.
- Compact JPEG rendering reduces Vision input size.
- Start with one worker for large scanned batches.

## Streamlit Cloud
Set `GROQ_API_KEY` in Streamlit Secrets.

Recommended models:
- Grading: `openai/gpt-oss-20b`
- Vision/OCR: `qwen/qwen3.6-27b`
