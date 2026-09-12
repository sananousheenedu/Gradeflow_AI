import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from services.config import get_config
from services.pdf_service import extract_pdf_text, get_pdf_page_count
from services.groq_service import grade_student_paper, vision_ocr_pdf
from services.analytics import results_to_dataframe, build_question_analytics, detect_review_flags
from services.exports import make_excel, make_json_zip

st.set_page_config(page_title="GradeFlow AI", page_icon="📝", layout="wide")
cfg = get_config()

if "results" not in st.session_state:
    st.session_state.results = []

st.title("📝 GradeFlow AI")
st.caption("100+ separate student PDFs • each PDF may contain 2+ pages • Vision OCR → grading → consolidated result sheet")

with st.sidebar:
    st.header("⚙️ Settings")
    grading_model = st.text_input("Grading model", value=cfg.grading_model)
    vision_model = st.text_input("Vision/OCR model", value=cfg.vision_model)
    workers = st.slider("Grading workers", 1, 2, 2, help="Up to 2 grading workers. OCR itself remains rate-safe and serialized.")
    max_marks = st.number_input("Exam maximum marks", min_value=1, max_value=10000, value=100)
    review_threshold = st.slider("Manual review threshold", 0.0, 1.0, cfg.review_threshold, 0.05)
    use_vision = st.checkbox("Use Vision OCR for scanned/handwritten PDFs", value=True)
    max_pages = st.number_input("Maximum pages per PDF for OCR", min_value=1, max_value=100, value=max(30, cfg.max_vision_pages))
    st.divider()
    if cfg.api_key:
        st.success("Groq API key loaded")
    else:
        st.error("GROQ_API_KEY missing")
    st.caption("Optimized mode: digital pages are processed locally, only scanned pages use Vision OCR, OCR calls are serialized, and grading is rate-controlled.")

st.markdown("## 1. Exam setup")
c1, c2 = st.columns(2)
with c1:
    question_file = st.file_uploader("Question paper PDF", type=["pdf"], key="question_file")
with c2:
    answer_key_file = st.file_uploader("Official answer key PDF", type=["pdf"], key="answer_key_file")

question_text = ""
answer_key_text = ""

if question_file and answer_key_file:
    with st.spinner("Reading question paper and answer key..."):
        question_text = extract_pdf_text(question_file.getvalue())
        answer_key_text = extract_pdf_text(answer_key_file.getvalue())

    # Scanned references need OCR too; otherwise the grader would have no reliable key.
    if use_vision and (len(question_text.strip()) < cfg.min_text_chars_for_ocr or len(answer_key_text.strip()) < cfg.min_text_chars_for_ocr):
        with st.spinner("Reference PDF text is limited; using Vision OCR..."):
            if len(question_text.strip()) < cfg.min_text_chars_for_ocr:
                question_text = vision_ocr_pdf(cfg.api_key, question_file.getvalue(), vision_model, int(max_pages))
            if len(answer_key_text.strip()) < cfg.min_text_chars_for_ocr:
                answer_key_text = vision_ocr_pdf(cfg.api_key, answer_key_file.getvalue(), vision_model, int(max_pages))

    a, b = st.columns(2)
    a.metric("Question paper characters", f"{len(question_text):,}")
    b.metric("Answer key characters", f"{len(answer_key_text):,}")

st.markdown("## 2. Upload 100+ separate student PDFs")
st.info("Important: upload each student's answer sheet as a separate PDF. Example: Student 1.pdf, Student 2.pdf, Student 3.pdf ... Student 100.pdf")
student_files = st.file_uploader(
    "Student answer-sheet PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="student_files",
    help="Each uploaded PDF is ONE student. That PDF may contain 2, 3, 8, or more pages. Do not split a student across files unless you group them first.",
)

if student_files:
    total_mb = sum(len(f.getvalue()) for f in student_files) / (1024 * 1024)
    st.success(f"✅ {len(student_files)} student PDF(s) selected • {total_mb:.1f} MB total")
    if len(student_files) >= 100:
        st.success("🎯 100+ paper batch detected.")
    with st.expander("Preview uploaded papers"):
        preview = []
        for f in student_files[:100]:
            data = f.getvalue()
            preview.append({"File": f.name, "Pages": get_pdf_page_count(data), "Size (MB)": round(len(data)/(1024*1024), 2)})
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

st.markdown("## 3. Run batch")
if st.button("🚀 Start bulk grading", type="primary", use_container_width=True):
    if not cfg.api_key:
        st.error("No Groq API key. Add GROQ_API_KEY to Streamlit Secrets.")
        st.stop()
    if not question_file or not answer_key_file:
        st.error("Upload the question paper and official answer key first.")
        st.stop()
    if not question_text.strip() or not answer_key_text.strip():
        st.error("The question paper or answer key could not be read. Enable Vision OCR and try again.")
        st.stop()
    if not student_files:
        st.error("Upload at least one student PDF.")
        st.stop()

    ordered_files = sorted(student_files, key=lambda x: x.name.lower())
    progress = st.progress(0)
    status = st.empty()
    results = []
    started = time.time()

    def process(uploaded):
        pdf_bytes = uploaded.getvalue()
        page_count = get_pdf_page_count(pdf_bytes)
        try:
            text = extract_pdf_text(pdf_bytes)
            if use_vision:
                # vision_ocr_pdf now OCRs only pages that lack selectable text.
                # This is much faster for mixed digital/scanned student PDFs.
                text = vision_ocr_pdf(cfg.api_key, pdf_bytes, vision_model, int(max_pages))
            elif not text.strip():
                raise ValueError("No selectable text found. Enable Vision OCR for scanned/handwritten PDFs.")
            if not text.strip():
                raise ValueError("No readable text could be extracted from this PDF.")

            result = grade_student_paper(
                api_key=cfg.api_key,
                grading_model=grading_model,
                question_paper=question_text,
                answer_key=answer_key_text,
                student_text=text,
                max_marks=int(max_marks),
            )
            result["filename"] = uploaded.name
            result["page_count"] = page_count
            result["review_required"] = (
                float(result.get("confidence", 0)) < review_threshold
                or result.get("student_name", "Unknown") in ["Unknown", ""]
                or result.get("roll_no", "Unknown") in ["Unknown", ""]
            )
            result["review_reasons"] = detect_review_flags(result, review_threshold)
            return result
        except Exception as exc:
            return {
                "filename": uploaded.name,
                "page_count": page_count,
                "student_name": "ERROR",
                "roll_no": "ERROR",
                "score": 0,
                "total_marks": int(max_marks),
                "percentage": 0,
                "grade": "ERROR",
                "confidence": 0,
                "feedback": str(exc),
                "question_results": [],
                "review_required": True,
                "review_reasons": [str(exc)],
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, f) for f in ordered_files]
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            progress.progress(i / len(futures))
            elapsed = time.time() - started
            rate = elapsed / i
            remaining = max(0, rate * (len(futures) - i))
            status.write(f"Processed {i}/{len(futures)} papers • estimated remaining: {remaining/60:.1f} min")

    # Keep the final result sheet in filename order, not completion order.
    results.sort(key=lambda r: str(r.get("filename", "")).lower())
    st.session_state.results = results
    st.success(f"🎉 Completed {len(results)} student paper(s).")

if st.session_state.results:
    df = results_to_dataframe(st.session_state.results)
    st.markdown("---")
    st.markdown("## 📊 Consolidated result sheet")
    valid = pd.to_numeric(df["Score"], errors="coerce")
    review_count = int(df["Review Required"].fillna(False).sum())
    error_count = int((df["Status"] == "ERROR").sum())
    a, b, c, d, e = st.columns(5)
    a.metric("Students", len(df))
    b.metric("Average", f"{valid.mean():.1f}")
    c.metric("Highest", f"{valid.max():.1f}")
    d.metric("Review", review_count)
    e.metric("Errors", error_count)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("## 📈 Class analytics")
    qdf = build_question_analytics(st.session_state.results)
    if not qdf.empty:
        st.bar_chart(qdf.set_index("Question")["Average Marks"])
        st.dataframe(qdf, use_container_width=True, hide_index=True)

    st.markdown("## ⚠️ Teacher review queue")
    review_items = [r for r in st.session_state.results if r.get("review_required")]
    if not review_items:
        st.success("No papers currently require manual review.")
    else:
        st.warning(f"{len(review_items)} paper(s) require review.")
        for r in review_items:
            with st.expander(f"{r.get('student_name')} • {r.get('roll_no')} • {r.get('score')}/{r.get('total_marks')}"):
                for reason in r.get("review_reasons", []):
                    st.write(f"- {reason}")
                st.write(r.get("feedback", ""))
                if r.get("question_results"):
                    st.dataframe(pd.DataFrame(r["question_results"]), use_container_width=True, hide_index=True)

    st.markdown("## ⬇️ Export")
    excel_bytes = make_excel(st.session_state.results)
    zip_bytes = make_json_zip(st.session_state.results)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download Excel result sheet", data=excel_bytes, file_name="gradeflow_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with d2:
        st.download_button("Download JSON feedback ZIP", data=zip_bytes, file_name="gradeflow_feedback.zip", mime="application/zip", use_container_width=True)
