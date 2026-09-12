import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, default=None):
    # Streamlit Cloud
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    # Local .env
    return os.getenv(name, default)


class Config:
    def __init__(self):
        self.api_key = get_secret("GROQ_API_KEY", "")
        self.grading_model = get_secret(
            "GRADING_MODEL",
            "openai/gpt-oss-20b"
        )
        self.vision_model = get_secret(
            "VISION_MODEL",
            "qwen/qwen3.6-27b"
        )
        self.max_workers = int(get_secret("MAX_WORKERS", "1"))
        self.review_threshold = float(
            get_secret("REVIEW_THRESHOLD", "0.75")
        )
        self.min_text_chars_for_ocr = int(
            get_secret("MIN_TEXT_CHARS_FOR_OCR", "300")
        )
        self.max_vision_pages = int(
            get_secret("MAX_VISION_PAGES", "20")
        )


def get_config():
    return Config()
