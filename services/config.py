import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


class Config:
    def __init__(self):
        self.api_key = get_secret("GROQ_API_KEY", "")
        self.grading_model = get_secret("GRADING_MODEL", "openai/gpt-oss-20b")
        self.vision_model = get_secret("VISION_MODEL", "qwen/qwen3.8-27b")
        self.review_threshold = float(get_secret("REVIEW_THRESHOLD", "0.75"))
        self.max_vision_pages = int(get_secret("MAX_VISION_PAGES", "20"))
        self.min_text_chars_for_ocr = int(get_secret("MIN_TEXT_CHARS_FOR_OCR", "300"))


def get_config():
    return Config()
