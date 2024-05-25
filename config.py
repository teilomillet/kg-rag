import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
