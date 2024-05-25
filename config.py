import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    EXA_API_KEY = os.getenv("EXA_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVBILY_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
