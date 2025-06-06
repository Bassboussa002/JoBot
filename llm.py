from langchain_groq import ChatGroq
from utils.config import GROQ_API_KEY

def initialize_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1024
    )

