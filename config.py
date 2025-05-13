import os
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize HuggingFace login
login(token="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")

# Embeddings instance as a module-level variable
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Memory instance as a module-level variable
_memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    return_messages=True
)

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

def get_embeddings():
    """Get the pre-configured embeddings instance"""
    return _embeddings

def get_memory():
    """Get the pre-configured memory instance"""
    return _memory