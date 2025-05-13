import json
from langchain.docstore.document import Document
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from utils.config import get_embeddings
from langchain.schema import Document
import json

def generate_docs_for_certifications(doc_name):
    with open(doc_name, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)
    
    docs_processed = []
    for chunk in loaded_chunks:
        if isinstance(chunk, dict):
            # Convert all key-value pairs into a string
            full_text = "\n".join(f"{key}: {value}" for key, value in chunk.items())
            docs_processed.append(Document(page_content=full_text))
        else:
            # Handle non-dict entries, just convert to string
            docs_processed.append(Document(page_content=str(chunk)))
    
    return docs_processed



def generate_docs(doc_name):
    with open(doc_name, "r", encoding="utf-8") as f:
        loaded_chunks= json.load(f)
    # print(type(loaded_chunks))       # Should be list
    # print(type(loaded_chunks[0]))    # List or dict?
    # print(loaded_chunks[0])          # See what it actually contains

    docs_processed = [chunk[0] for chunk in loaded_chunks]  # Extract text from nested lists
    return [Document(page_content=text) for text in docs_processed]

embedding_model=get_embeddings()
vectordb_freelance = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\freelance.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
vectordb_freelance_tunisia = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\freelance_tunisia.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
vectordb_startup_tunisia = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\startup_tunisia.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
vectordb_startup = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\startup.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
# vectordb_certifications = FAISS.from_documents(
#     documents=generate_docs_for_certifications(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\Certifications.json'),
#     embedding=embedding_model,
#     distance_strategy=DistanceStrategy.COSINE,
# )
vectordb_Code_de_travail = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\Code_Travail.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
vectordb_cv_enhancement = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\Cv_enhancement.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
vectordb_international_labor_market = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\International_labor_Market.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
vectordb_tunisian_labor_market = FAISS.from_documents(
    documents=generate_docs(r'C:\Users\betta\OneDrive\Bureau\projet_agents\data\Tunisian_labor_market.json'),
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)