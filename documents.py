from langchain.tools import Tool
from langchain_community.document_loaders import PyMuPDFLoader
from data.docs import *
def fetch_pdf_content_func(pdf_path: str) -> str:
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()[0]
    return data.page_content

fetch_pdf_content_tool = Tool(
    name="fetch_pdf_content",
    func=fetch_pdf_content_func,
    description="Fetches text content from PDF files"
)

def retrieve_documents_freelance(query: str) -> str:
    docs = vectordb_freelance.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."
def retrieve_documents_freelance_tunisia(query: str) -> str:
    docs = vectordb_freelance_tunisia.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."
def retrieve_documents_startup(query: str) -> str:
    docs = vectordb_startup.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."
def retrieve_documents_startup_tunisia(query: str) -> str:
    docs = vectordb_startup_tunisia.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."

# def retrieve_documents_certifications(query: str) -> str:
#     docs = vectordb_certifications.similarity_search(query, k=1)
#     if docs :
#         return "\n".join([doc.page_content for doc in docs])
#     else:
#         return "Could not find relevant information."

def retrieve_documents_Code_de_travail(query: str) -> str:
    docs = vectordb_Code_de_travail.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."

def retrieve_documents_cv_enhancement(query: str) -> str:
    docs = vectordb_cv_enhancement.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."

def retrieve_documents_international_labor_market(query: str) -> str:
    docs = vectordb_international_labor_market.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."

def retrieve_documents_tunisian_labor_market(query: str) -> str:
    docs = vectordb_tunisian_labor_market.similarity_search(query, k=1)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."


# Create the Retriever Tools
retriever_tool_freelance = Tool(
    name="international freelance retriever",
    func=retrieve_documents_freelance,
    description="Retrieve relevant documents from the vector store related to freelancing internationally based on semantic similarity."
)
retriever_tool_freelance_tunisia = Tool(
    name="tunisian freelance retriever",
    func=retrieve_documents_freelance_tunisia,
    description="Retrieve relevant documents from the vector store related to freelancing in tunisia based on semantic similarity."
)
retriever_tool_international_labor_market = Tool(
    name="international labor market retriever",
    func=retrieve_documents_international_labor_market,
    description="Retrieve relevant documents from the vector store related to international labor market based on semantic similarity."
)
# retriever_tool_certifications = Tool(
#     name="certifications retriever",
#     func=retrieve_documents_certifications,
#     description="Retrieve relevant documents from the vector store related to certifications based on semantic similarity."
# )
retriever_tool_startup = Tool(
    name="international startup retriever",
    func=retrieve_documents_startup,
    description="Retrieve relevant documents from the vector store related to startup internationally based on semantic similarity."
)
retriever_tool_startup_tunisia = Tool(
    name="tunisian startup retriever",
    func=retrieve_documents_startup,
    description="Retrieve relevant documents from the vector store related to startup in tunisia based on semantic similarity."
)
retriever_tool_resume = Tool(
    name="resume retriever",
    func=retrieve_documents_cv_enhancement,
    description="Retrieve relevant documents from the vector store related to resume enhancement based on semantic similarity."
)
retriever_tool_Code_de_travail = Tool(
    name="labor code retriever",
    func=retrieve_documents_Code_de_travail,
    description="Retrieve relevant documents from the vector store related to the Tunisian labor code (Code du Travail en Tunisie) based on semantic similarity."
)
