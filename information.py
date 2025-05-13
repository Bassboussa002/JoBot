from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from tools.documents import *
from tools.search import search_tool
from utils.config import get_memory
from tools.web import web_search



def create_information_agent(llm):
    tools = [
        retriever_tool_startup_tunisia,
        retriever_tool_freelance_tunisia,
        # retriever_tool_certifications,
        retriever_tool_international_labor_market,
        retriever_tool_freelance,
        retriever_tool_Code_de_travail,
        retriever_tool_startup,
        retriever_tool_resume,
        Tool(
            name="web_search",
            func=web_search,
            description="Effectue une recherche web"
        )
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=get_memory(),
        handle_parsing_errors=True
    )