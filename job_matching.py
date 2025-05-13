from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from tools.web import web_search
from utils.config import get_memory
def create_job_mztching_agent(llm):
    tools = [
        Tool(
            name="web_search",
            func=web_search,
            description="Effectue une recherche web"
        )
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=get_memory()
    )