from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain.agents import initialize_agent
from agents.information import *
from agents.cv_enhancer import *
from agents.job_matching import *
from agents.motivation import *
from utils.config import get_memory
import logging
from langchain.schema import BaseMessage

routing_prompt = PromptTemplate.from_template("""
Tu es un routeur intelligent. Ton but est de lire la demande de l'utilisateur et de déterminer à quel agent elle doit être envoyée.

Tu dois répondre uniquement avec :
- "motivation_letter" : si l'utilisateur demande de générer une lettre de motivation et fournit à la fois un CV et une offre d'emploi (ou un lien vers une offre).
- "ask_cv" : si l'utilisateur veut une lettre de motivation mais n’a pas fourni de CV.
- "ask_offer" : si l'utilisateur veut une lettre de motivation mais n’a pas fourni d'offre d'emploi.
- "information" : si l'utilisateur pose une question ou demande une recherche documentaire.
- "job_matching" : si l'utilisateur veut trouver des offres d'emploi.
- "enhace_resume": si l'utilisateur veut modifier son resume ou de l'ameliorer ou meme des recommendations pour le faire.
- "resume_enquire": si l'utilisateur pose une question ou demande une recherche documentaire sur le resume ou cv mais n’a pas fourni de CV et une offre d'emploi.
- "unknown" : si tu ne sais pas quoi faire.

Examine attentivement les informations fournies et choisis la bonne réponse.

Entrée utilisateur :
{input_text}
""")
logger = logging.getLogger(__name__)
async def orchestrator_smart(input_text: str, llm):
    # Determine route
    router_chain = LLMChain(
        llm=llm,
        prompt=routing_prompt
    )
    route = (await router_chain.arun(input_text)).strip().lower()
    logger.info(f"Routing to: {route}")
    # Initialize appropriate agent
    if "motivation_letter" in route:
        # from agents.motivation import create_motivation_agent
        agent = create_motivation_agent(llm)
    elif "information" in route:
        # from agents.information import create_information_agent
        agent = create_information_agent(llm)
    elif "ask_cv" in route:
        # from agents.motivation import create_motivation_agent
        agent = create_motivation_agent(llm)
    elif "ask_offer" in route:
        # from agents.motivation import create_motivation_agent
        agent = create_motivation_agent(llm)
    elif "resume_enquire" in route:
        # from agents.motivation import create_motivation_agent
        agent = create_information_agent(llm)
    elif "enhace_resume" in route:
        # from agents.cv_enhancer import create_cv_enhancer_agent
        agent = create_cv_enhancer_agent(llm)
    elif "job_matching" in route:
        # from agents.job_matching import create_job_mztching_agent
        agent = create_job_mztching_agent(llm)
    else:
        return "I didn't understand your request. Could you please rephrase?"
    
    # return await agent.ainvoke(input_text)
    response = await agent.ainvoke(input_text)
    # If the response is a HumanMessage or contains one
    if isinstance(response, BaseMessage):
       return response.content
    elif isinstance(response, dict):
       if "output" in response:
         return response["output"]
       elif "content" in response:
         return response["content"]
       else:
         return str(response)
    elif isinstance(response, str):
       return response
    else:
       return str(response)

 