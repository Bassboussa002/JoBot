from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from tools.documents import fetch_pdf_content_tool
# from tools.web import extract_job_information, enhance_cv
from utils.config import get_memory
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from bs4 import BeautifulSoup
import json
from utils.llm import initialize_llm

llm=initialize_llm()

def create_cv_enhancer_agent(llm):
    def enhance_cv_tool(input_str: str) -> str:
        """
        Améliore un CV en se basant uniquement sur son contenu existant.
        """
        prompt = f"""
Tu es un expert en rédaction et optimisation de CV.

Ta mission est de réécrire ce CV pour le rendre plus professionnel, clair, structuré et percutant, **sans inventer de nouvelles expériences ou compétences**.

Ta version améliorée doit :
- Mettre en valeur les expériences et compétences clés déjà présentes.
- Uniformiser la présentation et la formulation.
- Optimiser chaque section pour maximiser l’impact auprès des recruteurs.
- Éliminer les formulations vagues, les répétitions et les éléments inutiles.
- Garder un ton professionnel, authentique et fluide.
- Respecter les informations données : **ne rien ajouter qui ne figure pas dans le CV d'origine**.

Voici le contenu du CV :

{input_str}

Génère uniquement le contenu du CV optimisé, sans encadré ni commentaire.
"""
        return llm.invoke(prompt).content.strip()

    def extract_job_info_tool(url: str) -> str:
        """
        Extrait les informations de l'offre d'emploi depuis une URL.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        prompt = f"""
Voici une offre d'emploi extraite du site :

{text}

Retourne un JSON contenant **toutes** les informations suivantes :
- titre
- entreprise
- compétences (liste)
- qualifications (liste)
- responsabilités (liste)
- lieu
- salaire (si mentionné)

**Ne retourne rien d'autre que ce JSON brut.**
"""
        result = llm.invoke(prompt)
        result_content = result.content.strip()
        try:
            job_info = json.loads(result_content)
            return json.dumps(job_info, indent=2)
        except json.JSONDecodeError:
            return f"Erreur: le modèle n'a pas renvoyé un JSON valide. Voici ce que j'ai reçu :\n\n{result_content}"

    tools = [
        fetch_pdf_content_tool,
        Tool(
            name="extract_job_info",
            func=extract_job_info_tool,
            description="Extracts job details from URLs"
        ),
        Tool(
            name="cv_enhancer",
            func=enhance_cv_tool,
            description="Enhances CV structure and clarity"
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=get_memory(),
        handle_parsing_errors=True
    )
