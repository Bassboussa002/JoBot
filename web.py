from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from bs4 import BeautifulSoup
import json
@tool
def extract_job_information(url: str) -> str:
    """Extracts job details from URLs"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    return f"""
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
    # result = llm.invoke(prompt)
    # result_content = result.content.strip()

    # # Ensure JSON output is correctly formatted
    # try:
    #     job_info = json.loads(result_content)
    #     return json.dumps(job_info, indent=2)  # Pretty print JSON output
    # except json.JSONDecodeError:
    #     return f"Erreur: le modèle n'a pas renvoyé un JSON valide. Voici ce que j'ai reçu :\n\n{result_content}"


@tool
def generate_cover_letter(input_str: str) -> str:
    """
    Génère une lettre de motivation adaptée en se basant sur le contenu du CV et l'offre d'emploi.

    L'entrée doit être une chaîne contenant d'abord le contenu du CV, puis celui de l'offre d'emploi, dans un ordre naturel.
    Aucun séparateur spécifique n'est requis.
    """

    return f"""
Tu es un expert en rédaction de lettres de motivation sur mesure.

À partir des deux blocs d’informations ci-dessous — le premier étant un CV, le second une offre d'emploi — rédige une lettre de motivation convaincante, claire et engageante, en t'appuyant **uniquement sur les éléments réellement présents dans le CV**.

Ne fais aucune supposition ni invention sur le parcours du candidat. Si une information n’apparaît pas dans le CV, ne l’utilise pas.

Ta lettre doit :
- Être fluide, professionnelle et authentique.
- Mettre en avant les expériences et compétences pertinentes pour l’offre.
- Montrer une motivation sincère et bien alignée avec la mission de l’entreprise.
- Être structurée naturellement (introduction, paragraphes cohérents, conclusion).
- Ne contenir **aucun encadré, ni titre, ni commentaire** : uniquement le texte de la lettre.

Voici les informations :

{input_str}

Génère uniquement le texte de la lettre de motivation.
"""

    # return llm.invoke(prompt).content.strip()

search = DuckDuckGoSearchRun()
@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))

@tool
def web_search(query: str) -> str:
    """Performs web searches"""
    return search.run(query)

@tool
def enhance_cv(input_str: str) -> str:
    """
    Améliore un CV en se basant uniquement sur son contenu existant.

    L'entrée doit être une chaîne contenant d'abord le contenu du CV, puis celui de l'offre d'emploi, dans un ordre naturel.
    Aucun séparateur spécifique n'est requis.
    """

    return f"""
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

    # return llm.invoke(prompt).content.strip()
