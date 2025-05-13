from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool


def search(query: str) -> str:
    search = SerpAPIWrapper(serpapi_api_key="")
    return search.run(query)

# You can create the tool to pass to an agent
search_tool = Tool(
    name="search",
    description="search on the web to get relevent informations",
    func=search,
)