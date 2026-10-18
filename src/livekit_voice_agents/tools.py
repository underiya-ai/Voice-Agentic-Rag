
from dotenv import load_dotenv
from config import settings
from langchain.tools import tool
from langchain_tavily import TavilySearch


tavily_search = TavilySearch(
    max_results=3,
    search_depth="advanced",
    tavily_api_key=settings.tavily_api_key
)


# search tool
@tool
def search_tool(query:str) -> str:
    """Search the web for user queries or real-time data.
    Agrs:
    query: the user's search query
    """
    result= tavily_search.invoke({"query":query})
    return str(result)


    