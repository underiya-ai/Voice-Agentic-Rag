
from dotenv import load_dotenv
from livekit_voice_agents.Rag.config import settings
from langchain.tools import tool
from langchain_tavily import TavilySearch
from livekit_voice_agents.Rag.Policy_rag import retriever
from langchain_groq import ChatGroq

tavily_search = TavilySearch(
    max_results=4,
    search_depth="general",
    tavily_api_key=settings.tavily_api_key,
    include_answers=True,
    include_raw=True

)
groq_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.4
)

@tool
def retrieve_knowledgebase(question:str):
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')}, page {doc.metadata.get('page', 0) + 1}\n{doc.page_content}"
        for doc in docs
    )

    prompt = f"""You are a voice assistant for employees.
    Answer the question using ONLY the context below.
    If the context does not contain the answer, say you could not find it in the policy.
    Answer in 2-3 short spoken sentences. No markdown, bullets, or URLs.
    Mention the page number.

    Context:
    {context}

    Question: {question}
   """
    answer = groq_llm.invoke(prompt).content

    return answer
    
    
# search tool
@tool
def search_tool(query: str) -> str:
    """Search the web for user queries or real-time data.
    Args:
    query: the user's search query
    """
    result = tavily_search.invoke({"query": query})
    return str(result)

    