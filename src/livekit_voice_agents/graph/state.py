# from typing import List, Literal, TypedDict

# from langchain_core.documents import Document
# from pydantic import BaseModel, Field


# class RouteDecision(BaseModel):
#     route: Literal["kb", "direct"] = Field(
#         description="Use 'kb' for questions that need the company policy document; "
#                     "'direct' for greetings or simple chat that needs no search."
#     )


# class EvidenceGrade(BaseModel):
#     grade: Literal["good", "weak"] = Field(
#         description="'good' means the evidence can answer the question; "
#                     "'weak' means there is not enough evidence."
#     )


# class AgentState(TypedDict):
#     question: str          # employee ka original sawaal
#     current_query: str     # search ke liye rewrite ki hui query
#     kb_docs: List[Document]  # Pinecone se mile chunks
#     web_results: str       # Tavily se mile results (abhi optional)
#     kb_grade: str          # "good" ya "weak"
#     web_grade:str
#     answer:str
#     source_used:str
#     retry_count:int
    