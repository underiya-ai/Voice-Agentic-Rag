from livekit_voice_agents.Rag.config import settings
from langchain_groq import ChatGroq
from livekit_voice_agents.graph.state import RouteDecision,EvidenceGrade,AgentState
from typing import Literal
from livekit_voice_agents.Rag.Policy_rag import retriever


llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    
)

router_llm = llm.with_structured_output(RouteDecision, method="json_mode")

kb_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

def route_decision(state:AgentState):
    question =  state["question"]

    decision = router_llm.invoke('''You are a router for an Agentic RAG assistant.

Route to "kb" if the user asks about:
- Agentic RAG
- LangGraph Agentic RAG workflow
- retrieval grading
- query rewriting
- RAG architecture
- retriever tools
- web fallback in RAG

Route to "direct" only for greetings, thanks, or very simple conversation.

Question:
{question}

Return your response as valid JSON.
Example:
{{"route": "kb"}}
''')

    return {
        "current_query": question,
        "route": decision.route
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"


# retrieve node ;- retrieve the documents from the kb using the retriever and the current query

def retrieve_kb(state: AgentState):
    query = state["current_query"]
    docs = retriever.invoke(query)

    print(f"[KB Retriever] Query: {query}")
    print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

    return {"kb_docs": docs}


def grade_kb_evidence(state: AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    grade = kb_grader_llm.invoke(f'''
You are an evidence grader.

Question:
{question}

Private KB evidence:
{context}

Can this private KB evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
''')

    print("[KB Grader]", grade.grade)

    return {"kb_grade": grade.grade}


def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"
