from pinecone import Pinecone
from config import settings
from src.livekit_voice_agents.Rag.embeddings import embed

POLICY_PDF = "Company-Policy-and-Procedure.pdf"

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index("rag-voice")


def upsert_chunks(chunks: list[dict]) -> None:
    vectors = embed([c["text"] for c in chunks])
    index.upsert(vectors=[
        {"id": c["id"], "values": v,
         "metadata": {"text": c["text"], "source": c["source"], "page": c["page"]}}
        for c, v in zip(chunks, vectors)
    ])


def search(query: str, top_k: int = 4) -> list[dict]:
    qv = embed([query])[0]
    res = index.query(
        vector=qv,
        top_k=top_k,
        include_metadata=True,
        filter={"source": {"$eq": POLICY_PDF}},   # sirf policy PDF se
    )
    return [
        {"text": m.metadata["text"], "page": int(m.metadata.get("page", 0)),
         "score": m.score}
        for m in res.matches
    ]