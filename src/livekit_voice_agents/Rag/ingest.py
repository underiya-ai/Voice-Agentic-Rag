from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.livekit_voice_agents.Rag.vectore import upsert_chunks

PDF_PATH = Path("docs/Company-Policy-and-Procedure.pdf")
BATCH_SIZE = 100


def load_pdf(path: Path):
    """PDF load karke pages return karo."""
    return PyPDFLoader(str(path)).load()


def split_pages(pages):
    """Pages ko chunks mein todo."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)


def to_records(chunks, source: str) -> list[dict]:
    """Chunks ko Pinecone ke format mein badlo."""
    return [
        {
            "id": f"{source}-{i}",
            "text": c.page_content,
            "source": source,
            "page": c.metadata.get("page", 0) + 1,
        }
        for i, c in enumerate(chunks)
    ]


def upload(records: list[dict]) -> None:
    """Records ko batches mein Pinecone par bhejo."""
    for s in range(0, len(records), BATCH_SIZE):
        upsert_chunks(records[s:s + BATCH_SIZE])


def main():
    pages = load_pdf(PDF_PATH)
    chunks = split_pages(pages)
    records = to_records(chunks, source=PDF_PATH.name)
    upload(records)
    print(f"Uploaded {len(records)} chunks to Pinecone index")


if __name__ == "__main__":
    main()