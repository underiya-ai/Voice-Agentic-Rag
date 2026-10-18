from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharracterTextSplitter
from src.livekit_voice_agents.Rag.vectore import upsert_chunks

PDF_PATH = Path("docs/Company-policy-and-procedure.pdf")
BATCH_SIZE=100

def main():

    # load pdf 
    docs = PyPDFLoader(str(PDF_PATH)).load()

    # convert to chunks 
    text_splitters = RecursiveCharracterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitters.split_documents(docs)

    # 3 convert chunks into pinecone format

    records = [
        {
            "id": f"policy-{i}",
            "text": c.page_content,
            "source": PDF_PATH.name,
            "page":c.metadata.get("page",0)+1,


        }
        for i, c in enumerate(chunks)
    ]

    # 4. upload in the batches
    for s in range(0,len(records),BATCH_SIZE):
        upsert_chunks(records[s:s+BATCH_SIZE])

    print(f"Uploaded {len(records)} chunks to Pinecone index")

