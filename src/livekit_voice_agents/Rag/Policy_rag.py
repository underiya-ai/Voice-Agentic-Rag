from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from livekit_voice_agents.Rag.fast_embeddings import embeddings
from langchain_pinecone import PineconeVectorStore
from livekit_voice_agents.Rag.vector import pc
from livekit_voice_agents.Rag.config import settings
INDEX_NAME="rag-voice"

pdf_path = "docs/Company-Policy-and-Procedure.pdf"

# load the documents in pdf formate 
Loader = PyPDFLoader(pdf_path)


docs = Loader.load()

# split the documents into chunks 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# create a free embeddings model in embeddings.py file and import it here 


# create the pine cone vector database in vector.py file and import it here 

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace="langgraph-agentic-rag"


)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4,
                                                    "namespace": "langgraph-agentic-rag"}
                                                    )

print("Retriever created successfully with k=5 and namespace = langgraph-agentic-rag")




# test the private kb retrieval
test_question = "What is the email of  Dr. Charu Raheja ?"
kb_docs = retriever.invoke(test_question)

for i, doc in enumerate(kb_docs):
    print(f"Document {i+1}:")
    print(doc.page_content[:500])  # Print the first 500 characters of the document content
    print("\n---\n")
