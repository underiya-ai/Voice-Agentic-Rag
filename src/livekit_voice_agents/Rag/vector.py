# create the pinecone vector database
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from livekit_voice_agents.Rag.config import settings
import time
INDEX_NAME="rag-voice"
NAMESPACE= "langgraph-agentic-rag"

pc = Pinecone(api_key=settings.pinecone_api_key)

existing_indexes = [index.name for index in pc.list_backups()]

if INDEX_NAME not in existing_indexes:
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384, # dimensssion of embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )

        except Exception:
            print("Error creating index. It may already exist.")
            pass
               
        

        while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)
print("Pinecone index ready:", INDEX_NAME)

# Upload the document chunks to the pinecone vectordatabase 


                

