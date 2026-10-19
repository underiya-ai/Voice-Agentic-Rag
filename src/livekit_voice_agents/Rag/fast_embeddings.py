
from langchain_community.embeddings import FastEmbedEmbeddings


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

sample_vector = embeddings.embed_query("Hello, World!")

print("embeddings dimenssion:", len(sample_vector))

