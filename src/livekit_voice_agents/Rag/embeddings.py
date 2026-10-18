from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

embedding_model = TextEmbedding(model_name=MODEL_NAME)

def embed(texts:list[str]) -> list[list[float]]:
    """ Convert Texts to Embeddings using BGE model"""
    return [vector.tolist() for vector in embedding_model.embed(texts)]