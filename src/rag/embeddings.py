from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = "nomic-embed-text"

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)