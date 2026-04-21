from langchain_chroma import Chroma
from .embeddings import get_embeddings


def get_vector_store():
    return Chroma(collection_name="documents", persist_directory="db/chroma", embedding_function=get_embeddings())