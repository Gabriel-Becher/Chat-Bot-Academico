from .vector_store import get_vector_store

def retrieve(query):
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=5)
    return results