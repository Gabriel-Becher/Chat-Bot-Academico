from ..rag import load_docs, chunk_data, get_vector_store

def ingest_data():
    documents = load_docs()
    chunked_data = chunk_data(documents)

    ids = [doc.metadata["chunk_id"] for doc in chunked_data]

    vector_store = get_vector_store()

    existing_ids = set(vector_store._collection.get(ids=ids)["ids"])
    new_docs = [doc for doc in chunked_data if doc.metadata["chunk_id"] not in existing_ids]

    if new_docs:
        new_ids = [doc.metadata["chunk_id"] for doc in new_docs]
        vector_store.add_documents(new_docs, ids=new_ids)
    
    print(f"Ingested {len(new_docs)} new documents. Total documents in vector store: {len(vector_store._collection.get()['ids'])}")


if __name__ == "__main__":
    ingest_data()