from fastapi import FastAPI
from src.rag import ingestion
from src.rag import get_vector_store

app = FastAPI()

@app.get("/")
def testResponse():
   return {"message": "Hello World"}

@app.get("/update")
def update_vector_store():
    ingestion.ingest_data()
    vector_store = get_vector_store()
    total_docs = len(vector_store._collection.get()['ids'])
    return {"message": f"Vector store updated. Total documents: {total_docs}"}