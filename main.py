from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.rag import ingestion
from src.rag import get_vector_store
from src.rag import ask_question
from src.rag import retrieve
from src.rag.evaluate import evaluate_interaction



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestion.ingest_data()

@app.get("/")
def testResponse():
   return {"message": "Hello World"}

@app.get("/update")
def update_vector_store():
    ingestion.ingest_data()
    vector_store = get_vector_store()
    total_docs = len(vector_store._collection.get()['ids'])
    return {"message": f"Vector store updated. Total documents: {total_docs}"}

@app.get("/query")
def query_vector_store(query: str):
    context_docs = retrieve(query)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    results = ask_question(query, context=context_text)
    return {"results": str(results), "context": [doc.page_content for doc in context_docs]}

@app.get("/evaluate")
def run_evaluation(query: str, reference: str):
    context_docs = retrieve(query)
    context_list = [doc.page_content for doc in context_docs]
    context_text = "\n\n---\n\n".join(context_list)

    response_text = ask_question(query, context=context_text)

    metrics_result = evaluate_interaction(
        query=query,
        contexts=context_list,
        response=str(response_text),
        reference=reference
    )

    return {
        "query": query,
        "reference_provided": reference,
        "generated_response": str(response_text),
        "metrics": metrics_result,
        "retrieved_context": context_list
    }