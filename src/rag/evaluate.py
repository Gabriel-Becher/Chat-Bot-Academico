import pandas as pd
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

def evaluate_interaction(query: str, contexts: list, response: str, reference: str):
    
    local_llm = ChatOllama(model="llama3.1:8b", temperature=0)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_emb = LangchainEmbeddingsWrapper(local_embeddings)

    data = {
        "user_input": [query],
        "retrieved_contexts": [contexts],
        "response": [response],
        "reference": [reference],
    }

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    df_results = result.to_pandas()

    # Substituição do NaN por None pro FastAPI não chorar
    df = df_results.replace({np.nan: None})

    return {
        "faithfulness": df["faithfulness"].iloc[0],
        "answer_relevancy": df["answer_relevancy"].iloc[0],
        "context_precision": df["context_precision"].iloc[0],
        "context_recall": df["context_recall"].iloc[0]
    }