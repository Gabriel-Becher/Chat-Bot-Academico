import pandas as pd
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag import ask_question, retrieve


def query_rag(question: str) -> tuple[str, list[str]]:
    context_docs = retrieve(question)
    response = ask_question(question, context="\n\n---\n\n".join(doc.page_content for doc in context_docs))
    return str(response), [doc.page_content for doc in context_docs]


def build_dataset_from_csv(csv_path: str) -> Dataset:
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=';', engine="python")

    required_cols = {"question", "reference"}

    if df.shape[0] == 0:
        df = pd.read_csv(csv_path, sep=None, engine="python", header=None)

    if not required_cols.issubset(set(df.columns)):
        if df.shape[1] >= 2:
            df = df.copy()
            new_cols = list(df.columns)
            new_cols[0] = "question"
            new_cols[1] = "reference"
            df.columns = new_cols
        else:
            raise ValueError(f"O CSV precisa ter as colunas: {required_cols}")

    user_inputs, references, responses, retrieved_contexts = [], [], [], []

    print(f"Processando {len(df)} perguntas...")

    for i, row in df.iterrows():
        question = str(row["question"])
        reference = str(row["reference"]) if "reference" in row else ""

        print(f"  [{i+1}/{len(df)}] {question[:60]}...")

        try:
            response, contexts = query_rag(question)
        except Exception as e:
            print(f"  ⚠️  Erro na pergunta {i+1}: {e}")
            response = ""
            contexts = []

        user_inputs.append(question)
        references.append(reference)
        responses.append(response)
        retrieved_contexts.append(contexts)

    return Dataset.from_dict({
        "user_input": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "response": responses,
        "reference": references,
    })


def evaluate_from_csv(csv_path: str, output_path: str = "resultados_ragas.csv") -> pd.DataFrame:
    local_llm        = ChatOllama(model="llama3.1:8b", temperature=0)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_emb = LangchainEmbeddingsWrapper(local_embeddings)

    dataset = build_dataset_from_csv(csv_path)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("\nAvaliando com RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df_results = result.to_pandas().replace({np.nan: None})
    df_results.insert(0, "question", dataset["user_input"])

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Médias gerais no final
    means = {col: df_results[col].mean() for col in metric_cols}
    print("\n── Médias gerais ──────────────────────────────")
    for metric, value in means.items():
        print(f"  {metric:<22} {value:.4f}")

    df_results.to_csv(output_path, index=False)
    print(f"\nResultados salvos em {output_path}")

    return df_results


if __name__ == "__main__":
    CSV_PATH = "perguntas_respostas.csv"

    df = evaluate_from_csv(CSV_PATH)

    print("\n── Resultado por pergunta ─────────────────────")
    print(df[["question", "faithfulness", "answer_relevancy",
              "context_precision", "context_recall"]].to_string(index=False))