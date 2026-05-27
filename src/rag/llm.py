import re
import ollama

LLM_MODEL = "phi4-mini"

SYSTEM_PROMPT = """
voce e um assistente de perguntas e respostas. use somente o contexto fornecido.

regras:
- nao use conhecimento externo.
- se a informacao nao estiver claramente no contexto, responda exatamente: "desculpe, nao tenho informacoes suficientes para responder a essa pergunta."
- se o contexto tiver um link ou trecho literal que responda, devolva esse trecho exatamente como aparece.
- se o contexto mencionar apenas fontes (ex.: links) e a pergunta pedir detalhes especificos, diga que o contexto nao traz esses detalhes.
- responda de forma direta e objetiva.
"""
def ask_question(question: str, context: str = "") -> str:
    if context:
        cleaned_context = re.sub(r"[ \t]+", " ", context).strip()
        user_content = f"Contexto:\n{cleaned_context}\n\nPergunta:\n{question}"
    else:
        user_content = question

    

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return resp["message"]["content"]