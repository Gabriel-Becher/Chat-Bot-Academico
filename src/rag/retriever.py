import re
from typing import List, Optional

from .vector_store import get_vector_store


def _normalize_query(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[\w@#]+", text.lower())
    return [w for w in words if len(w) >= 4]


def _contains_keywords(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def retrieve(
    query: str,
    k: int = 10,
    fetch_k: int = 60,
    use_mmr: bool = True,
    lambda_mult: float = 0.35,
    max_distance: Optional[float] = 0.6,
) -> List:
    vector_store = get_vector_store()
    normalized_query = _normalize_query(query)
    keywords = _extract_keywords(normalized_query)

    if use_mmr:
        results = vector_store.max_marginal_relevance_search(
            normalized_query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        if keywords and not any(_contains_keywords(doc.page_content, keywords) for doc in results):
            scored = vector_store.similarity_search_with_score(normalized_query, k=max(fetch_k, 200))
            keyword_hits = [doc for doc, _ in scored if _contains_keywords(doc.page_content, keywords)]
            if keyword_hits:
                seen = {doc.metadata.get("chunk_id") for doc in results}
                for doc in keyword_hits:
                    chunk_id = doc.metadata.get("chunk_id")
                    if chunk_id in seen:
                        continue
                    results.append(doc)
                    seen.add(chunk_id)
                    if len(results) >= k:
                        break
        return results

    scored = vector_store.similarity_search_with_score(normalized_query, k=fetch_k)
    if max_distance is not None:
        filtered = [doc for doc, score in scored if score <= max_distance]
        if filtered:
            return filtered[:k]

    return [doc for doc, _ in scored[:k]]