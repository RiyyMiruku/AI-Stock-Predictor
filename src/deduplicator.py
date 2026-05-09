from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer("all-MiniLM-L6-v2")


def deduplicate(news: list[dict]) -> list[dict]:
    """以新聞標題的語意相似度（cos>=0.8）去除重複新聞。"""
    titles = [n["title"] for n in news]
    embeddings = _model.encode(titles, convert_to_tensor=True)
    keep = []
    for i, emb in enumerate(embeddings):
        if all(util.cos_sim(emb, embeddings[j]) < 0.8 for j in keep):
            keep.append(i)
    return [news[i] for i in keep]
