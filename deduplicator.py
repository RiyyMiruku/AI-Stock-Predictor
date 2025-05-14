from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def deduplicate(news):
    titles = [n['title'] for n in news]
    embeddings = model.encode(titles, convert_to_tensor=True)
    keep = []
    for i, emb in enumerate(embeddings):
        if all(util.cos_sim(emb, embeddings[j]) < 0.9 for j in keep):
            keep.append(i)
    return [news[i] for i in keep]
