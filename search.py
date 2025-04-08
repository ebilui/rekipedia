# search.py
from embedder import collection, embedding_function

query = "ジェイコム東京　八王子・日野局の2024/12/01の入金状況"  # 🔍 ここを変えて検索
query_embedding = embedding_function([query])  # tolist()を削除

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

# 結果の表示
for i, doc in enumerate(results["documents"][0]):
    print(f"\n🔹 Top {i+1}")
    print("📝 Document:", doc)
    print("📄 ID:", results["ids"][0][i])
    print("🔢 Distance (approx.):", results.get("distances", [[None]*len(doc)])[0][i])
