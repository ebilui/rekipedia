# query.py
from embedder import collection, embedding_function
from openai import OpenAI
import os

# 必要に応じて OPENAI_API_KEY を環境変数などでセット
client = OpenAI()

def query_documents(query_text, top_k=5):
    query_embedding = embedding_function([query_text])
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        id_ = results["ids"][0][i]
        dist = results["distances"][0][i]
        hits.append({"document": doc, "id": id_, "distance": dist})
    return hits

def generate_answer(query_text, hits):
    context = "\n---\n".join([hit["document"] for hit in hits])
    prompt = f"""
あなたはCSVデータに詳しいアシスタントです。
以下はCSVから得られた関連情報です：

{context}

この情報を元に、次の質問にできるだけ正確に答えてください：

「{query_text}」
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # または gpt-4
        messages=[
            {"role": "system", "content": "あなたはCSVに詳しいデータアシスタントです。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# CLIテスト
if __name__ == "__main__":
    query = input("🔍 質問を入力してください: ")
    results = query_documents(query)

    for i, hit in enumerate(results, 1):
        print(f"\n🔹 Top {i}")
        print("📝 Document:", hit['document'][:300])
        print("📄 ID:", hit['id'])
        print("🔢 Distance:", hit['distance'])
