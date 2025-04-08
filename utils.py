from query import embedding_function
from embedder import collection
from collections import Counter

def select_best_csv_file_for_question(question: str) -> str:
    query_embedding = embedding_function([question])
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    file_ids = [doc_id for doc_id in results["ids"][0]]  # ファイル名全体を取得
    most_common_file = Counter(file_ids).most_common(1)[0][0]
    print("most_common_file:", most_common_file)

    # 拡張子の.csv以降を削除
    if ".csv" in most_common_file:
        most_common_file = most_common_file.split(".csv")[0] + ".csv"

    return most_common_file