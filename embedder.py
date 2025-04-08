from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
from typing import List
import chromadb

# SentenceTransformer のモデルを読み込み
model = SentenceTransformer("intfloat/multilingual-e5-base")

# ChromaDBが要求する形式に合った埋め込み関数クラスを定義
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
	def __call__(self, input: List[str]) -> List[List[float]]:
		return model.encode(input, convert_to_numpy=True).tolist()

# インスタンスを生成
embedding_function = SentenceTransformerEmbeddingFunction()

# ChromaDB クライアントとコレクションを作成（or取得）
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
	name="rekipedia",  # 任意のコレクション名
	embedding_function=embedding_function
)

# チャンクのリストをベクトル化してDBへ追加する関数
def add_chunks_to_chroma(chunks: List[str], source_id: str = "csv_upload"):
	ids = [f"{source_id}_{i}" for i in range(len(chunks))]
	collection.add(documents=chunks, ids=ids)
	print(f"✅ {len(chunks)} 個のチャンクを ChromaDB に追加しました。")
