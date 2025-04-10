import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List, Generator
import os
import torch
from tqdm import tqdm
from processor import process_csv_file

# SentenceTransformer モデル
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda" if torch.cuda.is_available() else "cpu")

# ChromaDB 用 EmbeddingFunction の定義
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
	def __call__(self, input: List[str]) -> List[List[float]]:
		return model.encode(input, convert_to_numpy=True, show_progress_bar=False).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

# ChromaDB 初期化
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
	name="rekipedia",
	embedding_function=embedding_function,
)

def show_gpu_info():
	if torch.cuda.is_available():
		print(f"🚀 GPU 使用中: {torch.cuda.get_device_name(0)}")
	else:
		print("⚠️ GPU が使用されていません（CPUモードで実行中）")

def send_progress(progress, socketio):
	socketio.emit('progress', {'data': progress})

# チャンクをバッチでベクトル化・ChromaDBへ登録
def add_chunks_to_chroma_streaming(df: pd.DataFrame, source_id: str, socketio, batch_size=128):
	show_gpu_info()

	resume_file = f".resume_{source_id}.txt"
	processed_count = 0

	if os.path.exists(resume_file):
		with open(resume_file, "r") as f:
			processed_count = int(f.read().strip())
		print(f"🔁 {processed_count}件目から再開")

	total_chunks = process_csv_file(df)
	total = len(total_chunks)  # リストの長さを取得
	print(f"📦 処理対象チャンク数: {total}")

	with tqdm(total=total - processed_count, desc="🔄 登録中", ncols=80) as pbar:
		for i in range(processed_count, total, batch_size):
			batch = total_chunks[i:i + batch_size]  # 修正
			batch_ids = [f"{source_id}_{j}" for j in range(i, i + len(batch))]
			batch_embeddings = embedding_function(batch)
			collection.add(documents=batch, ids=batch_ids, embeddings=batch_embeddings, metadatas=[{"source": source_id, "row_index": i} for i in range(len(batch))])

			processed_count += len(batch)
			progress = (processed_count / total) * 100
			send_progress(progress, socketio)

			with open(resume_file, "w") as f:
				f.write(str(processed_count))

			pbar.update(len(batch))

	print("✅ ChromaDBへの登録完了")
