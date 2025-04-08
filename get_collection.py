from embedder import collection

print(collection.count())
# コレクション内のすべてのデータを取得
results = collection.get()

if results is None or not results.get("ids"):
    print("🚨 データが見つかりませんでした。")
else:
    # ID, ドキュメント（テキスト）、埋め込みベクトルを確認
    print("📄 IDs:", results["ids"])
    print("📜 Documents:", results["documents"])