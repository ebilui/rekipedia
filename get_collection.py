from chromadb import PersistentClient

client = PersistentClient(path="./chroma_db")
collection = client.get_collection("rekipedia")

data = collection.get()

sources = set(
    m["source"]
    for m in data["metadatas"]
    if m is not None and "source" in m
)

print("Sources in collection:")
for s in sources:
    print(s)
