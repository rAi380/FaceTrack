import pickle

with open("embeddings.pkl","rb") as f:
    data = pickle.load(f)

print("Total embeddings:", len(data["embeddings"]))
print("Total names:", len(data["names"]))
print("Sample names:", data["names"][:10])
print("Embedding length:", len(data["embeddings"][0]))