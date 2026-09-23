from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love eating pizza.",
    "I enjoy eating burgers.",
    "Pizza is my favorite food.",
    "I want to learn Python programming.",
    "Python is easy to learn.",
    "Artificial intelligence is changing technology.",
    "Machine learning is a part of AI.",
    "The dog is playing in the garden."
]

embeddings = model.encode(sentences)

print("Total number of sentences:", len(sentences))
print("Embedding dimension:", len(embeddings[0]))

print("\n--- Embeddings ---")

for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    print("Embedding:", embeddings[i])

similarity = cosine_similarity(embeddings)

print("\n--- Semantic Similarity ---")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):

        if similarity[i][j] > 0.5:
            print(
                f"\nSentence 1: {sentences[i]}"
                f"\nSentence 2: {sentences[j]}"
                f"\nSimilarity: {similarity[i][j]:.4f}"
            )