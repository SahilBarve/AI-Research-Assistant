from app.services.embeddings.embedding_service import EmbeddingService


embedding_service = EmbeddingService()

text = "Machine learning is a subset of artificial intelligence."

vector = embedding_service.embed_text(text)

print("Vector dimension:", len(vector))
print("First 10 values:", vector[:10])