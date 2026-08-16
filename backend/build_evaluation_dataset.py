from app.repositories.vector_repository import VectorRepository


# =========================================================
# LOAD ALL CHUNKS FROM QDRANT
# =========================================================

vector_repository = VectorRepository()

print("Loading chunks from Qdrant...")

stored_points = vector_repository.get_all_points(
    limit=1000
)

print(f"Total chunks loaded: {len(stored_points)}")


# =========================================================
# SORT CHUNKS BY CHUNK ID
# =========================================================

stored_points.sort(
    key=lambda point: point.payload["chunk_id"]
)


# =========================================================
# SAVE CHUNKS TO FILE
# =========================================================

output_file = "all_chunks.txt"

with open(output_file, "w", encoding="utf-8") as f:

    for point in stored_points:

        payload = point.payload

        chunk_id = payload["chunk_id"]
        text = payload["text"]

        f.write("\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(f"CHUNK ID: {chunk_id}\n")
        f.write("=" * 80)
        f.write("\n")

        f.write(text)
        f.write("\n")


print()
print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Chunks saved to: {output_file}")