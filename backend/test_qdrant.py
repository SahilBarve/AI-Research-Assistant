from app.repositories.vector_repository import VectorRepository


# Create the VectorRepository.
vector_repository = VectorRepository()


print("\n========================================")
print("BEFORE DELETE")
print("========================================")

# Get all currently stored points.
points = vector_repository.get_all_points(
    limit=1000
)

print(f"Stored points: {len(points)}")


# Delete every point from the collection.
vector_repository.delete_all_points()


print("\n========================================")
print("AFTER DELETE")
print("========================================")

# Check Qdrant again.
points_after_delete = vector_repository.get_all_points(
    limit=1000
)

print(
    f"Stored points after deletion: "
    f"{len(points_after_delete)}"
)

if len(points_after_delete) == 0:
    print("\nSUCCESS: Qdrant is empty.")
else:
    print("\nWARNING: Qdrant still contains points.")