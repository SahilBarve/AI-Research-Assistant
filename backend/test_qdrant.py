from app.repositories.vector_repository import VectorRepository


vector_repository = VectorRepository()

points = vector_repository.get_all_points()

print("\n========== STORED POINTS ==========\n")

for point in points:
    print("Point ID:", point.id)
    print("Payload:", point.payload)

    if point.vector:
        print("Vector dimension:", len(point.vector))

    print("-----------------------------------")

print("\n===================================\n")