docker run --name qdrant_db -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.20
