import json
import os
import random
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


load_dotenv("./.env")

# Configuration
if not os.getenv("QDRANT_HOST"):
    raise ValueError("QDRANT_HOST is missing from .env")

FILE_PATH = "../data/arxiv-metadata-oai-snapshot.json"

COLLECTION_NAME = "arxiv_abstracts"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

BATCH_SIZE = 64
UPLOAD_BATCH_SIZE = 256

TARGET_CATEGORIES = {"cs.LG"}


# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load embedding model on GPU
model = SentenceTransformer(
    MODEL_NAME,
    device=device,
    trust_remote_code=True,
)

embedding_dim = model.get_embedding_dimension()

print(f"Embedding dimension: {embedding_dim}")


# Initialize Qdrant
client = QdrantClient(url=os.getenv("QDRANT_HOST"))

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=embedding_dim,
        distance=models.Distance.COSINE,
    ),
)


# Load and filter arXiv papers
def load_arxiv_papers(file_path):
    documents = []
    metadata = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            paper = json.loads(line)

            categories = paper["categories"].split()

            if TARGET_CATEGORIES.intersection(categories):
                title = " ".join(paper["title"].split())
                abstract = " ".join(paper["abstract"].split())

                if not title or not abstract:
                    continue

                # TODO: Remove
                if random.random() > 0.1:
                    continue
                
                documents.append(abstract)

                metadata.append(
                    {
                        "source": "arXiv",
                        "arxiv_id": paper["id"],
                        "title": title,
                        "authors": paper["authors"].split(", "),
                        "published": paper["update_date"],
                        "categories": categories,
                    }
                )

            if i % 100_000 == 0:
                print(
                    f"Processed: {i:,} | "
                    f"Collected: {len(documents):,}"
                )

    return documents, metadata


print("Loading arXiv dataset...")

docs, metadata = load_arxiv_papers(FILE_PATH)

print(f"Total papers collected: {len(docs):,}")


# Generate embeddings
def generate_embeddings(documents):
    all_embeddings = []

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]

        embeddings = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )

        all_embeddings.extend(embeddings)

        if start % (BATCH_SIZE * 100) == 0:
            print(
                f"Embedded: {min(start + BATCH_SIZE, len(documents)):,}"
                f" / {len(documents):,}"
            )

    return all_embeddings


print(f"Generating embeddings on {device}...")

embeddings = generate_embeddings(docs)


# Upload to Qdrant in batches
print("Uploading embeddings to Qdrant...")

for start in range(0, len(docs), UPLOAD_BATCH_SIZE):
    end = min(start + UPLOAD_BATCH_SIZE, len(docs))

    points = []

    for idx in range(start, end):
        points.append(
            models.PointStruct(
                id=idx + 1,
                vector=embeddings[idx].tolist(),
                payload={
                    "document": docs[idx],
                    "metadata": metadata[idx],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(f"Uploaded: {end:,} / {len(docs):,}")


print("Finished!")
