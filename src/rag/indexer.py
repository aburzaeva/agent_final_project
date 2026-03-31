"""Indexes the product nutrition CSV into ChromaDB for RAG lookups."""

import csv
import os

import chromadb
from chromadb.config import Settings


DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "products.csv")
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma"))
COLLECTION_NAME = "products"


def build_index(csv_path: str = DATA_CSV, persist_dir: str = CHROMA_DIR):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            doc_text = (
                f"{row['name']} — категория: {row['category']}. "
                f"Калории: {row['calories']} ккал, "
                f"белки: {row['protein']} г, "
                f"жиры: {row['fat']} г, "
                f"углеводы: {row['carbs']} г, "
                f"клетчатка: {row['fiber']} г, "
                f"сахар: {row['sugar']} г."
            )
            ids.append(f"product_{i}")
            documents.append(doc_text)
            metadatas.append({
                "name": row["name"],
                "calories": float(row["calories"]),
                "protein": float(row["protein"]),
                "fat": float(row["fat"]),
                "carbs": float(row["carbs"]),
                "fiber": float(row["fiber"]),
                "sugar": float(row["sugar"]),
                "sodium": float(row["sodium"]),
                "category": row["category"],
            })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(ids)} products into ChromaDB at {persist_dir}")
    return collection


if __name__ == "__main__":
    build_index()
