"""Retriever for looking up product nutrition data from ChromaDB."""

import os

import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma"))
COLLECTION_NAME = "products"


class ProductRetriever:
    def __init__(self, persist_dir: str = CHROMA_DIR):
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search for products matching the query. Returns list of product dicts."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(query_texts=[query], n_results=n_results)

        products = []
        if results and results["metadatas"]:
            for meta in results["metadatas"][0]:
                products.append({
                    "name": meta["name"],
                    "calories": meta["calories"],
                    "protein": meta["protein"],
                    "fat": meta["fat"],
                    "carbs": meta["carbs"],
                    "fiber": meta["fiber"],
                    "sugar": meta["sugar"],
                    "sodium": meta["sodium"],
                    "category": meta["category"],
                })
        return products

    def get_product(self, product_name: str) -> dict | None:
        """Get the closest matching product by name."""
        results = self.search(product_name, n_results=1)
        return results[0] if results else None

    def calculate_nutrition(self, ingredients: list[dict]) -> dict:
        """
        Calculate total nutrition for a list of ingredients.
        Each ingredient: {"name": str, "grams": float}
        Nutrition values in the DB are per 100g.
        """
        total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0, "sugar": 0, "sodium": 0}
        details = []

        for ing in ingredients:
            product = self.get_product(ing["name"])
            if product:
                factor = ing["grams"] / 100.0
                item = {
                    "name": ing["name"],
                    "matched_product": product["name"],
                    "grams": ing["grams"],
                    "calories": round(product["calories"] * factor, 1),
                    "protein": round(product["protein"] * factor, 1),
                    "fat": round(product["fat"] * factor, 1),
                    "carbs": round(product["carbs"] * factor, 1),
                }
                details.append(item)
                for key in total:
                    total[key] += product[key] * factor
            else:
                details.append({"name": ing["name"], "grams": ing["grams"], "error": "product not found in database"})

        for key in total:
            total[key] = round(total[key], 1)

        return {"total": total, "details": details}
