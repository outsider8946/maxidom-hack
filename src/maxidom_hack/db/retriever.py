"""Семантический поиск товаров каталога по текстовому запросу."""

from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.errors import NotFoundError

from maxidom_hack.categories import db_categories, query
from maxidom_hack.utils.clients import EmbeddingClient

# Бада лежит в пакете, поэтому путь считается от модуля, а не от рабочей директории.
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent / "chroma_db")


class ProductRetriever:
    """Ищет товары в ChromaDB и возвращает карточки с метаданными.

    Единственный источник товаров для UI: JSON-каталог не используется.
    """

    def __init__(
        self,
        embed_client: EmbeddingClient,
        db_path: str = DEFAULT_DB_PATH,
        db_name: str = "products",
        metric: str = "cosine",
    ) -> None:
        self.embed_client = embed_client
        self.db_name = db_name
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=db_name,
            metadata={"hnsw:space": metric},
        )

    @property
    def size(self) -> int:
        return self.collection.count()

    def search(self, query: str, top_k: int = 20, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Возвращает карточки товаров по убыванию релевантности запроса."""
        if not query.strip() or not self.size:
            return []

        embedding = self.embed_client.embed_query(query).tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.size),
            where=where or None,
            include=["metadatas", "documents", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]

        items = []
        for product_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
            text = str(document or "")
            items.append({
                "id": int(product_id),
                # cosine distance -> similarity
                "score": round(1 - float(distance), 4),
                # Название товара — первая строка документа, в метаданных его нет.
                "name": text.split("\n", 1)[0],
                "description": text.split("\n", 1)[1].strip() if "\n" in text else "",
                "category": metadata.get("category", ""),
                "brand": metadata.get("brand", ""),
                "price": metadata.get("price", 0),
                "unit": metadata.get("unit", ""),
                "surface": metadata.get("surface", ""),
                "moisture_resistant": bool(metadata.get("moisture_resistant", False)),
            })
        return items

    def by_category(self, category: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Ищет товары одного типа калькулятора: запрос + фильтр по метаданным."""
        return self.search(
            query(category),
            top_k=top_k,
            where={"category": {"$in": db_categories(category)}},
        )

    def available(self) -> bool:
        """Проверяет, что коллекция инициализирована и не пуста."""
        try:
            return self.collection.count() > 0
        except (NotFoundError, Exception):
            return False