"""Семантический поиск товаров каталога по текстовому запросу."""

from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.errors import NotFoundError

from maxidom_hack.utils.categories import db_categories
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

    def filter(self, category_keys: list[str], limit: int | None = None) -> Any:
        """Возвращает сырой ответ ChromaDB по списку категорий калькулятора.

        ``limit`` ограничивает выборку, чтобы не читать всю категорию целиком.
        """
        db_to_key: dict[str, str] = {}
        all_db_cats: list[str] = []
        for key in category_keys:
            for db_cat in db_categories(key):
                db_to_key[db_cat] = key
                all_db_cats.append(db_cat)

        if not all_db_cats:
            return {"ids": [], "metadatas": [], "documents": []}

        return self.collection.get(
            where={"category": {"$in": all_db_cats}},  # type: ignore[dict-item]
            include=["metadatas", "documents"],
            limit=limit,
        )

    def search(
        self, query: str, top_k: int = 20, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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

        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        items = []
        for product_id, metadata, document, distance in zip(
            ids, metadatas, documents, distances
        ):
            text = str(document or "")
            items.append(
                {
                    "id": int(product_id),
                    # cosine distance -> similarity
                    "score": round(1 - float(distance), 4),
                    # Название товара — первая строка документа, в метаданных его нет.
                    "name": text.split("\n", 1)[0],
                    "description": text.split("\n", 1)[1].strip()
                    if "\n" in text
                    else "",
                    "category": metadata.get("category", ""),
                    "brand": metadata.get("brand", ""),
                    "price": metadata.get("price", 0),
                    "unit": metadata.get("unit", ""),
                    "surface": metadata.get("surface", ""),
                    "moisture_resistant": bool(
                        metadata.get("moisture_resistant", False)
                    ),
                }
            )
        return items

    def by_category(self, category: str) -> list[dict[str, Any]]:
        """Ищет товары одного типа калькулятора: запрос + фильтр по метаданным."""
        return self.filter(db_categories(category))

    def available(self) -> bool:
        """Проверяет, что коллекция инициализирована и не пуста."""
        try:
            return self.collection.count() > 0
        except NotFoundError:
            return False
