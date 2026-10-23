import json
from tqdm import tqdm
from time import time
from maxidom_hack.utils.clients import EmbeddingClient
from typing import Any

from chromadb import PersistentClient
from chromadb.errors import NotFoundError


class DBLoader:
    def __init__(self, embed_client: EmbeddingClient, db_path: str = './db/chroma_db', db_name: str= 'products', metric: str = 'cosine'):
        self.embed_client = embed_client
        self.db_name=db_name
        self.metric=metric
        self.db_client = PersistentClient(path=db_path)
    @staticmethod
    def _get_item_text(item: dict[str, Any]) -> str:
        """Формирует текст товара для эмбеддинга."""
        name = item.get("name", "")
        description = item.get('marketing_description', "")

        return f'{name}\n{description}'

    @staticmethod
    def _get_metadata(item: dict[str, Any]) -> dict[str, Any]:
        """Get metadata from a product item."""
        categories = item.get("categories", {})
        category = ""
        if isinstance(categories, dict):
            category = next(iter(categories.values()), "")

        brand = item.get("brand", {})
        brand_name = brand.get("name", "") if isinstance(brand, dict) else ""

        country = item.get("country", {})
        country_name = country.get("name", "") if isinstance(country, dict) else ""

        judge = item.get("judge", {})
        moisture_resistant = judge.get("moisture", False) if isinstance(judge, dict) else False
        surface = judge.get("surface", "") if isinstance(judge, dict) else "другое"

        return {
            "product_id": item["id"],
            "cod_good": item.get("cod_good", "") or "",
            "price": item.get("price", 0) or 0,
            "category": category,
            "brand": brand_name,
            "country": country_name,
            "moisture_resistant": moisture_resistant,
            "surface": surface,
            "unit": item.get("unit", "") or "",
        }


    def _delete(self):
        """Удаляет все данные из коллекции."""
        try:
            self.db_client.delete_collection(self.db_name)
            print("Все данные удалены из коллекции.")
        except NotFoundError:
            print("Коллекция не найдена.")

    def load(self, data_path: str, batch_size: int = 512):
        collection = self.db_client.get_or_create_collection(
            name=self.db_name,
            metadata={"hnsw:space": self.metric},
        )

        data = json.load(open(data_path))
        print(f"Полуечно товаров: {len(data)}")

        for i in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size + 1, desc="Обработка батчей"):
            batch = data[i : i + batch_size]
            ids = [str(item["id"]) for item in batch]
            embeddings = self.embed_client.embed_documents(
                [self._get_item_text(item) for item in batch],
                progress_bar=False,
                prompt='document',
                batch_size=batch_size,
            )
            
            metadatas = [self._get_metadata(p) for p in batch]
            documents = [self._get_item_text(p) for p in batch]
            add_t = time()
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

        print(f"Готово. В коллекции: {collection.count()} товаров")


    def delete_and_load(self, data_path: str):
        self._delete()
        self.load(data_path)