"""Загрузка каталога товаров в ChromaDB."""

import json
from pathlib import Path
from typing import Any

import gdown
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from tqdm import tqdm

from maxidom_hack.utils.clients import EmbeddingClient

# Каталог хранится вне пакета и подтягивается с Google Drive при первом запуске.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "ann_data_v1.json"
DATA_URL = "https://drive.google.com/file/d/1lsjmJu-fAXFqwENtYtfD8a_6EZ_A_s8-/view"


class DBLoader:
    def __init__(
        self,
        embed_client: EmbeddingClient,
        db_path: str = str(Path(__file__).resolve().parent / "chroma_db"),
        db_name: str = "products",
        metric: str = "cosine",
        data_path: str | Path = DEFAULT_DATA_PATH,
    ):
        self.embed_client = embed_client
        self.db_name = db_name
        self.metric = metric
        self.db_client = PersistentClient(path=db_path)
        self.data_path = Path(data_path)
        self.ensure_data(self.data_path)

    @staticmethod
    def ensure_data(data_path: Path | str = DEFAULT_DATA_PATH) -> Path:
        """Скачивает каталог с Google Drive, если файла ещё нет на диске.

        Файл весит ~170 МБ, поэтому качается в временный файл и перемещается
        только при успешной загрузке: оборванная выгрузка не оставит битый JSON.
        """
        path = Path(data_path)
        if path.exists() and path.stat().st_size > 0:
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")

        print(f"Каталог не найден — скачиваю в {path}")
        try:
            gdown.download(DATA_URL, str(tmp_path), quiet=False)
        except Exception as exc:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Не удалось скачать каталог: {exc}") from exc

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Google Drive не отдал файл. Проверьте доступ к ссылке или "
                "положите ann_data_v1.json в ./data/ вручную."
            )

        tmp_path.replace(path)
        print(f"Каталог сохранён: {path} ({path.stat().st_size / 1e6:.1f} МБ)")
        return path

    @staticmethod
    def _get_item_text(item: dict[str, Any]) -> str:
        """Формирует текст товара для эмбеддинга."""
        name = item.get("name", "")
        description = item.get("marketing_description", "")

        return f"{name}\n{description}"

    @staticmethod
    def _get_item_characteristics(item: list[dict[str, str]]):
        characteristics = []
        for characteristic in item:
            characteristics.append(
                f"{characteristic['name']}: {characteristic['value']}"
            )
        return "\n".join(characteristics)

    def _get_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
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
        moisture_resistant = (
            judge.get("moisture", False) if isinstance(judge, dict) else False
        )
        surface = judge.get("surface", "") if isinstance(judge, dict) else "другое"
        characteristics_item = item.get("characteristics", [])
        if characteristics_item:
            characteristics = self._get_item_characteristics(characteristics_item)
        else:
            characteristics = ""

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
            "characteristics": characteristics,
        }

    def _delete(self):
        """Удаляет все данные из коллекции."""
        try:
            self.db_client.delete_collection(self.db_name)
            print("Все данные удалены из коллекции.")
        except NotFoundError:
            print("Коллекция не найдена.")

    def load(self, data_path: str | Path | None = None, batch_size: int = 512):
        path = Path(data_path) if data_path else self.data_path
        collection = self.db_client.get_or_create_collection(
            name=self.db_name,
            metadata={"hnsw:space": self.metric},
        )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Полуечно товаров: {len(data)}")

        for i in tqdm(
            range(0, len(data), batch_size),
            total=len(data) // batch_size + 1,
            desc="Обработка батчей",
        ):
            batch = data[i : i + batch_size]
            ids = [str(item["id"]) for item in batch]
            embeddings = self.embed_client.embed_documents(
                [self._get_item_text(item) for item in batch],
                progress_bar=False,
                prompt="document",
                batch_size=batch_size,
            )

            metadatas = [self._get_metadata(p) for p in batch]
            documents = [self._get_item_text(p) for p in batch]
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,  # type: ignore[arg-type]
                documents=documents,
            )

        print(f"Готово. В коллекции: {collection.count()} товаров")

    def delete_and_load(self, data_path: str | Path | None = None):
        self._delete()
        self.load(data_path)
