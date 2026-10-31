"""FastAPI-сервер: калькулятор, семантический поиск и матчинг товаров."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from maxidom_hack.calculator import Calculator
from maxidom_hack.db.retriever import ProductRetriever
from maxidom_hack.matching.matcher import Matcher
from maxidom_hack.utils.clients import EmbeddingClient, JudgeClient, LLMClient
from maxidom_hack.utils.llm_filter import select_best_product
from maxidom_hack.utils.parser import ProductParser

# ---------- пути ----------

DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent.parent / "src" / "maxidom_hack" / "db" / "chroma_db"
)

# ---------- pydantic-модели ----------


class RoomRequest(BaseModel):
    length: float
    width: float
    height: float = 2.7
    purpose: str = "living"
    humidity: str = "medium"
    doors: int = 1
    windows: int = 1


class EstimateRequest(BaseModel):
    room: RoomRequest
    variant: str = "capital"
    products: list[dict[str, Any]] = Field(default_factory=list)


class ProductRequest(BaseModel):
    query: str = ""
    category_keys: list[str]
    limit: int = 20


class MatchRequest(BaseModel):
    items: list[dict[str, Any]]
    threshold: float = 0.5
    limit: int = 10


class WishRequest(BaseModel):
    """Запрос на подбор товаров: wishes + room + variant."""

    room: RoomRequest
    variant: str = "capital"
    wishes: dict[str, str]  # category_key → wish text
    use_llm: bool = False


class EstimateResponse(BaseModel):
    room: dict[str, Any]
    works: list[dict[str, Any]]
    total: float
    problems: list[str] = Field(default_factory=list)
    llm_reasons: dict[str, str] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    room: dict[str, Any]
    works: list[dict[str, Any]]
    total: float
    picked: list[dict[str, Any]] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    llm_reasons: dict[str, str] = Field(default_factory=dict)


# ---------- инициализация сервисов ----------

calculator = Calculator()
parser = ProductParser()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка ML-моделей при старте."""
    # Embedding-клиент и retriever (ChromaDB)
    embed_client = EmbeddingClient()
    db_path = os.environ.get("CHROMA_DB_PATH", DEFAULT_DB_PATH)
    retriever = ProductRetriever(embed_client, db_path=db_path)
    app.state.retriever = retriever
    app.state.llm_client: LLMClient | None = None
    app.state.matcher: Matcher | None = None
    yield
    # Очистка ресурсов при необходимости


app = FastAPI(
    title="Maxidom API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- endpoints ----------


@app.post("/api/v1/estimate", response_model=EstimateResponse)
def estimate(request: EstimateRequest):
    """Расчёт полной сметы: подбор товаров + расчёт количества + итог."""
    from maxidom_hack.utils.models import Room, CatalogProduct

    retriever = app.state.retriever
    if not retriever.available():
        raise HTTPException(503, "ChromaDB не инициализирована")

    room = Room(
        length=request.room.length,
        width=request.room.width,
        height=request.room.height,
        purpose=request.room.purpose,
        humidity=request.room.humidity,
        doors=request.room.doors,
        windows=request.room.windows,
    )

    products = [CatalogProduct(**p) for p in request.products]
    estimate = calculator.calc_estimate(room, request.variant, products)
    result = estimate.to_dict()

    return EstimateResponse(
        room=result["room"],
        works=result["works"],
        total=result["total"],
    )


@app.post("/api/v1/search")
def search_products(request: ProductRequest):
    """Поиск товаров: семантический поиск + фильтр по категориям."""
    retriever = app.state.retriever
    if not retriever.available():
        raise HTTPException(503, "ChromaDB не инициализирована")

    if request.query.strip():
        result = retriever.search_raw(request.query, request.category_keys, limit=request.limit)
    else:
        result = retriever.filter(request.category_keys, limit=request.limit)

    return result


@app.post("/api/v1/match")
def match_products(request: MatchRequest):
    """Матчинг товаров: определение поверхности и влагостойкости."""
    if app.state.matcher is None:
        app.state.matcher = Matcher(JudgeClient())

    results: list[dict[str, Any]] = []
    for item in request.items[: request.limit]:
        try:
            parsed = app.state.matcher.match_sync(item, request.threshold)
            results.append(
                {
                    "name": item.get("name", ""),
                    "surface": parsed["surface"],
                    "surface_source": parsed.get("surface_source"),
                    "moisture_resistant": parsed["moisture_resistant"],
                    "moisture_source": parsed.get("moisture_source"),
                    "status": "ok",
                    "raw": parsed.get("raw_model_response"),
                    "state": (
                        f"Регулярки: surface={parsed.get('surface_source')}, "
                        f"moisture={parsed.get('moisture_source')}"
                    ),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": item.get("name", ""),
                    "surface": None,
                    "surface_source": None,
                    "moisture_resistant": None,
                    "moisture_source": None,
                    "status": f"error: {exc}",
                    "raw": None,
                    "state": "",
                }
            )
    return results


@app.post("/api/v1/select-llm")
def select_best_llm(request: dict[str, Any]):
    """Выбор лучшего товара через LLM из списка кандидатов."""
    if app.state.llm_client is None:
        app.state.llm_client = LLMClient()

    candidates = request.get("candidates", [])
    llm_query = request.get("query", "Выбери оптимальный вариант из списка")

    best = select_best_product(app.state.llm_client, llm_query, candidates)
    return best


@app.get("/api/v1/health")
def health():
    """Проверка доступности сервисов."""
    retriever = app.state.retriever
    return {
        "status": "ok",
        "products_count": retriever.size if retriever else 0,
    }


@app.post("/api/v1/pipeline")
def pipeline(request: WishRequest):
    """Полный пайплайн: поиск товаров → парсинг → опц. LLM → расчёт сметы."""
    from maxidom_hack.utils.categories import label
    from maxidom_hack.utils.models import CatalogProduct, Room

    retriever = app.state.retriever
    if not retriever.available():
        raise HTTPException(503, "ChromaDB не инициализирована")

    room = Room(
        length=request.room.length,
        width=request.room.width,
        height=request.room.height,
        purpose=request.room.purpose,
        humidity=request.room.humidity,
        doors=request.room.doors,
        windows=request.room.windows,
    )

    picked: list[tuple[str, CatalogProduct]] = []
    problems: list[str] = []
    llm_reasons: dict[str, str] = {}

    for category, wish in request.wishes.items():
        if wish:
            found = retriever.search_raw(wish, [category], limit=5)
        else:
            found = retriever.filter([category], limit=5)

        raw_metadatas = found.get("metadatas") or [[]]
        metadatas = (
            raw_metadatas[0]
            if raw_metadatas and isinstance(raw_metadatas[0], list)
            else raw_metadatas
        )
        raw_documents = found.get("documents") or [[]]
        documents = (
            raw_documents[0]
            if raw_documents and isinstance(raw_documents[0], list)
            else raw_documents
        )

        if not metadatas or not documents:
            problems.append(f"{label(category)}: ничего не найдено")
            continue

        candidates = []
        for meta, doc in zip(metadatas, documents):
            if not meta.get("price") or not meta.get("unit"):
                continue
            candidates.append(
                {
                    "id": meta.get("product_id"),
                    "name": doc.split("\n")[0],
                    "description": doc.split("\n")[1] if "\n" in doc else "",
                    "price": meta.get("price"),
                    "unit": meta.get("unit"),
                    "meta": meta,
                    "doc": doc,
                }
            )

        if not candidates:
            problems.append(f"{label(category)}: нет валидных карточек")
            continue

        if request.use_llm:
            if app.state.llm_client is None:
                app.state.llm_client = LLMClient()
            llm_query = wish if wish else "Выбери оптимальный вариант из списка для ремонта"
            best = select_best_product(app.state.llm_client, llm_query, candidates)
            product = parser.parse_product(best["meta"], best["doc"])
            llm_reasons[label(category)] = best.get("llm_reason", "")
        else:
            product = parser.parse_product(candidates[0]["meta"], candidates[0]["doc"])

        picked.append((category, product))

    estimate = calculator.calc_estimate(
        room, request.variant, [p for _, p in picked]
    ).to_dict()

    picked_dicts = []
    for category, product in picked:
        picked_dicts.append(
            {
                "category": category,
                "id": product.id,
                "name": product.name,
                "calc_category": product.category,
                "package_area": product.package_area,
                "quantity_in_package": product.quantity_in_package,
                "consumption_per_m2": product.consumption_per_m2,
                "price": product.price,
                "unit": product.unit,
                "moisture_resistant": product.moisture_resistant,
            }
        )

    return PipelineResponse(
        room=estimate["room"],
        works=estimate["works"],
        total=estimate["total"],
        picked=picked_dicts,
        problems=problems,
        llm_reasons=llm_reasons,
    )
