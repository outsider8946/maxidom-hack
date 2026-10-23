"""Точка входа FastAPI для endpoint'ов калькулятора и ML-сервисов."""

import json
import os

from pathlib import Path
from time import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from maxidom_hack.calculator import Calculator
from maxidom_hack.models import Room
from maxidom_hack.parser import parse_product
from maxidom_hack.llm_client import LLMClient
from maxidom_hack.db.loader import DBLoader

init_t = time()
embed_client = EmbeddingClient()
print(f'LLMClient initialized in {time() - init_t:.2f} seconds')

db_t = time()
db_loader = DBLoader(llm_client=llm_client)
print(f'DBLoader initialized in {time() - db_t:.2f} seconds')

# db_loader.delete_and_load('/Users/dolor.src/Downloads/rel_data_annotated.json')
# ИСПРАВЛЕНО: Берем путь из env, либо используем дефолтный, чтобы не падало на других ОС
DB_DATA_PATH = os.environ.get("DB_DATA_PATH", "data/rel_data_annotated.json") 
try:
    db_loader.delete_and_load(DB_DATA_PATH)
except FileNotFoundError:
    print(f"Внимание: Файл {DB_DATA_PATH} не найден. Пропускаем загрузку в ChromaDB (калькулятор продолжит работать).")

    
app = FastAPI(title="Maxidom calculator")
calculator = Calculator()

# data_path = Path(__file__).resolve().parents[2] / "data" / "products_final.json"
# ИСПРАВЛЕНО: Ищем products_final.json в папке data рядом с проектом
data_path = Path(__file__).resolve().parent / "data" / "products_final.json"


class EstimateRequest(BaseModel):
	"""Тело запроса для полной сметы помещения."""

	room: Room
	variant: str = "capital"


class ProductRequest(BaseModel):
	"""Тело запроса для расчета одного товара из каталога."""

	product_id: int
	area: float
	waste: float = 0.05
	layers: int | None = None


def _products():
	"""Загружает и нормализует каталог проекта для запросов калькулятора."""
	with data_path.open(encoding="utf-8") as file:
		return [parse_product(item) for item in json.load(file)]


@app.post("/api/v1/calc/estimate")
def estimate(request: EstimateRequest):
	"""Возвращает полную смету для указанного помещения."""
	result = calculator.calc_estimate(request.room, request.variant, _products())
	return result.to_dict()


@app.post("/api/v1/calc/product")
def product(request: ProductRequest):
	"""Возвращает расчет упаковок для одного товара из каталога."""
	selected = next((item for item in _products() if item.id == request.product_id), None)
	if selected is None:
		raise HTTPException(status_code=404, detail="Product not found")
	return calculator.calc_by_product(selected, request.area, request.waste, request.layers).__dict__