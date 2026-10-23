"""Точка входа FastAPI для endpoint'ов калькулятора и ML-сервисов."""

import json

from pathlib import Path

from fastapi import FastAPI, HTTPException

from maxidom_hack.utils.models import EstimateRequest, ProductRequest
from maxidom_hack.calculator import Calculator
from maxidom_hack.parser import parse_product

    
app = FastAPI(title="Maxidom calculator")
calculator = Calculator()

data_path = Path(__file__).resolve().parent / "data" / "products_final.json"

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