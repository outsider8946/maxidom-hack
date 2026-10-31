"""Общие утилиты для Streamlit-страниц: вызовы API backend-сервера."""

import os
from typing import Any

import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


# ==================== RAG-страница ====================


def get_estimate_pipeline(
    room: dict,
    variant: str,
    wishes: dict[str, str],
    use_llm: bool = False,
) -> dict[str, Any]:
    """Запрос полного пайплайна: поиск + парсинг + LLM + расчёт сметы."""
    resp = requests.post(
        f"{API_BASE}/api/v1/pipeline",
        json={
            "room": room,
            "variant": variant,
            "wishes": wishes,
            "use_llm": use_llm,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def get_estimate(room: dict, variant: str, products: list[dict]) -> dict[str, Any]:
    """Запрос расчёта сметы на backend (по уже подобранным товарам)."""
    resp = requests.post(
        f"{API_BASE}/api/v1/estimate",
        json={"room": room, "variant": variant, "products": products},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def search_products(query: str, category_keys: list[str], limit: int = 20) -> dict:
    """Поиск товаров через backend."""
    resp = requests.post(
        f"{API_BASE}/api/v1/search",
        json={"query": query, "category_keys": category_keys, "limit": limit},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_health() -> dict:
    """Проверка доступности backend."""
    try:
        resp = requests.get(f"{API_BASE}/api/v1/health", timeout=5)
        return resp.json()
    except requests.RequestException:
        return {"status": "error"}


# ==================== Matching-страница ====================


def match_products(items: list[dict], threshold: float = 0.5, limit: int = 10) -> list[dict]:
    """Запуск матчинга товаров через backend."""
    resp = requests.post(
        f"{API_BASE}/api/v1/match",
        json={"items": items, "threshold": threshold, "limit": limit},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


# ==================== LLM ====================


def select_best_llm(candidates: list[dict], query: str) -> dict:
    """Выбор лучшего товара через LLM на backend."""
    resp = requests.post(
        f"{API_BASE}/api/v1/select-llm",
        json={"candidates": candidates, "query": query},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
