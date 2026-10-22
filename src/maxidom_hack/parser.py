"""Преобразование сырых записей каталога в товары для калькулятора."""

import re
from typing import Any

from .models import CatalogProduct


def _number(value: Any) -> float | None:
    """Извлекает первое десятичное число из значения с русскими единицами."""
    if value is None:
        return None
    match = re.search(r"\d+(?:[.,]\d+)?", str(value).replace(" ", ""))
    return float(match.group().replace(",", ".")) if match else None


def _characteristics(raw: dict[str, Any]) -> dict[str, str]:
    """Приводит список или словарь характеристик к единому виду с именами в нижнем регистре."""
    values = raw.get("characteristics", [])
    if isinstance(values, dict):
        return {str(key).lower(): str(value) for key, value in values.items()}
    result: dict[str, str] = {}
    for item in values if isinstance(values, list) else []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("title")
            value = item.get("value")
            if name is not None and value is not None:
                result[str(name).lower()] = str(value)
    return result


def _category(raw: dict[str, Any]) -> str:
    """Переводит название категории каталога во внутреннюю категорию калькулятора."""
    categories = raw.get("categories", [])
    values = list(categories.values()) if isinstance(categories, dict) else categories
    text = str(values[0] if values else raw.get("category", "")).lower()
    if any(word in text for word in ("краск", "эмал")):
        return "wall_paint"
    if "обо" in text:
        return "wall_wallpaper"
    if "ламинат" in text or "линолеум" in text:
        return "floor_laminate"
    if "плит" in text:
        return "wall_tile" if "стен" in text else "floor_tile"
    if "клей" in text:
        return "tile_glue"
    if "затир" in text:
        return "tile_grout"
    if "шпакл" in text:
        return "putty"
    if "грунт" in text:
        return "primer"
    if "плинтус" in text:
        return "skirting"
    return str(raw.get("category", ""))


def parse_product(raw_json: dict[str, Any]) -> CatalogProduct:
    """Преобразует одну запись датасета в нормализованную модель товара.

    Объем упаковки, площадь упаковки и расход могут находиться в характеристиках
    или в свободном тексте. Аннотация judge используется как запасной источник
    влагостойкости, поскольку формируется ML-пайплайном.
    """
    characteristics = _characteristics(raw_json)
    name = str(raw_json.get("name", ""))
    description = str(raw_json.get("description", raw_json.get("marketing_description", "")))
    quantity = next((_number(characteristics[key]) for key in ("объем", "литраж банки", "вес") if key in characteristics), None)
    package_area = next((_number(value) for key, value in characteristics.items() if "площад" in key), None)
    area_match = re.search(r"(\d+[.,]\d*)\s*м2", name, re.I)
    # В некоторых записях площадь упаковки указана только в названии товара.
    package_area = package_area or (_number(area_match.group(1)) if area_match else None)
    consumption = next((_number(value) for key, value in characteristics.items() if "расход" in key), None)
    if consumption is None:
        # В описаниях используются разные обозначения: г, кг и л на квадратный метр.
        match = re.search(r"(\d+[.,]?\d*)\s*(?:гр|г|кг|л)\s*/\s*м[²2]", description, re.I)
        consumption = _number(match.group(1)) if match else None
    judge = raw_json.get("judge", {})
    moisture = any("влаг" in key and str(value).lower() in ("да", "true", "1") for key, value in characteristics.items())
    # Сохраняем совместимость с аннотированными записями, загруженными DBLoader.
    moisture = moisture or (isinstance(judge, dict) and bool(judge.get("moisture", False)))
    washable = any("моющ" in key and str(value).lower() in ("да", "true", "1") for key, value in characteristics.items())
    return CatalogProduct(
        id=int(raw_json.get("id", 0)), name=name,
        price=float(raw_json.get("price", raw_json.get("result_price", 0)) or 0),
        unit=str(raw_json.get("unit", "")), package_area=package_area,
        quantity_in_package=quantity, consumption_per_m2=consumption,
        moisture_resistant=moisture, washable=washable, category=_category(raw_json),
    )