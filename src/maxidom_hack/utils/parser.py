"""Преобразование сырых записей каталога в товары для калькулятора."""

import re
from typing import Any

from .models import CatalogProduct

# Порядок важен: узкие признаки проверяются раньше широких. Иначе
# «Затирки для плитки» уйдёт в плитку, а «Обои под покраску» — в краску.
CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("затир",), "tile_grout"),
    (("клей", "клеи"), "tile_glue"),
    (("шпатл", "шпакл"), "putty"),
    (("грунт",), "primer"),
    (("плинтус",), "skirting"),
    (("обои",), "wall_wallpaper"),
    (("краск", "эмал"), "wall_paint"),
    (("ламинат", "линолеум", "spc", "кварцвинил"), "floor_laminate"),
    (("розетк",), "socket"),
    (("выключател",), "switch"),
    (("кабель", "провод"), "cable"),
)

# Плитка различается по назначению, поэтому проверяется отдельно.
TILE_WORDS = ("плит", "керамогранит", "кафель", "мозаик")
WALL_TILE_WORDS = ("настен", "ванной", "кухн", "фасадн", "клинкер", "мозаик")


class ProductParser:
    number_re = re.compile(r"\d+(?:[.,]\d+)?")

    def _number(self, value: Any) -> float | None:
        """Извлекает первое десятичное число из значения с русскими единицами."""
        if value is None:
            return None
        match = self.number_re.search(str(value).replace(" ", ""))
        return float(match.group().replace(",", ".")) if match else None

    def _characteristics(self, raw_str: str) -> dict[str, str]:
        """Разбирает блок характеристик «имя: значение» в словарь нижним регистром.

        Строки без разделителя и маркетинговый текст отбрасываются: первый ломает
        разбор, второй содержит слово «расход» в свободном тексте.
        """
        result: dict[str, str] = {}
        for line in str(raw_str or "").split("\n"):
            name, separator, value = line.partition(":")
            if not separator:
                continue
            name = name.strip().lower()
            if not name or "маркетинг" in name:
                continue
            result[name] = value.strip()
        return result

    def _category(self, raw: str) -> str:
        """Переводит название категории каталога во внутреннюю категорию калькулятора."""
        text = str(raw).lower()
        for words, category in CATEGORY_RULES:
            if any(word in text for word in words):
                return category
        if any(word in text for word in TILE_WORDS):
            if any(word in text for word in WALL_TILE_WORDS):
                return "wall_tile"
            return "floor_tile"
        return str(raw)

    def parse_product(self, meta: dict[str, Any], doc: str) -> CatalogProduct:
        """Преобразует одну запись датасета в нормализованную модель товара.

        Объем упаковки, площадь упаковки и расход могут находиться в характеристиках
        или в свободном тексте. Аннотация judge используется как запасной источник
        влагостойкости, поскольку формируется ML-пайплайном.
        """
        characteristics = self._characteristics(meta.get("characteristics", ""))
        doc_split = doc.split("\n")
        name = doc_split[0].strip()
        description = doc_split[1].strip() if len(doc_split) > 1 else ""
        quantity = next(
            (
                self._number(characteristics[key])
                for key in ("объем", "литраж банки", "вес", "метраж упаковки") # добавил метраж, чтобы корректно учитывать бухты кабеля
                if key in characteristics
            ),
            None,
        )
        package_area = next(
            (
                self._number(value)
                for key, value in characteristics.items()
                if "площад" in key
            ),
            None,
        )
        area_match = re.search(r"(\d+[.,]\d*)\s*м2", name, re.IGNORECASE)
        # В некоторых записях площадь упаковки указана только в названии товара.
        package_area = package_area or (
            self._number(area_match.group(1)) if area_match else None
        )
        consumption = next(
            (
                self._number(value)
                for key, value in characteristics.items()
                if "расход" in key
            ),
            None,
        )
        if consumption is None:
            # В описаниях используются разные обозначения: г, кг и л на квадратный метр.
            match = re.search(
                r"(\d+[.,]?\d*)\s*(?:гр|г|кг|л)\s*/\s*м[²2]", description, re.IGNORECASE
            )
            consumption = self._number(match.group(1)) if match else None

        moisture = bool(meta.get("moisture_resistant", False))

        # Ищем "моющаяся" в характеристиках, названии или описании
        name_and_desc = f"{name} {description} {meta.get('marketing_description', '')}".lower()
        # Если вдруг ML не разметил моющуюся, но в тексте есть "ванная" или "санузел" - это влагостойкий материал
        if "ванн" in name_and_desc or "сануз" in name_and_desc:
            moisture = True
        washable = any(
            "моющ" in key and str(value).lower() in ("да", "true", "1")
            for key, value in characteristics.items()
        ) or ("моющ" in name_and_desc)
        
        return CatalogProduct(
            id=int(meta.get("product_id", 0)),
            name=name,
            price=float(meta.get("price", 0.0)),
            unit=str(meta.get("unit", "")),
            package_area=package_area,
            quantity_in_package=quantity,
            consumption_per_m2=consumption,
            moisture_resistant=moisture,
            washable=washable,
            category=self._category(meta["category"]),
        )
