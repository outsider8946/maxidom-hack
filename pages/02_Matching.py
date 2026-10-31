"""Страница матчинга товаров: классы поверхности и влагостойкости через FRIDA-Decisions.

Поддерживает загрузку JSON (отдельный файл или вставка) и ручной ввод.
Результаты выводятся отдельно по каждому товару + общая таблица.
"""

import json
from typing import Any

import pandas as pd
import streamlit as st

from maxidom_hack.matching.matcher import Matcher
from maxidom_hack.utils.clients import JudgeClient
from maxidom_hack.utils.devices import detect_device

DEVICE = detect_device()

SAMPLE_JSON = {
    "name": "Плитка настенная керамическая ALMA CERAMICA",
    "marketing_description": "Влагостойкая плитка для ванной комнаты, 20х60 см, глянцевая поверхность",
    "characteristics": [
        {"name": "Тип", "value": "настенная"},
        {"name": "Размер", "value": "20x60x0.9"},
        {"name": "Влагостойкость", "value": "да"},
        {"name": "Площадь упаковки", "value": "1.68 м2"},
        {"name": "Маркетинговое описание товара", "value": "Описание..."},
    ],
}


def _characteristics_from_text(text: str) -> list[dict[str, str]]:
    """Разбирает характеристики вида «имя: значение», по одной на строку."""
    items: list[dict[str, str]] = []
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if not sep or not name.strip():
            continue
        items.append({"name": name.strip(), "value": value.strip()})
    return items


def _parse_json_input(
    uploaded: Any, pasted: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Пытается извлечь список товаров из файла или текста.

    Accepts: single dict, list of dicts, or a JSON string.
    Returns (items, error_message).
    """
    if uploaded is not None:
        try:
            data = json.load(uploaded)
        except json.JSONDecodeError as exc:
            return [], f"Не удалось распарсить файл: {exc}"

        if isinstance(data, dict):
            return [data], None
        if isinstance(data, list):
            return data, None
        return [], "JSON содержит объект, ожидаем список товаров или одиночную карточку"

    if not pasted.strip():
        return [], None

    try:
        data = json.loads(pasted)
    except json.JSONDecodeError as exc:
        return [], f"Не удалось распарсить вставку: {exc}"

    if isinstance(data, dict):
        return [data], None
    if isinstance(data, list):
        return data, None
    return [], "JSON содержит объект, ожидаем список товаров или одиночную карточку"


# ---------- judge и matcher: ленивая инициализация ----------


@st.cache_resource(show_spinner="Загружаем judge-модель FRIDA-Decisions...")
def _init_matcher() -> Matcher:
    """Возвращает Matcher, инициализированный JudgeClient для текущего устройства."""
    return Matcher(JudgeClient(device=DEVICE))


def _run_match(
    matcher: Matcher, items: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Прогоняет список товаров через matcher (регулярки + judge)."""
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            # Используем наш новый синхронный метод!
            parsed = matcher.match_sync(item, threshold)
            results.append(
                {
                    "name": item.get("name", ""),
                    "surface": parsed["surface"],
                    "surface_source": parsed.get("surface_source"),
                    "moisture_resistant": parsed["moisture_resistant"],
                    "moisture_source": parsed.get("moisture_source"),
                    "status": "ok",
                    "raw": parsed.get("raw_model_response"),
                    "state": f"Регулярки: surface={parsed.get('surface_source')}, moisture={parsed.get('moisture_source')}"
                }
            )
        except Exception as exc:
            results.append({
                "name": item.get("name", ""),
                "surface": None, "surface_source": None,
                "moisture_resistant": None, "moisture_source": None,
                "status": f"error: {exc}", "raw": None, "state": ""
            })
    return results


# ---------- рендер результатов ----------


def _render_single_class(result: dict[str, Any]) -> None:
    """Выводит классы отдельно и показывает, что сработало: регулярка или модель."""
    left, right = st.columns(2)

    surface_src = result.get("surface_source")
    surface_badge = f" ({surface_src})" if surface_src else ""
    left.metric("🏗 Поверхность", f"{result['surface'] or '—'}{surface_badge}")

    moisture_src = result.get("moisture_source")
    moisture_badge = f" ({moisture_src})" if moisture_src else ""
    
    if result["moisture_resistant"] is not None:
        label = "Влагостойкий" if result["moisture_resistant"] else "Обычный"
        right.metric("💧 Влагостойкость", f"{label}{moisture_badge}")
    else:
        right.metric("💧 Влагостойкость", "—")

    with st.expander("📋 Детали (Регулярки / Модель)", expanded=False):
        st.code(result.get("state", ""))
        if result.get("raw"):
            st.json(result["raw"])


def _render_results(results: list[dict[str, Any]]) -> None:
    """Отображает таблицу результатов для пакета и отдельно по каждому товару."""
    if not results:
        st.info("Нет результатов для отображения.")
        return

    table = pd.DataFrame(
        [
            {
                "Товар": r["name"],
                "Поверхность": r["surface"] or "—",
                "Влагостойкий": r["moisture_resistant"] if r["status"] == "ok" else "—",
                "noul": r.get("noul_score"),
                "Статус": r["status"],
            }
            for r in results
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    # Отдельно по каждому
    st.subheader("Детали по товарам")
    for i, result in enumerate(results):
        with st.container():
            st.markdown(f"**{i + 1}. {result['name']}**")
            _render_single_class(result)
            st.divider()


# ---------- страница ----------


st.title("Матчинг товаров")
st.caption(
    "Judge-модель FRIDA-Decisions определяет класс поверхности и "
    "влагостойкость. Модель загружается по кнопке «Определить классы»."
)

with st.sidebar:
    st.header("Параметры")
    threshold = st.slider("Порог влагостойкости", 0.0, 1.0, 0.5, 0.05)
    limit = st.slider("Товаров за прогон", 1, 100, 10)

tab_json, tab_manual = st.tabs(["JSON (файл / вставка)", "Вручную"])

# ---------- вкладка: JSON ----------

with tab_json:
    uploaded = st.file_uploader(
        "Загрузить JSON", type=["json"], accept_multiple_files=False
    )
    st.caption("Примечание: JSON может содержать одиночный объект или список объектов.")
    pasted = st.text_area(
        "Или вставьте JSON",
        placeholder=json.dumps(SAMPLE_JSON, ensure_ascii=False, indent=2),
        height=220,
    )

    items, json_error = _parse_json_input(uploaded, pasted)

    if json_error:
        st.error(json_error)
        items = []
    elif items:
        st.success(f"Загружено {len(items)} товаров")

    # Превью товаров
    if items:
        preview = pd.DataFrame(
            [
                {
                    "id": item.get("id", ""),
                    "Название": str(item.get("name", ""))[:60],
                    "Категория": str(item.get("category", item.get("categories", "")))[
                        :40
                    ],
                    "Характеристик": len(item.get("characteristics", [])),
                }
                for item in items[:limit]
            ]
        )
        st.dataframe(preview, hide_index=True, width="stretch")
        if len(items) > limit:
            st.caption(f"Показаны первые {limit} из {len(items)}")

    if st.button(
        "Определить классы", key="match_json", type="primary", disabled=not items
    ):
        try:
            matcher = _init_matcher()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось загрузить модель: {exc}")
            st.stop()

        with st.status("Запускаю judge-модель...", expanded=False):
            results = _run_match(matcher, items[:limit], threshold)

        _render_results(results)

    # Пример JSON
    with st.expander("Пример формата JSON"):
        st.code(json.dumps(SAMPLE_JSON, ensure_ascii=False, indent=2), language="json")

# ---------- вкладка: вручную ----------

with tab_manual:
    st.subheader("Введите данные товара")
    name = st.text_input("Название", placeholder="Плитка настенная керамическая")
    description = st.text_area(
        "Описание (marketing_description)",
        placeholder="Влагостойкая плитка для ванной комнаты, 20х60 см, глянцевая",
        height=80,
    )
    char_text = st.text_area(
        "Характеристики (каждая строка: «имя: значение»)",
        placeholder=(
            "Тип: настенная\n"
            "Размер: 20x60x0.9\n"
            "Влагостойкость: да\n"
            "Площадь упаковки: 1.68 м2"
        ),
        height=160,
    )

    # Превью разбора характеристик
    if char_text.strip():
        chars = _characteristics_from_text(char_text)
        if chars:
            st.caption(f"Разобрано: {len(chars)} характеристик")
            st.dataframe(
                pd.DataFrame(chars, columns=["name", "value"]),
                hide_index=True,
                width="stretch",
            )

    item: dict[str, Any] = {
        "name": name,
        "marketing_description": description,
        "characteristics": _characteristics_from_text(char_text),
    }

    if st.button(
        "Определить классы", key="match_manual", type="primary", disabled=not name
    ):
        try:
            matcher = _init_matcher()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось загрузить модель: {exc}")
            st.stop()

        with st.status("Запускаю judge-модель...", expanded=False):
            results = _run_match(matcher, [item], threshold)

        _render_results(results)
