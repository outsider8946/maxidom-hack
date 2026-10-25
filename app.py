"""Streamlit UI: выбор типов материалов, подбор из ChromaDB и расчёт сметы."""

import platform
import sys
from typing import Literal

import pandas as pd
import streamlit as st

from maxidom_hack.calculator import Calculator
from maxidom_hack.db.retriever import ProductRetriever
from maxidom_hack.utils.categories import CATEGORIES, label
from maxidom_hack.utils.clients import EmbeddingClient
from maxidom_hack.utils.models import CatalogProduct, Room
from maxidom_hack.utils.parser import ProductParser


def detect_device() -> Literal["cpu", "cuda", "mps"]:
    # Apple Silicon
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    # NVIDIA
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


DEVICE = detect_device()
COLUMNS = 4

PURPOSES = {
    "living": "Жилая комната",
    "kitchen": "Кухня",
    "bathroom": "Санузел",
    "hallway": "Коридор",
}
HUMIDITY = {"low": "Низкая", "medium": "Средняя", "high": "Высокая"}
VARIANTS = {"capital": "Капитальный", "cosmetic": "Косметический"}
WORK_TITLES = {
    "floor_tile": "Плитка на пол",
    "wall_tile": "Плитка на стены",
    "floor_laminate": "Ламинат",
    "wall_paint": "Покраска стен",
    "ceiling_paint": "Покраска потолка",
    "wall_wallpaper": "Обои",
    "skirting": "Плинтус",
    "electrics": "Электрика",
}


# ---------- инициализация классов один раз на процесс ----------


@st.cache_resource(show_spinner="Инициализируем RAG: эмбеддинги и ChromaDB...")
def init_retriever() -> ProductRetriever | None:
    """Поднимает эмбеддинг-модель и коллекцию ChromaDB.

    Вызывается на самом верху страницы, до отрисовки интерфейса.
    """
    try:
        retriever = ProductRetriever(EmbeddingClient(device=DEVICE))
    except Exception as exc:  # noqa: BLE001 - UI не должен падать из-за ML-слоя
        st.error(f"RAG не инициализирован: {exc}")
        return None
    if not retriever.available():
        st.error("Коллекция ChromaDB пуста — сначала загрузи товары через DBLoader.")
        return None
    return retriever


@st.cache_resource(show_spinner="Инициализируем Калькулятора...")
def init_calculator() -> Calculator:
    """Инициализирует калькулятор.

    Вызывается на самом верху страницы, до отрисовки интерфейса.
    """
    return Calculator()


@st.cache_resource(show_spinner="Инициализируем парсер...")
def init_parser() -> ProductParser:
    """Инициализирует парсер.

    Вызывается на самом верху страницы, до отрисовки интерфейса.
    """
    return ProductParser()


# ---------- виджеты ----------


def category_buttons() -> list[str]:
    """Toggle-кнопки: клик добавляет/убирает тип из выбора."""
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = []

    selected: list[str] = st.session_state.selected_categories
    cols = st.columns(COLUMNS)
    for index, category in enumerate(CATEGORIES):
        with cols[index % COLUMNS]:
            is_active = category in selected
            if st.button(
                f"{'✓ ' if is_active else ''}{label(category)}",
                key=f"cat_{category}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                if is_active:
                    selected.remove(category)
                else:
                    selected.append(category)
                st.rerun()
    return list(selected)


def room_form() -> tuple[Room, str]:
    """Параметры помещения и вариант ремонта — вход для калькулятора."""
    purpose = st.selectbox("Назначение", list(PURPOSES), format_func=PURPOSES.__getitem__)
    humidity = st.selectbox("Влажность", list(HUMIDITY), format_func=HUMIDITY.__getitem__)
    length = st.number_input("Длина, м", 1.0, 30.0, 4.0, 0.1)
    width = st.number_input("Ширина, м", 1.0, 30.0, 3.0, 0.1)
    height = st.number_input("Высота, м", 2.0, 6.0, 2.7, 0.1)
    doors = st.number_input("Двери, шт.", 0, 10, 1)
    windows = st.number_input("Окна, шт.", 0, 10, 1)
    variant = st.selectbox("Тип ремонта", list(VARIANTS), format_func=VARIANTS.__getitem__)

    room = Room(
        length=length,
        width=width,
        height=height,
        purpose=purpose,
        humidity=humidity,
        doors=int(doors),
        windows=int(windows),
    )
    return room, variant


# ---------- пайплайн: БД -> парсер -> калькулятор ----------


def pick_products(
    retriever: ProductRetriever,
    parser: ProductParser,
    selected: list[str],
    limit: int = 50,
) -> tuple[list[tuple[str, CatalogProduct]], list[str]]:
    """Берёт по одному товару на выбранный тип и нормализует их парсером.

    Берётся первая запись с заполненными ценой и единицей измерения — без них
    калькулятор не может посчитать ни упаковки, ни стоимость.

    Returns:
        ``([(тип, товар)], [проблемы])``.
    """
    picked: list[tuple[str, CatalogProduct]] = []
    problems: list[str] = []

    for category in selected:
        found = retriever.filter([category], limit=limit)
        metadatas = found.get("metadatas") or []
        documents = found.get("documents") or []

        product = None
        for meta, doc in zip(metadatas, documents):
            if not meta.get("price") or not meta.get("unit"):
                continue
            try:
                product = parser.parse_product(meta, doc)
            except Exception as exc:  # noqa: BLE001 - битая карточка не роняет смету
                problems.append(
                    f"{label(category)}: парсер не разобрал карточку — "
                    f"{type(exc).__name__}: {exc}"
                )
                break
            break

        if product is not None:
            picked.append((category, product))
        else:
            problems.append(f"{label(category)}: нет расчитанной карточки")

    return picked, problems


def render_parsed(picked: list[tuple[str, CatalogProduct]]) -> None:
    """Показывает, что парсер извлёк из карточек — по этим полям идёт расчёт."""
    frame = pd.DataFrame(
        [
            {
                "Тип": label(category),
                "Товар": product.name[:55],
                "Категория расчёта": product.category,
                "Площадь упак., м²": product.package_area,
                "Объём упак.": product.quantity_in_package,
                "Расход": product.consumption_per_m2,
                "Цена": product.price,
                "Ед.": product.unit,
            }
            for category, product in picked
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")

    # Калькулятор считает только по известным внутренним категориям.
    unknown = [
        product.name[:40] for _, product in picked if product.category not in CATEGORIES
    ]
    if unknown:
        st.warning(
            "Категория не распознана калькулятором, товар в смету не попадёт: "
            + ", ".join(unknown)
        )


def render_estimate(estimate: dict, picked: list[tuple[str, CatalogProduct]]) -> None:
    """Выводит смету: итог, разбивку по работам и непопавшие в неё типы."""
    works = [work for work in estimate["works"] if work["items"]]
    st.metric("Итого по смете", f"{estimate['total']:,.0f} ₽")

    if not works:
        st.warning(
            "Смета пустая: выбранные товары не подходят под работы этого варианта "
            "ремонта либо калькулятор не распознал их категорию."
        )
        return

    for work in works:
        title = WORK_TITLES.get(work["work"], work["work"])
        with st.expander(
            f"{title} — {work['total']:,.0f} ₽ ({work['area']} м²)", expanded=True
        ):
            frame = pd.DataFrame(work["items"])
            st.dataframe(frame, hide_index=True, width="stretch")

    # Товар мог быть подобран и распознан, но не войти в набор работ сценария.
    counted = {item["product_id"] for work in works for item in work["items"]}
    missed = [
        label(category)
        for category, product in picked
        if product.id not in counted
    ]
    if missed:
        st.info(
            "Не вошло в смету этого варианта ремонта: " + ", ".join(missed)
            + ". Эти работы калькулятор считает только в другом сценарии "
            "или не имеет для них формул."
        )


# ---------- страница ----------

st.set_page_config(page_title="Подбор материалов", page_icon="🧱", layout="wide")

retriever = init_retriever()
calculator = init_calculator()
parser = init_parser()

with st.sidebar:
    st.header("RAG")
    if retriever:
        st.metric("Товаров в базе", f"{retriever.size:,}")
        if st.button("Перезагрузить RAG"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("RAG недоступен.")

    st.header("Помещение")
    room, variant = room_form()
    st.caption(
        f"Пол: {room.floor_area:.1f} м² · Стены: {room.wall_area:.1f} м² · "
        f"Периметр: {room.perimeter:.1f} м"
    )

st.title("Подбор материалов")
st.caption("Выбери типы товаров — по одному товару каждого типа попадёт в смету.")

selected = category_buttons()

st.subheader("Выбрано")
if not selected:
    st.info("Выбери типы товаров кнопками выше.")
elif retriever:
    with st.status("Собираю по товару на тип и считаю смету...", expanded=False):
        picked, problems = pick_products(retriever, parser, selected)
        estimate = calculator.calc_estimate(
            room, variant, [product for _, product in picked]
        ).to_dict()

    for problem in problems:
        st.warning(problem)

    st.subheader("Товары из базы")
    render_parsed(picked)

    st.subheader("Смета")
    render_estimate(estimate, picked)
