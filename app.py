"""Streamlit UI: выбор типов материалов и подбор товаров из ChromaDB."""

import sys
import platform
import pandas as pd
import streamlit as st

from maxidom_hack.categories import CATEGORIES, label
from maxidom_hack.db.retriever import ProductRetriever
from maxidom_hack.utils.clients import EmbeddingClient


def detect_device() -> str:
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


# ---------- инициализация RAG один раз на процесс ----------

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
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if is_active:
                    selected.remove(category)
                else:
                    selected.append(category)
                st.rerun()
    return list(selected)

def render_selection(selected: list[str]) -> None:
    """Показывает, что именно выбрал пользователь."""
    if not selected:
        st.info("Выбери типы товаров кнопками выше.")
        return

    st.markdown(" ".join(f"`{label(c)}`" for c in selected))
    frame = pd.DataFrame([
        {
            "Тип товара": label(c),
            "Ключ": c,
            "Категорий в БД": len(CATEGORIES[c]["db_categories"]),
            "Поисковый запрос": CATEGORIES[c]["query"],
        }
        for c in selected
    ])
    st.dataframe(frame, hide_index=True, use_container_width=True)


def render_products(retriever: ProductRetriever, selected: list[str], top_k: int) -> None:
    """Тянет товары из БД по каждому выбранному типу."""
    with st.status("Ищем товары в ChromaDB...", expanded=False):
        found = {c: retriever.by_category(c, top_k) for c in selected}

    for category in selected:
        items = found[category]
        with st.expander(f"{label(category)} — {len(items)} товаров", expanded=len(selected) == 1):
            if not items:
                st.warning("Ничего не найдено: проверь `db_categories` в categories.py.")
                continue
            frame = pd.DataFrame([
                {
                    "id": i["id"],
                    "Товар": i["name"],
                    "Бренд": i["brand"],
                    "Цена": i["price"],
                    "Ед.": i["unit"],
                    "Категория в БД": i["category"],
                    "score": i["score"],
                }
                for i in items
            ])
            st.dataframe(frame, hide_index=True, use_container_width=True)


# ---------- страница ----------

st.set_page_config(page_title="Подбор материалов", page_icon="🧱", layout="wide")

retriever = init_retriever()

with st.sidebar:
    st.header("RAG")
    if retriever:
        st.metric("Товаров в базе", f"{retriever.size:,}")
        top_k = st.slider("Товаров на тип", 3, 50, 10)
        if st.button("Перезагрузить RAG"):
            st.cache_resource.clear()
            st.rerun()
    else:
        top_k = 10
        st.warning("RAG недоступен.")

st.title("Подбор материалов")
st.caption("Выбери типы товаров из калькулятора — подборка придет из ChromaDB.")

selected = category_buttons()

st.subheader("Выбрано")
render_selection(selected)

if retriever and selected:
    st.subheader("Товары из базы")
    render_products(retriever, selected, top_k)