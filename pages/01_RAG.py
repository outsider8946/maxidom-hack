"""Streamlit UI: поэтапный мастер подбора материалов и расчёт сметы."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from maxidom_hack.calculator import Calculator
from maxidom_hack.db.retriever import ProductRetriever
from maxidom_hack.utils.categories import CATEGORIES, label, db_categories
from maxidom_hack.utils.clients import EmbeddingClient
from maxidom_hack.utils.devices import detect_device
from maxidom_hack.utils.models import CatalogProduct, Room
from maxidom_hack.utils.parser import ProductParser

from maxidom_hack.utils.llm_filter import select_best_product
from maxidom_hack.utils.clients import LLMClient

DEVICE = detect_device()

PURPOSES = {
    "living": "Жилая комната",
    "kitchen": "Кухня",
    "bathroom": "Санузел",
    "hallway": "Коридор",
}
HUMIDITY = {"low": "Низкая", "medium": "Средняя", "high": "Высокая"}
VARIANTS = {"capital": "Капитальный", "cosmetic": "Косметический"}

# Назначения, где стены и пол отделываются плиткой вместо обоев и ламината.
WET_PURPOSES = ("bathroom", "kitchen")
SKIP_LABEL = "Пропустить этап"

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


# ---------- этапы мастера ----------


@dataclass(frozen=True)
class Option:
    """Один вариант этапа: что выбирает пользователь и какие категории это добавляет."""

    value: str
    title: str
    adds: tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    """Этап мастера — набор взаимоисключающих вариантов одной зоны."""

    key: str
    title: str
    question: str
    options: tuple[Option, ...]
    wish_hint: str = "Например: моющиеся, светлые, под покраску"
    hint: str = ""

    def option(self, value: str | None) -> Option | None:
        for option in self.options:
            if option.value == value:
                return option
        return None


STAGE_VARIANT = Stage(
    key="variant",
    title="Тип ремонта",
    question="Какой ремонт считаем?",
    options=(
        Option("capital", "Капитальный", ()),
        Option("cosmetic", "Косметический", ()),
    ),
    wish_hint="Например: меняем всё, включая проводку",
    hint="Набор следующих шагов зависит от этого выбора. Пропуск = капитальный.",
)

@st.cache_resource(show_spinner="Инициализируем LLM...")
def init_llm() -> LLMClient:
    return LLMClient()

def option(
    value: str, title: str, adds: tuple[str, ...]
) -> Option:
    return Option(value, title, adds)


def stage_plan(variant: str, purpose: str, chosen: set[str]) -> list[Stage]:
    """Этапы, доступные при выбранном типе ремонта и назначении комнаты.

    Взаимоисключающие материалы собраны в один этап: стены можно либо оклеить,
    либо покрасить, поэтому в косметическом ремонте жилой комнаты они даны
    двумя вариантами одного шага.
    """
    # Плитка в сценариях калькулятора считается только в капитальном ремонте,
    # поэтому в косметическом санузел остаётся с обоями/краской и ламинатом.
    wet = purpose in WET_PURPOSES and variant == "capital"

    if wet:
        walls = Stage(
            key="walls",
            title="Стены",
            question="Чем отделываем стены?",
            options=(option("wall_tile", "Плитка на стены", ("wall_tile",)),),
            wish_hint="Например: светлая керамика 20x30, для ванной",
            hint="В санузле и на кухне стены отделываются плиткой.",
        )
    elif variant == "cosmetic":
        walls = Stage(
            key="walls",
            title="Стены",
            question="Чем отделываем стены? Варианты исключают друг друга.",
            options=(
                option("wall_wallpaper", "Обои", ("wall_wallpaper",)),
                option("wall_paint", "Краска для стен", ("wall_paint",)),
            ),
            wish_hint="Например: флизелиновые под покраску, тёплый серый",
            hint="Краска и обои считаются по одной и той же площади стен — "
            "выбрать можно только одно.",
        )
    else:
        walls = Stage(
            key="walls",
            title="Стены",
            question="Чем отделываем стены?",
            options=(option("wall_paint", "Краска для стен", ("wall_paint",)),),
            wish_hint="Например: матовая водно-дисперсионная, белый",
        )

    floor = Stage(
        key="floor",
        title="Пол",
        question="Какое покрытие на полу?",
        options=(
            option("floor_tile", "Плитка на пол", ("floor_tile",))
            if wet
            else option("floor_laminate", "Ламинат", ("floor_laminate",)),
        ),
        wish_hint="Например: ламинат 33 класса, дуб натуральный",
    )

    stages = [STAGE_VARIANT, walls, floor]

    if not wet and variant == "capital":
        stages.append(
            Stage(
                key="ceiling",
                title="Потолок",
                question="Что делаем с потолком?",
                options=(
                    option("ceiling_paint", "Покраска потолка", ("wall_paint",)),
                ),
                wish_hint="Например: матовая, без бликов",
                hint="Для потолка считается та же краска, что и для стен.",
            )
        )

    # Клей и затирка — не самостоятельные работы, а спутники плитки.
    if {"wall_tile", "floor_tile"} & chosen:
        stages.append(
            Stage(
                key="chemistry",
                title="Клей и затирка",
                question="Что добавить к плитке?",
                options=(
                    option("both", "Клей и затирка", ("tile_glue", "tile_grout")),
                    option("glue", "Только клей", ("tile_glue",)),
                    option("grout", "Только затирку", ("tile_grout",)),
                ),
                wish_hint="Например: клей для керамогранита, затирка влагостойкая",
            )
        )

    stages.append(
        Stage(
            key="skirting",
            title="Плинтус",
            question="Нужен ли плинтус?",
            options=(option("skirting", "Плинтус", ("skirting",)),),
            wish_hint="Например: ПВХ под покраску, белый",
        )
    )

    if variant == "capital":
        stages.append(
            Stage(
                key="electrics",
                title="Электрика",
                question="Считаем ли электрику?",
                options=(
                    option(
                        "electrics",
                        "Розетки, выключатели и кабель",
                        ("socket", "switch", "cable"),
                    ),
                ),
                wish_hint="Например: розетки со шторками, кабель ВВГнг 3x2.5",
                hint="Количество считается эвристикой от площади и числа дверей.",
            )
        )

    return stages


# ---------- инициализация классов один раз на процесс ----------


@st.cache_resource(show_spinner="Инициализируем RAG: эмбеддинги и ChromaDB...")
def init_retriever() -> ProductRetriever | None:
    """Поднимает эмбеддинг-модель и коллекцию ChromaDB."""
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
    return Calculator()


@st.cache_resource(show_spinner="Инициализируем парсер...")
def init_parser() -> ProductParser:
    return ProductParser()


# ---------- виджеты ----------


def room_form() -> Room:
    """Параметры помещения — вход для калькулятора."""
    purpose = st.selectbox(
        "Назначение", list(PURPOSES), format_func=PURPOSES.__getitem__
    )
    humidity = st.selectbox(
        "Влажность", list(HUMIDITY), format_func=HUMIDITY.__getitem__
    )
    length = st.number_input("Длина, м", 1.0, 30.0, 4.0, 0.1)
    width = st.number_input("Ширина, м", 1.0, 30.0, 3.0, 0.1)
    height = st.number_input("Высота, м", 2.0, 6.0, 2.7, 0.1)
    doors = st.number_input("Двери, шт.", 0, 10, 1)
    windows = st.number_input("Окна, шт.", 0, 10, 1)

    return Room(
        length=length,
        width=width,
        height=height,
        purpose=purpose,
        humidity=humidity,
        doors=int(doors),
        windows=int(windows),
    )


ALL_OPTION_STAGES: tuple[Stage, ...] = tuple(
    stage
    for variant in VARIANTS
    for purpose in PURPOSES
    for stage in stage_plan(variant, purpose, {"wall_tile", "floor_tile"})
)

ANSWERS_KEY = "wizard_answers"
WISHES_KEY = "wizard_wishes"


def reset_wizard() -> None:
    """Очищает состояние мастера."""
    st.session_state[ANSWERS_KEY] = {}
    st.session_state[WISHES_KEY] = {}
    st.session_state["wizard_step"] = 0
    for key in list(st.session_state):
        if str(key).startswith(("stage_", "wish_")):
            st.session_state.pop(key, None)


def context_suffix() -> str:
    """Контекст мастера: комната и тип ремонта меняют набор вариантов этапов.

    Ответы хранятся по контексту, поэтому выбор другой комнаты или другого
    типа ремонта не подмешивается к текущему расчёту.
    """
    return (
        f"{st.session_state.get('room_purpose', '')}"
        f"_{st.session_state.get('room_humidity', '')}"
        f"_{st.session_state.get('picked_variant', 'capital')}"
    )


def stage_widget_key(stage_key: str) -> str:
    """Ключ виджета выбора этапа.

    Первый этап статичен, поэтому его ключ не зависит от контекста: иначе
    выбор типа ремонта терялся бы сразу после его же применения.
    """
    if stage_key == "variant":
        return "stage_variant"
    return f"stage_{context_suffix()}_{stage_key}"


def wish_widget_key(stage_key: str) -> str:
    """Ключ поля пожеланий этапа."""
    return f"wish_{context_suffix()}_{stage_key}"


def read_answers(stages: list[Stage]) -> dict[str, str | None]:
    """Выбор пользователя по этапам: ключ этапа -> value варианта.

    Streamlit убирает из session_state значения виджетов, которых не было в
    текущем прогоне, поэтому пройденные шаги читаются из сохранённых ответов,
    а текущий шаг — из виджета.
    """
    saved = st.session_state.get(ANSWERS_KEY) or {}
    answers: dict[str, str | None] = dict(saved.get(context_suffix(), {}))
    for stage in stages:
        title = st.session_state.get(stage_widget_key(stage.key))
        if title is None:
            answers.setdefault(stage.key, None)
            continue
        answers[stage.key] = next(
            (o.value for o in stage.options if o.title == title), None
        )
    return answers


def read_wishes(stages: list[Stage]) -> dict[str, str]:
    """Текстовые пожелания по этапам, включая сохранённые с пройденных шагов."""
    saved = st.session_state.get(WISHES_KEY) or {}
    wishes: dict[str, str] = dict(saved.get(context_suffix(), {}))
    for stage in stages:
        value = st.session_state.get(wish_widget_key(stage.key))
        if value is not None:
            wishes[stage.key] = str(value)
    return wishes


def save_progress(
    stages: list[Stage], answers: dict[str, str | None], wishes: dict[str, str]
) -> None:
    """Сохраняет ответы и пожелания текущего контекста перед переходом."""
    (st.session_state.setdefault(ANSWERS_KEY, {}))[context_suffix()] = dict(answers)
    (st.session_state.setdefault(WISHES_KEY, {}))[context_suffix()] = dict(wishes)
    if answers.get("variant"):
        st.session_state["picked_variant"] = answers["variant"]


def chosen_categories(answers: dict[str, str | None]) -> set[str]:
    """Категории калькулятора, которые добавил выбор пользователя."""
    out: set[str] = set()
    for stage in ALL_OPTION_STAGES:
        picked = stage.option(answers.get(stage.key))
        if picked:
            out.update(picked.adds)
    return out


def collect_wishes(
    stages: list[Stage], answers: dict[str, str | None], wishes: dict[str, str]
) -> dict[str, str]:
    """Итоговый словарь «категория калькулятора → текстовые пожелания»."""
    result: dict[str, str] = {}
    for stage in stages:
        picked = stage.option(answers.get(stage.key))
        if not picked:
            continue
        wish = str(wishes.get(stage.key, "")).strip()
        for category in picked.adds:
            # Категория может попасть из двух этапов (краска стен и потолка).
            merged = f"{result.get(category, '')} {wish}".strip()
            result[category] = merged
    return result


# ---------- пайплайн: БД -> парсер -> калькулятор ----------


def pick_products(
    retriever: ProductRetriever,
    parser: ProductParser,
    wishes: dict[str, str],
    limit: int = 20,
    use_llm: bool = False,
    llm: LLMClient | None = None,
) -> tuple[list[tuple[str, CatalogProduct]], list[str], dict[str, str]]: # <-- Добавили dict
    
    picked: list[tuple[str, CatalogProduct]] = []
    problems: list[str] = []
    llm_reasons: dict[str, str] = {} # <-- Сюда будем складывать причины

    for category, wish in wishes.items():
        if wish:
            found = retriever.search_raw(wish, [category], limit=5)
        else:
            found = retriever.filter([category], limit=5)
            
        raw_metadatas = found.get("metadatas") or [[]]
        metadatas = raw_metadatas[0] if raw_metadatas and isinstance(raw_metadatas[0], list) else raw_metadatas
        raw_documents = found.get("documents") or [[]]
        documents = raw_documents[0] if raw_documents and isinstance(raw_documents[0], list) else raw_documents

        if not metadatas or not documents:
            problems.append(f"{label(category)}: ничего не найдено")
            continue

        candidates = []
        for meta, doc in zip(metadatas, documents):
            if not meta.get("price") or not meta.get("unit"):
                continue
            candidates.append({
                "id": meta.get("product_id"),
                "name": doc.split("\n")[0],
                "description": doc.split("\n")[1] if "\n" in doc else "",
                "price": meta.get("price"),
                "unit": meta.get("unit"),
                "meta": meta,
                "doc": doc
            })

        if not candidates:
            problems.append(f"{label(category)}: нет валидных карточек")
            continue

        if use_llm and llm:
            # Если пожеланий нет, просим LLM выбрать просто лучший вариант
            llm_query = wish if wish else "Выбери оптимальный вариант из списка для ремонта"
            
            best = select_best_product(llm, llm_query, candidates)
            product = parser.parse_product(best["meta"], best["doc"])
            llm_reasons[label(category)] = best.get("llm_reason", "")
        else:
            product = parser.parse_product(candidates[0]["meta"], candidates[0]["doc"])

        picked.append((category, product))

    return picked, problems, llm_reasons


def render_wishes(wishes: dict[str, str]) -> None:
    """Показывает итоговый словарь выбора: этап, товарная группа, пожелания."""
    frame = pd.DataFrame(
        [
            {
                "Категория": label(category),
                "Пожелания": wish or "—",
            }
            for category, wish in wishes.items()
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")



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

def render_llm_reasons(llm_reasons: dict[str, str]) -> None:
    """Отображает отдельный бар с обоснованием выбора от LLM."""
    if not llm_reasons:
        return
        
    st.subheader("Обоснование от LLM")
    for category, reason in llm_reasons.items():
        # Если причина пустая, вообще не выводим эту плашку
        if not reason or not str(reason).strip():
            continue 
            
        if "нет подходящего" in reason.lower() or "не подходит" in reason.lower() or "⚠️" in reason:
            st.warning(f"**{category}**: {reason}")
        else:
            st.info(f"**{category}**: {reason}")

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
    missed_products = [(category, product) for category, product in picked if product.id not in counted]
    
    if missed_products:
        missed_text = []
        for category, product in missed_products:
            reason = "не подходит под работы этого сценария"
            # Проверяем, не отсеяла ли его влажность
            if room.humidity == "high" and product.category in ("wall_tile", "wall_paint") and not product.moisture_resistant:
                reason = "отфильтрован: не является влагостойким (для помещений с высокой влажностью)"
            
            missed_text.append(f"{label(category)} ({reason})")
            
        st.info(
            "Не вошло в смету этого варианта ремонта: "
            + ", ".join(missed_text)
        )


# ---------- мастер ----------


def render_stage(
    stage: Stage,
    index: int,
    total: int,
    answer: str | None,
    wish: str,
) -> None:
    """Один шаг мастера: взаимоисключающие варианты и поле пожеланий.

    Сохранённые выбор и пожелание передаются начальными значениями: при
    возврате «Назад» виджет создаётся заново и сам по себе ничего не помнит.
    """
    st.progress((index + 1) / (total + 1), text=f"Шаг {index + 1} из {total + 1}")
    st.subheader(stage.title)
    if stage.hint:
        st.caption(stage.hint)

    titles = [o.title for o in stage.options] + [SKIP_LABEL]
    saved = next(
        (o.title for o in stage.options if o.value == answer),
        titles[0],
    )
    st.radio(
        stage.question,
        titles,
        index=titles.index(saved),
        key=stage_widget_key(stage.key),
    )

    st.text_input(
        "Пожелания к материалу",
        value=wish,
        key=wish_widget_key(stage.key),
        placeholder=stage.wish_hint,
        help="Текст уходит в семантический поиск по каталогу.",
    )


# ---------- страница ----------

retriever = init_retriever()
calculator = init_calculator()
parser = init_parser()

with st.sidebar:
    st.header("RAG")
    if retriever:
        st.metric("Товаров в базе", f"{retriever.size:,}")
    if st.button("Пройти заново", width="stretch"):
        reset_wizard()
        st.rerun()

    st.header("Помещение")
    room = room_form()
    st.caption(
        f"Пол: {room.floor_area:.1f} м² · Стены: {room.wall_area:.1f} м² · "
        f"Периметр: {room.perimeter:.1f} м"
    )

    st.header("Настройки")
    use_llm = st.toggle("Использовать LLM для выбора", value=False, help="Qwen3-4B выберет лучший товар из топ-3 и объяснит выбор.")

st.title("Поэтапный подбор материалов")
st.caption(
    "На каждом шаге выбирается один вариант или шаг пропускается. "
    "Пожелания текстом уточняют подбор товара в каталоге."
)

if not retriever:
    st.stop()

# Ключи комнат входят в ключи виджетов этапов: при их смене набор вариантов
# меняется, и ответы прошлой комнаты не должны остаться в новых виджетах.
st.session_state["room_purpose"] = room.purpose
st.session_state["room_humidity"] = room.humidity
st.session_state.setdefault("picked_variant", "capital")
st.session_state.setdefault("wizard_step", 0)

# Первый шаг статичен, поэтому тип ремонта читается до сборки остальных этапов:
# именно он решает, какие шаги идут следом. После первого шага виджет пропадает,
# и тип ремонта держится в picked_variant.
variant_title = st.session_state.get("stage_variant")
if variant_title is not None:
    variant = next(
        (o.value for o in STAGE_VARIANT.options if o.title == variant_title),
        st.session_state.picked_variant,
    )
    st.session_state.picked_variant = variant
variant = str(st.session_state.picked_variant)

# Этап «Клей и затирка» появляется только когда выбрана плитка, поэтому список
# этапов собирается в два прохода: сначала без него, затем по выбору стен и пола.
base_plan = stage_plan(variant, room.purpose, set())
stages = stage_plan(variant, room.purpose, chosen_categories(read_answers(base_plan)))
answers = read_answers(stages)
wishes = read_wishes(stages)

step = min(int(st.session_state.get("wizard_step", 0)), len(stages))

if step < len(stages):
    current = stages[step]
    render_stage(
        current,
        step,
        len(stages),
        answers.get(current.key),
        wishes.get(current.key, ""),
    )

    left, right = st.columns(2)
    with left:
        if st.button("Назад", disabled=step == 0, width="stretch"):
            save_progress(stages, answers, wishes)
            st.session_state.wizard_step = step - 1
            st.rerun()
    with right:
        if st.button("Далее", type="primary", width="stretch"):
            save_progress(stages, answers, wishes)
            st.session_state.wizard_step = step + 1
            st.rerun()
else:
    if st.button("Начать заново"):
        reset_wizard()
        st.rerun()

    st.subheader("Итог выбора")
    selection = collect_wishes(stages, answers, wishes)
    if not selection:
        st.info("Все этапы пропущены — считать нечего.")
        st.stop()

    render_wishes(selection)

    with st.status("Подбираю товары и считаю смету...", expanded=False):
        picked, problems, llm_reasons = pick_products(
            retriever, 
            parser, 
            selection, 
            use_llm=use_llm, 
            llm=init_llm() if use_llm else None
        )
        estimate = calculator.calc_estimate(
            room, variant, [product for _, product in picked]
        ).to_dict()

    for problem in problems:
        st.warning(problem)

    # === БЕЗОПАСНЫЙ ВЫВОД ОБОСНОВАНИЙ ===
    # Если LLM выключена, llm_reasons будет пустым, и функция просто завершится
    render_llm_reasons(llm_reasons)
    # ===================================

    st.subheader("Товары из базы")
    render_parsed(picked)

    st.subheader("Смета")
    render_estimate(estimate, picked)
