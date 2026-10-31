"""Streamlit UI: поэтапный мастер подбора материалов и расчёт сметы (через API)."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

# Добавляем корень проекта в путь для импорта ui/utils
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.utils import get_estimate_pipeline

# ---------- константы ----------

PURPOSES = {
    "living": "Жилая комната",
    "kitchen": "Кухня",
    "bathroom": "Санузел",
    "hallway": "Коридор",
}
HUMIDITY = {"low": "Низкая", "medium": "Средняя", "high": "Высокая"}
VARIANTS = {"capital": "Капитальный", "cosmetic": "Косметический"}

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

LABELS = {
    "floor_tile": "Плитка на пол",
    "wall_tile": "Плитка на стены",
    "floor_laminate": "Ламинат",
    "wall_paint": "Краска для стен",
    "ceiling_paint": "Краска для потолка",
    "wall_wallpaper": "Обои",
    "tile_glue": "Клей для плитки",
    "tile_grout": "Затирка для плитки",
    "skirting": "Плинтус",
    "socket": "Розетки",
    "switch": "Выключатели",
    "cable": "Кабель",
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


def option(
    value: str, title: str, adds: tuple[str, ...]
) -> Option:
    return Option(value, title, adds)


def stage_plan(variant: str, purpose: str, chosen: set[str]) -> list[Stage]:
    """Этапы, доступные при выбранном типе ремонта и назначении комнаты."""
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
        wish_hint="Например: ламинат 33 класс, дуб натуральный",
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


# ---------- состояние ----------

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
    """Контекст мастера: комната и тип ремонта меняют набор вариантов этапов."""
    return (
        f"{st.session_state.get('room_purpose', '')}"
        f"_{st.session_state.get('room_humidity', '')}"
        f"_{st.session_state.get('picked_variant', 'capital')}"
    )


def stage_widget_key(stage_key: str) -> str:
    """Ключ виджета выбора этапа."""
    if stage_key == "variant":
        return "stage_variant"
    return f"stage_{context_suffix()}_{stage_key}"


def wish_widget_key(stage_key: str) -> str:
    """Ключ поля пожеланий этапа."""
    return f"wish_{context_suffix()}_{stage_key}"


def read_answers(stages: list[Stage]) -> dict[str, str | None]:
    """Выбор пользователя по этапам."""
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
    """Текстовые пожелания по этапам."""
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
    """Сохраняет ответы и пожелания текущего контекста."""
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
    """Итоговый словарь «категория → пожелания»."""
    result: dict[str, str] = {}
    for stage in stages:
        picked = stage.option(answers.get(stage.key))
        if not picked:
            continue
        wish = str(wishes.get(stage.key, "")).strip()
        for category in picked.adds:
            merged = f"{result.get(category, '')} {wish}".strip()
            result[category] = merged
    return result


# ---------- рендер ----------


def render_wishes(wishes: dict[str, str]) -> None:
    """Показывает итоговый словарь выбора."""
    frame = pd.DataFrame(
        [
            {
                "Категория": LABELS.get(cat, cat),
                "Пожелания": wish or "—",
            }
            for cat, wish in wishes.items()
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")


def render_llm_reasons(llm_reasons: dict[str, str]) -> None:
    """Отображает обоснование выбора от LLM."""
    if not llm_reasons:
        return
    st.subheader("Обоснование от LLM")
    for category, reason in llm_reasons.items():
        if not reason or not str(reason).strip():
            continue
        if "нет подходящего" in reason.lower() or "не подходит" in reason.lower() or "⚠️" in reason:
            st.warning(f"**{category}**: {reason}")
        else:
            st.info(f"**{category}**: {reason}")


def render_parsed(picked: list[dict]) -> None:
    """Показывает товары, подобранные из базы."""
    frame = pd.DataFrame(
        [
            {
                "Тип": LABELS.get(p.get("category", ""), p.get("category", "")),
                "Товар": p.get("name", "")[:55],
                "Категория расчёта": p.get("calc_category", ""),
                "Площадь упак., м²": p.get("package_area"),
                "Объём упак.": p.get("quantity_in_package"),
                "Расход": p.get("consumption_per_m2"),
                "Цена": p.get("price"),
                "Ед.": p.get("unit", ""),
            }
            for p in picked
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")


def render_estimate(estimate: dict, picked: list[dict]) -> None:
    """Выводит смету."""
    works = [work for work in estimate.get("works", []) if work.get("items")]
    st.metric("Итого по смете", f"{estimate.get('total', 0):,.0f} ₽")

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

    counted = {item["product_id"] for work in works for item in work.get("items", [])}
    missed = [p for p in picked if p.get("id") not in counted]

    if missed:
        missed_text = []
        for p in missed:
            reason = "не подходит под работы этого сценария"
            missed_text.append(f"{LABELS.get(p.get('category', ''), p.get('category', ''))} ({reason})")
        st.info(
            "Не вошло в смету этого варианта ремонта: "
            + ", ".join(missed_text)
        )


def render_stage(
    stage: Stage,
    index: int,
    total: int,
    answer: str | None,
    wish: str,
) -> None:
    """Один шаг мастера."""
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

# Проверка доступности backend
from ui.utils import get_health

health = get_health()
if health.get("status") != "ok":
    st.error(
        "Backend недоступен. Запустите: `uvicorn server.main:app --host 0.0.0.0 --port 8000`"
    )
    st.stop()

with st.sidebar:
    st.header("RAG")
    st.metric("Товаров в базе", f"{health.get('products_count', '?'):,}")
    if st.button("Пройти заново", width="stretch"):
        reset_wizard()
        st.rerun()

    st.header("Помещение")
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

    room = {
        "length": length,
        "width": width,
        "height": height,
        "purpose": purpose,
        "humidity": humidity,
        "doors": int(doors),
        "windows": int(windows),
    }
    floor_area = length * width
    perimeter = 2 * (length + width)
    wall_area_gross = perimeter * height
    door_area = 1.8
    window_area = 1.5
    wall_area = max(wall_area_gross - doors * door_area - windows * window_area, 0.0)
    st.caption(
        f"Пол: {floor_area:.1f} м² · Стены: {wall_area:.1f} м² · "
        f"Периметр: {perimeter:.1f} м"
    )

    st.header("Настройки")
    use_llm = st.toggle(
        "Использовать LLM для выбора",
        value=False,
        help="Qwen3-4B выберет лучший товар из топ-3 и объяснит выбор.",
    )

st.title("Поэтапный подбор материалов")
st.caption(
    "На каждом шаге выбирается один вариант или шаг пропускается. "
    "Пожелания текстом уточняют подбор товара в каталоге."
)

st.session_state["room_purpose"] = purpose
st.session_state["room_humidity"] = humidity
st.session_state.setdefault("picked_variant", "capital")
st.session_state.setdefault("wizard_step", 0)

variant_title = st.session_state.get("stage_variant")
if variant_title is not None:
    variant = next(
        (o.value for o in STAGE_VARIANT.options if o.title == variant_title),
        st.session_state.picked_variant,
    )
    st.session_state.picked_variant = variant
variant = str(st.session_state.picked_variant)

base_plan = stage_plan(variant, purpose, set())
stages = stage_plan(variant, purpose, chosen_categories(read_answers(base_plan)))
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
        result = get_estimate_pipeline(room, variant, selection, use_llm)

    for problem in result.get("problems", []):
        st.warning(problem)

    render_llm_reasons(result.get("llm_reasons", {}))

    st.subheader("Товары из базы")
    render_parsed(result.get("picked", []))

    st.subheader("Смета")
    render_estimate(result, result.get("picked", []))
