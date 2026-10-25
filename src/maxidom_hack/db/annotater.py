"""Разметка товаров: паттерны → Judge для незакрытых полей.

Логика:
1. Паттерны пытаются определить surface и moisture независимо.
2. Если поле определено паттерном — модель по нему не спрашиваем.
3. Модель получает только те вопросы, которые паттерны не закрыли:
   - оба поля не определены → спрашиваем оба;
   - только surface → спрашиваем surface;
   - только moisture → спрашиваем moisture.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from maxidom_hack.utils.clients import JudgeClient
from maxidom_hack.db.patterns import PatternRules, CategoryRules

class DBAnnotater:
    def __init__(
        self,
        judge_client: JudgeClient,
        relevant_cateogires: dict[str, str] | None = None,
    ):
        self.judge_client = judge_client
        self.relevant_cateogires = relevant_cateogires
        self.pattern_rules = PatternRules()
        self.cat_rules = CategoryRules()


    def _get_choice_filter_request(self, cat: str, samples: list[dict[str, Any]]):
        samples_descr = []
        for sample in samples:
            descr = sample.get("marketing_description") or sample.get("name")
            if descr:
                samples_descr.append(descr)
        joined = "\n".join(samples_descr)  # вынесено из f-string: на py3.11 backslash в {}
        state = (
            f"Категория каталога: {cat}\n"
            f"Примеры товаров из каталога:\n{joined}"
        )
        return {
            "state": state,
            "questions": {
                "construction": {
                    "type": "choice",
                    "instructions": "Как эта категория относится к ремонту помещений?",
                    "criteria": {
                        "core": "материалы и работы: плитка, краска, обои, ГКЛ, кабель, розетки, автоматы, трубы, смесители, сухие смеси, плинтусы, светильники для монтажа",
                        "adjacent": "сопутствующее ремонту: инструмент, крепёж, расходники, встраиваемая техника, кондиционеры, бойлеры, вентиляция, спецодежда, защита",
                        "excluded": "не относится к ремонту: отдельно стоящая бытовая техника, электроника, растения, косметика, одежда, зоо, продукты, мебель, декор, игрушки",
                    },
                }
            },
        }

    def get_choice_clf_request(
        self,
        item: dict[str, Any],
        *,
        ask_surface: bool = True,
        ask_moisture: bool = True,
    ):
        name = item["name"]
        description = item.get("marketing_description") or ""
        features_list = []
        for feature in item.get("characteristics") or []:
            feature_name = feature.get("name") or ""
            if "описание" in feature_name.lower():
                continue
            features_list.append(f'{feature_name}: {feature.get("value", "")}')
        features = "\n".join(features_list)
        state = (
            f"Название товара: {name}\n"
            f"Описание товара: {description}\n"
            f"Характеристики товара:\n{features}"
        )

        questions: dict[str, Any] = {}
        if ask_surface:
            questions["surface"] = {
                "type": "choice",
                "instructions": "К какой поверхности относится товар?",
                "criteria": {
                    "напольное":  "материалы для пола: ламинат, линолеум, плитка для пола, стяжка, подложка",
                    "стеновое":   "материалы для стен: обои, штукатурка, краска, ГКЛ, стеновая плитка",
                    "потолочное": "материалы для потолка: натяжные, потолочная плитка, потолочный плинтус",
                    "электрика":  "кабель, розетки, выключатели, автоматы, щиты, светильники",
                    "другое":     "клей, крепёж, инструмент, сантехника, товары не для отделки",
                },
            }
        if ask_moisture:
            questions["moisture_resistant"] = {
                "type": "noul",
                "instructions": "Предназначен ли товар для помещений с высокой влажностью?",
                "criteria": {
                    "true":  "влагостойкий ГКЛВ, влагостойкая краска, гидроизоляция, плитка для санузла, затирка на цементной основе",
                    "false": "обычные материалы без специальных влагостойких свойств",
                },
            }
        return {"state": state, "questions": questions}

    # ---------- медоиды для фильтрации категорий ----------

    def _medoids(self, products: list[dict[str, Any]], k: int = 3):
        if not products:
            return []
        texts = [
            f"{p.get('name') or ''} {(p.get('marketing_description') or '')[:100]}"
            for p in products
        ]
        tfidf_matrix = TfidfVectorizer().fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        scores = sim_matrix.sum(axis=1)
        idx = np.argsort(scores)[::-1][:k]
        return [products[i] for i in idx]

    def _get_medoids_from_cat(self, data: list[dict[str, Any]], target_cat: str):
        products = []
        for item in data:
            cats = item.get("categories") or {}
            if not cats:
                continue
            catalog_cat = next(iter(cats.values()))
            if catalog_cat == target_cat:
                products.append(item)
        return self._medoids(products)

    def _get_filter_requests_batch(self, data: list[dict[str, Any]], cats: list[str]):
        medoids_batch = []
        for cat in tqdm(cats, desc="Computing medoids..."):
            medoids_batch.append(self._get_medoids_from_cat(data, cat))

        requests_batch = []
        for cat, medoids in tqdm(
            zip(cats, medoids_batch), desc="Formatting requests..."
        ):
            requests_batch.append(self._get_choice_filter_request(cat, medoids))
        return requests_batch


    def _postprocessing_filter_batch(self, out_batch: list[dict[str, Any]]):
        outs = []
        for out in out_batch:
            answers = out.get("answers", {})
            construction = answers.get("construction")
            if not construction:
                outs.append("excluded")
                continue
            probs = construction["probabilities"]
            label = max(probs, key=probs.get)
            outs.append(label)
        return outs

    def _postprocessing_clf_batch(
        self,
        out_batch: list[dict[str, Any]],
        *,
        ask_surface: bool = True,
        ask_moisture: bool = True,
    ) -> list[dict[str, Any]]:
        outs = []
        for out in out_batch:
            answers = out.get("answers", {})
            rec: dict[str, Any] = {}
            if ask_surface and "surface" in answers:
                rec["surface"] = answers["surface"]["choice"]
            if ask_moisture and "moisture_resistant" in answers:
                rec["moisture"] = answers["moisture_resistant"]["noul"] >= 0.5
            outs.append(rec)
        return outs


    async def _afilter(self, request_batch: list[dict[str, Any]]):
        out_judge = await self.judge_client.judge_batch(request_batch)
        return self._postprocessing_filter_batch(out_judge)

    async def _aclf(
        self,
        request_batch: list[dict[str, Any]],
        *,
        ask_surface: bool = True,
        ask_moisture: bool = True,
    ):
        out_judge = await self.judge_client.judge_batch(request_batch)
        return self._postprocessing_clf_batch(
            out_judge, ask_surface=ask_surface, ask_moisture=ask_moisture
        )


    async def _aclassify(self, relevant_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ann: list[dict[str, Any]] = []
        for item in relevant_data:
            text = PatternRules.build_text(item)
            surface = PatternRules.detect_surface(text, item.get("categories"))
            moisture = PatternRules.detect_moisture(text)
            ann.append({
                "surface": surface,
                "moisture": moisture,
                "surface_source": "pattern" if surface is not None else None,
                "moisture_source": "pattern" if moisture is not None else None,
            })

        both: list[int] = []
        surface_only: list[int] = []
        moisture_only: list[int] = []
        for idx, rec in enumerate(ann):
            need_s = rec["surface"] is None
            need_m = rec["moisture"] is None
            if need_s and need_m:
                both.append(idx)
            elif need_s:
                surface_only.append(idx)
            elif need_m:
                moisture_only.append(idx)

        tasks: list[Any] = []
        plan: list[tuple[str, list[int]]] = []

        if both:
            reqs = [
                self.get_choice_clf_request(relevant_data[i], ask_surface=True, ask_moisture=True)
                for i in both
            ]
            tasks.append(self._aclf(reqs, ask_surface=True, ask_moisture=True))
            plan.append(("both", both))
        if surface_only:
            reqs = [
                self.get_choice_clf_request(relevant_data[i], ask_surface=True, ask_moisture=False)
                for i in surface_only
            ]
            tasks.append(self._aclf(reqs, ask_surface=True, ask_moisture=False))
            plan.append(("surface_only", surface_only))
        if moisture_only:
            reqs = [
                self.get_choice_clf_request(relevant_data[i], ask_surface=False, ask_moisture=True)
                for i in moisture_only
            ]
            tasks.append(self._aclf(reqs, ask_surface=False, ask_moisture=True))
            plan.append(("moisture_only", moisture_only))

        if tasks:
            results = await asyncio.gather(*tasks)
            for (kind, indices), out in zip(plan, results):
                for idx, res in zip(indices, out):
                    if "surface" in res:
                        ann[idx]["surface"] = res["surface"]
                        ann[idx]["surface_source"] = "model"
                    if "moisture" in res:
                        ann[idx]["moisture"] = res["moisture"]
                        ann[idx]["moisture_source"] = "model"

        n_pat_s = sum(1 for r in ann if r["surface_source"] == "pattern")
        n_pat_m = sum(1 for r in ann if r["moisture_source"] == "pattern")
        n_mod_s = sum(1 for r in ann if r["surface_source"] == "model")
        n_mod_m = sum(1 for r in ann if r["moisture_source"] == "model")
        print(
            f"[annotate] surface: pattern={n_pat_s}, model={n_mod_s} | "
            f"moisture: pattern={n_pat_m}, model={n_mod_m} | "
            f"items={len(relevant_data)}"
        )

        for item, rec in zip(relevant_data, ann):
            item["surface"] = rec["surface"]
            item["moisture"] = rec["moisture"]
            item["_annotation"] = {
                "surface_source": rec["surface_source"],
                "moisture_source": rec["moisture_source"],
            }
        return relevant_data


    async def _set_relevant_categories(self, data: list[dict[str, Any]]) -> dict[str, str]:
        unique_cats = np.unique(
            [
                next(iter(item["categories"].values()))
                for item in data
                if item.get("categories")
            ]
        ).tolist()
        # ---------- 1. Паттерны по имени категории ----------
        pattern_result: dict[str, str] = {}
        cats_to_ask: list[Any] = []
        for cat in unique_cats:
            label = self.cat_rules.detect(str(cat))
            if label is not None:
                pattern_result[str(cat)] = label
            else:
                cats_to_ask.append(cat)

        # ---------- 2. LLM только для тех, кого не закрыл паттерн ----------
        llm_result: dict[str, str] = {}
        if cats_to_ask:
            requests_batch = self._get_filter_requests_batch(data, cats_to_ask)
            tasks = [self._afilter(batch) for batch in requests_batch]
            results = await tqdm_asyncio.gather(*tasks, desc="Filtering categories...")
            flatten = [item for sublist in results for item in sublist]
            llm_result = {str(cat): label for cat, label in zip(cats_to_ask, flatten)}

         # ---------- 3. Статистика ----------
        total = len(unique_cats)
        print(
            f"[filter] categories: pattern={len(pattern_result)}, "
            f"model={len(llm_result)}, total={total}"
        )

        return {**pattern_result, **llm_result}


    def annotate(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.relevant_cateogires:
            self.relevant_cateogires = asyncio.run(self._set_relevant_categories(data))

        relevant_data = []
        for item in data:
            cats = item.get("categories") or {}
            if not cats:
                continue
            cat = next(iter(cats.values()))
            if (
                cat not in self.relevant_cateogires
                or self.relevant_cateogires[cat] == "excluded"
            ):
                continue
            if not item.get('unit', '') or not item.get('price', 0):
                continue
            relevant_data.append(item)

        print(f'Skipped: {len(data) - len(relevant_data)}/{len(data)} items')

        return asyncio.run(self._aclassify(relevant_data))