from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
import asyncio
from typing import Any
import numpy as np
from maxidom_hack.utils.clients import JudgeClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DBAnnotater:
    def __init__(self, judge_client: JudgeClient, relevant_cateogires: list[str] | None = None):
        self.judge_client = judge_client
        self.relevant_cateogires = relevant_cateogires


    def _get_choice_filter_request(self, cat: str, samples: list[dict[str, Any]]):
        samples_descr = []
        for sample in samples:
            descr = sample.get("marketing_description") or sample.get("name")
            samples_descr.append(descr)
        descr = '\n'.join(samples_descr)
        state = (
            f"Категория каталога: {cat}\n"
            f"Примеры товаров из каталога:\n{descr}"
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

    def get_choice_clf_request(self, item: dict[str, Any]):
        name = item['name']
        description= item['marketing_description']
        features_list = []
        for feature in item['characteristics']:
            feature_name = feature['name']
            feature_value = feature['value']
            if 'описание' in feature_name:
                continue
            features_list.append(f'{feature_name}: {feature_value}')
        features = '\n'.join(features_list)
        state = (
                f"Название товара: {name}\n"
                f"Описание товара: {description}"
                f"Характеристики товара:\n{features}"
            )
        return {
            "state": state,
            "questions": {
                "surface": {
                    "type": "choice",
                    "instructions": "К какой поверхности относится товар?",
                    "criteria": {
                        "напольное":  "материалы для пола: ламинат, линолеум, плитка для пола, стяжка, подложка",
                        "стеновое":   "материалы для стен: обои, штукатурка, краска, ГКЛ, стеновая плитка",
                        "потолочное": "материалы для потолка: натяжные, потолочная плитка, потолочный плинтус",
                        "электрика":  "кабель, розетки, выключатели, автоматы, щиты, светильники",
                        "другое":     "клей, крепёж, инструмент, сантехника, товары не для отделки",
                    },
                },
                "moisture_resistant": {
                    "type": "noul",
                    "instructions": "Предназначен ли товар для помещений с высокой влажностью?",
                    "criteria": {
                        "true":  "влагостойкий ГКЛВ, влагостойкая краска, гидроизоляция, плитка для санузла, затирка на цементной основе",
                        "false": "обычные материалы без специальных влагостойких свойств",
                    },
                },
            }
        }

    def _medoids(self, products: list[dict[str, Any]], k=3):
        texts = [f"{p['name']} {(p.get('description') or '')[:100]}" for p in products]
        tfidf_matrix = TfidfVectorizer().fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        scores = sim_matrix.sum(axis=1)
        idx = np.argsort(scores)[::-1][:k]
        return [products[i] for i in idx]

    def _get_medoids_from_cat(self, data: list[dict[str, Any]], target_cat: str):
        products = []
        for item in data:
            catalog_cat = list(item["categories"].values())[0]
            if catalog_cat == target_cat:
                products.append(item)
        return self._medoids(products)

    def _get_filter_requests_batch(self, data: list[dict[str, Any]], cats: list[str]):
        medoids_batch = []

        for cat in tqdm(cats, desc="Computing medioids..."):
            medoids_batch.append(self._get_medoids_from_cat(data, cat))

        requests_batch = []
        for cat, medoids in tqdm(zip(cats, medoids_batch), desc="Formatting requests..."):
            requests_batch.append(self._get_choice_filter_request(cat, medoids))

        return requests_batch

    def _postprocessing_filter_batch(self, out_batch: list[dict[str, Any]]):
        outs = []
        for out in out_batch:
            probs = out["answers"]["construction"]["probabilities"]
            label = max(probs, key=probs.get)
            outs.append(label)

        return outs

    def _postprocessing_clf_batch(self, out_batch: list[dict[str, Any]]) -> list[dict[str, str]]:
        outs = []
        for out in out_batch:
            surface = out['answers']['surface']['choice']
            moisture = out['answers']['moisture_resistant']['noul'] >= 0.5
            outs.append({'surface': surface, 'moisture': moisture})
        return outs

    async def _afilter(self, request_batch: list[dict[str, Any]]):
        out_judge = await self.judge_client.judge_batch(request_batch)
        return self._postprocessing_filter_batch(out_judge)

    async def _aclf(self, request_batch: list[dict[str, Any]]):
        out_judge = await self.judge_client.judge_batch(request_batch)
        return self._postprocessing_clf_batch(out_judge)

    
    async def _aclassify(self, relevant_data: list[dict[str, Any]]):
        pass


    
    async def _set_relevant_categories(self, data: list[dict[str, Any]]) -> dict[str, str]:
        unique_cats = np.unique([list(item["categories"].values())[0] for item in data]).tolist()
        requests_batch = self._get_filter_requests_batch(data, unique_cats)
        tasks = [self._afilter(batch) for batch in requests_batch]
        results =  await tqdm_asyncio.gather(*tasks, desc="Filtering...")
        flatten_results = [item for sublist in results for item in sublist]
        return {str(cat): label for cat, label in zip(unique_cats, flatten_results)}
        

    def annotate(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.relevant_cateogires:
            self.relevant_cateogires = asyncio.run(self._set_relevant_categories(data))

        relevant_data = []
        for item in data:
            cat = list(item['categories'].values())[0]
            if cat not in self.relevant_cateogires or self.relevant_cateogires[cat] == 'excluded':
                continue   
            relevant_data.append(item)

        clf_outs = asyncio.run(self._aclassify(relevant_data))
        

        

        