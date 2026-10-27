from typing import Any

from maxidom_hack.utils.clients import JudgeClient


class Matcher:
    """Class to match product by classes items using the JudgeClient."""

    def __init__(self, judge_client: JudgeClient):
        self.judge_client = judge_client

    def parse_result(
        self, result: dict[str, Any], threshold: float = 0.5
    ) -> dict[str, Any]:
        """Parse the result from the judge model."""
        surface = result["answers"]["surface"]["choice"]
        moisture = result["answers"]["moisture_resistant"]["noul"] >= threshold
        return {
            "surface": surface,
            "moisture_resistant": moisture,
        }

    def format_reqeust(self, item: dict[str, Any]) -> dict[str, Any]:
        """Format the request for the judge model."""
        name = item.get("name", "")
        description = item.get("marketing_description", "")
        features_list = []
        for feature in item.get("characteristics", []):
            feature_name = str(feature.get("name", ""))
            feature_value = feature.get("value", "")
            if "описание" in feature_name:
                continue
            features_list.append(f"{feature_name}: {feature_value}")
        features = "\n".join(features_list)
        state = (
            f"Название товара: {name}\n"
            f"Описание товара: {description}\n"
            f"Характеристики товара:\n{features}"
        )
        return {
            "state": state,
            "questions": {
                "surface": {
                    "type": "choice",
                    "instructions": "К какой поверхности относится товар?",
                    "criteria": {
                        "напольное": "материалы для пола: ламинат, линолеум, плитка для пола, стяжка, подложка",
                        "стеновое": "материалы для стен: обои, штукатурка, краска, ГКЛ, стеновая плитка",
                        "потолочное": "материалы для потолка: натяжные, потолочная плитка, потолочный плинтус",
                        "электрика": "кабель, розетки, выключатели, автоматы, щиты, светильники",
                        "другое": "клей, крепёж, инструмент, сантехника, товары не для отделки",
                    },
                },
                "moisture_resistant": {
                    "type": "noul",
                    "instructions": "Предназначен ли товар для помещений с высокой влажностью?",
                    "criteria": {
                        "true": "влагостойкий ГКЛВ, влагостойкая краска, гидроизоляция, плитка для санузла, затирка на цементной основе",
                        "false": "обычные материалы без специальных влагостойких свойств",
                    },
                },
            },
        }

    async def match(self, item: dict[str, Any]) -> dict[str, Any]:
        """Match a product item using the JudgeClient."""
        request = self.format_reqeust(item)
        clf_result = await self.judge_client.judge(request)
        return self.parse_result(clf_result)
