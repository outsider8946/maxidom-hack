from typing import Any
from maxidom_hack.utils.clients import JudgeClient
from maxidom_hack.db.patterns import PatternRules, CategoryRules # <-- ИМПОРТИРУЕМ РЕГУЛЯРКИ

class Matcher:
    """Матчит товары: сначала регулярки, потом Judge."""

    def __init__(self, judge_client: JudgeClient):
        self.judge_client = judge_client
        self.pattern_rules = PatternRules()
        self.cat_rules = CategoryRules()

    def match_sync(self, item: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
        """Синхронный матчинг: регулярки + модель (для Streamlit)."""
        
        # 1. Прогоняем регулярки (как в annotater.py)
        text = PatternRules.build_text(item)
        surface = PatternRules.detect_surface(text, item.get("categories"))
        moisture = PatternRules.detect_moisture(text)
        
        surface_source = "pattern" if surface is not None else None
        moisture_source = "pattern" if moisture is not None else None
        
        # 2. Если регулярки всё нашли - модель не нужна!
        if surface is not None and moisture is not None:
            return {
                "surface": surface,
                "moisture_resistant": moisture,
                "surface_source": surface_source,
                "moisture_source": moisture_source,
                "raw_model_response": None
            }

        # 3. Если чего-то не хватает - формируем запрос к модели (только по пропущенным полям)
        # Используем метод из annotater (или пишем локально)
        request = self._format_request(item, ask_surface=(surface is None), ask_moisture=(moisture is None))
        
        # Отправляем в модель (JudgeClient)
        try:
            raw = self.judge_client.judge(request)
            answers = raw.get("answers", {})
            
            if surface is None and "surface" in answers:
                surface = answers["surface"]["choice"]
                surface_source = "model"
                
            if moisture is None and "moisture_resistant" in answers:
                moisture = answers["moisture_resistant"]["noul"] >= threshold
                moisture_source = "model"
        except Exception as e:
            print(f"Judge error: {e}")
            raw = None

        return {
            "surface": surface,
            "moisture_resistant": moisture,
            "surface_source": surface_source,
            "moisture_source": moisture_source,
            "raw_model_response": raw
        }

    def _format_request(self, item: dict[str, Any], ask_surface: bool, ask_moisture: bool) -> dict[str, Any]:
        """Формирует запрос для Judge только по нужным вопросам."""
        name = item.get("name", "")
        description = item.get("marketing_description", "")
        features_list = []
        for feature in item.get("characteristics", []):
            if "описание" in str(feature.get("name", "")).lower():
                continue
            features_list.append(f'{feature.get("name")}: {feature.get("value", "")}')
        features = "\n".join(features_list)
        
        state = f"Название: {name}\nОписание: {description}\nХарактеристики:\n{features}"
        
        questions = {}
        if ask_surface:
            questions["surface"] = {
                "type": "choice",
                "instructions": "К какой поверхности относится товар?",
                "criteria": {
                    "напольное": "материалы для пола",
                    "стеновое": "материалы для стен",
                    "потолочное": "материалы для потолка",
                    "электрика": "кабель, розетки, выключатели",
                    "другое": "клей, инструмент, сантехника",
                },
            }
        if ask_moisture:
            questions["moisture_resistant"] = {
                "type": "noul",
                "instructions": "Предназначен ли товар для помещений с высокой влажностью?",
                "criteria": {
                    "true": "влагостойкая плитка, краска, гидроизоляция",
                    "false": "обычные материалы",
                },
            }
            
        return {"state": state, "questions": questions}