import re
from maxidom_hack.utils.clients import LLMClient

def select_best_product(llm: LLMClient, query: str, candidates: list[dict]) -> dict:
    if not candidates:
        return None
    if len(candidates) == 1:
        # Даже если товар один, вернем его с заглушкой для обоснования
        res = candidates[0].copy()
        res["llm_reason"] = "Единственный подходящий товар в базе."
        return res

    context = ""
    for i, c in enumerate(candidates):
        context += f"[{i}] {c['name']}\n"
        context += f"    Цена: {c['price']} руб.\n"
        context += f"    Описание: {c['description'][:200]}...\n\n"

    system_prompt = (
        "Ты — эксперт по ремонту и строительным материалам. "
        "Тебе дан запрос пользователя и список товаров. "
        "Выбери ОДИН самый подходящий товар. "
        "Ответь строго в формате (БЕЗ КВАДРАТНЫХ СКОБОК):\n"
        "ВЫБОР: 3\n"
        "ПРИЧИНА: [краткое объяснение]"
    )
    
    user_prompt = f"Запрос пользователя: {query}\n\nДоступные товары:\n{context}"

    try:
        response = llm.generate(system_prompt, user_prompt, think=False)
        
        # 1. Извлекаем причину (берем всё после слова ПРИЧИНА, очищаем от пробелов)
        reason_match = re.search(r"ПРИЧИНА:\s*(.*)", response, re.DOTALL)
        reason_text = reason_match.group(1).strip() if reason_match else "Выбор LLM"
        
        # Если LLM написала бред и причина пустая, ставим заглушку
        if not reason_text:
            reason_text = "LLM выбрала этот товар как наиболее подходящий."

        # 2. Ищем номер (с учетом возможных скобок)
        match = re.search(r"ВЫБОР:\s*\[?(\d+)\]?", response)
        
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(candidates):
                chosen = candidates[idx].copy()
                chosen["llm_reason"] = reason_text
                return chosen
                
        # 3. Обработка случая "нет подходящего"
        if "нет подходящего" in response.lower() or "не существует" in response.lower():
            fallback = candidates[0].copy()
            fallback["llm_reason"] = f"⚠️ {reason_text}"
            return fallback
                
    except Exception as e:
        print(f"[ERROR] Ошибка LLM: {e}")
    
    # 4. Fallback
    fallback = candidates[0].copy()
    fallback["llm_reason"] = "Выбор по умолчанию (топ-1 RAG), так как LLM не справилась."
    return fallback