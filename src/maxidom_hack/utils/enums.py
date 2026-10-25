"""Эмбеддинг-подсказки для модели multilingual-e5."""

from typing import ClassVar


class EmbeddingPrompts:
    QUERY = "query: "
    DOCUMENT = "passage: "
    VANILLA = ""

    _MAP: ClassVar[dict[str, str]] = {
        "query": QUERY,
        "document": DOCUMENT,
        "vanilla": VANILLA,
    }

    @classmethod
    def get(cls, prompt: str) -> str:
        return cls._MAP.get(prompt, cls.VANILLA)
