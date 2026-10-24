class EmbeddingPrompts:
    QUERY = "query: "
    DOCUMENT = "passage: "
    VANILLA = ""

    _MAP = {
        "query": QUERY,
        "document": DOCUMENT,
        "vanilla": VANILLA,
    }

    @classmethod
    def get(cls, prompt: str) -> str:
        return cls._MAP.get(prompt, cls.VANILLA)
