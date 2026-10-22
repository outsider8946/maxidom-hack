class EmbeddingPrompts:
    QUERY = 'task: search result | query: '
    DOCUMENT = 'title: none | text: '
    CLUSTERING = 'task: clustering | query: '
    CLASSIFICATION = 'task: classification | query: '
    INSTRUCTION_RETRIEVAL = 'task: code retrieval | query: '
    STS = 'task: sentence similarity | query: '

    _MAP = {
        'query': QUERY,
        'document': DOCUMENT,
        'clustering': CLUSTERING,
        'classification': CLASSIFICATION,
        'instruction_retrieval': INSTRUCTION_RETRIEVAL,
        'sts': STS,

    }

    @classmethod
    def get(cls, prompt: str) -> str:
        return cls._MAP.get(prompt, cls.STS)
