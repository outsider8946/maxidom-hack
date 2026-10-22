import os
import torch
from ollama import Client
from typing import Literal
from sentence_transformers import SentenceTransformer
from frida_decisions import Judge
from maxidom_hack.enums import EmbeddingPrompts
import numpy as np
from dotenv import load_dotenv

load_dotenv()
class LLMClient:
    def __init__(self, device: Literal['cpu', 'gpu', 'mps'] = 'mps') -> None:
        self.device = device
        self.client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        self.embed_model = self._load_embedding_model()
        self.llm_model = os.environ.get("LLM", "qwen3.5:4b")
        self.judge_model = self._load_judge_model()

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model using SentenceTransformer."""
        model_name = os.environ.get("EMBED", "intfloat/multilingual-e5-small")
        model = SentenceTransformer(
            model_name, 
            device=self.device, 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
        model.max_seq_length = 256
        print("model name: ", model_name)
        print("device:", model.device)
        print("dtype:", model[0].auto_model.dtype)
        print("max_seq:", model.max_seq_length)
        print("Реальное устройство модели:", next(model.parameters()).device)
        print("Устройство первого слоя:", model[0].auto_model.device)
        return model

    def _load_judge_model(self) -> Judge:
        """Load the judge model."""
        return Judge.from_pretrained(
            'ai-forever/FRIDA-Decisions',
            device=self.device,
        )
        
    def embed_query(self, text: str, prompt: str = 'query: ') -> np.ndarray:
        return self.embed_model.encode(
            text, convert_to_numpy=True,
            normalize_embeddings=True,
            prompt=prompt,#EmbeddingPrompts.get(prompt)
        )
    
    def embed_documents(self, texts: list[str], 
                        prompt: str= 'query: ', 
                        batch_size: int = 128, 
                        progress_bar: bool = False
        ) -> np.ndarray:
        return self.embed_model.encode(
            texts, batch_size=batch_size, 
            prompt=prompt, #EmbeddingPrompts.get(prompt),
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=progress_bar,
        )


    def generate(self, system_prompt: str, user_prompt: str, think: bool=False) -> str:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat(model=self.llm_model, messages=msgs, think=think)
        print("eval_count:", response.get("eval_count"))
        print("eval_sec:", response.get("eval_duration", 0) / 1e9)
        print("tok/s:", response.get("eval_count", 0) / max(response.get("eval_duration", 1) / 1e9, 1e-9))
        print("content:", response["message"]["content"])
        print("thinking:", response["message"].get("thinking", ""))
        return response["message"]["content"]