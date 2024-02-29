"""Load data functions."""

import logging
import os
from typing import Dict
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

DATA_DIR = "data"


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, host: str, model_name: str = "nomic-embed-text"):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. Please install it with `pip install ollama`"
            )

        self._client = ollama.Client(host)
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        # Call Ollama Embedding API for each document.
        for document in input:
            embedding = self._client.embeddings(model=self._model_name, prompt=document)
            embeddings.append(embedding["embedding"])

        return embeddings


def load_data() -> Dict:
    client = chromadb.PersistentClient(path=f"{DATA_DIR}/chroma_db")

    return None
