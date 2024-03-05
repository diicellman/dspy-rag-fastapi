"""Load data functions."""

import logging
import os
import re

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
DATA_DIR = "data"
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "localhost")


# Custom Embedding function that supports Ollama embeddings
class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to get embeddings for a list of texts using Ollama Python Library.
    It requires a host url and a model name. The default model name is "nomic-embed-text".
    """

    def __init__(
        self, host: str = "http://localhost:11434", model_name: str = "nomic-embed-text"
    ):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. Please install it with `pip install ollama`"
            )

        self._client = ollama.Client(host)
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.
        Args:
            input (Documents): A list of texts to get embeddings for.
        Returns:
            Embeddings: The embeddings for the texts.
        Example:
            >>> ollama = OllamaEmbeddingFunction(host="http://localhost:11434")
            >>> texts = ["Hello, world!", "How are you?"]
            >>> embeddings = ollama(texts)
        """

        embeddings = []
        # Call Ollama Embedding API for each document.
        for document in input:
            embedding = self._client.embeddings(model=self._model_name, prompt=document)
            embeddings.append(embedding["embedding"])

        return embeddings


def load_data() -> None:
    """
    Loads data from /data/example to Chroma Vector store.
    """

    logger.info("Loading data.")
    # Split document into single sentences
    chunks = []
    with open(
        f"{DATA_DIR}/example/paul_graham_essay.txt", "r", encoding="utf-8"
    ) as file:
        text = file.read()
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        chunks.extend(sentences)

    logger.info("Creating embeddings.")
    ollama_ef = OllamaEmbeddingFunction(host=ollama_base_url)
    chunks_embeddings = ollama_ef(chunks)

    db = chromadb.PersistentClient(path=f"{DATA_DIR}/chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    logger.info("Loading data in Chroma.")
    chroma_collection.add(
        ids=[f"id{i}" for i in range(1, len(chunks) + 1)],
        embeddings=chunks_embeddings,
        documents=chunks,
    )
    logger.info("Successfully loaded embeddings in the Chroma.")


if __name__ == "__main__":
    load_data()
