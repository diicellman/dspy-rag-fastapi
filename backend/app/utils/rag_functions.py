"""DSPy functions."""

import os

import dspy
import ollama
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.teleprompt import BootstrapFewShot

from app.utils.load import OllamaEmbeddingFunction
from app.utils.rag_modules import RAG
from app.utils.models import MessageData, RAGResponse, QAList

load_dotenv()


from typing import Dict

# Global settings
DATA_DIR = "data"
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "localhost")
ollama_embedding_function = OllamaEmbeddingFunction(host=ollama_base_url)

retriever_model = ChromadbRM(
    "quickstart",
    f"{DATA_DIR}/chroma_db",
    embedding_function=ollama_embedding_function,
    k=5,
)

dspy.settings.configure(rm=retriever_model)


def get_zero_shot_query(payload: MessageData):
    rag = RAG()
    # Global settings
    ollama_lm = dspy.OllamaLocal(
        model=payload.ollama_model_name,
        base_url=ollama_base_url,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    # parsed_chat_history = ", ".join(
    #     [f"{chat['role']}: {chat['content']}" for chat in payload.chat_history]
    # )
    with dspy.context(lm=ollama_lm):
        pred = rag(
            question=payload.query,  # chat_history=parsed_chat_history
        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[c[:200] + "..." for c in pred.context],
    )


def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def compile_rag(qa_items: QAList) -> Dict:
    # Global settings
    ollama_lm = dspy.OllamaLocal(
        model=qa_items.ollama_model_name,
        base_url=ollama_base_url,
        temperature=qa_items.temperature,
        top_p=qa_items.top_p,
        max_tokens=qa_items.max_tokens,
    )

    trainset = [
        dspy.Example(
            question=item.question,
            answer=item.answer,
        ).with_inputs("question")
        for item in qa_items.items
    ]

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    with dspy.context(lm=ollama_lm):
        compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    # Saving
    compiled_rag.save(f"{DATA_DIR}/compiled_rag.json")

    return {"message": "Successfully compiled RAG program!"}


def get_compiled_rag(payload: MessageData):
    # Loading:
    rag = RAG()
    rag.load(f"{DATA_DIR}/compiled_rag.json")

    # Global settings
    ollama_lm = dspy.OllamaLocal(
        model=payload.ollama_model_name,
        base_url=ollama_base_url,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    # parsed_chat_history = ", ".join(
    #     [f"{chat['role']}: {chat['content']}" for chat in payload.chat_history]
    # )
    with dspy.context(lm=ollama_lm):
        pred = rag(
            question=payload.query,  # chat_history=parsed_chat_history
        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[c[:200] + "..." for c in pred.context],
    )


def get_list_ollama_models():
    client = ollama.Client(host=ollama_base_url)

    models = []
    models_list = client.list()
    for model in models_list["models"]:
        models.append(model["name"])

    return {"models": models}
