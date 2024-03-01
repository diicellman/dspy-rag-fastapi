"""DSPy functions."""

import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.teleprompt import BootstrapFewShot
from app.utils.load import OllamaEmbeddingFunction

from typing import Dict

# Global settings
ollama_lm = dspy.OllamaLocal(model="phi:latest")
ollama_embedding_function = OllamaEmbeddingFunction()

retriever_model = ChromadbRM(
    "quickstart",
    "../../data/chroma_db",
    embedding_function=ollama_embedding_function,
    k=5,
)

dspy.settings.configure(lm=ollama_lm, rm=retriever_model)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def compile_rag() -> Dict:
    # Small training set with question and answer pairs
    trainset = [
        dspy.Example(
            question="What were the two main things the author worked on before college?",
            answer="Writing and programming",
        ).with_inputs("question"),
        dspy.Example(
            question="What kind of writing did the author do before college?",
            answer="Short stories",
        ).with_inputs("question"),
        dspy.Example(
            question="What was the first computer language the author learned?",
            answer="Fortran",
        ).with_inputs("question"),
        dspy.Example(
            question="What kind of computer did the author's father buy?",
            answer="TRS-80",
        ).with_inputs("question"),
        dspy.Example(
            question="What was the author's original plan for college?",
            answer="Study philosophy",
        ).with_inputs("question"),
    ]

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)

    # Compile!
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    # Saving
    compiled_rag.save("../../data/checkpoints/compiled_rag.json")

    return {"message": "Successfully compiled RAG program!"}


def get_compiled_rag():
    # Loading:
    rag = RAG()
    rag.load("../../data/checkpoints/compiled_rag.json")

    return rag
