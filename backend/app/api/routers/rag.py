"""Endpoints."""

from fastapi import APIRouter

from app.utils.rag_modules import RAG, compile_rag, get_compiled_rag

rag_router = APIRouter()


@rag_router.get("/healthcheck")
async def healthcheck():

    return {"message": "Thanks for playing."}


@rag_router.get("/zero-shot-query")
async def zero_shot_query(query: str):
    rag = RAG()
    pred = rag(query)

    return {
        "question": query,
        "predicted answer": pred.answer,
        "retrieved contexts (truncated)": [c[:200] + "..." for c in pred.context],
    }


@rag_router.get("/compiled-query")
async def compiled_query(query: str):
    compiled_rag = get_compiled_rag()
    pred = compiled_rag(query)

    return {
        "question": query,
        "predicted answer": pred.answer,
        "retrieved contexts (truncated)": [c[:200] + "..." for c in pred.context],
    }


@rag_router.post("/compile-program")
async def compile_program():
    return compile_rag()
