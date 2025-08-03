from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from ..rag.chain import get_rag_chain

router = APIRouter()

@router.get("/chat")
async def stream_chat(q: str):
    chain = get_rag_chain()
    response = chain.run(q)

    async def event_generator():
        yield {"data": response}
    return EventSourceResponse(event_generator())
