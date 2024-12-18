import pretty_errors
from fastapi import FastAPI
from pydantic import BaseModel
from backend.worker import conversation_retrieval_chain

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/generate")
async def generate_response(request: QueryRequest):
    return {"response": conversation_retrieval_chain(request.query)}