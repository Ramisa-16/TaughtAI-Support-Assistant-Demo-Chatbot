from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import get_support_response

app = FastAPI()

# Allow frontend to connect (adjust this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with actual frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history (simple in-memory session)
conversation_history = []

# Request schema
class ChatRequest(BaseModel):
    query: str

# ------------------ Endpoints ------------------

@app.post("/chat")
def chat(request: ChatRequest):
    ai_reply, _ = get_support_response(request.query, conversation_history)
    return {"response": ai_reply}