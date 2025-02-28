from fastapi import FastAPI , HTTPException 
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware

import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str

@app.get('/debug-summary')
async def ask_query():
    # reply = "**Summary** The conversation is between a human user and an AI assistant, LexAi. The human asks about the Indian Penal Code (IPC) and specifically inquires about Section 420. The AI responds by explaining the meaning of Section 420, which deals with Cheating and Dishonestly, and provides examples of cases that fall under this section. Additionally, the AI discusses the punishment for theft under IPC and other related sections. **Key Points** * The human user asks about IPC Section 420. * LexAi explains the meaning and application of Section 420. * The AI discusses the punishment for theft under IPC. * Other related sections (486-489 and 376E) are also mentioned. **User Intentions** * The human user seeks information on a specific section of the Indian Penal Code (IPC). * They may be interested in understanding the legal implications of certain actions or behaviors. **AI Responses** * LexAi provides accurate and helpful information on the requested topic. * The AI responds to follow-up questions about other related sections. **Title** python { title: Indian Law }",
    reply="hellooo"
    time.sleep(5)
    return {
        "response":reply,
        "title":"sample-title",
        "lawyer":"example-lawyer",
        "status":"success"
    }
@app.post("/debug-reply")
async def reply_message(request:QueryRequest):
    reply = request.question
    time.sleep(3)
    return {
        "response":reply,
        "status":"success"
    }