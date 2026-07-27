from fastapi import FastAPI
from pydantic import BaseModel #inherited Class
from backend.rag import ask_question

app=FastAPI()

class Question(BaseModel):  #inheriting the particular qn from base model
    question:str

@app.post("/ask")
def ask(q:Question): #FastAPI automatically converts JSON->Question
    answer=ask_question(q.question)
    return {"answer":answer}
