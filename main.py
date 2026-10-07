import os
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="AgentFlow Video API",
    version="1.0.0"
)

# مؤقتًا أثناء التطوير
# هنقيد الـorigin لاحقًا على Hugging Face Space بتاعنا فقط.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoRequest(BaseModel):
    idea: str


@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AgentFlow Video",
        "message": "Backend is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate-prompt")
async def generate_prompt(request: VideoRequest):

    idea = request.idea.strip()

    if not idea:
        raise HTTPException(
            status_code=400,
            detail="Video idea cannot be empty."
        )

    if len(idea) > 500:
        raise HTTPException(
            status_code=400,
            detail="Video idea is too long."
        )

    return {
        "success": True,
        "idea": idea,

        "agents": {
            "script_agent": {
                "status": "ready"
            },

            "director_agent": {
                "status": "ready"
            },

            "camera_agent": {
                "status": "ready"
            },

            "prompt_agent": {
                "status": "ready"
            }
        },

        "message": "AgentFlow backend pipeline is ready."
    }
