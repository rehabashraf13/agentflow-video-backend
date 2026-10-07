import os
import json
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="AgentFlow Video API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# If Groq changes/retire models later, this can be changed
# from Render without modifying the source code.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


class VideoRequest(BaseModel):
    idea: str


class VideoGenerationRequest(BaseModel):
    prompt: str
@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AgentFlow Video",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY)
    }


async def call_agent(system_prompt: str, user_prompt: str) -> str:

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.7,
        "max_tokens": 350,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }

    try:

        async with httpx.AsyncClient(timeout=60.0) as client:

            response = await client.post(
                GROQ_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Groq API error: {response.text}"
            )

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


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

    # -------------------------
    # AGENT 1 — SCRIPT WRITER
    # -------------------------

    script = await call_agent(
        """
You are the Script Agent in an AI filmmaking system.

Convert the user's idea into ONE visually clear scene that can
actually happen within exactly five seconds.

Focus on:
- one main subject
- one clear action
- visual storytelling
- physical movement
- beginning and end of the 5-second shot

Do not write dialogue.
Do not write multiple scenes.
Do not explain your reasoning.

Return only the short scene description.
""",
        idea
    )

    # -------------------------
    # AGENT 2 — DIRECTOR
    # -------------------------

    direction = await call_agent(
        """
You are the Director Agent in an AI filmmaking system.

Take the five-second scene written by the Script Agent and create
a concise visual direction.

Specify:
- environment
- subject appearance
- lighting
- atmosphere
- color palette
- cinematic mood

Preserve the original action.
Do not add another scene.
Do not explain your reasoning.

Return only the visual direction.
""",
        script
    )

    # -------------------------
    # AGENT 3 — CINEMATOGRAPHER
    # -------------------------

    camera = await call_agent(
        """
You are the Cinematography Agent for a professional AI video system.

Design the camera treatment for the provided five-second scene.

Specify:
- shot type
- camera angle
- camera movement
- lens/look
- depth of field
- composition
- motion characteristics

The movement must be achievable in one continuous five-second shot.

Do not change the story.
Do not explain your reasoning.

Return only the cinematography instructions.
""",
        f"""
SCENE:
{script}

DIRECTOR:
{direction}
"""
    )

    # -------------------------
    # AGENT 4 — VIDEO PROMPT
    # -------------------------

    final_prompt = await call_agent(
        """
You are the final Prompt Engineer for a state-of-the-art
text-to-video model.

Combine the provided scene, direction and cinematography into
ONE production-ready English video-generation prompt.

Requirements:
- one continuous five-second shot
- 16:9 cinematic composition
- explicit subject and action
- environment
- camera movement
- lighting
- realistic temporal motion
- coherent physics
- high visual detail
- cinematic quality

Avoid:
- multiple shots
- scene cuts
- text
- subtitles
- logos
- watermarks
- duplicated subjects
- unnecessary adjectives

Return ONLY the final video prompt.
Do not use headings.
Do not explain anything.
""",
        f"""
ORIGINAL IDEA:
{idea}

SCRIPT AGENT:
{script}

DIRECTOR AGENT:
{direction}

CAMERA AGENT:
{camera}
"""
    )

    return {
        "success": True,
        "idea": idea,

        "agents": {

            "script_agent": {
                "status": "completed",
                "output": script
            },

            "director_agent": {
                "status": "completed",
                "output": direction
            },

            "camera_agent": {
                "status": "completed",
                "output": camera
            },

            "prompt_agent": {
                "status": "completed",
                "output": final_prompt
            }
        },

        "final_prompt": final_prompt
    }


from fastapi.responses import Response


HF_TOKEN = os.getenv("HF_TOKEN")

HF_VIDEO_MODEL = os.getenv(
    "HF_VIDEO_MODEL",
    "Wan-AI/Wan2.2-TI2V-5B"
)


@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):

    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Video prompt cannot be empty."
        )

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured."
        )

    url = (
        "https://router.huggingface.co/"
        f"hf-inference/models/{HF_VIDEO_MODEL}"
    )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "num_frames": 81,
            "num_inference_steps": 20,
            "guidance_scale": 5.0
        }
    }

    try:

        async with httpx.AsyncClient(
            timeout=300.0
        ) as client:

            response = await client.post(
                url,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:

            raise HTTPException(
                status_code=response.status_code,
                detail=f"Hugging Face error: {response.text}"
            )

        return Response(
            content=response.content,
            media_type="video/mp4",
            headers={
                "Content-Disposition":
                'inline; filename="agentflow-video.mp4"'
            }
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Video generation error: {str(e)}"
        )
