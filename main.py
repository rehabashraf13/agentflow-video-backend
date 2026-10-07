import os
import asyncio
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from gradio_client import Client


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AgentFlow Video API",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

HF_TOKEN = os.getenv("HF_TOKEN")

ZEROGPU_SPACE = "numanajmal0/wan-video-api"


# =========================================================
# REQUEST MODELS
# =========================================================

class VideoRequest(BaseModel):
    idea: str


class VideoGenerationRequest(BaseModel):
    prompt: str


# =========================================================
# BASIC ROUTES
# =========================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AgentFlow Video",
        "version": "4.0.0",
        "video_backend": "Hugging Face ZeroGPU"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "huggingface_configured": bool(HF_TOKEN),
        "groq_model": GROQ_MODEL,
        "video_space": ZEROGPU_SPACE,
        "video_model": "wan-base"
    }


# =========================================================
# GROQ AGENT FUNCTION
# =========================================================

async def call_agent(
    system_prompt: str,
    user_prompt: str
) -> str:

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

        async with httpx.AsyncClient(
            timeout=60.0
        ) as client:

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

        return (
            data["choices"][0]["message"]["content"]
            .strip()
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


# =========================================================
# MULTI-AGENT PROMPT PIPELINE
# =========================================================

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


    # =====================================================
    # AGENT 1 — SCRIPT
    # =====================================================

    script = await call_agent(
        """
You are the Script Agent in an AI filmmaking system.

Convert the user's idea into ONE visually clear scene
that can happen within exactly five seconds.

Focus on:
- one main subject
- one clear action
- visual storytelling
- physical movement
- a clear beginning and ending

Do not write dialogue.
Do not create multiple scenes.
Do not explain your reasoning.

Return only the short scene description.
""",
        idea
    )


    # =====================================================
    # AGENT 2 — DIRECTOR
    # =====================================================

    direction = await call_agent(
        """
You are the Director Agent in an AI filmmaking system.

Transform the provided five-second scene into
professional visual direction.

Specify:
- environment
- subject appearance
- lighting
- atmosphere
- color palette
- cinematic mood

Preserve the original action.

Do not create another scene.
Do not explain your reasoning.

Return only the visual direction.
""",
        f"""
ORIGINAL IDEA:
{idea}

SCRIPT:
{script}
"""
    )


    # =====================================================
    # AGENT 3 — CINEMATOGRAPHER
    # =====================================================

    camera = await call_agent(
        """
You are the Cinematography Agent in a professional
AI filmmaking system.

Design the cinematography for the provided scene.

Specify:
- shot type
- camera angle
- camera movement
- lens/look
- depth of field
- composition
- motion characteristics

Everything must be achievable in ONE continuous
five-second shot.

Do not change the story.
Do not add scene cuts.
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


    # =====================================================
    # AGENT 4 — PROMPT ENGINEER
    # =====================================================

    final_prompt = await call_agent(
        """
You are the final Prompt Engineer for an advanced
AI text-to-video model.

Combine the supplied script, direction and
cinematography into ONE production-ready
English video-generation prompt.

Requirements:

- exactly one continuous five-second shot
- cinematic 16:9 composition
- explicit main subject
- explicit physical action
- environment
- camera movement
- lighting
- depth
- realistic temporal motion
- coherent physics
- consistent subject appearance
- high visual detail
- professional cinematic quality

Avoid:

- multiple shots
- scene cuts
- text
- subtitles
- logos
- watermarks
- duplicated subjects
- deformed anatomy
- unnecessary adjectives

Return ONLY the final video prompt.

Do not use headings.
Do not explain your reasoning.
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


# =========================================================
# VIDEO AGENT — HUGGING FACE ZEROGPU
# =========================================================

def run_video_generation(prompt: str):

    # Connect to the public Hugging Face ZeroGPU Space.
    #
    # HF_TOKEN allows the request to use the logged-in
    # Hugging Face account quota.

    client = Client(
        ZEROGPU_SPACE,
        token=HF_TOKEN
    )


    # The parameters below match the REAL /generate
    # API schema returned by the Space.

    result = client.predict(

        # Use the normal Wan base model.
        model_key="wan-base",

        # Final prompt produced by our Prompt Agent.
        prompt=prompt,

        # Things we want the model to avoid.
        negative_prompt=(
            "text, subtitles, captions, logos, watermarks, "
            "low quality, blurry image, JPEG artifacts, "
            "distorted anatomy, deformed hands, malformed face, "
            "extra fingers, extra limbs, duplicated subjects, "
            "static image, frozen motion, flickering, "
            "inconsistent appearance, inconsistent motion"
        ),

        # 16:9 landscape video.
        width=832,
        height=480,

        # Valid value according to the Space schema (4n+1).
        num_frames=49,

        # Generation quality settings.
        steps=30,
        guidance_scale=5.0,

        # Random seed.
        seed=-1,

        # Not relevant for wan-base,
        # but required by the endpoint schema.
        lora_scale=1.0,

        # No custom checkpoint.
        custom_ckpt="",

        api_name="/generate"
    )

    return result


# =========================================================
# GENERATE VIDEO ENDPOINT
# =========================================================

@app.post("/generate-video")
async def generate_video(
    request: VideoGenerationRequest
):

    prompt = request.prompt.strip()


    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Video prompt cannot be empty."
        )


    if len(prompt) > 4000:
        raise HTTPException(
            status_code=400,
            detail="Video prompt is too long."
        )


    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured."
        )


    try:

        # gradio_client is synchronous.
        # Run it outside FastAPI's async event loop.

        result = await asyncio.to_thread(
            run_video_generation,
            prompt
        )


        if not result:
            raise RuntimeError(
                "ZeroGPU returned an empty result."
            )


        # =================================================
        # READ GRADIO RESPONSE
        # =================================================
        #
        # According to the /generate API schema,
        # the Space returns:
        #
        # 1. Video file
        # 2. Seed used
        # 3. Status message
        #
        # gradio_client normally returns these as a tuple.

        if isinstance(result, (list, tuple)):

            if len(result) == 0:
                raise RuntimeError(
                    "ZeroGPU returned no output."
                )

            video_result = result[0]

        else:

            # Defensive fallback in case a future
            # Gradio version returns only the file.

            video_result = result


        # =================================================
        # GET LOCAL VIDEO PATH
        # =================================================

        video_path = None


        # Most common gradio_client response:
        # a downloaded temporary filepath.

        if isinstance(video_result, str):

            video_path = video_result


        # Some Gradio versions may return a dictionary.

        elif isinstance(video_result, dict):

            video_path = video_result.get("path")


        # FileData-like object.

        elif hasattr(video_result, "path"):

            video_path = video_result.path


        if not video_path:

            raise RuntimeError(
                "ZeroGPU returned no usable video file path."
            )


        if not os.path.exists(video_path):

            raise RuntimeError(
                f"Generated video file was not downloaded: {video_path}"
            )


        # =================================================
        # READ MP4
        # =================================================

        with open(video_path, "rb") as video_file:

            video_bytes = video_file.read()


        if not video_bytes:

            raise RuntimeError(
                "Generated video file is empty."
            )


        # =================================================
        # RETURN MP4 TO OUR WEBSITE
        # =================================================

        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={
                "Content-Disposition":
                    'inline; filename="agentflow-video.mp4"',

                "Cache-Control":
                    "no-store"
            }
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "ZeroGPU video generation error: "
                f"{str(e)}"
            )
        )
