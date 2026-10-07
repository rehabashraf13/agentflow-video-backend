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
    version="4.2.0"
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
        "version": "4.2.0",
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
# EMPTY + TRUNCATED OUTPUT RETRY
# =========================================================

async def call_agent(
    system_prompt: str,
    user_prompt: str,
    agent_name: str,
    min_length: int = 40,
    max_retries: int = 4
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

    last_error = None

    for attempt in range(1, max_retries + 1):

        # Add an explicit retry instruction after
        # an empty/truncated response.
        retry_instruction = ""

        if attempt > 1:
            retry_instruction = f"""

IMPORTANT RETRY INSTRUCTION:
Your previous response was empty or incomplete.

Return a COMPLETE response now.

Do not stop mid-sentence.
Do not return an empty response.
The response must contain at least {min_length} characters.
"""

        payload = {
            "model": GROQ_MODEL,

            "temperature": 0.55,

            # More room for the reasoning model.
            "max_tokens": 1000,

            "messages": [
                {
                    "role": "system",
                    "content":
                        system_prompt +
                        retry_instruction
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        try:

            async with httpx.AsyncClient(
                timeout=120.0
            ) as client:

                response = await client.post(
                    GROQ_URL,
                    headers=headers,
                    json=payload
                )


            # =============================================
            # HTTP ERROR
            # =============================================

            if response.status_code != 200:

                last_error = (
                    f"Groq HTTP {response.status_code}: "
                    f"{response.text}"
                )

                if attempt < max_retries:

                    await asyncio.sleep(attempt)
                    continue

                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"{agent_name} failed after "
                        f"{max_retries} attempts. "
                        f"{last_error}"
                    )
                )


            # =============================================
            # PARSE RESPONSE
            # =============================================

            data = response.json()


            try:

                choice = data["choices"][0]

                message = choice.get("message", {})

                content = message.get("content", "")

                finish_reason = choice.get(
                    "finish_reason"
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                AttributeError
            ):

                content = ""
                finish_reason = None


            if content is None:
                content = ""

            if not isinstance(content, str):
                content = str(content)

            content = content.strip()


            # =============================================
            # VALIDATION 1 — EMPTY OUTPUT
            # =============================================

            if not content:

                last_error = (
                    f"{agent_name} returned empty output "
                    f"on attempt {attempt}."
                )

                if attempt < max_retries:

                    await asyncio.sleep(attempt)
                    continue

                break


            # =============================================
            # VALIDATION 2 — TOO SHORT / TRUNCATED
            # =============================================

            if len(content) < min_length:

                last_error = (
                    f"{agent_name} returned only "
                    f"{len(content)} characters "
                    f"on attempt {attempt}."
                )

                if attempt < max_retries:

                    await asyncio.sleep(attempt)
                    continue

                break


            # =============================================
            # VALIDATION 3 — TOKEN LIMIT
            # =============================================

            if finish_reason == "length":

                last_error = (
                    f"{agent_name} response was truncated "
                    f"because the token limit was reached."
                )

                if attempt < max_retries:

                    await asyncio.sleep(attempt)
                    continue

                break


            # =============================================
            # SUCCESS
            # =============================================

            return content


        except HTTPException:
            raise


        except Exception as e:

            last_error = (
                f"{agent_name} error: {str(e)}"
            )

            if attempt < max_retries:

                await asyncio.sleep(attempt)
                continue


    # =============================================
    # ALL RETRIES FAILED
    # =============================================

    raise HTTPException(
        status_code=502,
        detail=(
            f"{agent_name} failed to produce a complete "
            f"response after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
    )


# =========================================================
# MULTI-AGENT PROMPT PIPELINE
# =========================================================

@app.post("/generate-prompt")
async def generate_prompt(
    request: VideoRequest
):

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

        system_prompt="""
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

Your response must be complete.

Never stop mid-sentence.

Return only the short scene description.
""",

        user_prompt=idea,

        agent_name="Script Agent",

        min_length=60
    )


    # =====================================================
    # AGENT 2 — DIRECTOR
    # =====================================================

    direction = await call_agent(

        system_prompt="""
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

Your response must be complete.

Never stop mid-sentence.

Return only the visual direction.
""",

        user_prompt=f"""
ORIGINAL IDEA:
{idea}

SCRIPT:
{script}
""",

        agent_name="Director Agent",

        min_length=100
    )


    # =====================================================
    # AGENT 3 — CINEMATOGRAPHER
    # =====================================================

    camera = await call_agent(

        system_prompt="""
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

Your response must be complete.

Never stop mid-sentence.

Return only the cinematography instructions.
""",

        user_prompt=f"""
ORIGINAL IDEA:
{idea}

SCENE:
{script}

DIRECTOR:
{direction}
""",

        agent_name="Camera Agent",

        min_length=100
    )


    # =====================================================
    # AGENT 4 — PROMPT ENGINEER
    # =====================================================

    final_prompt = await call_agent(

        system_prompt="""
You are the final Prompt Engineer for an advanced
AI text-to-video model.

Combine the supplied script, direction and
cinematography into ONE production-ready English
video-generation prompt.

The prompt must describe exactly ONE continuous
five-second shot.

Include:

- the main subject
- the subject's physical action
- environment
- composition
- camera angle
- camera movement
- lighting
- depth of field
- realistic temporal motion
- coherent physics
- consistent subject appearance
- cinematic visual quality

The final prompt must be detailed enough for a
text-to-video model to understand the entire shot.

Avoid:

- multiple shots
- scene cuts
- dialogue
- text
- subtitles
- logos
- watermarks
- duplicated subjects
- deformed anatomy

CRITICAL:

Return a COMPLETE final video prompt.

Never return an empty response.

Never return only a fragment.

Never stop mid-sentence.

Return ONLY the final English video prompt.

Do not use headings.

Do not explain your reasoning.
""",

        user_prompt=f"""
ORIGINAL IDEA:
{idea}

SCRIPT AGENT:
{script}

DIRECTOR AGENT:
{direction}

CAMERA AGENT:
{camera}
""",

        agent_name="Prompt Agent",

        # Much stricter validation for the final prompt.
        min_length=180
    )


    # =====================================================
    # FINAL PROMPT VALIDATION
    # =====================================================

    final_prompt = final_prompt.strip()


    if len(final_prompt) < 180:

        raise HTTPException(
            status_code=502,
            detail=(
                "Final video prompt did not pass "
                "quality validation."
            )
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

def run_video_generation(
    prompt: str
):

    client = Client(
        ZEROGPU_SPACE,
        token=HF_TOKEN
    )


    result = client.predict(

        model_key="wan-base",

        prompt=prompt,

        negative_prompt=(
            "text, subtitles, captions, logos, watermarks, "
            "low quality, blurry image, JPEG artifacts, "
            "distorted anatomy, deformed hands, malformed face, "
            "extra fingers, extra limbs, duplicated subjects, "
            "static image, frozen motion, flickering, "
            "inconsistent appearance, inconsistent motion"
        ),

        width=832,

        height=480,

        num_frames=49,

        steps=30,

        guidance_scale=5.0,

        seed=-1,

        lora_scale=1.0,

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

        result = await asyncio.to_thread(
            run_video_generation,
            prompt
        )


        if not result:

            raise RuntimeError(
                "ZeroGPU returned an empty result."
            )


        # =================================================
        # EXTRACT VIDEO
        # =================================================

        if isinstance(
            result,
            (list, tuple)
        ):

            if len(result) == 0:

                raise RuntimeError(
                    "ZeroGPU returned no output."
                )

            video_result = result[0]

        else:

            video_result = result


        # =================================================
        # EXTRACT VIDEO FILE PATH
        # =================================================

        video_path = None


        if isinstance(
            video_result,
            str
        ):

            video_path = video_result


        elif isinstance(
            video_result,
            dict
        ):

            video_path = video_result.get(
                "path"
            )


        elif hasattr(
            video_result,
            "path"
        ):

            video_path = video_result.path


        if not video_path:

            raise RuntimeError(
                "ZeroGPU returned no usable "
                "video file path."
            )


        if not os.path.exists(
            video_path
        ):

            raise RuntimeError(
                "Generated video file was "
                f"not downloaded: {video_path}"
            )


        # =================================================
        # READ MP4
        # =================================================

        with open(
            video_path,
            "rb"
        ) as video_file:

            video_bytes = (
                video_file.read()
            )


        if not video_bytes:

            raise RuntimeError(
                "Generated video file is empty."
            )


        # =================================================
        # RETURN MP4
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
