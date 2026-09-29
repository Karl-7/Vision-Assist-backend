import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx

# Optional: if python-dotenv is installed, it automatically reads environment variables from a .env file during local development.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()

# Allow your GitHub Pages frontend to call this API across origins. Start with "*" for quick setup,
# and later replace it with your specific domain, for example:
# ["https://karl-7.github.io"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read the key from environment variables; never hardcode it or commit it to Git.
# Local development: create a .env file in the project directory with a line like GEMINI_API_KEY=your_key
# Deployment to Render/Railway or similar platforms: add the same variable in the platform's Environment Variables settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not found. For local development, set it in a .env file; "
        "for production deployment, add it in the platform's Environment Variables settings."
    )

GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_INSTRUCTION = (
    "You are a real-time visual assistant for a blind person navigating their surroundings. "
    "They may ask follow-up questions about the same scene; use the conversation history for context. "
    "Rules for your answer:\n"
    "1. Respond with exactly ONE short sentence in English. No preamble, no extra explanation.\n"
    "2. Give a concrete, actionable instruction, not a description. Prefer clock positions "
    "(e.g. '10 o'clock', '2 o'clock') or simple directions (left / right / straight ahead / slightly up / down), "
    "plus a rough distance when useful (e.g. 'about two steps ahead').\n"
    "3. Safety first: if there is an obstacle, step, drop-off, moving vehicle, or other hazard in the frame, "
    "mention it before anything else, even if it's not what they asked about.\n"
    "4. If what they're looking for is not visible, say so plainly and suggest one specific next action "
    "(e.g. 'No door visible, try turning right') instead of guessing.\n"
    "5. Do not mention colors, aesthetics, or other purely visual details unless the person explicitly asks for them.\n"
    "6. If the image is too blurry, dark, or unclear to answer confidently, say that briefly and suggest "
    "please move the camera a bit."
)


class HistoryTurn(BaseModel):
    role: str          # "user" or "model"
    text: str


class AnalyzeRequest(BaseModel):
    prompt: str
    image_base64: str          # Current frame without the "data:image/jpeg;base64," prefix
    history: List[HistoryTurn] = []


def history_to_gemini_content(turn: HistoryTurn):
    return {"role": turn.role, "parts": [{"text": turn.text}]}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    contents = [history_to_gemini_content(t) for t in req.history]
    contents.append({
        "role": "user",
        "parts": [
            {"text": req.prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": req.image_base64}},
        ],
    })

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": contents,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GEMINI_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                json=body,
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to request Gemini: {e}")

    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=data["error"].get("message", "Gemini API returned an error"))

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        answer = "(No response received)"

    return {"answer": answer}


@app.get("/")
def health():
    return {"status": "ok"}