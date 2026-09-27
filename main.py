import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx

# 本地开发时可选：如果装了 python-dotenv，会自动从 .env 文件读取环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()

# 允许你的 GitHub Pages 前端跨域调用。先用 "*" 跑通，
# 稳定后建议改成你的具体域名，比如：
# ["https://karl-7.github.io"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Key 从环境变量读取，绝不写进代码或提交到 Git。
# 本地跑：在项目目录建一个 .env 文件，写一行 GEMINI_API_KEY=你的key
# 部署到 Render/Railway 等平台：在平台的 Environment Variables 设置里添加同名变量
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "未找到环境变量 GEMINI_API_KEY。本地开发请在 .env 文件里设置，"
        "线上部署请在平台的 Environment Variables 里添加。"
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
    "moving the camera or turning on more light."
)


class HistoryTurn(BaseModel):
    role: str          # "user" 或 "model"
    text: str


class AnalyzeRequest(BaseModel):
    prompt: str
    image_base64: str          # 当前帧，不带 "data:image/jpeg;base64," 前缀
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
        raise HTTPException(status_code=502, detail=f"请求 Gemini 失败：{e}")

    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=data["error"].get("message", "Gemini API 返回错误"))

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        answer = "（没有收到回答）"

    return {"answer": answer}


@app.get("/")
def health():
    return {"status": "ok"}