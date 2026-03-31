"""
AstroWeek Backend — FastAPI сервер для Render.
Хранит ANTHROPIC_API_KEY в переменных окружения (не в коде).
Фронтенд отправляет уже рассчитанные натальные данные и транзиты,
бэкенд проксирует запрос к Claude API и возвращает прогноз.
"""

import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="AstroWeek API", version="1.0.0")

# CORS — разрешаем запросы с любого источника (Telegram Mini App)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"


# ── Схемы запроса ──────────────────────────────────────────────

class PlanetInfo(BaseModel):
    name: str
    sign: str
    retrograde: bool = False

class NatalData(BaseModel):
    sun: dict
    moon: dict
    ascendant: dict
    planets: list[PlanetInfo]

class TransitPlanet(BaseModel):
    n: str
    s: str
    sign: str

class AspectInfo(BaseModel):
    transit: str
    transitSign: str
    natal: str
    natalSign: str
    aspect: str
    symbol: str
    orb: str
    angle: int

class ForecastRequest(BaseModel):
    natal: NatalData
    transits: list[TransitPlanet]
    aspects: list[AspectInfo]
    week_start: str
    week_end: str


# ── Эндпоинты ──────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "AstroWeek API"}


@app.post("/api/forecast")
async def generate_forecast(req: ForecastRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY не задан на сервере")

    # Формируем промпт
    natal_desc = ", ".join(
        f"{p.name}: {p.sign}{'  ℞' if p.retrograde else ''}"
        for p in req.natal.planets
    )
    asc_str = ""
    if req.natal.ascendant.get("sign", "неизвестен") != "неизвестен":
        asc_str = f", Асцендент: {req.natal.ascendant['sign']}"

    transit_desc = ", ".join(f"{t.s} {t.n} в {t.sign}" for t in req.transits)

    if req.aspects:
        aspect_desc = "; ".join(
            f"{a.transit} ({a.transitSign}) {a.symbol} натальный {a.natal} ({a.natalSign}), орб {a.orb}°"
            for a in req.aspects[:10]
        )
    else:
        aspect_desc = "значимых транзитных аспектов на этой неделе нет"

    prompt = f"""Ты профессиональный астролог. Составь персональный астрологический прогноз на неделю с {req.week_start} по {req.week_end} на русском языке.

НАТАЛЬНАЯ КАРТА:
☉ Солнце: {req.natal.sun.get('sign')}, ☽ Луна: {req.natal.moon.get('sign')}{asc_str}
Планеты: {natal_desc}

ТРАНЗИТЫ НА ТЕКУЩУЮ НЕДЕЛЮ (реальные позиции планет):
{transit_desc}

КЛЮЧЕВЫЕ АСПЕКТЫ ТРАНЗИТОВ К НАТАЛЬНОЙ КАРТЕ (сортировка по силе):
{aspect_desc}

Опирайся на конкретные аспекты и транзиты при составлении прогноза. Упоминай планеты по имени.
Верни ТОЛЬКО валидный JSON без markdown и без ```json блоков:
{{
  "summary": "2-3 предложения общего прогноза, конкретные планеты и аспекты",
  "favorable_days": ["день недели", ...],
  "career": {{"rating": 1-5, "text": "2-3 предложения"}},
  "relationships": {{"rating": 1-5, "text": "2-3 предложения"}},
  "health": {{"rating": 1-5, "text": "2-3 предложения"}}
}}"""

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    if resp.status_code != 200:
        raise HTTPException(502, f"Claude API вернул {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    raw_text = "".join(b.get("text", "") for b in data.get("content", []))
    # Убираем возможные markdown-обёртки
    clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        forecast = json.loads(clean)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Не удалось разобрать ответ Claude: {e}")

    return forecast
