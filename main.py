"""
AstroWeek Backend v2 — Kerykeion + Claude API
- Натальная карта через Swiss Ephemeris (Kerykeion)
- Транзиты текущей недели через Kerykeion
- Аспекты транзит → натал
- Прогноз через Claude API
"""

import os
import json
import httpx
import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from kerykeion import AstrologicalSubjectFactory, AspectsFactory

app = FastAPI(title="AstroWeek API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Русские названия знаков
SIGN_RU = {
    "Ari": "Овен", "Tau": "Телец", "Gem": "Близнецы", "Can": "Рак",
    "Leo": "Лев", "Vir": "Дева", "Lib": "Весы", "Sco": "Скорпион",
    "Sag": "Стрелец", "Cap": "Козерог", "Aqu": "Водолей", "Pis": "Рыбы",
}

# Русские названия планет
PLANET_RU = {
    "Sun": "Солнце", "Moon": "Луна", "Mercury": "Меркурий",
    "Venus": "Венера", "Mars": "Марс", "Jupiter": "Юпитер",
    "Saturn": "Сатурн", "Uranus": "Уран", "Neptune": "Нептун",
    "Pluto": "Плутон",
}

PLANET_EMOJI = {
    "Sun": "☉", "Moon": "☽", "Mercury": "☿", "Venus": "♀",
    "Mars": "♂", "Jupiter": "♃", "Saturn": "♄", "Uranus": "♅",
    "Neptune": "♆", "Pluto": "♇",
}

ASPECT_RU = {
    "conjunction": "конъюнкция", "opposition": "оппозиция",
    "trine": "трин", "square": "квадрат", "sextile": "секстиль",
}

ASPECT_SYMBOL = {
    "conjunction": "☌", "opposition": "☍", "trine": "△",
    "square": "□", "sextile": "⚹",
}

ASPECT_ANGLE = {
    "conjunction": 0, "opposition": 180, "trine": 120,
    "square": 90, "sextile": 60,
}


# ── Схемы запроса ──────────────────────────────────────────────

class BirthRequest(BaseModel):
    birth_date: str        # "ДД.ММ.ГГГГ"
    birth_time: str        # "ЧЧ:ММ" или ""
    lat: float             # широта
    lon: float             # долгота
    timezone: str          # IANA-зона, например "Europe/Moscow"
    city_name: str         # для отображения


# ── Вспомогательные функции ────────────────────────────────────

def get_planet_list(subject) -> list[dict]:
    """Извлекает список планет из субъекта Kerykeion."""
    planets = []
    for key in ["sun", "moon", "mercury", "venus", "mars",
                "jupiter", "saturn", "uranus", "neptune", "pluto"]:
        p = getattr(subject, key, None)
        if p is None:
            continue
        name_en = p.name  # "Sun", "Moon", etc.
        planets.append({
            "name_en": name_en,
            "name": PLANET_RU.get(name_en, name_en),
            "symbol": PLANET_EMOJI.get(name_en, ""),
            "sign": SIGN_RU.get(p.sign, p.sign),
            "sign_en": p.sign,
            "abs_pos": round(p.abs_pos, 4),
            "position": round(p.position, 4),
            "house": getattr(p, "house", None),
            "retrograde": bool(getattr(p, "retrograde", False)),
        })
    return planets


def get_ascendant(subject) -> Optional[dict]:
    """Возвращает асцендент."""
    asc = getattr(subject, "first_house", None)
    if asc is None:
        return None
    return {
        "sign": SIGN_RU.get(asc.sign, asc.sign),
        "sign_en": asc.sign,
        "abs_pos": round(asc.abs_pos, 4),
        "position": round(asc.position, 4),
    }


def calc_transit_aspects(natal_subject, transit_subject) -> list[dict]:
    """Считает аспекты транзит → натал через Kerykeion AspectsFactory."""
    result = AspectsFactory.dual_chart_aspects(transit_subject, natal_subject)
    aspects = []
    for asp in result.aspects:
        asp_key = asp.aspect.lower()
        if asp_key not in ASPECT_ANGLE:
            continue
        aspects.append({
            "transit": PLANET_RU.get(asp.p1_name, asp.p1_name),
            "transit_en": asp.p1_name,
            "transitSign": SIGN_RU.get(
                getattr(asp, "p1_sign", ""), getattr(asp, "p1_sign", "")
            ),
            "natal": PLANET_RU.get(asp.p2_name, asp.p2_name),
            "natal_en": asp.p2_name,
            "natalSign": SIGN_RU.get(
                getattr(asp, "p2_sign", ""), getattr(asp, "p2_sign", "")
            ),
            "aspect": ASPECT_RU.get(asp_key, asp_key),
            "symbol": ASPECT_SYMBOL.get(asp_key, ""),
            "angle": ASPECT_ANGLE[asp_key],
            "orb": str(round(asp.orbit, 1)),
            "movement": getattr(asp, "aspect_movement", ""),
        })
    # Сортируем по орбу (слабейший орб = сильнейший аспект)
    aspects.sort(key=lambda a: float(a["orb"]))
    return aspects[:12]


def week_bounds() -> tuple[str, str, datetime.date, datetime.date]:
    """Возвращает даты начала и конца текущей недели."""
    today = datetime.date.today()
    mon = today - datetime.timedelta(days=today.weekday())
    sun = mon + datetime.timedelta(days=6)
    fmt = lambda d: d.strftime("%d.%m.%Y")
    return fmt(mon), fmt(sun), mon, sun


async def call_claude(prompt: str) -> dict:
    """Вызывает Claude API и возвращает распарсенный JSON."""
    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 1200,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"Claude API: {resp.status_code}")
    raw = "".join(b.get("text", "") for b in resp.json().get("content", []))
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Не удалось разобрать ответ Claude: {e}")


# ── Эндпоинты ──────────────────────────────────────────────────

@app.get("/")
@app.head("/")
def health():
    return {"status": "ok", "service": "AstroWeek API v2"}


@app.post("/api/chart")
async def get_chart(req: BirthRequest):
    """
    Основной эндпоинт: принимает данные рождения,
    возвращает натальную карту + транзиты + аспекты + прогноз Claude.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY не задан")

    # 1. Парсим дату и время рождения
    try:
        day, month, year = map(int, req.birth_date.split("."))
    except Exception:
        raise HTTPException(400, "Неверный формат даты. Ожидается ДД.ММ.ГГГГ")

    has_time = bool(req.birth_time and ":" in req.birth_time)
    if has_time:
        try:
            hour, minute = map(int, req.birth_time.split(":"))
        except Exception:
            raise HTTPException(400, "Неверный формат времени. Ожидается ЧЧ:ММ")
    else:
        hour, minute = 12, 0  # полдень по умолчанию

    # 2. Натальная карта через Kerykeion (Swiss Ephemeris)
    try:
        natal = AstrologicalSubjectFactory.from_birth_data(
            name="Native",
            year=year, month=month, day=day,
            hour=hour, minute=minute,
            lat=req.lat,
            lng=req.lon,
            tz_str=req.timezone,
            online=False,
        )
    except Exception as e:
        raise HTTPException(500, f"Ошибка расчёта натальной карты: {e}")

    natal_planets = get_planet_list(natal)
    ascendant = get_ascendant(natal) if has_time else None

    # 3. Транзиты — середина текущей недели (среда, полдень UTC)
    week_start, week_end, mon_date, _ = week_bounds()
    wed_date = mon_date + datetime.timedelta(days=2)

    try:
        transit = AstrologicalSubjectFactory.from_birth_data(
            name="Transit",
            year=wed_date.year, month=wed_date.month, day=wed_date.day,
            hour=12, minute=0,
            lat=req.lat,
            lng=req.lon,
            tz_str="UTC",
            online=False,
        )
    except Exception as e:
        raise HTTPException(500, f"Ошибка расчёта транзитов: {e}")

    transit_planets = get_planet_list(transit)

    # 4. Аспекты транзит → натал
    try:
        aspects = calc_transit_aspects(natal, transit)
    except Exception as e:
        aspects = []
        print(f"Ошибка расчёта аспектов: {e}")

    # 5. Прогноз через Claude
    sun_sign = next((p["sign"] for p in natal_planets if p["name_en"] == "Sun"), "—")
    moon_sign = next((p["sign"] for p in natal_planets if p["name_en"] == "Moon"), "—")
    asc_str = f", Асцендент: {ascendant['sign']}" if ascendant else ""

    natal_desc = ", ".join(
        f"{p['name']}: {p['sign']}{'  ℞' if p['retrograde'] else ''}"
        for p in natal_planets
    )
    transit_desc = ", ".join(
        f"{p['symbol']} {p['name']} в {p['sign']}"
        for p in transit_planets
    )

    if aspects:
        aspect_desc = "; ".join(
            f"{a['transit']} ({a['transitSign']}) {a['symbol']} натальный {a['natal']} ({a['natalSign']}), "
            f"орб {a['orb']}°{' [' + a['movement'] + ']' if a.get('movement') else ''}"
            for a in aspects[:10]
        )
    else:
        aspect_desc = "значимых транзитных аспектов на этой неделе нет"

    prompt = f"""Ты профессиональный астролог. Составь персональный астрологический прогноз на неделю с {week_start} по {week_end} на русском языке.

НАТАЛЬНАЯ КАРТА (Swiss Ephemeris, Placidus):
☉ Солнце: {sun_sign}, ☽ Луна: {moon_sign}{asc_str}
Планеты: {natal_desc}

ТРАНЗИТЫ НА ТЕКУЩУЮ НЕДЕЛЮ (реальные позиции планет, середина недели):
{transit_desc}

КЛЮЧЕВЫЕ АСПЕКТЫ ТРАНЗИТОВ К НАТАЛЬНОЙ КАРТЕ (сортировка по силе орба):
{aspect_desc}

Инструкции:
- Опирайся ТОЛЬКО на конкретные аспекты и транзиты выше
- Упоминай планеты и знаки по имени
- Учитывай применяющиеся (Applying) аспекты как более важные
- Пиши живым языком, без штампов

Верни ТОЛЬКО валидный JSON без markdown:
{{"summary":"2-3 предложения общего прогноза с конкретными планетами","favorable_days":["день недели"],"career":{{"rating":3,"text":"2-3 предложения"}},"relationships":{{"rating":3,"text":"2-3 предложения"}},"health":{{"rating":3,"text":"2-3 предложения"}}}}"""

    forecast = await call_claude(prompt)

    # 6. Формируем финальный ответ
    return {
        "natal": {
            "sun": {"sign": sun_sign},
            "moon": {"sign": moon_sign},
            "ascendant": ascendant or {"sign": "неизвестен"},
            "planets": natal_planets,
        },
        "transits": transit_planets,
        "aspects": aspects,
        "forecast": forecast,
        "week_start": week_start,
        "week_end": week_end,
        "_meta": {
            "has_time": has_time,
            "city": req.city_name,
            "timezone": req.timezone,
            "engine": "kerykeion+swisseph",
        }
    }
