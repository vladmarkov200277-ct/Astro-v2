"""
AstroWeek Backend v2 — Kerykeion + Claude API
- Натальная карта через Swiss Ephemeris (Kerykeion)
- Транзиты текущей недели через Kerykeion
- Аспекты транзит → натал
- Прогноз через Claude API
- Интерпретация аспектов через Claude API
"""

import os
import json
import httpx
import datetime
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


class AspectsRequest(BaseModel):
    aspects: list[dict]    # список аспектов из /api/chart
    sun_sign: str          # натальный знак Солнца
    week_start: str        # "28.04.2026"
    week_end: str          # "04.05.2026"


# ── Вспомогательные функции ────────────────────────────────────

def get_planet_list(subject) -> list[dict]:
    """Извлекает список планет из субъекта Kerykeion."""
    planets = []
    for key in ["sun", "moon", "mercury", "venus", "mars",
                "jupiter", "saturn", "uranus", "neptune", "pluto"]:
        p = getattr(subject, key, None)
        if p is None:
            continue
        name_en = p.name
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
        hour, minute = 12, 0

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

    prompt = f"""Ты астролог с чувством юмора, пишешь для молодых девушек 18-30 лет. Стиль: дружеский, лёгкий, с иронией — как подруга, которая разбирается в звёздах. Никаких штампов типа "звёзды благоволят" или "будьте осторожны".

НАТАЛЬНАЯ КАРТА (Swiss Ephemeris, Placidus):
☉ Солнце: {sun_sign}, ☽ Луна: {moon_sign}{asc_str}
Планеты: {natal_desc}

ТРАНЗИТЫ НА ТЕКУЩУЮ НЕДЕЛЮ:
{transit_desc}

КЛЮЧЕВЫЕ АСПЕКТЫ (сила → слабее):
{aspect_desc}

ПРАВИЛА:
- Опирайся на конкретные аспекты и транзиты, упоминай планеты по имени
- Текст каждого раздела: 2 коротких предложения максимум, без воды
- Правильные склонения и согласования на русском языке
- Применяющиеся (Applying) аспекты важнее разделяющихся
- Юмор уместный, не пошлый
- favorable_days — только реальные дни когда аспекты позитивны

Верни ТОЛЬКО валидный JSON без markdown, строго эта структура:
{{
  "summary": "1-2 предложения — главная тема недели с конкретной планетой",
  "favorable_days": ["день недели"],
  "love": {{"rating": 3, "text": "2 предложения про романтику и отношения"}},
  "friends": {{"rating": 3, "text": "2 предложения про дружбу и общение"}},
  "career": {{"rating": 3, "text": "2 предложения про работу и учёбу"}},
  "finance": {{"rating": 3, "text": "2 предложения про деньги и покупки"}},
  "health": {{"rating": 3, "text": "2 предложения про энергию и самочувствие"}}
}}"""

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


@app.post("/api/aspects")
async def get_aspects_text(req: AspectsRequest):
    """
    Принимает список транзитных аспектов и генерирует
    через Claude профессиональные русскоязычные интерпретации.
    Вызывается фронтендом асинхронно после /api/chart.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY не задан")

    if not req.aspects:
        return {"aspects": []}

    asp_list = "\n".join(
        f"{i + 1}. {a.get('transit', '')} ({a.get('transitSign', '')}) "
        f"{a.get('symbol', '')} натальный {a.get('natal', '')} ({a.get('natalSign', '')}), "
        f"орб {a.get('orb', '?')}°{', движение: ' + a['movement'] if a.get('movement') else ''}"
        for i, a in enumerate(req.aspects[:5])
    )

    prompt = f"""Ты профессиональный астролог. Напиши интерпретацию транзитных аспектов недели для человека с натальным Солнцем в знаке {req.sun_sign}. Период: {req.week_start} — {req.week_end}.

Аспекты (от сильнейшего к слабейшему по орбу):
{asp_list}

Требования к каждому аспекту:
- Заголовок: одно чёткое предложение на русском с правильными падежами и родом. Формат: "Транзитный [планета] [аспект] натальный/натальную/натальное [планета]". Примеры: "Транзитный Марс образует квадратуру к натальному Сатурну", "Транзитная Венера формирует трин с натальной Луной", "Транзитное Солнце соединяется с натальным Меркурием"
- Текст: 2 конкретных предложения — что означает этот аспект на практике и как с ним работать на этой неделе. Без эзотерики и штампов, пишите как грамотный специалист. Используй "вы" и правильные падежи.

Верни ТОЛЬКО валидный JSON-массив без markdown-разметки:
[
  {{"title": "...", "text": "..."}},
  {{"title": "...", "text": "..."}},
  ...
]"""

    result = await call_claude(prompt)

    if not isinstance(result, list):
        raise HTTPException(500, "Claude вернул неожиданный формат")

    enriched = []
    for i, asp in enumerate(req.aspects[:5]):
        claude_data = result[i] if i < len(result) else {}
        enriched.append({
            **asp,
            "title": claude_data.get("title", f"{asp.get('transit', '')} {asp.get('symbol', '')} {asp.get('natal', '')}"),
            "text": claude_data.get("text", ""),
        })

    return {"aspects": enriched}
