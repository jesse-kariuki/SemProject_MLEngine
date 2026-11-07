"""SMS wording. Every message fits one 160-character segment, because each extra segment costs credit.

The Kiswahili text should be reviewed by a native speaker before launch.
"""
from __future__ import annotations

from app.risk import SEVERE, WARNING, WATCH

SMS_LIMIT = 160
LEVEL_WORD = {
    "en": {WATCH: "WATCH", WARNING: "WARNING", SEVERE: "SEVERE WARNING"},
    "sw": {WATCH: "ANGALIZO", WARNING: "ONYO", SEVERE: "HATARI KUBWA"},
}
FLOOD = {
    "en": "FloodGuard {word}: {place}: {reason}. Avoid flooded roads, rivers, drains.",
    "sw": "FloodGuard {word}: {place}: {reason}. Epuka barabara zilizofurika, mito, mitaro.",
}
RAIN_REASON = {"en": "up to {mm:.0f}mm rain in 24h", "sw": "mvua hadi {mm:.0f}mm kwa saa 24"}
OTHER_REASON = {"en": "flood risk raised", "sw": "hatari ya mafuriko imeongezeka"}
ALL_CLEAR = {
    "en": "FloodGuard: flood risk near {place} has eased. Stay careful around water.",
    "sw": "FloodGuard: hatari ya mafuriko karibu na {place} imepungua. Endelea kuwa mwangalifu.",
}
OTP = {
    "en": "FloodGuard code: {code}. It expires in {minutes} minutes. Do not share it.",
    "sw": "Nambari ya FloodGuard: {code}. Itaisha baada ya dakika {minutes}. Usimpe mtu.",
}
OFFICIAL_PREFIX = {"en": "Kenya Met alert: ", "sw": "Tahadhari ya Kenya Met: "}
DEFAULT_PLACE = {"en": "your area", "sw": "eneo lako"}
RAIN_REASON_MIN_MM = 10.0           # below this, a rain figure would not explain the alert


def _fit(text: str, limit: int = SMS_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _place(place: str | None, language: str, max_len: int = 24) -> str:
    return ((place or "").strip() or DEFAULT_PLACE[language])[:max_len]


def flood_message(level: int, place: str | None, rain_24h_mm: float | None, language: str, opt_out: str = "") -> str:
    reason = (RAIN_REASON[language].format(mm=rain_24h_mm) if (rain_24h_mm or 0) >= RAIN_REASON_MIN_MM
              else OTHER_REASON[language])
    body = FLOOD[language].format(word=LEVEL_WORD[language][level], place=_place(place, language), reason=reason)
    return _fit(f"{body} {opt_out}".strip())


def all_clear_message(place: str | None, language: str) -> str:
    return _fit(ALL_CLEAR[language].format(place=_place(place, language)))


def otp_message(code: str, minutes: int, language: str) -> str:
    return OTP[language].format(code=code, minutes=minutes)


def official_message(headline: str, language: str) -> str:
    return _fit(OFFICIAL_PREFIX[language] + " ".join(headline.split()))
