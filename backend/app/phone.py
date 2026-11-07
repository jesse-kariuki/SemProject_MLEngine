"""Kenyan mobile numbers, normalised to E.164."""
from __future__ import annotations

import re

_LOCAL = re.compile(r"^(?:\+?254|0)?([17]\d{8})$")


def normalise_phone(raw: str) -> str | None:
    """Accept 0712345678, 712345678, 254712345678 or +254712345678. Returns +254... or None."""
    match = _LOCAL.match(re.sub(r"[\s\-()]", "", raw))
    return f"+254{match.group(1)}" if match else None
