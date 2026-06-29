from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache

from ..config import settings

logger = logging.getLogger("nids.geoip")

# open the MaxMind db if configured, otherwise stay off
_reader = None
if settings.GEOIP_DB_PATH and settings.GEOIP_DB_PATH.exists():
    try:
        import geoip2.database

        _reader = geoip2.database.Reader(str(settings.GEOIP_DB_PATH))
        logger.info("GeoIP database loaded")
    except Exception as exc:
        logger.warning("GeoIP disabled: %s", exc)


@lru_cache(maxsize=10000)
def lookup(ip: str) -> dict | None:
    if _reader is None:
        return None
    try:
        if ipaddress.ip_address(ip).is_private:
            return None
    except ValueError:
        return None
    try:
        resp = _reader.city(ip)
    except Exception:
        return None
    lat, lng = resp.location.latitude, resp.location.longitude
    if lat is None or lng is None:
        return None
    return {"lat": float(lat), "lng": float(lng), "city": resp.city.name or "Unknown"}
